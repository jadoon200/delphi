"""Read-only FastAPI surface over the baked snapshot.

Hardened the way the sibling projects are: no mutation endpoints, bounded responses,
explicit CORS, and **every response that shows a comparison carries its assumptions as
structured data** so a client cannot render the numbers without them.

The one computational endpoint is ``POST /diagnostic/score``. It runs the predictability
diagnostic on a posted series — an autocorrelation profile and a newsvendor sizing — which
is cheap, stateless, and needs no model. That is deliberate: the diagnostic is the most
useful thing this project produced, and it is the part that costs nothing to run.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from delphi.api.snapshot import (
    DemandPoint,
    DemandSeriesPayload,
    DiagnosticBand,
    Snapshot,
    classify,
    classify_band,
)
from delphi.config import get_settings
from delphi.control.newsvendor import CostRatio
from delphi.forecast.predictability import optional_lag_autocorrelation

SNAPSHOT_ENV = "DELPHI_SNAPSHOT_PATH"
DEFAULT_SNAPSHOT = Path("data/snapshot.json")
MAX_SERIES_POINTS = 4000
MAX_POSTED_POINTS = 20_000


@lru_cache(maxsize=1)
def load_snapshot() -> Snapshot:
    settings = get_settings()
    path = Path(getattr(settings, "snapshot_path", None) or DEFAULT_SNAPSHOT)
    if not path.exists():
        raise RuntimeError(
            f"snapshot not found at {path}; run `python scripts/build_snapshot.py` first"
        )
    return Snapshot.model_validate(json.loads(path.read_text()))


app = FastAPI(
    title="DELPHI",
    version="0.1.0",
    description=(
        "Read-only capacity-planning decision support. Predicts whether a workload is "
        "forecastable at all, then sizes capacity from the price of failure rather than "
        "from convention."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness plus an honest statement of what is being served."""
    try:
        snapshot = load_snapshot()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "snapshot_mode": snapshot.snapshot_mode,
        "generated_at": snapshot.generated_at.isoformat(),
        "workloads": len(snapshot.workloads),
        "delphi_version": snapshot.delphi_version,
    }


@app.get("/assumptions")
def assumptions() -> dict[str, list[str]]:
    return {"assumptions": load_snapshot().assumptions}


@app.get("/workloads")
def workloads() -> dict[str, object]:
    snapshot = load_snapshot()
    return {
        "snapshot_mode": snapshot.snapshot_mode,
        "generated_at": snapshot.generated_at,
        "workloads": snapshot.workloads,
    }


@app.get("/workloads/{workload_id}")
def workload_detail(workload_id: str) -> dict[str, object]:
    snapshot = load_snapshot()
    summary = snapshot.workload(workload_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"unknown workload {workload_id!r}")
    return {
        "workload": summary,
        "horizons": snapshot.horizons.get(workload_id, []),
        "frontier": snapshot.frontiers.get(workload_id, []),
        "assumptions": snapshot.assumptions,
    }


@app.get("/demand/{workload_id}")
def demand(
    workload_id: str,
    limit: Annotated[int, Query(ge=100, le=MAX_SERIES_POINTS)] = 1500,
) -> DemandSeriesPayload:
    """Demand series, downsampled to a bounded number of points."""
    snapshot = load_snapshot()
    if snapshot.workload(workload_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown workload {workload_id!r}")
    path = Path(f"data/series/{workload_id}.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="series not baked into this snapshot")
    payload = DemandSeriesPayload.model_validate(json.loads(path.read_text()))
    if len(payload.points) > limit:
        factor = len(payload.points) // limit + 1
        payload = DemandSeriesPayload(
            workload_id=payload.workload_id,
            step_seconds=payload.step_seconds * factor,
            downsample_factor=factor,
            points=payload.points[::factor],
        )
    return payload


@app.get("/findings")
def findings() -> dict[str, object]:
    """The pre-registered questions and what the measurements actually said."""
    snapshot = load_snapshot()
    return {"findings": snapshot.findings, "assumptions": snapshot.assumptions}


class DiagnosticRequest(BaseModel):
    """A demand series to diagnose, plus the economics of getting it wrong."""

    values: list[float] = Field(..., min_length=48, max_length=MAX_POSTED_POINTS)
    step_seconds: int = Field(..., gt=0, le=86400)
    underage_cost: float = Field(
        19.0, gt=0, description="Cost of one unit of unmet demand for one step."
    )
    overage_cost: float = Field(
        1.0, gt=0, description="Cost of one unit of idle capacity for one step."
    )
    capacity_per_replica: float = Field(1.0, gt=0)


class DiagnosticResponse(BaseModel):
    bins: int
    days: float
    minute_autocorrelation: float | None
    daily_autocorrelation: float | None
    weekly_autocorrelation: float | None
    peak_to_mean: float
    forecastable: bool
    band: DiagnosticBand
    verdict: str
    q_star: float
    demand_at_q_star: float
    recommended_replicas: int
    notes: list[str]


@app.post("/diagnostic/score")
def score(request: Annotated[DiagnosticRequest, Body()]) -> DiagnosticResponse:
    """Diagnose a posted series and size it from the cost ratio.

    Stateless and model-free: an autocorrelation profile plus the newsvendor identity. This
    is the cheapest useful thing the project does, and the answer it gives — *is this
    workload forecastable at all* — is the one worth having before building anything.
    """
    values = np.asarray(request.values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise HTTPException(status_code=422, detail="demand must be finite and non-negative")
    if float(np.std(values)) <= 0:
        raise HTTPException(
            status_code=422, detail="a constant series has no structure to diagnose"
        )

    def autocorrelation(lag: int) -> float | None:
        return optional_lag_autocorrelation(values, lag)

    per_day = max(round(86400 / request.step_seconds), 1)
    daily = autocorrelation(per_day)
    ratio = CostRatio(
        underage_per_unit=request.underage_cost, overage_per_unit=request.overage_cost
    )
    q_star = ratio.critical_ratio
    demand_at_q = float(np.quantile(values, q_star))

    notes: list[str] = []
    if daily is None:
        notes.append(
            "Series is shorter than one day at this cadence, so the daily diagnostic "
            "could not be computed — the forecastability verdict is unavailable, not negative."
        )
        forecastable, verdict = False, "Insufficient history to judge."
        band: DiagnosticBand = "none"
    else:
        forecastable, verdict = classify(daily)
        band = classify_band(daily)
    if len(values) <= 7 * per_day + 8:
        notes.append(
            "Series is shorter than two weeks, so weekly structure is untestable — this is "
            "the same limitation that made week-ahead forecasting unmeasurable on the "
            "7-day Azure traces."
        )
    notes.append(
        f"q* = {q_star:.4f} is derived from your cost ratio "
        f"({request.underage_cost:g}:{request.overage_cost:g}), not from convention."
    )

    return DiagnosticResponse(
        bins=len(values),
        days=len(values) * request.step_seconds / 86400.0,
        minute_autocorrelation=autocorrelation(1),
        daily_autocorrelation=daily,
        weekly_autocorrelation=autocorrelation(7 * per_day),
        peak_to_mean=float(np.quantile(values, 0.95) / values.mean()),
        forecastable=forecastable,
        band=band,
        verdict=verdict,
        q_star=q_star,
        demand_at_q_star=demand_at_q,
        recommended_replicas=int(np.ceil(demand_at_q / request.capacity_per_replica - 1e-9)),
        notes=notes,
    )


@app.get("/geoint/evidence")
def evidence(limit: Annotated[int, Query(ge=1, le=200)] = 50) -> dict[str, object]:
    """ARGUS-shaped evidence export, so the all-source lane can consume capacity findings."""
    snapshot = load_snapshot()
    items = [
        {
            "id": f"delphi:{w.workload_id}",
            "source": "DELPHI",
            "kind": "capacity_diagnostic",
            "observed_at": snapshot.generated_at.isoformat(),
            "summary": (
                f"{w.label}: daily autocorrelation {w.daily_autocorrelation:.3f} — {w.verdict}"
            ),
            "confidence": "measured",
            "attributes": {
                "workload_id": w.workload_id,
                "daily_autocorrelation": w.daily_autocorrelation,
                "forecastable": w.forecastable,
                "band": w.band,
                "peak_to_mean": w.peak_to_mean,
                "licence": w.licence,
            },
        }
        for w in snapshot.workloads[:limit]
    ]
    return {"items": items, "assumptions": snapshot.assumptions}


__all__ = ["DemandPoint", "app", "load_snapshot"]


# --- single-container SPA serving -------------------------------------------------
# The deploy is one container: the API serves the built dashboard same-origin, so there
# is no CORS surface and no second service to keep alive on a free tier. Mounted last so
# every API route above wins; unknown paths fall through to the SPA entry point.
_DIST = Path(get_settings().dashboard_dist or "frontend/dist")

if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    _DIST_ROOT = _DIST.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        """Serve the SPA shell for any unmatched path.

        **The candidate is resolved and confined to the dist directory.** Without that,
        ``_DIST / path`` walks straight out of it: percent-encoded traversal such as
        ``/..%2f..%2frequirements-serve.txt`` or ``/%2e%2e/%2e%2e/etc/hostname`` survives
        URL normalisation, reaches this handler intact, and served arbitrary files off the
        container — as root, in the deploy image. Render's edge happened to reject those
        requests with a 400, but an accident at the CDN is not a security control, and
        anyone running the documented ``docker run`` locally had no such cover.

        ``resolve()`` also collapses symlinks, so a link inside ``dist`` cannot be used to
        step outside it either. An absolute ``path`` would make ``/`` discard the root
        entirely; the containment check catches that case too.
        """
        index = _DIST_ROOT / "index.html"
        if not path:
            return FileResponse(index)
        candidate = (_DIST_ROOT / path).resolve()
        if candidate.is_file() and candidate.is_relative_to(_DIST_ROOT):
            return FileResponse(candidate)
        return FileResponse(index)
