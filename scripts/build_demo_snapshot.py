"""Bake a demo snapshot for the deploy image, from synthetic data only.

The full snapshot needs the raw traces — 2.5 GB of them — which are gitignored and far too
large to ship in a container. So the deployed site runs on **synthetic stand-ins whose
autocorrelation structure is set to match the real workloads**, and says so: the snapshot
mode is ``demo``, the masthead shows it, and every workload label carries the word
synthetic.

That is the honest arrangement. The alternative — shipping a stale copy of real results and
calling it live — is the exact failure this project has flagged in a sibling repo.

The *findings* are the real ones. Those are measured conclusions, not data, and they travel
with the code.

Run: ``python scripts/build_demo_snapshot.py``
"""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
from build_snapshot import (  # type: ignore[import-not-found]
    OUT_SERIES,
    OUT_SNAPSHOT,
    findings,
    horizon_rows,
    summarise,
    write_series,
)

from delphi.api.snapshot import (
    STANDING_ASSUMPTIONS,
    Snapshot,
)
from delphi.data.series import DemandSeries

#: (id, label, daily autocorrelation to imitate, peakiness, step seconds, days)
PROFILES: tuple[tuple[str, str, float, float, int, int], ...] = (
    ("synthetic-strong-daily", "Synthetic — strong daily rhythm", 0.73, 2.8, 60, 7),
    ("synthetic-moderate-daily", "Synthetic — moderate daily rhythm", 0.47, 1.15, 300, 29),
    ("synthetic-weak-daily", "Synthetic — weak daily rhythm", 0.35, 1.45, 60, 7),
    ("synthetic-persistent", "Synthetic — persistent, no rhythm", 0.22, 2.4, 300, 30),
)


def _ar1(count: int, tau_steps: float, rng: np.random.Generator) -> np.ndarray:
    """Mean-reverting AR(1) noise with a tunable correlation time.

    A cumulative random walk was the obvious choice and it is wrong: a walk never
    decorrelates, so adding more of it *raises* the one-day autocorrelation instead of
    lowering it. AR(1) decays as ``exp(-lag / tau)``, which is what makes the
    non-seasonal share controllable.
    """
    phi = math.exp(-1.0 / max(tau_steps, 1e-6))
    out = np.empty(count, dtype=np.float64)
    out[0] = rng.normal()
    scale = math.sqrt(max(1 - phi * phi, 1e-9))
    noise = rng.normal(0.0, scale, count)
    for i in range(1, count):
        out[i] = phi * out[i - 1] + noise[i]
    return (out - out.mean()) / (out.std() or 1.0)


def _daily_autocorrelation(values: np.ndarray, per_day: int) -> float:
    centred = values - values.mean()
    return float(np.corrcoef(centred[:-per_day], centred[per_day:])[0, 1])


def make_series(
    workload_id: str,
    *,
    target_daily: float,
    peakiness: float,
    step_seconds: int,
    days: int,
    seed: int,
) -> DemandSeries:
    """Build a synthetic fleet whose measured daily autocorrelation hits ``target_daily``.

    The seasonal share is solved for numerically rather than assumed. The demo exists to
    illustrate the diagnostic, so a workload labelled "weak daily rhythm" that measures
    0.74 would make the whole page incoherent.
    """
    rng = np.random.default_rng(seed)
    per_day = max(round(86400 / step_seconds), 1)
    count = per_day * days
    index = np.arange(count, dtype=np.float64)

    daily = np.sin(2 * math.pi * index / per_day - math.pi / 2)
    weekly = 0.25 * np.sin(2 * math.pi * index / (7 * per_day))
    seasonal = daily + weekly
    # tau of four hours: strongly persistent minute to minute, essentially decorrelated
    # by one day, so the seasonal weight alone controls the daily lag.
    base = _ar1(count, tau_steps=4 * 3600 / step_seconds, rng=rng)
    jitter = rng.normal(0, 1, count)

    # Spikes are folded into the calibrated quantity, not added afterwards: they perturb
    # the autocorrelation, so calibrating without them leaves the label wrong.
    spikes = 20 * np.where(rng.random(count) < 0.004, rng.gamma(2.0, peakiness, count), 0.0)

    def build(weight: float) -> np.ndarray:
        signal = weight * seasonal + (1 - weight) * base
        return 100 + 30 * signal + 4 * jitter + spikes

    low, high = 0.0, 1.0
    values = build(0.5)
    for _ in range(40):  # bisection on a monotone relationship
        mid = (low + high) / 2
        values = build(mid)
        if _daily_autocorrelation(values, per_day) < target_daily:
            low = mid
        else:
            high = mid
    return DemandSeries(
        workload_id=workload_id,
        source_id="synthetic-demo",
        resource_kind="cpu",
        unit="units",
        step_seconds=step_seconds,
        timestamps=tuple(
            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=step_seconds * i)
            for i in range(count)
        ),
        values=np.maximum(values, 1.0),
        is_imputed=np.zeros(count, dtype=np.bool_),
        quality=np.ones(count, dtype=np.float64),
    )


def main() -> None:
    workloads = []
    horizons = {}
    for seed, (workload_id, label, strength, peak, step, days) in enumerate(PROFILES):
        series = make_series(
            workload_id,
            target_daily=strength,
            peakiness=peak,
            step_seconds=step,
            days=days,
            seed=20260806 + seed,
        )
        workloads.append(
            summarise(
                workload_id,
                label,
                series,
                "Synthetic — generated locally, no external data.",
            )
        )
        horizons[workload_id] = horizon_rows(series)
        write_series(workload_id, series)

    snapshot = Snapshot(
        generated_at=datetime.now(UTC),
        snapshot_mode="demo",
        delphi_version="0.1.0",
        assumptions=[
            "DEMO DATA: these workloads are synthetic stand-ins whose autocorrelation "
            "structure imitates the real traces. The raw traces are 2.5 GB and are not "
            "shipped in the container. The *findings* below are the real measured ones.",
            *STANDING_ASSUMPTIONS,
        ],
        workloads=sorted(workloads, key=lambda w: -w.daily_autocorrelation),
        horizons=horizons,
        frontiers={},
        findings=findings(),
    )
    OUT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SERIES.mkdir(parents=True, exist_ok=True)
    OUT_SNAPSHOT.write_text(snapshot.model_dump_json(indent=2))
    print(f"wrote demo snapshot: {len(snapshot.workloads)} synthetic workloads")
    for w in snapshot.workloads:
        print(
            f"  {w.workload_id:<28} r_day={w.daily_autocorrelation:+.3f} "
            f"{'forecastable' if w.forecastable else 'not forecastable'}"
        )


if __name__ == "__main__":
    main()
