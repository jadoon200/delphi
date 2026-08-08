"""Read-only API: contract, honesty fields, and the diagnostic endpoint."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from delphi.api import app as app_module
from delphi.api.snapshot import (
    FORECASTABLE_AUTOCORRELATION,
    STANDING_ASSUMPTIONS,
    Finding,
    Snapshot,
    WorkloadSummary,
    classify,
    classify_band,
)


def _snapshot() -> Snapshot:
    return Snapshot(
        generated_at=datetime(2026, 8, 5, tzinfo=UTC),
        snapshot_mode="replay",
        delphi_version="test",
        assumptions=STANDING_ASSUMPTIONS,
        workloads=[
            WorkloadSummary(
                workload_id="w-high",
                label="high structure",
                source_id="s",
                licence="CC-BY-4.0",
                resource_kind="cpu",
                unit="MHz",
                step_seconds=300,
                bins=8640,
                days=30.0,
                daily_autocorrelation=0.73,
                minute_autocorrelation=0.98,
                weekly_autocorrelation=0.4,
                peak_to_mean=2.1,
                mean_demand=100.0,
                forecastable=True,
                verdict="Strong daily structure.",
            ),
            WorkloadSummary(
                workload_id="w-low",
                label="low structure",
                source_id="s",
                licence="unverified",
                resource_kind="cpu",
                unit="MHz",
                step_seconds=300,
                bins=8640,
                days=30.0,
                daily_autocorrelation=0.20,
                minute_autocorrelation=0.98,
                weekly_autocorrelation=0.1,
                peak_to_mean=1.3,
                mean_demand=50.0,
                forecastable=False,
                verdict="Effectively no daily structure.",
            ),
        ],
        horizons={},
        frontiers={},
        findings=[
            Finding(
                question_id="Q1",
                question="q",
                prior="p",
                answer="a",
                verdict="refuted",
                evidence="e",
            )
        ],
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    path = tmp_path / "snapshot.json"
    path.write_text(_snapshot().model_dump_json())
    monkeypatch.setattr(app_module, "DEFAULT_SNAPSHOT", path)
    app_module.load_snapshot.cache_clear()

    class _Settings:
        snapshot_path = path

    monkeypatch.setattr(app_module, "get_settings", lambda: _Settings())
    yield TestClient(app_module.app)
    app_module.load_snapshot.cache_clear()


def test_health_states_what_is_being_served(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    # snapshot mode must be explicit, never inferred from freshness
    assert body["snapshot_mode"] == "replay"
    assert body["workloads"] == 2


def test_missing_snapshot_returns_503_not_a_lie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_module.load_snapshot.cache_clear()

    class _Settings:
        snapshot_path = tmp_path / "absent.json"

    monkeypatch.setattr(app_module, "get_settings", lambda: _Settings())
    response = TestClient(app_module.app, raise_server_exceptions=False).get("/health")
    assert response.status_code == 503
    app_module.load_snapshot.cache_clear()


def test_every_comparison_response_carries_its_assumptions(client: TestClient) -> None:
    """A client must not be able to render a comparison without the caveats."""
    for path in ("/assumptions", "/findings", "/workloads/w-high", "/geoint/evidence"):
        body = client.get(path).json()
        assert body["assumptions"], f"{path} omitted assumptions"
        assert any("open-loop" in a.lower() for a in body["assumptions"])


def test_unknown_workload_is_404(client: TestClient) -> None:
    assert client.get("/workloads/nope").status_code == 404
    assert client.get("/demand/nope").status_code == 404


def test_diagnostic_derives_the_quantile_from_the_price_ratio(client: TestClient) -> None:
    values = (50 + 20 * np.sin(np.arange(2000) * 2 * np.pi / 288)).tolist()
    body = client.post(
        "/diagnostic/score",
        json={
            "values": values,
            "step_seconds": 300,
            "underage_cost": 9.0,
            "overage_cost": 1.0,
        },
    ).json()
    assert body["q_star"] == pytest.approx(0.9)
    assert body["daily_autocorrelation"] > 0.9  # a pure daily sine
    assert body["forecastable"] is True


def test_diagnostic_flags_a_series_too_short_for_weekly_structure(client: TestClient) -> None:
    """The limitation that made week-ahead untestable on the 7-day traces, surfaced."""
    values = (50 + 5 * np.sin(np.arange(600) * 2 * np.pi / 288)).tolist()
    body = client.post("/diagnostic/score", json={"values": values, "step_seconds": 300}).json()
    assert any("weekly" in note.lower() for note in body["notes"])


def test_diagnostic_rejects_degenerate_input(client: TestClient) -> None:
    flat = [10.0] * 500
    assert (
        client.post("/diagnostic/score", json={"values": flat, "step_seconds": 300}).status_code
        == 422
    )
    negative = [1.0, -5.0] * 250
    assert (
        client.post("/diagnostic/score", json={"values": negative, "step_seconds": 300}).status_code
        == 422
    )


def test_diagnostic_refuses_a_series_too_short_to_judge(client: TestClient) -> None:
    assert (
        client.post(
            "/diagnostic/score", json={"values": [1.0] * 10, "step_seconds": 300}
        ).status_code
        == 422
    )


def test_classification_threshold_is_the_one_the_evaluation_established() -> None:
    high, _ = classify(FORECASTABLE_AUTOCORRELATION + 0.01)
    low, _ = classify(FORECASTABLE_AUTOCORRELATION - 0.01)
    assert high and not low


def test_borderline_verdicts_disclose_the_measured_exception() -> None:
    """The shipped verdict must not claim more than `docs/EVAL.md` supports.

    `materna-2` sat at r = 0.450 — below the cutoff — and still won 2 of 4 settings at a
    twelve-hour commitment, while `materna-1` at 0.494 won none. An earlier verdict string
    asserted that nothing below the threshold ever won "at any commitment length", which
    the project's own table refutes.
    """
    for value in (0.450, 0.494):
        _, verdict = classify(value)
        assert "borderline" in verdict.lower()
        assert "0.450" in verdict and "0.494" in verdict

    _, confident = classify(0.73)
    assert "borderline" not in confident.lower()

    for value in (0.10, 0.35):
        _, verdict = classify(value)
        assert "at any commitment length" not in verdict


def test_band_is_the_single_source_of_the_threshold(client: TestClient) -> None:
    """The UI must not re-derive the cutoff; it renders whatever band the API reports."""
    assert classify_band(0.73) == "strong"
    assert classify_band(0.470) == "borderline"
    assert classify_band(0.35) == "weak"
    assert classify_band(0.10) == "none"

    values = [60 + 25 * math.sin(i / 45.8) for i in range(2000)]
    body = client.post("/diagnostic/score", json={"values": values, "step_seconds": 300}).json()
    assert body["band"] == classify_band(body["daily_autocorrelation"])
    assert (body["band"] == "strong") == body["forecastable"]


def test_evidence_export_is_argus_shaped(client: TestClient) -> None:
    items = client.get("/geoint/evidence").json()["items"]
    assert items
    for item in items:
        assert {"id", "source", "kind", "observed_at", "summary", "attributes"} <= set(item)
        assert item["source"] == "DELPHI"


def test_demand_response_is_bounded(client: TestClient, tmp_path: Path) -> None:
    series_dir = Path("data/series")
    series_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "workload_id": "w-high",
        "step_seconds": 300,
        "downsample_factor": 1,
        "points": [
            {"ts": "2026-08-05T00:00:00Z", "value": float(i), "is_imputed": False}
            for i in range(5000)
        ],
    }
    path = series_dir / "w-high.json"
    existed = path.exists()
    original = path.read_text() if existed else None
    path.write_text(json.dumps(payload))
    try:
        body = client.get("/demand/w-high?limit=500").json()
        assert len(body["points"]) <= 500
        assert body["downsample_factor"] > 1
    finally:
        if original is not None:
            path.write_text(original)
        else:
            path.unlink()


def test_spa_route_cannot_be_walked_out_of_the_dist_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Percent-encoded traversal escaped `dist` and served arbitrary container files.

    `/..%2f..%2frequirements-serve.txt` and `/%2e%2e/%2e%2e/etc/hostname` both survived URL
    normalisation, reached the handler, and were served — as root, in the deploy image.
    Render's edge rejected them with a 400, but that is an accident of the CDN, not a
    control, and `docker run` locally had no such cover.
    """
    import importlib

    from delphi import config as config_module

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<title>DELPHI</title>")
    (dist / "assets" / "app.js").write_text("// bundle")
    (tmp_path / "SECRET.txt").write_text("SENSITIVE-CONTENTS")

    monkeypatch.setenv("DELPHI_DASHBOARD_DIST", str(dist))
    monkeypatch.setenv("DELPHI_SNAPSHOT_PATH", str(tmp_path / "snapshot.json"))
    config_module.get_settings.cache_clear()
    reloaded = importlib.reload(app_module)
    try:
        client = TestClient(reloaded.app)
        for probe in (
            "/../SECRET.txt",
            "/..%2fSECRET.txt",
            "/%2e%2e/SECRET.txt",
            "/..%2f..%2fSECRET.txt",
            "/%2e%2e%2fSECRET.txt",
        ):
            body = client.get(probe).text
            assert "SENSITIVE-CONTENTS" not in body, f"{probe} escaped the dist directory"
            assert "DELPHI" in body, f"{probe} should fall through to the SPA shell"

        # Under /assets Starlette's StaticFiles guards its own mount and answers 404
        # rather than falling through — also safe, just a different shape of refusal.
        assets_probe = client.get("/assets/..%2f..%2fSECRET.txt")
        assert "SENSITIVE-CONTENTS" not in assets_probe.text
        assert assets_probe.status_code == 404

        # A genuine asset inside dist must still be served.
        assert "// bundle" in client.get("/assets/app.js").text
    finally:
        config_module.get_settings.cache_clear()
        importlib.reload(app_module)
