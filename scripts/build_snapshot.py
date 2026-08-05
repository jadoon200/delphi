"""Bake the read-only snapshot the API serves.

Runs from cached trace aggregates, so it is fast enough to run at image build — which is
how the deploy stays honest: the snapshot is regenerated from current code on every deploy
rather than shipping a stale artifact that can drift from the repo.

What goes in is what the evaluation actually established, including the negative results.
Nothing here is aspirational.

Run: ``python scripts/build_snapshot.py``
"""

from datetime import timedelta
from pathlib import Path

import numpy as np

from delphi.api.snapshot import (
    STANDING_ASSUMPTIONS,
    DemandPoint,
    DemandSeriesPayload,
    Finding,
    HorizonRow,
    Snapshot,
    WorkloadSummary,
    classify,
)
from delphi.control.serving import ServingProfile, gpu_seconds_demand
from delphi.data.bitbrains import load_fleet as bitbrains_fleet
from delphi.data.bitbrains import to_demand_series as bitbrains_series
from delphi.data.llm_inference import CITATION_2024, load_binned
from delphi.data.materna import TRACES as MATERNA_TRACES
from delphi.data.materna import load_fleet as materna_fleet
from delphi.data.materna import to_demand_series as materna_series
from delphi.data.series import DemandSeries
from delphi.timeutil import utc_now

OUT_SNAPSHOT = Path("data/snapshot.json")
OUT_SERIES = Path("data/series")
SERIES_POINTS = 3000


def _autocorrelation(values: np.ndarray, lag: int) -> float | None:
    if lag < 1 or len(values) <= lag + 8:
        return None
    centred = values - values.mean()
    return float(np.corrcoef(centred[:-lag], centred[lag:])[0, 1])


def summarise(
    workload_id: str,
    label: str,
    series: DemandSeries,
    licence: str,
) -> WorkloadSummary:
    per_day = max(round(86400 / series.step_seconds), 1)
    daily = _autocorrelation(series.values, per_day) or 0.0
    forecastable, verdict = classify(daily)
    return WorkloadSummary(
        workload_id=workload_id,
        label=label,
        source_id=series.source_id,
        licence=licence,
        resource_kind=series.resource_kind,
        unit=series.unit,
        step_seconds=series.step_seconds,
        bins=len(series),
        days=len(series) * series.step_seconds / 86400.0,
        daily_autocorrelation=daily,
        minute_autocorrelation=_autocorrelation(series.values, 1) or 0.0,
        weekly_autocorrelation=_autocorrelation(series.values, 7 * per_day),
        peak_to_mean=float(np.quantile(series.values, 0.95) / series.values.mean()),
        mean_demand=float(series.values.mean()),
        forecastable=forecastable,
        verdict=verdict,
    )


def horizon_rows(series: DemandSeries) -> list[HorizonRow]:
    """Which predictor wins at each lead time — the crossover, per workload."""
    values = series.values
    per_day = max(round(86400 / series.step_seconds), 1)
    scale = float(np.mean(np.abs(np.diff(values[: len(values) // 2]))))
    if scale <= 0:
        return []
    rows: list[HorizonRow] = []
    candidates = [
        ("5 min", max(round(300 / series.step_seconds), 1)),
        ("1 hour", max(round(3600 / series.step_seconds), 1)),
        ("6 hours", max(round(6 * 3600 / series.step_seconds), 1)),
        ("1 day", per_day),
    ]
    for label, lead in candidates:
        if len(values) <= 2 * per_day + lead + 8:
            continue
        index = np.arange(2 * per_day, len(values))
        seasonal = values[index - per_day]
        # One hour of trailing history, matching the window used in docs/EVAL.md. The
        # window length is a real lever: a 6-hour mean loses to seasonal even at a 5-minute
        # lead, which would make this table contradict the evaluation it summarises.
        window = max(round(3600 / series.step_seconds), 1)
        trailing = np.asarray([values[max(0, i - lead - window) : i - lead].mean() for i in index])
        trailing_mase = float(np.mean(np.abs(values[index] - trailing)) / scale)
        seasonal_mase = float(np.mean(np.abs(values[index] - seasonal)) / scale)
        rows.append(
            HorizonRow(
                horizon_label=label,
                horizon_steps=lead,
                trailing_mase=trailing_mase,
                seasonal_mase=seasonal_mase,
                winner="trailing" if trailing_mase < seasonal_mase else "seasonal",
            )
        )
    return rows


def write_series(workload_id: str, series: DemandSeries) -> None:
    OUT_SERIES.mkdir(parents=True, exist_ok=True)
    factor = max(len(series) // SERIES_POINTS, 1)
    points = [
        DemandPoint(
            ts=series.timestamps[index],
            value=float(series.values[index : index + factor].mean()),
            is_imputed=bool(series.is_imputed[index : index + factor].any()),
        )
        for index in range(0, len(series), factor)
    ]
    payload = DemandSeriesPayload(
        workload_id=workload_id,
        step_seconds=series.step_seconds * factor,
        downsample_factor=factor,
        points=points,
    )
    (OUT_SERIES / f"{workload_id}.json").write_text(payload.model_dump_json())


def findings() -> list[Finding]:
    """The pre-registered questions, with the answers the measurements gave."""
    return [
        Finding(
            question_id="Q1",
            question="Does forecasting beat reaction at minute-scale autoscaling horizons?",
            prior="Yes, and the margin should grow with actuation delay.",
            answer=(
                "No. Against a properly tuned sliding-window percentile recommender — the "
                "incumbent that Kubernetes VPA and Borg Autopilot already ship — forecasting "
                "lost in 8 of 8 settings, and the gap widened with cold start."
            ),
            verdict="refuted",
            evidence=(
                "Minute-scale autocorrelation is ~0.98, so a trailing percentile already "
                "extracts nearly all the signal; an explicit model adds its own error. "
                "Confirmed across four forecasters."
            ),
        ),
        Finding(
            question_id="Q4",
            question="Does a price-derived compliance target beat a conventional fixed one?",
            prior="Only when prices are asymmetric enough to matter.",
            answer=(
                "Yes in 19 of 24 settings. The newsvendor identity q* = C_u/(C_u+C_o) sets "
                "the percentile that convention normally fixes at p95."
            ),
            verdict="confirmed",
            evidence=(
                "At C_u/C_o = 19 the derived target equals the p95 convention exactly and "
                "the two agree to the cent — a forced algebraic identity that serves as a "
                "consistency check on the harness."
            ),
        ),
        Finding(
            question_id="Q10",
            question="Is there a workload where no controller beats static provisioning?",
            prior="Expected yes.",
            answer=(
                "Yes. On a stable Azure Functions workload, static provisioning was the "
                "cheapest point on the entire frontier and neither calibrated controller "
                "reached a 1% violation rate at all."
            ),
            verdict="confirmed",
            evidence="Demand p95/mean was near 1.2; there was nothing for a forecaster to add.",
        ),
        Finding(
            question_id="Q11",
            question="Does forecasting ever pay, and if so where?",
            prior="Unknown after Q1 refuted the autoscaling case.",
            answer=(
                "Yes, in the commitment regime: sizing once and holding for 6-12 hours on a "
                "workload with strong daily structure. Forward forecasting dominated a "
                "trailing window — cheaper AND fewer violations — in 5 of 7 and 6 of 7 "
                "quantile settings, halving the violation rate at lower cost at 12 hours."
            ),
            verdict="confirmed",
            evidence=(
                "Two conditions are both necessary: daily autocorrelation above ~0.5, and a "
                "commitment long enough that reaction is impossible. Either alone is not "
                "enough."
            ),
        ),
        Finding(
            question_id="Q12",
            question="Does week-ahead forecasting work?",
            prior="Seasonality should dominate at long horizons.",
            answer=(
                "No. On the only fleet long enough to test it (13 weeks), a flat average of "
                "the prior four weeks beat every structured predictor: 53.1% error versus "
                "56-60%."
            ),
            verdict="refuted",
            evidence=(
                "Weekly autocorrelation was 0.271 against a daily 0.248 — both weak. There "
                "was no long-range structure to exploit."
            ),
        ),
        Finding(
            question_id="Q13",
            question="Does daily autocorrelation predict whether forecasting will pay?",
            prior="Proposed as a diagnostic after the mixed results above.",
            answer=(
                "It predicted every outcome measured so far. High-autocorrelation workloads "
                "reward forecasting at long commitments; low-autocorrelation ones never did, "
                "at any commitment length."
            ),
            verdict="open",
            evidence=(
                "Strong so far but the sample is small, and the threshold near 0.5 is drawn "
                "from a handful of workloads rather than fitted. Treat as a rule of thumb "
                "with a measurement behind it, not a calibrated boundary."
            ),
        ),
    ]


def main() -> None:
    workloads: list[WorkloadSummary] = []
    horizons: dict[str, list[HorizonRow]] = {}

    for trace in ("code", "conv"):
        binned = load_binned(trace, bin_seconds=60)
        truth = gpu_seconds_demand(binned, ServingProfile())
        series = DemandSeries(
            workload_id=f"azure-llm-{trace}",
            source_id="azure-llm-inference-2024",
            resource_kind="gpu_seconds",
            unit="gpu-seconds/bin",
            step_seconds=60,
            timestamps=tuple(binned.start + timedelta(seconds=60 * i) for i in range(len(truth))),
            values=truth,
            is_imputed=np.zeros(len(truth), dtype=np.bool_),
            quality=np.ones(len(truth), dtype=np.float64),
        )
        workload_id = f"azure-llm-{trace}"
        workloads.append(
            summarise(
                workload_id,
                f"Azure LLM inference — {trace} (GPU work)",
                series,
                f"CC-BY-4.0 — {CITATION_2024}",
            )
        )
        horizons[workload_id] = horizon_rows(series)
        write_series(workload_id, series)

    for trace in ("rnd", "fastStorage"):
        fleet = bitbrains_fleet(trace, bin_seconds=300)
        series = bitbrains_series(fleet, trace=trace)
        workload_id = f"bitbrains-{trace}"
        workloads.append(
            summarise(
                workload_id,
                f"Bitbrains {trace} — enterprise VM fleet CPU",
                series,
                "Terms unverified: canonical Grid Workloads Archive host unreachable.",
            )
        )
        horizons[workload_id] = horizon_rows(series)
        write_series(workload_id, series)

    for trace in MATERNA_TRACES:
        fleet = materna_fleet(trace, bin_seconds=300)
        series = materna_series(fleet, trace=trace)
        workload_id = f"materna-{trace.split('-')[-1]}"
        workloads.append(
            summarise(
                workload_id,
                f"Materna {trace.split('-')[-1]} — enterprise VM fleet CPU",
                series,
                "Terms unverified: canonical Grid Workloads Archive host unreachable.",
            )
        )
        horizons[workload_id] = horizon_rows(series)
        write_series(workload_id, series)

    snapshot = Snapshot(
        generated_at=utc_now(),
        snapshot_mode="replay",
        delphi_version="0.1.0",
        assumptions=STANDING_ASSUMPTIONS,
        workloads=sorted(workloads, key=lambda w: -w.daily_autocorrelation),
        horizons=horizons,
        frontiers={},
        findings=findings(),
    )
    OUT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SNAPSHOT.write_text(snapshot.model_dump_json(indent=2))
    print(
        f"wrote {OUT_SNAPSHOT} — {len(snapshot.workloads)} workloads, "
        f"{len(snapshot.findings)} findings"
    )
    for w in snapshot.workloads:
        flag = "forecastable" if w.forecastable else "not forecastable"
        print(f"  {w.workload_id:<24} r_day={w.daily_autocorrelation:+.3f}  {flag}")


if __name__ == "__main__":
    main()
