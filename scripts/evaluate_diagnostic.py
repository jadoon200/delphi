"""The predictability diagnostic, across every workload, with every forecaster.

Two gaps in the record this closes.

**Gap 1 — the diagnostic rested on three workloads.** The claim that daily autocorrelation
predicts whether forecasting pays was drawn from Azure LLM `code`/`conv` and one Bitbrains
fleet. Three points is not a relationship. This runs seven independent workloads from three
providers.

**Gap 2 — every controller experiment in this project used `SeasonalNaiveForecaster`.**
Every headline finding — "forecasting loses at minute horizons", "forecasting wins at 6-12h
commitments" — was measured with the crudest forecaster in the codebase, while ARIMA, ETS,
LightGBM and drift models sat built and tested but unused in any control loop. A conclusion
about *forecasting* drawn from one weak forecaster is a conclusion about that forecaster.

So this sweeps forecaster x workload x commitment window, and reports whether the
diagnostic survives a better model.

Run: ``python scripts/evaluate_diagnostic.py``
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from delphi.control.commitment import (
    BackwardCommitmentController,
    ForwardCommitmentController,
)
from delphi.control.controllers import ControlContext
from delphi.control.serving import ServingProfile, gpu_seconds_demand
from delphi.control.simulator import CapacityProfile, CostModel, simulate
from delphi.data.bitbrains import load_fleet as bitbrains_fleet
from delphi.data.bitbrains import to_demand_series as bitbrains_series
from delphi.data.llm_inference import load_binned
from delphi.data.materna import TRACES as MATERNA_TRACES
from delphi.data.materna import load_fleet as materna_fleet
from delphi.data.materna import to_demand_series as materna_series
from delphi.data.series import DemandSeries
from delphi.forecast.baselines import (
    DriftNaiveForecaster,
    LightGBMQuantileForecaster,
    SeasonalNaiveForecaster,
    StatsForecastForecaster,
)
from delphi.forecast.contracts import QuantileForecaster
from delphi.forecast.features import FeatureSpec

QUANTILES: tuple[float, ...] = (0.80, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class Workload:
    name: str
    series: DemandSeries
    profile: CapacityProfile
    costs: CostModel
    day_steps: int

    @property
    def daily_autocorrelation(self) -> float:
        values = self.series.values - self.series.values.mean()
        return float(np.corrcoef(values[: -self.day_steps], values[self.day_steps :])[0, 1])

    @property
    def peak_to_mean(self) -> float:
        return float(np.quantile(self.series.values, 0.95) / self.series.values.mean())


def _fleet_workload(name: str, series: DemandSeries, price: float) -> Workload:
    profile = CapacityProfile(
        workload_id=series.workload_id,
        step_seconds=series.step_seconds,
        capacity_per_replica=max(float(np.median(series.values)) / 8.0, 1.0),
        startup_seconds=600.0,
        teardown_seconds=300.0,
        utilisation_target=0.85,
        min_replicas=0,
        max_replicas=100_000,
        scale_to_zero=True,
    )
    costs = CostModel(price_per_replica_hour=price, churn_cost_per_action=price / 6.0)
    return Workload(name, series, profile, costs, 86400 // series.step_seconds)


def build_workloads(include_gpu: bool = True) -> list[Workload]:
    workloads: list[Workload] = []
    if include_gpu:
        for trace in ("code", "conv"):
            binned = load_binned(trace, bin_seconds=60)
            truth = gpu_seconds_demand(binned, ServingProfile())
            series = DemandSeries(
                workload_id=f"llm-{trace}",
                source_id="azure-llm-inference-2024",
                resource_kind="gpu_seconds",
                unit="gpu-seconds/bin",
                step_seconds=60,
                timestamps=tuple(
                    binned.start + timedelta(seconds=60 * i) for i in range(len(truth))
                ),
                values=truth,
                is_imputed=np.zeros(len(truth), dtype=np.bool_),
                quality=np.ones(len(truth), dtype=np.float64),
            )
            profile = ServingProfile(startup_seconds=240.0).capacity_profile(
                workload_id=f"llm-{trace}", bin_seconds=60
            )
            costs = CostModel(price_per_replica_hour=3.40, churn_cost_per_action=3.40 * 240 / 3600)
            workloads.append(Workload(f"azure-llm-{trace}", series, profile, costs, 1440))

    for trace in ("rnd", "fastStorage"):
        fleet = bitbrains_fleet(trace, bin_seconds=300)
        workloads.append(
            _fleet_workload(f"bitbrains-{trace}", bitbrains_series(fleet, trace=trace), 0.0416)
        )
    for trace in MATERNA_TRACES:
        fleet = materna_fleet(trace, bin_seconds=300)
        workloads.append(
            _fleet_workload(
                f"materna-{trace.split('-')[-1]}", materna_series(fleet, trace=trace), 0.0416
            )
        )
    return workloads


def forecasters(day_steps: int) -> dict[str, QuantileForecaster]:
    """Every forecaster the codebase has, not just the crude one used until now."""
    return {
        "seasonal-naive": SeasonalNaiveForecaster(day_steps),
        "drift": DriftNaiveForecaster(residual_window=day_steps),
        "ets": StatsForecastForecaster(kind="auto_ets", season_length=min(day_steps, 96)),
        "lightgbm": LightGBMQuantileForecaster(
            feature_spec=FeatureSpec(
                lags=(1, 2, 3, 6, 12, day_steps // 2, day_steps),
                windows=(6, 12, day_steps),
            ),
            n_estimators=60,
            max_training_rows=8000,
        ),
    }


def dominance(
    workload: Workload, window_steps: int, forecaster: QuantileForecaster
) -> tuple[int, float, float]:
    """Return (times forward dominated, best backward viol., best forward viol.)."""
    context = ControlContext(
        series=workload.series, profile=workload.profile, start_step=workload.day_steps * 2
    )
    scored = slice(workload.day_steps * 3, None)
    wins = 0
    best_back, best_fwd = 1.0, 1.0
    for quantile in QUANTILES:
        back = simulate(
            demand=workload.series.values[scored],
            requested=BackwardCommitmentController(
                window_steps=window_steps, quantile=quantile
            ).plan(context)[scored],
            profile=workload.profile,
            cost_model=workload.costs,
        )
        fwd = simulate(
            demand=workload.series.values[scored],
            requested=ForwardCommitmentController(
                window_steps=window_steps, forecaster=forecaster, quantile=quantile
            ).plan(context)[scored],
            profile=workload.profile,
            cost_model=workload.costs,
        )
        if fwd.cost <= back.cost and fwd.violation_rate <= back.violation_rate:
            wins += 1
        best_back = min(best_back, back.violation_rate)
        best_fwd = min(best_fwd, fwd.violation_rate)
    return wins, best_back, best_fwd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--windows", nargs="+", type=int, default=[6, 12])
    args = parser.parse_args()

    print("# The predictability diagnostic across seven workloads and four forecasters\n")
    print("Does daily autocorrelation predict whether forecasting pays — and does the")
    print("answer survive replacing seasonal-naive with a real model?\n")

    workloads = build_workloads(include_gpu=not args.skip_gpu)
    print("## Workloads\n")
    print("| workload | bins | days | step | daily autocorr | p95/mean |")
    print("|---|---:|---:|---:|---:|---:|")
    for w in sorted(workloads, key=lambda x: -x.daily_autocorrelation):
        print(
            f"| `{w.name}` | {len(w.series):,} | "
            f"{len(w.series) * w.series.step_seconds / 86400:.0f} | {w.series.step_seconds}s "
            f"| **{w.daily_autocorrelation:.3f}** | {w.peak_to_mean:.2f} |"
        )

    for window_hours in args.windows:
        print(
            f"\n## {window_hours}-hour commitment — times forward dominated "
            f"(of {len(QUANTILES)} quantiles)\n"
        )
        names = list(forecasters(1440))
        print("| workload | daily autocorr | " + " | ".join(names) + " | any |")
        print("|---|---:|" + "---:|" * (len(names) + 1))
        for w in sorted(workloads, key=lambda x: -x.daily_autocorrelation):
            steps = window_hours * 3600 // w.series.step_seconds
            cells, total = [], 0
            for name, forecaster in forecasters(w.day_steps).items():
                try:
                    wins, _, _ = dominance(w, steps, forecaster)
                except Exception as exc:
                    cells.append("err")
                    print(f"<!-- {w.name}/{name}: {type(exc).__name__}: {exc} -->")
                    continue
                cells.append(f"**{wins}**" if wins else "0")
                total = max(total, wins)
            print(
                f"| `{w.name}` | {w.daily_autocorrelation:.3f} | "
                + " | ".join(cells)
                + f" | {total} |"
            )


if __name__ == "__main__":
    main()
