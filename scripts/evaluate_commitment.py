"""Does forecasting pay in the commitment regime? Full sweep through the validated simulator.

The directional experiment (2026-08-05) showed forward forecasting dominating a trailing
window at 6-12 hour commitments on a workload with strong daily structure. That comparison
was demand-versus-capacity arithmetic with no simulator, no actuation delay and no cost
model, so it could not go in the record as a result.

This is the confirmation run: the same question through the same replay simulator that
scores every other controller in the project, with prices, churn and actuation delay, over
three workloads chosen to span the diagnostic axis.

| workload | daily autocorrelation | prediction |
|---|---|---|
| Azure LLM `code` | 0.730 | forecasting should win clearly |
| Azure LLM `conv` | 0.349 | forecasting should win marginally, or trade off |
| Bitbrains fleet CPU | 0.248 | forecasting should not help |

If that ordering holds, the daily autocorrelation is a usable *a priori* diagnostic for
whether a workload rewards forecasting at all — which is worth more to an operator than
another controller.

Reported per L5: cost at matched violation rate, Pareto frontiers, and the **spread across
held-out windows** rather than a single pooled number.

Run: ``python scripts/evaluate_commitment.py``
"""

import argparse
from datetime import timedelta

import numpy as np

from delphi.control.commitment import (
    BackwardCommitmentController,
    ForwardCommitmentController,
)
from delphi.control.controllers import ControlContext
from delphi.control.serving import ServingProfile, gpu_seconds_demand
from delphi.control.simulator import CapacityProfile, CostModel
from delphi.data.bitbrains import load_fleet, to_demand_series
from delphi.data.llm_inference import load_binned
from delphi.data.series import DemandSeries
from delphi.evaluation.frontier import (
    FrontierPoint,
    cost_at_matched_violation,
    format_frontier_table,
    pareto_front,
    score_plan,
)
from delphi.forecast.baselines import SeasonalNaiveForecaster

QUANTILES: tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
GPU_PRICE_PER_HOUR = 3.40
VM_PRICE_PER_HOUR = 0.0416


def gpu_workload(trace: str) -> tuple[DemandSeries, CapacityProfile, CostModel, int]:
    binned = load_binned(trace, bin_seconds=60)
    truth = gpu_seconds_demand(binned, ServingProfile())
    profile = ServingProfile(startup_seconds=240.0).capacity_profile(
        workload_id=f"llm-{trace}", bin_seconds=60
    )
    series = DemandSeries(
        workload_id=f"llm-{trace}",
        source_id="azure-llm-inference-2024",
        resource_kind="gpu_seconds",
        unit="gpu-seconds/bin",
        step_seconds=60,
        timestamps=tuple(
            binned.start + timedelta(seconds=60 * index) for index in range(len(truth))
        ),
        values=truth,
        is_imputed=np.zeros(len(truth), dtype=np.bool_),
        quality=np.ones(len(truth), dtype=np.float64),
    )
    costs = CostModel(
        price_per_replica_hour=GPU_PRICE_PER_HOUR,
        churn_cost_per_action=GPU_PRICE_PER_HOUR * 240.0 / 3600.0,
    )
    return series, profile, costs, 1440  # one day in bins


def fleet_workload() -> tuple[DemandSeries, CapacityProfile, CostModel, int]:
    fleet = load_fleet("rnd", bin_seconds=300)
    series = to_demand_series(fleet, trace="rnd")
    profile = CapacityProfile(
        workload_id=series.workload_id,
        step_seconds=300,
        capacity_per_replica=2000.0,  # MHz per provisioned unit
        startup_seconds=600.0,
        teardown_seconds=300.0,
        utilisation_target=0.85,
        min_replicas=0,
        max_replicas=100_000,
        scale_to_zero=True,
    )
    costs = CostModel(
        price_per_replica_hour=VM_PRICE_PER_HOUR,
        churn_cost_per_action=VM_PRICE_PER_HOUR * 600.0 / 3600.0,
    )
    return series, profile, costs, 288  # one day in 5-minute bins


def daily_autocorrelation(series: DemandSeries, day_steps: int) -> float:
    values = series.values - series.values.mean()
    return float(np.corrcoef(values[:-day_steps], values[day_steps:])[0, 1])


def evaluate(
    label: str,
    series: DemandSeries,
    profile: CapacityProfile,
    costs: CostModel,
    day_steps: int,
    windows: tuple[tuple[int, str], ...],
) -> None:
    print(f"\n## {label}\n")
    print(f"- Daily autocorrelation: **{daily_autocorrelation(series, day_steps):.3f}**")
    print(
        f"- {len(series):,} bins x {series.step_seconds}s "
        f"({len(series) * series.step_seconds / 86400:.1f} days)"
    )
    print(
        f"- Actuation delay {profile.startup_seconds:.0f}s, "
        f"${costs.price_per_replica_hour}/replica-hour\n"
    )

    context = ControlContext(series=series, profile=profile, start_step=day_steps * 2)
    scored = slice(day_steps * 3, None)
    forecaster = SeasonalNaiveForecaster(day_steps)

    for window_steps, window_label in windows:
        points: list[FrontierPoint] = []
        for quantile in QUANTILES:
            for family, controller in (
                (
                    "backward",
                    BackwardCommitmentController(window_steps=window_steps, quantile=quantile),
                ),
                (
                    "forward",
                    ForwardCommitmentController(
                        window_steps=window_steps,
                        forecaster=forecaster,
                        quantile=quantile,
                    ),
                ),
            ):
                points.append(
                    score_plan(
                        controller_id=controller.controller_id,
                        family=family,
                        setting=quantile,
                        plan=controller.plan(context),
                        context=context,
                        cost_model=costs,
                        scored=scored,
                    )
                )

        print(f"### Commitment window {window_label}\n")
        print("| family | cost @ ≤20% viol. | @ ≤10% | @ ≤5% | best viol. | frontier pts |")
        print("|---|---:|---:|---:|---:|---:|")
        front = pareto_front(points)
        for family in ("backward", "forward"):
            arm = [point for point in points if point.family == family]
            cells = []
            for target in (0.20, 0.10, 0.05):
                value = cost_at_matched_violation(arm, target)
                cells.append("—" if value is None else f"${value:,.0f}")
            best = min(point.violation_rate for point in arm)
            count = sum(1 for point in front if point.family == family)
            print(f"| {family} | " + " | ".join(cells) + f" | {best:.4f} | {count} |")

        # dominance check: is any forward point strictly better on both axes?
        dominating = [
            f
            for f in points
            if f.family == "forward"
            and any(
                f.cost <= b.cost
                and f.violation_rate <= b.violation_rate
                and (f.cost < b.cost or f.violation_rate < b.violation_rate)
                for b in points
                if b.family == "backward" and abs(b.setting - f.setting) < 1e-9
            )
        ]
        print(
            f"\n**Forward dominates backward at the same setting in "
            f"{len(dominating)} of {len(QUANTILES)} quantiles.**\n"
        )
        print(format_frontier_table(front))
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fleet", action="store_true")
    args = parser.parse_args()

    print("# Commitment-interval capacity control: does forecasting pay?\n")
    print("Confirmation run through the validated replay simulator — prices, churn and")
    print("actuation delay included. Backward and forward controllers perform the *same*")
    print("arithmetic (q-quantile of a window of demand), differing only in whether that")
    print("window is observed or predicted.\n")
    print("**Open-loop assumption applies**: replaying a trace against a new policy is a")
    print("counterfactual, valid here because all three arrival processes are externally")
    print("driven. Rankings transfer; absolute dollars carry a sim-to-real disclaimer.\n")

    gpu_windows = ((60, "1 h"), (120, "2 h"), (360, "6 h"), (720, "12 h"))
    for trace in ("code", "conv"):
        series, profile, costs, day = gpu_workload(trace)
        evaluate(
            f"Azure LLM inference `{trace}` (GPU-seconds)", series, profile, costs, day, gpu_windows
        )

    if not args.skip_fleet:
        series, profile, costs, day = fleet_workload()
        fleet_windows = ((12, "1 h"), (24, "2 h"), (72, "6 h"), (144, "12 h"))
        evaluate(
            "Bitbrains `rnd` fleet CPU (the low-autocorrelation control)",
            series,
            profile,
            costs,
            day,
            fleet_windows,
        )


if __name__ == "__main__":
    main()
