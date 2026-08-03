"""M8 — the cost-versus-violation frontier for every controller, on real demand.

This is the sentence the project has to be able to say: *here is the frontier for six
controllers on a real trace, and here is why ours sits where it does.*

Run: ``python scripts/evaluate_frontier.py``  (add ``--synthetic`` to skip the archive)
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from delphi.control.adaptive import BudgetPacedController, NewsvendorController
from delphi.control.controllers import (
    ControlContext,
    PercentileRecommender,
    ProactiveQuantileController,
    ReactiveController,
    StaticController,
)
from delphi.control.newsvendor import CostRatio, implied_ratio_of_controller
from delphi.control.simulator import CapacityProfile, CostModel
from delphi.data.azure_functions import load_archive_cohort, select_cohort
from delphi.data.series import DemandSeries, aggregate_series
from delphi.data.synthetic import generate_synthetic
from delphi.evaluation.frontier import (
    FrontierPoint,
    ParitySpec,
    cost_at_matched_violation,
    format_frontier_table,
    frontier_dynamic_range,
    pareto_front,
    score_plan,
)
from delphi.forecast.baselines import SeasonalNaiveForecaster

ARCHIVE = Path("data/raw/azure-functions-2019.tar.xz")
#: 5-minute bins: 1-minute Azure data is noisy enough that per-minute replica decisions are
#: dominated by integer rounding. Aggregation is declared, not silent.
AGGREGATION = 5
SEASON = 24 * 60 // AGGREGATION  # one day
PRICE_PER_REPLICA_HOUR = 0.0416  # Azure B2s, southeastasia, list, retrieved 2026-08-03

#: Risk settings swept for every controller. Same count for every arm — that is the parity.
SETTINGS: tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
PARITY = ParitySpec(
    configurations_per_controller=len(SETTINGS),
    selection_metric="newsvendor loss at the lane's cost ratio",
    validation_fraction=0.0,
)


def build_context(series: DemandSeries, *, capacity_per_replica: float) -> ControlContext:
    profile = CapacityProfile(
        workload_id=series.workload_id,
        step_seconds=series.step_seconds,
        capacity_per_replica=capacity_per_replica,
        startup_seconds=float(series.step_seconds * 3),
        teardown_seconds=float(series.step_seconds),
        utilisation_target=0.9,
        min_replicas=0,
        max_replicas=100_000,
        scale_to_zero=True,
    )
    return ControlContext(series=series, profile=profile, start_step=SEASON * 4)


def evaluate(context: ControlContext, label: str) -> list[FrontierPoint]:
    costs = CostModel(price_per_replica_hour=PRICE_PER_REPLICA_HOUR, churn_cost_per_action=0.001)
    scored = slice(SEASON * 7, None)
    forecaster = SeasonalNaiveForecaster(SEASON)
    points: list[FrontierPoint] = []

    for setting in SETTINGS:
        arms = {
            "static": (StaticController(percentile=setting), "C0 static"),
            "reactive": (
                ReactiveController(tolerance=1.0 - setting, stabilisation_steps=6),
                "C1 reactive",
            ),
            "percentile": (
                PercentileRecommender(
                    window_steps=SEASON,
                    percentile=setting,
                    half_life_steps=SEASON // 2,
                    margin=0.0,
                ),
                "C2 percentile",
            ),
            "proactive": (
                ProactiveQuantileController(
                    forecaster=forecaster, quantile=setting, refit_stride=SEASON
                ),
                "C3 proactive",
            ),
            "budget_paced": (
                BudgetPacedController(
                    forecaster=forecaster,
                    target_violation_rate=round(1.0 - setting, 4),
                    refit_stride=SEASON,
                    score_window=SEASON * 3,
                ),
                "C5 budget-paced",
            ),
            "newsvendor": (
                NewsvendorController(
                    forecaster=forecaster,
                    ratio=CostRatio.from_quantile(setting),
                    refit_stride=SEASON,
                    score_window=SEASON * 3,
                ),
                "C6 newsvendor",
            ),
        }
        for controller, family in arms.values():
            plan = controller.plan(context)
            points.append(
                score_plan(
                    controller_id=controller.controller_id,
                    family=family,
                    setting=setting,
                    plan=plan,
                    context=context,
                    cost_model=costs,
                    scored=scored,
                )
            )
    print(f"\n### {label}\n")
    print(format_frontier_table(points))
    return points


def report(points: list[FrontierPoint], label: str) -> None:
    print(f"\n#### Frontier summary — {label}\n")
    front = pareto_front(points)
    print("Non-dominated points (minimising both cost and violations):\n")
    print(format_frontier_table(front))

    spread = frontier_dynamic_range(points)
    cheapest = min(point.cost for point in points)
    print(
        f"\n- Dynamic range across all points: **{spread:.1f}** "
        f"({spread / cheapest:.1%} of the cheapest point)"
    )
    if spread / cheapest < 0.05:
        print(
            "- **Saturation warning:** every controller costs nearly the same here, so "
            "this trace has no headroom to separate them (L5 Trap 3)."
        )

    print("\n**Cost at a matched violation rate** — the honest scalar comparison:\n")
    print("| family | cost @ <=10% viol. | cost @ <=5% viol. | cost @ <=1% viol. |")
    print("|---|---:|---:|---:|")
    for family in sorted({point.family for point in points}):
        arm = [point for point in points if point.family == family]
        cells = []
        for target in (0.10, 0.05, 0.01):
            value = cost_at_matched_violation(arm, target)
            cells.append("never reached" if value is None else f"{value:.1f}")
        print(f"| {family} | " + " | ".join(cells) + " |")

    print("\n**Implicit price of failure** — what each arm's realised violation rate asserts:\n")
    print("| family | median violations | implied C_u / C_o |")
    print("|---|---:|---:|")
    for family in sorted({point.family for point in points}):
        arm = [point for point in points if point.family == family]
        median = float(np.median([point.violation_rate for point in arm]))
        implied = implied_ratio_of_controller(min(median, 0.999))
        shown = "inf" if implied == float("inf") else f"{implied:.1f}"
        print(f"| {family} | {median:.4f} | {shown} |")


def report_newsvendor_question(context: ControlContext, label: str) -> None:
    """Q4 — does a price-derived target beat a conventional fixed one?

    Sweeping C6's ``q*`` over the same grid as C5's target makes the two arms *identical
    by construction*: q* is the setting, so C6 reduces to C5. That is not a bug, it is the
    point — the newsvendor contribution is not a different control law, it is that the
    setting is **derived from prices rather than chosen by convention**.

    So the honest experiment is this one: hold the industry-default P95 fixed, vary the
    real cost ratio, and measure what the default costs when the economics disagree with it.
    """
    print(f"\n#### Q4 — price-derived target vs the conventional P95 default — {label}\n")
    costs = CostModel(price_per_replica_hour=PRICE_PER_REPLICA_HOUR, churn_cost_per_action=0.001)
    scored = slice(SEASON * 7, None)
    forecaster = SeasonalNaiveForecaster(SEASON)

    default = NewsvendorController(
        forecaster=forecaster,
        ratio=CostRatio.from_quantile(0.95),
        refit_stride=SEASON,
        score_window=SEASON * 3,
    )
    default_point = score_plan(
        controller_id="convention P95",
        family="default",
        setting=0.95,
        plan=default.plan(context),
        context=context,
        cost_model=costs,
        scored=scored,
    )

    print("`C_u/C_o` is the price of one unit of unmet demand relative to one unit of idle")
    print("capacity. The default is only right where `q*` happens to land on 0.95.\n")
    print("| C_u / C_o | derived q* | violations | cost | total cost vs P95 default |")
    print("|---:|---:|---:|---:|---:|")
    for multiple in (1.0, 3.0, 9.0, 19.0, 49.0, 99.0):
        ratio = CostRatio(underage_per_unit=multiple, overage_per_unit=1.0)
        point = score_plan(
            controller_id=f"newsvendor C_u/C_o={multiple:g}",
            family="derived",
            setting=ratio.critical_ratio,
            plan=NewsvendorController(
                forecaster=forecaster,
                ratio=ratio,
                refit_stride=SEASON,
                score_window=SEASON * 3,
            ).plan(context),
            context=context,
            cost_model=costs,
            scored=scored,
        )
        # total economic cost = capacity bill + priced breaches, on the operator's own terms
        unit_price = costs.price_per_replica_hour / (3600 / context.profile.step_seconds)
        derived_total = point.cost + multiple * unit_price * point.violation_sum / max(
            context.profile.capacity_per_replica, 1e-9
        )
        default_total = default_point.cost + multiple * unit_price * (
            default_point.violation_sum / max(context.profile.capacity_per_replica, 1e-9)
        )
        delta = derived_total - default_total
        verdict = f"{delta:+.2f}" + (" (derived better)" if delta < 0 else "")
        print(
            f"| {multiple:g} | {ratio.critical_ratio:.3f} | {point.violation_rate:.4f} | "
            f"{point.cost:.1f} | {verdict} |"
        )
    print(
        f"\nP95 default: violations {default_point.violation_rate:.4f}, "
        f"cost {default_point.cost:.1f}."
    )


def synthetic_context() -> ControlContext:
    series = generate_synthetic(
        "clean_daily",
        periods=SEASON * 14,
        step_seconds=60 * AGGREGATION,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return build_context(series, capacity_per_replica=8.0)


def azure_contexts(max_workloads: int) -> list[ControlContext]:
    """Load the busiest real workloads, with the selection rule stated.

    The decile-stratified tail of Azure Functions is overwhelmingly near-dead — many
    functions are literally zero for days. Those are not a capacity-planning problem
    (scale-to-zero plus a cold-start budget is the whole answer), so they are excluded here
    *by a stated rule* rather than quietly dropped, and the exclusion is itself a finding.
    """
    selection = select_cohort(ARCHIVE, top_n=8, per_decile=0, seed=20260802, selection_day=1)
    loaded = load_archive_cohort(ARCHIVE, selection)
    contexts: list[ControlContext] = []
    for series in loaded:
        if float(np.mean(series.values)) < 1.0:
            continue
        binned = aggregate_series(series, AGGREGATION, reducer="sum")
        capacity = max(1.0, float(np.quantile(binned.values, 0.5)) / 4.0)
        contexts.append(build_context(binned, capacity_per_replica=capacity))
        if len(contexts) >= max_workloads:
            break
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="skip the real archive")
    parser.add_argument("--workloads", type=int, default=2)
    args = parser.parse_args()

    print("# Cost-versus-violation frontier\n")
    print("Generated by `scripts/evaluate_frontier.py`.\n")
    print(f"- **Tuning parity:** {PARITY.describe()}.")
    print(
        f"- **Price:** ${PRICE_PER_REPLICA_HOUR}/replica-hour "
        "(Azure B2s, southeastasia, retail list, retrieved 2026-08-03). "
        "List price is not what an enterprise pays; the *shape* of the frontier is the "
        "result, not the absolute dollars."
    )
    print(f"- **Aggregation:** {AGGREGATION}-minute bins, declared not silent.")
    print(
        "- **Open-loop assumption:** replaying a trace against a new policy is a "
        "counterfactual, valid only because Azure Functions arrivals are externally "
        "driven and cannot respond to our provisioning. Rankings transfer; absolute "
        "savings carry a sim-to-real disclaimer.\n"
    )

    synthetic_ctx = synthetic_context()
    synthetic = evaluate(synthetic_ctx, "Synthetic `clean_daily` (ceiling by construction)")
    report(synthetic, "synthetic")
    report_newsvendor_question(synthetic_ctx, "synthetic")

    if not args.synthetic:
        if not ARCHIVE.exists():
            print(f"\n_Archive missing at {ARCHIVE}; run `make fetch-azure` first._")
            return
        for index, context in enumerate(azure_contexts(args.workloads)):
            label = (
                f"Azure Functions 2019 — workload {index + 1} "
                f"(`{context.series.workload_id[-16:]}`)"
            )
            points = evaluate(context, label)
            report(points, label)
            report_newsvendor_question(context, label)


if __name__ == "__main__":
    main()
