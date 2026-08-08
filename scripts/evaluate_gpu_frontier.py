"""The decisive question: does forecasting beat a percentile recommender, ever?

Earlier sweeps compared proactive control against reactive HPA. That is the wrong
benchmark — reactive HPA is a strawman and beating it proves nothing. The real incumbent is
the sliding-window percentile recommender that Borg Autopilot and Kubernetes VPA already
ship, and on every measurement so far it has matched or beaten forecasting.

So this script puts every family on one cost-versus-violation frontier, at several
cold-start times, and asks a single falsifiable question:

    **Is any forecast-driven controller ever non-dominated — cheaper at the same violation
    rate, or safer at the same cost — than the best percentile recommender?**

If the answer is no at every cold start, forecasting does not earn its place in this
project and that gets published as the finding.

Run: ``python scripts/evaluate_gpu_frontier.py``
"""

import argparse
from datetime import timedelta

import numpy as np

from delphi.control.adaptive import NewsvendorController
from delphi.control.controllers import (
    ControlContext,
    PercentileRecommender,
    ProactiveQuantileController,
    ReactiveController,
    StaticController,
)
from delphi.control.newsvendor import CostRatio
from delphi.control.serving import ServingProfile, gpu_seconds_demand
from delphi.control.simulator import CostModel
from delphi.data.llm_inference import BinnedInference, load_binned
from delphi.data.series import DemandSeries
from delphi.evaluation.frontier import (
    FrontierPoint,
    cost_at_matched_violation,
    pareto_front,
    score_plan,
)
from delphi.forecast.baselines import SeasonalNaiveForecaster

BIN_SECONDS = 60
SEASON = 1440
REFIT_STRIDE = 240
GPU_PRICE_PER_HOUR = 3.40
SETTINGS: tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
#: Window lengths the percentile recommender is allowed to tune over. Giving the incumbent
#: its own tuning dimension is the point: a baseline that only got one configuration would
#: not be a fair opponent.
PERCENTILE_WINDOWS: tuple[int, ...] = (60, 240, SEASON // 2, SEASON)
#: Above this violation rate no operator would run, so frontier points there prove nothing.
USABLE_VIOLATION_CEILING = 0.10


def _series(values: np.ndarray, binned: BinnedInference, name: str) -> DemandSeries:
    return DemandSeries(
        workload_id=name,
        source_id="azure-llm-inference-2024",
        resource_kind="gpu_seconds",
        unit="gpu-seconds/bin",
        step_seconds=binned.bin_seconds,
        timestamps=tuple(
            binned.start + timedelta(seconds=binned.bin_seconds * index)
            for index in range(len(values))
        ),
        values=np.maximum(values, 0.0),
        is_imputed=np.zeros(len(values), dtype=np.bool_),
        quality=np.ones(len(values), dtype=np.float64),
    )


def frontier_for(
    binned: BinnedInference,
    truth: np.ndarray,
    startup_seconds: float,
    trace: str,
) -> list[FrontierPoint]:
    profile = ServingProfile(startup_seconds=startup_seconds)
    capacity = profile.capacity_profile(workload_id=f"llm-{trace}", bin_seconds=binned.bin_seconds)
    costs = CostModel(
        price_per_replica_hour=GPU_PRICE_PER_HOUR,
        churn_cost_per_action=GPU_PRICE_PER_HOUR * startup_seconds / 3600.0,
    )
    context = ControlContext(
        series=_series(truth, binned, f"llm-{trace}"), profile=capacity, start_step=SEASON * 2
    )
    scored = slice(SEASON * 3, None)
    forecaster = SeasonalNaiveForecaster(SEASON)
    points: list[FrontierPoint] = []

    def add(controller_id: str, family: str, setting: float, plan: np.ndarray) -> None:
        points.append(
            score_plan(
                controller_id=controller_id,
                family=family,
                setting=setting,
                plan=plan,
                context=context,
                cost_model=costs,
                scored=scored,
            )
        )

    for setting in SETTINGS:
        add(
            f"static:p{setting}",
            "static",
            setting,
            StaticController(percentile=setting).plan(context),
        )
        add(
            f"reactive:tol={1 - setting:.2f}",
            "reactive",
            setting,
            ReactiveController(tolerance=1.0 - setting, stabilisation_steps=5).plan(context),
        )
        # the incumbent gets its own tuning dimension, not a single configuration
        for window in PERCENTILE_WINDOWS:
            add(
                f"percentile:w={window}:p{setting}",
                "percentile",
                setting,
                PercentileRecommender(
                    window_steps=window,
                    percentile=setting,
                    half_life_steps=max(window // 2, 1),
                    margin=0.0,
                ).plan(context),
            )
        add(
            f"proactive:q{setting}",
            "proactive",
            setting,
            ProactiveQuantileController(
                forecaster=forecaster, quantile=setting, refit_stride=REFIT_STRIDE
            ).plan(context),
        )
        add(
            f"newsvendor:q{setting}",
            "newsvendor",
            setting,
            NewsvendorController(
                forecaster=forecaster,
                ratio=CostRatio.from_quantile(setting),
                refit_stride=REFIT_STRIDE,
                score_window=SEASON * 2,
            ).plan(context),
        )
    return points


def report(points: list[FrontierPoint], trace: str, startup: float) -> dict[str, int]:
    front = pareto_front(points)
    on_front: dict[str, int] = {}
    for point in front:
        on_front[point.family] = on_front.get(point.family, 0) + 1

    print(f"\n### `{trace}` — cold start {startup:.0f}s\n")
    print(
        "| family | points on the Pareto frontier | cost @ ≤5% viol. | cost @ ≤1% viol. "
        "| best violation rate |"
    )
    print("|---|---:|---:|---:|---:|")
    for family in ("static", "reactive", "percentile", "proactive", "newsvendor"):
        arm = [point for point in points if point.family == family]
        if not arm:
            continue
        five = cost_at_matched_violation(arm, 0.05)
        one = cost_at_matched_violation(arm, 0.01)
        best = min(point.violation_rate for point in arm)
        print(
            f"| {family} | {on_front.get(family, 0)} | "
            f"{'—' if five is None else f'${five:,.0f}'} | "
            f"{'—' if one is None else f'${one:,.0f}'} | {best:.4f} |"
        )

    forecast_families = {"proactive", "newsvendor"}
    # Counting raw frontier points flatters the loser: a controller that is very cheap and
    # very unsafe is non-dominated simply because nothing else is that reckless. Only the
    # operationally usable region counts, and cost at a matched violation rate is the
    # headline, because that is the comparison an operator would actually make.
    usable = [point for point in points if point.violation_rate <= USABLE_VIOLATION_CEILING]
    usable_front = pareto_front(usable)
    forecast_usable = sum(1 for point in usable_front if point.family in forecast_families)
    incumbent_best_5 = cost_at_matched_violation(
        [p for p in points if p.family == "percentile"], 0.05
    )
    forecast_best_5 = cost_at_matched_violation(
        [p for p in points if p.family in forecast_families], 0.05
    )
    print(
        f"\n- Frontier points inside the usable region "
        f"(≤{USABLE_VIOLATION_CEILING:.0%} violations): "
        f"**{forecast_usable} forecast-driven** of {len(usable_front)} total"
    )
    verdict = "no"
    if incumbent_best_5 is not None and forecast_best_5 is not None:
        delta = forecast_best_5 - incumbent_best_5
        verdict = "YES" if delta < 0 else "no"
        better = "cheaper" if delta < 0 else "more expensive"
        print(
            f"- At ≤5% violations: forecasting ${forecast_best_5:,.0f} vs percentile "
            f"${incumbent_best_5:,.0f} — forecasting is **${abs(delta):,.0f} {better}** "
            f"({abs(delta) / incumbent_best_5:.1%})"
        )
    elif forecast_best_5 is not None and incumbent_best_5 is None:
        verdict = "YES"
        print("- Only the forecast family reached ≤5% violations at all.")
    print(f"- **Does forecasting earn its place here? {verdict}**")
    return {
        "forecast_usable_front": forecast_usable,
        "beats_incumbent": int(verdict == "YES"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", default=["code", "conv"])
    parser.add_argument("--startups", nargs="+", type=float, default=[60.0, 240.0, 600.0, 1200.0])
    args = parser.parse_args()

    print("# Does forecasting beat the incumbent percentile recommender?\n")
    print(
        "Every family on one cost-versus-violation frontier. The percentile recommender "
        "gets its own tuning dimension (4 window lengths x 7 percentiles = 28 "
        "configurations) because a baseline given one configuration is not an opponent.\n"
    )
    print(
        "A family 'earns its place' only if it contributes a **non-dominated** point: "
        "cheaper at the same violation rate, or safer at the same cost.\n"
    )

    summary: dict[tuple[str, float], dict[str, int]] = {}
    for trace in args.traces:
        binned = load_binned(trace, bin_seconds=BIN_SECONDS)
        truth = gpu_seconds_demand(binned, ServingProfile())
        for startup in args.startups:
            points = frontier_for(binned, truth, startup, trace)
            summary[(trace, startup)] = report(points, trace, startup)

    print("\n## Verdict\n")
    print("| trace | cold start | forecast points in usable region | beats percentile? |")
    print("|---|---:|---:|:--:|")
    for (trace, startup), row in summary.items():
        beats = "yes" if row["beats_incumbent"] else "**no**"
        print(f"| `{trace}` | {startup:.0f}s | {row['forecast_usable_front']} | {beats} |")
    wins = sum(row["beats_incumbent"] for row in summary.values())
    print(
        f"\n**Forecasting beat the incumbent percentile recommender in {wins} of "
        f"{len(summary)} settings.**"
    )
    if wins == 0:
        print(
            "\nAgainst a properly tuned percentile recommender, forecast-driven capacity "
            "control does not earn its place in this lane. That is the finding, and it is "
            "published rather than reframed."
        )


if __name__ == "__main__":
    main()
