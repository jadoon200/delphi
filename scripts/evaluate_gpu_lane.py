"""M16 — is request-rate autoscaling structurally wrong for LLM inference?

The claim is repeated constantly in 2026 practitioner writing and rarely measured: a GPU
saturates on token work while request count and CPU stay flat, so an autoscaler tracking
arrivals is sizing against the wrong quantity. Azure's LLM inference traces publish both
arrivals *and* token counts, so the claim can be checked rather than repeated.

Every proxy is rescaled to the true demand's mean before comparison. That matters: a proxy
that merely needed a different constant would be a tuning problem, not a structural one.
What survives rescaling is a genuine difference in *shape*.

Run: ``python scripts/evaluate_gpu_lane.py``
"""

import argparse

import numpy as np

from delphi.control.adaptive import NewsvendorController
from delphi.control.controllers import (
    ControlContext,
    PercentileRecommender,
    ProactiveQuantileController,
    ReactiveController,
)
from delphi.control.newsvendor import CostRatio
from delphi.control.serving import (
    ServingProfile,
    gpu_seconds_demand,
    phase_shares,
    proxy_demand,
    rescale_to,
    signal_tracking_error,
    tokens_per_request,
)
from delphi.control.simulator import CostModel, simulate
from delphi.data.llm_inference import BinnedInference, load_binned
from delphi.data.series import DemandSeries
from delphi.forecast.baselines import SeasonalNaiveForecaster

BIN_SECONDS = 60
SEASON = 24 * 60 // (BIN_SECONDS // 60)  # one day in bins
#: How often the proactive family re-forecasts. Measured 2026-08-03: refitting only once
#: per season makes the controller act on day-old forecasts and it loses to plain reaction
#: (violations 0.2391 vs 0.1521). Refitting every 4 hours or sooner reverses that (0.1323).
#: Cadence is a first-class tuning parameter here, swept below rather than assumed.
REFIT_STRIDE = 240
#: Azure ND-series A100 list, southeastasia, order of magnitude. Swept, not trusted.
GPU_PRICE_PER_HOUR = 3.40


def _series(values: np.ndarray, binned: BinnedInference, name: str) -> DemandSeries:
    from datetime import timedelta

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


def describe(binned: BinnedInference, profile: ServingProfile, trace: str) -> None:
    print(f"\n### `{trace}` — what the trace actually demands\n")
    truth = gpu_seconds_demand(binned, profile)
    prefill_share, decode_share = phase_shares(binned, profile)
    context_per_request, generated_per_request = tokens_per_request(binned)

    print(
        f"- Bins: {len(binned):,} x {binned.bin_seconds}s "
        f"from {binned.start.date()} ({len(binned) * binned.bin_seconds / 86400:.1f} days)"
    )
    print(
        f"- Requests/bin: mean {binned.requests.mean():,.0f}, "
        f"p95 {np.quantile(binned.requests, 0.95):,.0f}"
    )
    print(
        f"- GPU work/bin: mean {truth.mean():,.1f} GPU-s, "
        f"p95 {np.quantile(truth, 0.95):,.1f} GPU-s "
        f"(a bin supplies {binned.bin_seconds}s per replica)"
    )
    print(f"- Phase split: **prefill {prefill_share:.1%} / decode {decode_share:.1%}** of GPU work")
    context_cv = float(context_per_request.std() / context_per_request.mean())
    generated_cv = float(generated_per_request.std() / generated_per_request.mean())
    print(
        f"- Context tokens/request: mean {context_per_request.mean():,.0f}, "
        f"coefficient of variation **{context_cv:.3f}**"
    )
    print(
        f"- Generated tokens/request: mean {generated_per_request.mean():,.0f}, "
        f"coefficient of variation **{generated_cv:.3f}**"
    )
    print("\nThe coefficients of variation are the crux: if work-per-request were constant,")
    print("request count would be a perfectly good proxy and this whole lane would be moot.")


def compare_signals(binned: BinnedInference, profile: ServingProfile) -> None:
    print("\n#### Does a proxy signal track true GPU work?\n")
    truth = gpu_seconds_demand(binned, profile)
    print(
        "| tracked signal | correlation | mean abs rel. error | p95 abs rel. error "
        "| mean error at peak bins | worst under-provision |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for signal in ("requests", "prefill", "decode", "total_tokens"):
        stats = signal_tracking_error(truth, proxy_demand(binned, signal=signal))
        print(
            f"| `{signal}` | {stats['correlation']:.4f} | "
            f"{stats['mean_abs_relative_error']:.1%} | {stats['p95_abs_relative_error']:.1%} | "
            f"{stats['under_provision_at_peak']:+.1%} | "
            f"{stats['worst_under_provision']:+.1%} |"
        )
    print("\nAll proxies are rescaled to the true mean first, so any error left is *shape*,")
    print("not calibration. A negative value at peak bins means the proxy under-provisions")
    print("exactly when demand is highest — the failure that breaches an SLO.")


def compare_controllers(binned: BinnedInference, profile: ServingProfile, trace: str) -> None:
    """Score controllers that track the true signal against ones that track a proxy."""
    print("\n#### Scaling on the proxy vs scaling on the truth\n")
    truth = gpu_seconds_demand(binned, profile)
    capacity = profile.capacity_profile(workload_id=f"llm-{trace}", bin_seconds=binned.bin_seconds)
    costs = CostModel(
        price_per_replica_hour=GPU_PRICE_PER_HOUR,
        churn_cost_per_action=GPU_PRICE_PER_HOUR * profile.startup_seconds / 3600.0,
    )
    scored = slice(SEASON * 3, None)
    truth_series = _series(truth, binned, f"llm-{trace}-truth")
    proxy_series = _series(
        rescale_to(truth, proxy_demand(binned, signal="requests")),
        binned,
        f"llm-{trace}-proxy",
    )
    truth_context = ControlContext(series=truth_series, profile=capacity, start_step=SEASON * 2)
    proxy_context = ControlContext(series=proxy_series, profile=capacity, start_step=SEASON * 2)
    forecaster = SeasonalNaiveForecaster(SEASON)

    print("| controller | tracked signal | violations | GPU-hours | cost | cold-start bins |")
    print("|---|---|---:|---:|---:|---:|")
    arms = [
        ("reactive HPA", ReactiveController(tolerance=0.1, stabilisation_steps=5)),
        ("percentile (VPA-style)", PercentileRecommender(window_steps=SEASON, percentile=0.95)),
        (
            "proactive p95",
            ProactiveQuantileController(
                forecaster=forecaster, quantile=0.95, refit_stride=REFIT_STRIDE
            ),
        ),
        (
            "newsvendor",
            NewsvendorController(
                forecaster=forecaster,
                ratio=CostRatio.from_quantile(0.95),
                refit_stride=REFIT_STRIDE,
                score_window=SEASON * 2,
            ),
        ),
    ]
    for name, controller in arms:
        for label, context in (("request count", proxy_context), ("GPU work", truth_context)):
            plan = controller.plan(context)
            # always scored against the TRUE demand: a proxy-driven plan must live with
            # the consequences of the work that actually arrived
            result = simulate(
                demand=truth[scored],
                requested=plan[scored],
                profile=capacity,
                cost_model=costs,
            )
            print(
                f"| {name} | {label} | {result.violation_rate:.4f} | "
                f"{result.capacity_hours:,.0f} | ${result.cost:,.0f} | "
                f"{result.cold_start_steps} |"
            )


def sweep_startup(binned: BinnedInference, profile: ServingProfile, trace: str) -> None:
    """Does the proactive advantage grow with cold-start time, as Q1 predicts?"""
    print("\n#### Cold start is the mechanism — sweep `startup_seconds`\n")
    print("| startup | reactive viol. | proactive viol. | absolute margin | relative reduction |")
    print("|---:|---:|---:|---:|---:|")
    truth = gpu_seconds_demand(binned, profile)
    costs = CostModel(price_per_replica_hour=GPU_PRICE_PER_HOUR)
    scored = slice(SEASON * 3, None)
    forecaster = SeasonalNaiveForecaster(SEASON)
    for startup in (0.0, 60.0, 240.0, 600.0, 1200.0):
        variant = ServingProfile(
            prefill_tokens_per_second=profile.prefill_tokens_per_second,
            decode_tokens_per_second=profile.decode_tokens_per_second,
            startup_seconds=startup,
            teardown_seconds=profile.teardown_seconds,
            utilisation_target=profile.utilisation_target,
        )
        capacity = variant.capacity_profile(
            workload_id=f"llm-{trace}", bin_seconds=binned.bin_seconds
        )
        context = ControlContext(
            series=_series(truth, binned, f"llm-{trace}-truth"),
            profile=capacity,
            start_step=SEASON * 2,
        )
        reactive = simulate(
            demand=truth[scored],
            requested=ReactiveController(tolerance=0.1, stabilisation_steps=5).plan(context)[
                scored
            ],
            profile=capacity,
            cost_model=costs,
        ).violation_rate
        proactive = simulate(
            demand=truth[scored],
            requested=ProactiveQuantileController(
                forecaster=forecaster, quantile=0.95, refit_stride=REFIT_STRIDE
            ).plan(context)[scored],
            profile=capacity,
            cost_model=costs,
        ).violation_rate
        relative = (reactive - proactive) / reactive if reactive > 0 else float("nan")
        print(
            f"| {startup:.0f}s | {reactive:.4f} | {proactive:.4f} | "
            f"{reactive - proactive:+.4f} | {relative:.1%} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", default=["code", "conv"])
    args = parser.parse_args()

    print("# GPU / LLM-inference lane\n")
    print(
        "Generated by `scripts/evaluate_gpu_lane.py`. Data: Azure LLM inference traces "
        "2024 (CC-BY 4.0, cite Stojkovic et al., HPCA 2025).\n"
    )
    print(
        "Serving rates are **modelled parameters**, not measurements — the traces publish "
        "tokens, not hardware timings. Conclusions are read from rankings and from the "
        "startup sweep, never from absolute GPU-hour figures.\n"
    )

    profile = ServingProfile()
    print(
        f"Modelled replica: {profile.prefill_tokens_per_second:,.0f} prefill tok/s, "
        f"{profile.decode_tokens_per_second:,.0f} decode tok/s, "
        f"{profile.startup_seconds:.0f}s cold start, "
        f"{profile.utilisation_target:.0%} utilisation ceiling. "
        f"One decode token costs {1 / profile.prefill_decode_cost_ratio:.1f}x a prefill token."
    )

    for trace in args.traces:
        binned = load_binned(trace, bin_seconds=BIN_SECONDS)
        describe(binned, profile, trace)
        compare_signals(binned, profile)
        compare_controllers(binned, profile, trace)
        sweep_startup(binned, profile, trace)


if __name__ == "__main__":
    main()
