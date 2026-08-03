"""The M4 validation gate: prove the simulator before trusting anything it produces.

Four checks, in increasing order of how easy they are to skip:

1. **Analytic** — reproduce the Erlang-C closed form for M/M/c. A queue that cannot do this
   is not modelling a queue, and every downstream capacity number inherits the error.
2. **Degenerate** — infinite capacity never violates, zero capacity always does, constant
   demand yields a constant plan, and zero actuation delay reproduces the plan exactly.
3. **Determinism** — identical inputs produce byte-identical output.
4. **Sensitivity** — vary every constant that was *guessed* rather than measured, and
   report how far the *conclusions* move, not just the numbers.

Check 4 is the one people skip, and it is the difference between a simulator and a
storytelling device. A conclusion that survives it can be published with confidence; one
that does not must be labelled fragile in ``docs/EVAL.md``.

Run: ``python scripts/validate_simulator.py``
"""

from datetime import UTC, datetime
from itertools import pairwise

import numpy as np

from delphi.control.controllers import (
    ControlContext,
    PercentileRecommender,
    ProactiveQuantileController,
    ReactiveController,
)
from delphi.control.queueing import (
    mmc_mean_wait_seconds,
    poisson_arrivals,
    simulate_multi_server_queue,
)
from delphi.control.simulator import (
    CapacityProfile,
    CostModel,
    apply_actuation_delay,
    expost_optimal_plan,
    simulate,
)
from delphi.data.synthetic import generate_synthetic
from delphi.forecast.baselines import SeasonalNaiveForecaster

STEP_SECONDS = 3600
SEASON = 24


def check_erlang_c() -> bool:
    print("## 1. Analytic — event-driven queue vs the Erlang-C closed form\n")
    print("| servers | utilisation | Erlang-C Wq (s) | simulated Wq (s) | rel. error |")
    print("|---:|---:|---:|---:|---:|")
    rng = np.random.default_rng(20260803)
    service_rate = 0.5
    passed = True
    for servers, utilisation in ((1, 0.5), (2, 0.6), (3, 0.7), (5, 0.8), (8, 0.85), (10, 0.6)):
        arrival_rate = utilisation * servers * service_rate
        duration = 1_500_000 / arrival_rate
        arrivals = poisson_arrivals(
            rate_per_second=arrival_rate, duration_seconds=duration, rng=rng
        )
        service = rng.exponential(1 / service_rate, size=len(arrivals))
        pool = np.full(int(duration // 60) + 2, servers, dtype=np.int64)
        outcome = simulate_multi_server_queue(
            arrival_seconds=arrivals,
            service_seconds=service,
            servers_per_step=pool,
            step_seconds=60.0,
        )
        theoretical = mmc_mean_wait_seconds(arrival_rate, service_rate, servers)
        error = abs(outcome.mean_wait_seconds - theoretical) / theoretical
        passed &= error < 0.05
        print(
            f"| {servers} | {utilisation:.2f} | {theoretical:.4f} | "
            f"{outcome.mean_wait_seconds:.4f} | {error:.2%} |"
        )
    print(f"\n**Erlang-C reproduction: {'PASS' if passed else 'FAIL'}** (tolerance 5%)\n")
    return passed


def check_degenerate() -> bool:
    print("## 2. Degenerate cases\n")
    profile = CapacityProfile(
        workload_id="w",
        step_seconds=STEP_SECONDS,
        capacity_per_replica=100.0,
        startup_seconds=0.0,
        teardown_seconds=0.0,
        utilisation_target=1.0,
    )
    demand = np.asarray([10.0, 500.0, 90.0, 250.0], dtype=np.float64)
    results = {
        "infinite capacity -> no violations": simulate(
            demand=demand, requested=np.full(4, 1000, dtype=np.int64), profile=profile
        ).violation_rate
        == 0.0,
        "zero capacity -> all violations": simulate(
            demand=demand, requested=np.zeros(4, dtype=np.int64), profile=profile
        ).violation_rate
        == 1.0,
        "constant demand -> no churn": simulate(
            demand=np.full(50, 250.0),
            requested=expost_optimal_plan(np.full(50, 250.0), profile),
            profile=profile,
        ).scaling_actions
        == 0,
        "zero delay -> plan is served exactly": bool(
            np.array_equal(
                apply_actuation_delay(np.asarray([1, 7, 2], dtype=np.int64), profile),
                np.asarray([1, 7, 2], dtype=np.int64),
            )
        ),
        "hindsight plan meets every step": simulate(
            demand=demand, requested=expost_optimal_plan(demand, profile), profile=profile
        ).violation_rate
        == 0.0,
    }
    for name, ok in results.items():
        print(f"- {'PASS' if ok else 'FAIL'} — {name}")
    passed = all(results.values())
    print(f"\n**Degenerate cases: {'PASS' if passed else 'FAIL'}**\n")
    return passed


def check_determinism() -> bool:
    print("## 3. Determinism\n")
    rng = np.random.default_rng(2)
    demand = np.abs(rng.normal(400, 90, 300))
    profile = CapacityProfile(
        workload_id="w",
        step_seconds=STEP_SECONDS,
        capacity_per_replica=100.0,
        startup_seconds=float(STEP_SECONDS * 2),
    )
    requested = np.full(300, 5, dtype=np.int64)
    runs = [
        simulate(
            demand=demand,
            requested=requested,
            profile=profile,
            fidelity="queue",
            wait_slo_seconds=30.0,
            seed=99,
        )
        for _ in range(2)
    ]
    identical = (
        runs[0].violation_magnitude.tobytes() == runs[1].violation_magnitude.tobytes()
        and runs[0].provisioned.tobytes() == runs[1].provisioned.tobytes()
        and runs[0].cost == runs[1].cost
    )
    print(f"- {'PASS' if identical else 'FAIL'} — repeated runs are byte-identical\n")
    return identical


def _score(
    controller_plan: np.ndarray,
    demand: np.ndarray,
    profile: CapacityProfile,
    costs: CostModel,
    scored: slice,
) -> tuple[float, float]:
    result = simulate(
        demand=demand[scored],
        requested=controller_plan[scored],
        profile=profile,
        cost_model=costs,
    )
    return result.violation_rate, result.cost


def check_sensitivity() -> bool:
    """Does 'proactive beats reactive' survive perturbing every guessed constant?"""
    print("## 4. Sensitivity — do the conclusions move when the guesses do?\n")
    series = generate_synthetic(
        "clean_daily",
        periods=SEASON * 40,
        step_seconds=STEP_SECONDS,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    demand = series.values
    scored = slice(SEASON * 15, None)
    start_step = SEASON * 10

    print("### 4a. Lead time — the pre-registered prediction (Q1)\n")
    print("The proactive margin must *grow* with `startup_seconds`. If it does not, the")
    print("simulator's actuation delay is not doing its job.\n")
    print("| startup (steps) | reactive viol. | proactive viol. | margin | rel. reduction |")
    print("|---:|---:|---:|---:|---:|")
    margins = []
    reactive_rates = []
    for lead_steps in (0, 1, 2, 4, 8):
        profile = CapacityProfile(
            workload_id="w",
            step_seconds=STEP_SECONDS,
            capacity_per_replica=10.0,
            startup_seconds=float(STEP_SECONDS * lead_steps),
            teardown_seconds=float(STEP_SECONDS),
            utilisation_target=1.0,
        )
        context = ControlContext(series=series, profile=profile, start_step=start_step)
        costs = CostModel(price_per_replica_hour=1.0)
        reactive, _ = _score(
            ReactiveController(tolerance=0.0, stabilisation_steps=0).plan(context),
            demand,
            profile,
            costs,
            scored,
        )
        proactive, _ = _score(
            ProactiveQuantileController(
                forecaster=SeasonalNaiveForecaster(SEASON), quantile=0.95, refit_stride=SEASON
            ).plan(context),
            demand,
            profile,
            costs,
            scored,
        )
        margins.append(reactive - proactive)
        reactive_rates.append(reactive)
        relative = (reactive - proactive) / reactive if reactive > 0 else float("nan")
        print(
            f"| {lead_steps} | {reactive:.3f} | {proactive:.3f} | "
            f"{reactive - proactive:+.3f} | {relative:.1%} |"
        )
    grows = margins[-1] > margins[0]
    monotone = all(later >= earlier for earlier, later in pairwise(margins))
    saturated = max(reactive_rates) > 0.5
    print(
        f"\n**Margin grows with lead time: {'PASS' if grows else 'FAIL'}** "
        f"({margins[0]:+.3f} at 0 steps -> {margins[-1]:+.3f} at 8 steps)"
    )
    print(f"- Monotone across every lead: {'yes' if monotone else 'NO — margin is non-monotone'}")
    if saturated:
        print(
            "- **Saturation warning:** the reactive baseline exceeds 50% violations at the "
            "longer leads, so both controllers are mostly failing there. Read the *ranking* "
            "in this table, not the levels: a tight utilisation ceiling plus integer "
            "replicas makes the violation rate hypersensitive. Recorded per L5 Trap 3."
        )
    print()

    print("### 4b. Guessed constants — does the ranking survive?\n")
    print("| perturbation | reactive viol. | proactive viol. | proactive wins? |")
    print("|---|---:|---:|:--:|")
    base = {
        "capacity_per_replica": 10.0,
        "startup_seconds": float(STEP_SECONDS * 3),
        "teardown_seconds": float(STEP_SECONDS),
        "utilisation_target": 1.0,
    }
    perturbations = {
        "baseline": {},
        "utilisation ceiling 0.7": {"utilisation_target": 0.7},
        "utilisation ceiling 0.9": {"utilisation_target": 0.9},
        "replica capacity x0.5": {"capacity_per_replica": 5.0},
        "replica capacity x2": {"capacity_per_replica": 20.0},
        "teardown x6": {"teardown_seconds": float(STEP_SECONDS * 6)},
    }
    rankings = []
    for name, override in perturbations.items():
        profile = CapacityProfile(
            workload_id="w",
            step_seconds=STEP_SECONDS,
            **{**base, **override},  # type: ignore[arg-type]
        )
        context = ControlContext(series=series, profile=profile, start_step=start_step)
        costs = CostModel(price_per_replica_hour=1.0)
        reactive, _ = _score(
            ReactiveController(tolerance=0.0, stabilisation_steps=0).plan(context),
            demand,
            profile,
            costs,
            scored,
        )
        proactive, _ = _score(
            ProactiveQuantileController(
                forecaster=SeasonalNaiveForecaster(SEASON), quantile=0.95, refit_stride=SEASON
            ).plan(context),
            demand,
            profile,
            costs,
            scored,
        )
        wins = proactive <= reactive
        rankings.append(wins)
        print(f"| {name} | {reactive:.3f} | {proactive:.3f} | {'yes' if wins else 'NO'} |")
    stable = all(rankings)
    print(
        f"\n**Ranking stable under perturbation: {'PASS' if stable else 'FRAGILE'}** "
        f"({sum(rankings)}/{len(rankings)})\n"
    )

    print("### 4c. Churn cost — does pricing thrashing change the cost ranking?\n")
    print("| churn cost/action | reactive cost | percentile cost | proactive cost |")
    print("|---:|---:|---:|---:|")
    profile = CapacityProfile(workload_id="w", step_seconds=STEP_SECONDS, **base)  # type: ignore[arg-type]
    context = ControlContext(series=series, profile=profile, start_step=start_step)
    plans = {
        "reactive": ReactiveController(tolerance=0.0, stabilisation_steps=0).plan(context),
        "percentile": PercentileRecommender(
            window_steps=SEASON * 3, half_life_steps=SEASON, margin=0.15
        ).plan(context),
        "proactive": ProactiveQuantileController(
            forecaster=SeasonalNaiveForecaster(SEASON), quantile=0.95, refit_stride=SEASON
        ).plan(context),
    }
    for churn in (0.0, 1.0, 10.0):
        costs = CostModel(price_per_replica_hour=1.0, churn_cost_per_action=churn)
        row = [_score(plan, demand, profile, costs, scored)[1] for plan in plans.values()]
        print(f"| {churn:.0f} | " + " | ".join(f"{value:.1f}" for value in row) + " |")
    print()
    return grows and stable


def main() -> None:
    print("# Simulator validation gate\n")
    print("Generated by `scripts/validate_simulator.py`. The M4 checkpoint is not green")
    print("until every section below passes.\n")
    outcomes = {
        "analytic (Erlang-C)": check_erlang_c(),
        "degenerate": check_degenerate(),
        "determinism": check_determinism(),
        "sensitivity": check_sensitivity(),
    }
    print("## Verdict\n")
    for name, ok in outcomes.items():
        print(f"- {'PASS' if ok else 'FAIL'} — {name}")
    verdict = "GREEN" if all(outcomes.values()) else "NOT GREEN"
    print(f"\n**M4 gate: {verdict}**")


if __name__ == "__main__":
    main()
