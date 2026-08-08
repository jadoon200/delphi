"""Degenerate, determinism, and actuation-delay checks for the replay simulator."""

import numpy as np
import pytest

from delphi.control.simulator import (
    CapacityProfile,
    CostModel,
    apply_actuation_delay,
    expost_optimal_plan,
    regret,
    simulate,
)


def _profile(**overrides: object) -> CapacityProfile:
    defaults: dict[str, object] = {
        "workload_id": "w",
        "step_seconds": 60,
        "capacity_per_replica": 100.0,
        "startup_seconds": 180.0,
        "teardown_seconds": 60.0,
        "utilisation_target": 1.0,
    }
    defaults.update(overrides)
    return CapacityProfile(**defaults)  # type: ignore[arg-type]


def test_scale_up_lands_only_after_the_startup_delay() -> None:
    """The parameter that makes forecasting worth anything must actually bite."""
    profile = _profile(startup_seconds=180.0)  # 3 steps at 60s
    requested = np.asarray([1, 1, 5, 5, 5, 5, 5], dtype=np.int64)
    provisioned = apply_actuation_delay(requested, profile)
    # requested at index 2, so it serves from index 5
    assert provisioned.tolist() == [1, 1, 1, 1, 1, 5, 5]


def test_scale_down_uses_the_teardown_delay_not_the_startup_delay() -> None:
    profile = _profile(startup_seconds=300.0, teardown_seconds=60.0)
    requested = np.asarray([5, 5, 1, 1, 1], dtype=np.int64)
    provisioned = apply_actuation_delay(requested, profile)
    assert provisioned.tolist() == [5, 5, 5, 1, 1]


def test_a_later_request_supersedes_one_still_in_flight() -> None:
    """A reconciliation loop acts on desired state, not a backlog of past intentions."""
    profile = _profile(startup_seconds=180.0)
    requested = np.asarray([1, 4, 9, 9, 9, 9, 9], dtype=np.int64)
    provisioned = apply_actuation_delay(requested, profile)
    assert 4 not in provisioned.tolist()
    assert provisioned.tolist()[-1] == 9


def test_a_continuously_changing_request_still_lands() -> None:
    """Superseding retargets the in-flight change; it must not restart its clock.

    Restarting it froze capacity forever under any monotone ramp — a 1..12 request served
    a flat 1 replica — which silently penalised every smooth, forecast-driven controller
    while leaving the deadbanded reactive baseline untouched. The ranking-based validation
    gate could not see it because it moved all controllers the same way.
    """
    profile = _profile(startup_seconds=180.0)  # three steps
    requested = np.arange(1, 13, dtype=np.int64)
    provisioned = apply_actuation_delay(requested, profile)

    assert provisioned[-1] > provisioned[0], "a rising request must eventually provision"
    assert provisioned.tolist() == [1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9]

    # And a ramp that settles must reach the level it settled on.
    settling = np.asarray([1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6], dtype=np.int64)
    assert apply_actuation_delay(settling, profile).tolist()[-1] == 6


def test_zero_actuation_delay_makes_the_plan_immediate() -> None:
    profile = _profile(startup_seconds=0.0, teardown_seconds=0.0)
    requested = np.asarray([1, 7, 2], dtype=np.int64)
    assert apply_actuation_delay(requested, profile).tolist() == [1, 7, 2]


def test_infinite_capacity_never_violates_and_zero_capacity_always_does() -> None:
    demand = np.asarray([10.0, 50.0, 90.0], dtype=np.float64)
    profile = _profile(startup_seconds=0.0, min_replicas=0)
    plenty = simulate(
        demand=demand,
        requested=np.full(3, 100, dtype=np.int64),
        profile=profile,
    )
    assert plenty.violation_rate == 0.0
    assert plenty.violation_sum == 0.0

    nothing = simulate(
        demand=demand,
        requested=np.zeros(3, dtype=np.int64),
        profile=profile,
    )
    assert nothing.violation_rate == 1.0
    assert nothing.violation_sum == pytest.approx(demand.sum())


def test_constant_demand_produces_a_constant_plan_and_no_churn() -> None:
    demand = np.full(50, 250.0, dtype=np.float64)
    profile = _profile(startup_seconds=0.0)
    optimal = expost_optimal_plan(demand, profile)
    assert len(set(optimal.tolist())) == 1
    result = simulate(demand=demand, requested=optimal, profile=profile)
    assert result.scaling_actions == 0
    assert result.violation_rate == 0.0


def test_simulation_is_byte_identical_across_runs() -> None:
    rng = np.random.default_rng(3)
    demand = np.abs(rng.normal(300, 60, 200))
    requested = np.full(200, 4, dtype=np.int64)
    profile = _profile()
    first = simulate(
        demand=demand,
        requested=requested,
        profile=profile,
        fidelity="queue",
        wait_slo_seconds=5.0,
        seed=11,
    )
    second = simulate(
        demand=demand,
        requested=requested,
        profile=profile,
        fidelity="queue",
        wait_slo_seconds=5.0,
        seed=11,
    )
    assert first.violation_magnitude.tobytes() == second.violation_magnitude.tobytes()
    assert first.provisioned.tobytes() == second.provisioned.tobytes()
    assert first.cost == second.cost


def test_expost_optimal_plan_is_the_cheapest_feasible_trajectory() -> None:
    demand = np.asarray([0.0, 100.0, 101.0, 250.0], dtype=np.float64)
    profile = _profile(startup_seconds=0.0, capacity_per_replica=100.0, utilisation_target=1.0)
    optimal = expost_optimal_plan(demand, profile)
    assert optimal.tolist() == [0, 1, 2, 3]
    # one replica fewer anywhere must break feasibility
    result = simulate(demand=demand, requested=optimal, profile=profile)
    assert result.violation_rate == 0.0
    thinner = np.maximum(optimal - 1, 0)
    assert simulate(demand=demand, requested=thinner, profile=profile).violation_rate > 0


def test_regret_is_zero_for_the_hindsight_plan_and_positive_for_over_provisioning() -> None:
    demand = np.full(60, 150.0, dtype=np.float64)
    profile = _profile(startup_seconds=0.0)
    costs = CostModel(price_per_replica_hour=1.0)
    optimal = expost_optimal_plan(demand, profile)
    assert regret(
        simulate(demand=demand, requested=optimal, profile=profile, cost_model=costs),
        profile,
        costs,
    ) == pytest.approx(0.0)
    wasteful = simulate(demand=demand, requested=optimal + 3, profile=profile, cost_model=costs)
    assert regret(wasteful, profile, costs) > 0


def test_churn_cost_penalises_thrashing() -> None:
    demand = np.full(20, 100.0, dtype=np.float64)
    profile = _profile(startup_seconds=0.0, teardown_seconds=0.0)
    flapping = np.asarray([1, 5] * 10, dtype=np.int64)
    steady = np.full(20, 3, dtype=np.int64)
    costs = CostModel(price_per_replica_hour=1.0, churn_cost_per_action=2.0)
    noisy = simulate(demand=demand, requested=flapping, profile=profile, cost_model=costs)
    calm = simulate(demand=demand, requested=steady, profile=profile, cost_model=costs)
    assert noisy.scaling_actions > calm.scaling_actions
    assert noisy.cost > calm.cost


def test_cold_start_steps_count_unlanded_scale_ups() -> None:
    profile = _profile(startup_seconds=180.0)
    demand = np.full(8, 10.0, dtype=np.float64)
    requested = np.asarray([1, 1, 6, 6, 6, 6, 6, 6], dtype=np.int64)
    result = simulate(demand=demand, requested=requested, profile=profile)
    assert result.cold_start_steps == 3


def test_queue_fidelity_requires_an_explicit_wait_slo() -> None:
    with pytest.raises(ValueError, match="wait_slo_seconds"):
        simulate(
            demand=np.ones(5, dtype=np.float64),
            requested=np.ones(5, dtype=np.int64),
            profile=_profile(),
            fidelity="queue",
        )


def test_profile_rejects_an_impossible_replica_band() -> None:
    with pytest.raises(ValueError, match="replica bounds"):
        _profile(min_replicas=5, max_replicas=2)


def test_plan_is_clamped_into_the_profile_band() -> None:
    profile = _profile(startup_seconds=0.0, min_replicas=2, max_replicas=6, scale_to_zero=False)
    result = simulate(
        demand=np.full(4, 10.0, dtype=np.float64),
        requested=np.asarray([0, 1, 99, 4], dtype=np.int64),
        profile=profile,
    )
    assert result.requested.tolist() == [2, 2, 6, 4]
