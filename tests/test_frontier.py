"""Pareto machinery: the comparison object must be a curve, computed correctly."""

import pytest

from delphi.evaluation.frontier import (
    FrontierPoint,
    ParitySpec,
    cost_at_matched_violation,
    dominated_hypervolume,
    format_frontier_table,
    frontier_dynamic_range,
    pareto_front,
)


def _point(cost: float, violation: float, family: str = "f", setting: float = 0.9) -> FrontierPoint:
    return FrontierPoint(
        controller_id=f"{family}@{setting}",
        family=family,
        setting=setting,
        violation_rate=violation,
        violation_sum=violation * 100,
        cost=cost,
        capacity_hours=cost,
        scaling_actions=0,
        mean_replicas=cost / 10,
        regret=cost - 10,
    )


def test_pareto_front_drops_dominated_points() -> None:
    points = [
        _point(10.0, 0.20),
        _point(20.0, 0.10),
        _point(30.0, 0.05),
        _point(25.0, 0.30),  # dominated: costs more than 20 and violates more
        _point(35.0, 0.12),  # dominated: costs more than 30 and violates more
    ]
    front = pareto_front(points)
    assert [(p.cost, p.violation_rate) for p in front] == [
        (10.0, 0.20),
        (20.0, 0.10),
        (30.0, 0.05),
    ]


def test_pareto_front_is_monotone_in_both_axes() -> None:
    points = [_point(float(c), 1.0 / c) for c in range(1, 30)]
    front = pareto_front(points)
    costs = [point.cost for point in front]
    violations = [point.violation_rate for point in front]
    assert costs == sorted(costs)
    assert violations == sorted(violations, reverse=True)


def test_pareto_front_of_a_single_point_is_that_point() -> None:
    assert len(pareto_front([_point(5.0, 0.1)])) == 1


def test_pareto_front_of_nothing_is_empty() -> None:
    assert pareto_front([]) == []


def test_cost_at_matched_violation_reports_unreachable_rather_than_the_nearest() -> None:
    """A controller that never hits the target must be shown as unable to."""
    points = [_point(10.0, 0.20), _point(20.0, 0.12)]
    assert cost_at_matched_violation(points, 0.25) == 10.0
    assert cost_at_matched_violation(points, 0.15) == 20.0
    assert cost_at_matched_violation(points, 0.01) is None


def test_cost_at_matched_violation_picks_the_cheapest_feasible_point() -> None:
    points = [_point(30.0, 0.02), _point(18.0, 0.03), _point(50.0, 0.01)]
    assert cost_at_matched_violation(points, 0.05) == 18.0


def test_dynamic_range_flags_a_saturated_benchmark() -> None:
    tied = [_point(100.0, 0.05), _point(100.4, 0.06), _point(100.2, 0.04)]
    assert frontier_dynamic_range(tied) == pytest.approx(0.4)
    assert frontier_dynamic_range(tied) / 100.0 < 0.05  # would trip the saturation warning

    spread = [_point(50.0, 0.30), _point(150.0, 0.01)]
    assert frontier_dynamic_range(spread) / 50.0 > 0.05


def test_hypervolume_rewards_a_frontier_closer_to_the_ideal_corner() -> None:
    good = [_point(10.0, 0.02), _point(20.0, 0.01)]
    poor = [_point(10.0, 0.40), _point(20.0, 0.30)]
    reference = 100.0
    assert dominated_hypervolume(good, cost_reference=reference) > dominated_hypervolume(
        poor, cost_reference=reference
    )


def test_hypervolume_of_an_empty_frontier_is_zero() -> None:
    assert dominated_hypervolume([], cost_reference=10.0) == 0.0


def test_parity_spec_states_the_budget_every_arm_was_held_to() -> None:
    spec = ParitySpec(
        configurations_per_controller=7,
        selection_metric="newsvendor loss",
        validation_fraction=0.25,
    )
    described = spec.describe()
    assert "7 configurations" in described
    assert "newsvendor loss" in described
    assert "25%" in described


def test_table_renders_every_point_sorted_by_violation_rate() -> None:
    table = format_frontier_table([_point(20.0, 0.10), _point(10.0, 0.30)])
    lines = table.splitlines()
    assert len(lines) == 4  # header, separator, two rows
    assert "0.1000" in lines[2]
    assert "0.3000" in lines[3]
