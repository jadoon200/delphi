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


def test_hypervolume_matches_the_area_computed_by_hand() -> None:
    """A relative check passes even when the area is double what it should be.

    One point at cost 5 with a 0.2 violation rate, against a reference corner at cost 10,
    dominates exactly the rectangle from cost 5 to 10 and violation 0.2 to 1.0 — an area of
    5 x 0.8 = 4.0. The previous implementation swept from cost zero, where nothing is
    affordable, and returned 8.0.
    """
    assert dominated_hypervolume([_point(5.0, 0.2)], cost_reference=10.0) == pytest.approx(4.0)

    staircase = [_point(2.0, 0.5), _point(5.0, 0.2), _point(8.0, 0.1)]
    expected = (5 - 2) * (1 - 0.5) + (8 - 5) * (1 - 0.2) + (10 - 8) * (1 - 0.1)
    assert dominated_hypervolume(staircase, cost_reference=10.0) == pytest.approx(expected)


def test_hypervolume_ranks_two_curves_the_way_the_area_does() -> None:
    """The off-by-one strip could reverse a ranking, not merely inflate both sides."""
    reaches_low_but_dear = [_point(1.0, 0.9), _point(9.0, 0.05)]
    cheap_and_middling = [_point(4.0, 0.5), _point(5.0, 0.45)]
    assert dominated_hypervolume(cheap_and_middling, cost_reference=10.0) > dominated_hypervolume(
        reaches_low_but_dear, cost_reference=10.0
    )


def test_hypervolume_ignores_points_outside_the_reference_box() -> None:
    """Only points strictly inside the reference corner contribute.

    Note what does *not* count as outside: a cheap point with a dreadful violation rate is
    still non-dominated and still dominates a thin strip of its own. Excluding it would
    understate the frontier, so the box test is against the reference corner alone.
    """
    inside = [_point(5.0, 0.2)]
    too_dear = _point(50.0, 0.01)
    no_better_than_reference = _point(1.0, 1.0)
    assert dominated_hypervolume(
        [*inside, too_dear, no_better_than_reference], cost_reference=10.0
    ) == pytest.approx(dominated_hypervolume(inside, cost_reference=10.0))

    # A cheap-but-bad point is inside the box and legitimately adds its own strip.
    assert dominated_hypervolume([*inside, _point(1.0, 0.99)], cost_reference=10.0) == (
        pytest.approx(4.0 + (5 - 1) * (1 - 0.99))
    )


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
