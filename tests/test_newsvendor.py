"""The newsvendor identity is the project's thesis, so it is verified, not assumed."""

import numpy as np
import pytest

from delphi.control.newsvendor import (
    CostRatio,
    expected_newsvendor_cost,
    implied_ratio_of_controller,
    interpolate_quantile,
    size_from_quantiles,
)
from delphi.control.simulator import CapacityProfile


def _profile(**overrides: object) -> CapacityProfile:
    defaults: dict[str, object] = {
        "workload_id": "w",
        "step_seconds": 60,
        "capacity_per_replica": 100.0,
        "utilisation_target": 1.0,
    }
    defaults.update(overrides)
    return CapacityProfile(**defaults)  # type: ignore[arg-type]


def test_critical_ratio_matches_the_closed_form() -> None:
    assert CostRatio(underage_per_unit=9.0, overage_per_unit=1.0).critical_ratio == 0.9
    assert CostRatio(underage_per_unit=1.0, overage_per_unit=1.0).critical_ratio == 0.5
    assert CostRatio(underage_per_unit=19.0, overage_per_unit=1.0).critical_ratio == 0.95


def test_from_quantile_inverts_the_identity() -> None:
    for quantile in (0.5, 0.8, 0.9, 0.95, 0.99):
        assert CostRatio.from_quantile(quantile).critical_ratio == pytest.approx(quantile)


def test_a_p95_controller_implicitly_prices_failure_at_19x() -> None:
    """Reading a fixed-target autoscaler's unstated economics back out."""
    ratio = CostRatio.from_quantile(0.95)
    assert ratio.underage_per_unit / ratio.overage_per_unit == pytest.approx(19.0)
    assert implied_ratio_of_controller(0.05) == pytest.approx(19.0)


@pytest.mark.parametrize("quantile", [0.5, 0.7, 0.9, 0.95, 0.99])
def test_the_critical_ratio_quantile_actually_minimises_expected_cost(quantile: float) -> None:
    """The identity is the whole argument, so it is checked numerically, not trusted.

    Sweeping capacity and taking the argmin must land on the critical-ratio quantile of
    the demand distribution.
    """
    rng = np.random.default_rng(20260803)
    demand = rng.gamma(shape=6.0, scale=25.0, size=200_000)
    ratio = CostRatio.from_quantile(quantile)

    grid = np.linspace(demand.min(), np.quantile(demand, 0.9999), 4000)
    costs = [
        expected_newsvendor_cost(capacity=float(c), demand_samples=demand, ratio=ratio)
        for c in grid
    ]
    empirical_best = float(grid[int(np.argmin(costs))])
    theoretical_best = float(np.quantile(demand, quantile))
    assert empirical_best == pytest.approx(theoretical_best, rel=0.02)


def test_interpolation_clamps_instead_of_inventing_a_tail() -> None:
    """A ratio implying p99.9 cannot be answered by a forecast that stops at p95."""
    levels = (0.5, 0.9, 0.95)
    values = np.asarray([100.0, 150.0, 170.0])
    assert interpolate_quantile(levels, values, 0.999) == 170.0
    assert interpolate_quantile(levels, values, 0.1) == 100.0
    assert interpolate_quantile(levels, values, 0.925) == pytest.approx(160.0)


def test_sizing_converts_a_cost_ratio_into_replicas() -> None:
    levels = (0.5, 0.9, 0.99)
    values = np.asarray([100.0, 250.0, 500.0])
    profile = _profile(capacity_per_replica=100.0)
    cheap_failure = size_from_quantiles(
        levels=levels, values=values, ratio=CostRatio.from_quantile(0.5), profile=profile
    )
    costly_failure = size_from_quantiles(
        levels=levels, values=values, ratio=CostRatio.from_quantile(0.99), profile=profile
    )
    assert cheap_failure.replicas == 1
    assert costly_failure.replicas == 5
    assert costly_failure.quantile > cheap_failure.quantile


def test_expensive_failure_never_buys_less_than_cheap_failure() -> None:
    """Monotonicity: raising the price of a breach can only raise capacity."""
    levels = (0.5, 0.8, 0.9, 0.95, 0.99)
    values = np.asarray([80.0, 120.0, 160.0, 210.0, 400.0])
    profile = _profile()
    previous = -1
    for quantile in (0.5, 0.8, 0.9, 0.95, 0.99):
        decision = size_from_quantiles(
            levels=levels,
            values=values,
            ratio=CostRatio.from_quantile(quantile),
            profile=profile,
        )
        assert decision.replicas >= previous
        previous = decision.replicas


def test_sizing_respects_the_profile_band_and_records_its_reasoning() -> None:
    profile = _profile(min_replicas=2, max_replicas=3, scale_to_zero=False)
    decision = size_from_quantiles(
        levels=(0.5, 0.99),
        values=np.asarray([10.0, 100_000.0]),
        ratio=CostRatio.from_quantile(0.99),
        profile=profile,
    )
    assert decision.replicas == 3
    evidence = decision.as_evidence()
    assert evidence["q_star"] == pytest.approx(0.99)
    assert evidence["underage_per_unit"] > evidence["overage_per_unit"]


def test_zero_or_negative_prices_are_refused() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        CostRatio(underage_per_unit=0.0, overage_per_unit=1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        CostRatio(underage_per_unit=1.0, overage_per_unit=-1.0)


def test_a_controller_that_never_violates_implies_an_infinite_price_of_failure() -> None:
    assert implied_ratio_of_controller(0.0) == float("inf")
