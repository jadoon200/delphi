"""Baseline controllers: causality, control law, and the behaviours that stop thrashing."""

from datetime import UTC, datetime

import numpy as np
import pytest

from delphi.control.controllers import (
    ControlContext,
    PercentileRecommender,
    ProactiveQuantileController,
    ReactiveController,
    StaticController,
)
from delphi.control.simulator import CapacityProfile
from delphi.data.series import DemandSeries
from delphi.data.synthetic import generate_synthetic
from delphi.forecast.baselines import SeasonalNaiveForecaster


def _series(values: np.ndarray, step_seconds: int = 60) -> DemandSeries:
    from datetime import timedelta

    start = datetime(2026, 1, 1, tzinfo=UTC)
    return DemandSeries(
        workload_id="w",
        source_id="synthetic-gold",
        resource_kind="invocations",
        unit="requests/minute",
        step_seconds=step_seconds,
        timestamps=tuple(
            start + timedelta(seconds=index * step_seconds) for index in range(len(values))
        ),
        values=values,
        is_imputed=np.zeros(len(values), dtype=np.bool_),
        quality=np.ones(len(values), dtype=np.float64),
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


def _context(values: np.ndarray, start_step: int = 50, **profile_kwargs: object) -> ControlContext:
    return ControlContext(
        series=_series(values),
        profile=_profile(**profile_kwargs),
        start_step=start_step,
    )


def test_observable_window_excludes_the_step_being_decided() -> None:
    """The actuation delay belongs to the simulator; counting it here too would double it."""
    context = _context(np.full(200, 100.0), startup_seconds=180.0)  # 3 steps
    assert context.observable_upto(100) == 100
    assert context.observable_upto(1) == 1
    assert context.observable_upto(0) == 0


def test_no_controller_can_see_the_step_it_is_planning() -> None:
    """A spike must not be anticipated by a purely reactive controller.

    If any of these reacted on the spike's own step they would be reading the future, and
    would beat the proactive family for entirely the wrong reason.
    """
    values = np.full(300, 50.0)
    values[200] = 5_000.0
    context = _context(values, startup_seconds=180.0)
    for controller in (
        ReactiveController(tolerance=0.0, stabilisation_steps=0),
        PercentileRecommender(window_steps=60, half_life_steps=30, margin=0.0),
    ):
        plan = controller.plan(context)
        assert plan[200] == plan[199], f"{controller.controller_id} reacted on the spike step"


def test_reactive_controller_tracks_the_last_observation() -> None:
    values = np.full(200, 250.0)
    context = _context(values, startup_seconds=0.0)
    plan = ReactiveController(tolerance=0.0, stabilisation_steps=0).plan(context)
    # 250 demand / 100 per replica -> 3 replicas
    assert set(plan[60:].tolist()) == {3}


def test_reactive_tolerance_suppresses_small_corrections() -> None:
    values = np.concatenate([np.full(100, 300.0), np.full(100, 310.0)])
    context = _context(values, startup_seconds=0.0)
    jumpy = ReactiveController(tolerance=0.0, stabilisation_steps=0).plan(context)
    calm = ReactiveController(tolerance=0.5, stabilisation_steps=0).plan(context)
    assert np.count_nonzero(np.diff(calm)) <= np.count_nonzero(np.diff(jumpy))


def test_stabilisation_window_delays_scale_down_but_not_scale_up() -> None:
    values = np.concatenate([np.full(100, 900.0), np.full(100, 100.0)])
    context = _context(values, startup_seconds=0.0)
    without = ReactiveController(tolerance=0.0, stabilisation_steps=0).plan(context)
    with_window = ReactiveController(tolerance=0.0, stabilisation_steps=30).plan(context)
    # the stabilised controller holds the higher capacity for longer
    assert with_window[105] > without[105]
    assert with_window[-1] == without[-1]


def test_percentile_recommender_sizes_above_the_median_of_its_window() -> None:
    rng = np.random.default_rng(5)
    values = np.abs(rng.normal(400, 120, 600))
    context = _context(values, startup_seconds=0.0)
    plan = PercentileRecommender(window_steps=120, half_life_steps=60, margin=0.0).plan(context)
    median_replicas = np.ceil(np.median(values) / 100.0)
    assert float(np.mean(plan[200:])) > median_replicas


def test_percentile_recommender_margin_only_adds_capacity() -> None:
    rng = np.random.default_rng(6)
    values = np.abs(rng.normal(400, 120, 400))
    context = _context(values, startup_seconds=0.0)
    lean = PercentileRecommender(window_steps=120, margin=0.0).plan(context)
    padded = PercentileRecommender(window_steps=120, margin=0.5).plan(context)
    assert np.all(padded >= lean)


def test_static_controller_is_constant_and_derived_only_from_warmup() -> None:
    values = np.concatenate([np.full(100, 100.0), np.full(100, 10_000.0)])
    context = _context(values, start_step=100, startup_seconds=0.0)
    plan = StaticController(percentile=0.95).plan(context)
    assert len(set(plan.tolist())) == 1
    # the post-warmup surge must not raise the level
    assert plan[0] == 1


def test_proactive_controller_anticipates_a_periodic_surge() -> None:
    """The point of forecasting: act before the ramp, not after it."""
    series = generate_synthetic(
        "clean_daily", periods=24 * 20, step_seconds=3600, start=datetime(2026, 1, 1, tzinfo=UTC)
    )
    profile = CapacityProfile(
        workload_id="w",
        step_seconds=3600,
        capacity_per_replica=10.0,
        startup_seconds=3600 * 3,  # 3 steps of lead time
        teardown_seconds=3600.0,
        utilisation_target=1.0,
    )
    context = ControlContext(series=series, profile=profile, start_step=24 * 8)
    proactive = ProactiveQuantileController(
        forecaster=SeasonalNaiveForecaster(24), quantile=0.95, refit_stride=24
    ).plan(context)
    reactive = ReactiveController(tolerance=0.0, stabilisation_steps=0).plan(context)

    from delphi.control.simulator import simulate

    scored = slice(24 * 10, None)
    proactive_result = simulate(
        demand=series.values[scored], requested=proactive[scored], profile=profile
    )
    reactive_result = simulate(
        demand=series.values[scored], requested=reactive[scored], profile=profile
    )
    assert proactive_result.violation_rate < reactive_result.violation_rate


def test_control_context_rejects_a_start_step_with_no_warmup() -> None:
    with pytest.raises(ValueError, match="warmup"):
        ControlContext(series=_series(np.full(10, 1.0)), profile=_profile(), start_step=0)


def test_every_controller_respects_the_profile_replica_band() -> None:
    rng = np.random.default_rng(9)
    values = np.abs(rng.normal(900, 400, 400))
    context = _context(values, startup_seconds=0.0, min_replicas=2, max_replicas=5)
    for controller in (
        StaticController(),
        ReactiveController(),
        PercentileRecommender(window_steps=60),
    ):
        plan = controller.plan(context)
        assert plan.min() >= 2 and plan.max() <= 5, controller.controller_id
