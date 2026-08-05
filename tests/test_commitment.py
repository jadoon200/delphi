"""Commitment-interval controllers: hold within a window, and never read the future."""

from datetime import UTC, datetime

import numpy as np
import pytest

from delphi.control.commitment import (
    BackwardCommitmentController,
    ForwardCommitmentController,
    commitment_boundaries,
)
from delphi.control.controllers import ControlContext
from delphi.control.simulator import CapacityProfile
from delphi.data.series import DemandSeries
from delphi.data.synthetic import generate_synthetic
from delphi.forecast.baselines import SeasonalNaiveForecaster

SEASON = 24


def _context(kind: str = "clean_daily", periods: int = SEASON * 30) -> ControlContext:
    series = generate_synthetic(
        kind,  # type: ignore[arg-type]
        periods=periods,
        step_seconds=3600,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    profile = CapacityProfile(
        workload_id="w",
        step_seconds=3600,
        capacity_per_replica=10.0,
        startup_seconds=3600.0,
        teardown_seconds=3600.0,
        utilisation_target=1.0,
    )
    return ControlContext(series=series, profile=profile, start_step=SEASON * 2)


def test_boundaries_start_at_the_planning_origin_and_step_by_the_window() -> None:
    context = _context()
    boundaries = list(commitment_boundaries(context, 6))
    assert boundaries[0] == context.start_step
    assert boundaries[1] - boundaries[0] == 6


def test_a_commitment_is_held_for_its_whole_window() -> None:
    """The defining property: capacity may only change at a boundary."""
    context = _context()
    for controller in (
        BackwardCommitmentController(window_steps=6, quantile=0.95),
        ForwardCommitmentController(
            window_steps=6, forecaster=SeasonalNaiveForecaster(SEASON), quantile=0.95
        ),
    ):
        plan = controller.plan(context)
        changes = np.diff(plan[context.start_step :]).nonzero()[0]
        assert np.all((changes + 1) % 6 == 0), controller.controller_id


def test_backward_and_forward_apply_the_same_arithmetic() -> None:
    """Both take a q-quantile of a window; only the window's provenance differs.

    A perfectly periodic series makes yesterday's window identical to the coming one, so
    the two controllers must agree — anything else would mean the sizing rules differ and
    the comparison would not be about information.
    """
    series = generate_synthetic(
        "clean_daily",
        periods=SEASON * 30,
        step_seconds=3600,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        seed=7,
    )
    exact = DemandSeries(
        workload_id=series.workload_id,
        source_id=series.source_id,
        resource_kind=series.resource_kind,
        unit=series.unit,
        step_seconds=series.step_seconds,
        timestamps=series.timestamps,
        # strictly periodic with period SEASON: yesterday == today
        values=np.tile(series.values[:SEASON], len(series) // SEASON),
        is_imputed=series.is_imputed,
        quality=series.quality,
    )
    profile = CapacityProfile(
        workload_id="w",
        step_seconds=3600,
        capacity_per_replica=10.0,
        startup_seconds=0.0,
        utilisation_target=1.0,
    )
    context = ControlContext(series=exact, profile=profile, start_step=SEASON * 2)
    backward = BackwardCommitmentController(window_steps=SEASON, quantile=0.9).plan(context)
    forward = ForwardCommitmentController(
        window_steps=SEASON, forecaster=SeasonalNaiveForecaster(SEASON), quantile=0.9
    ).plan(context)
    assert np.array_equal(backward[SEASON * 3 :], forward[SEASON * 3 :])


def test_neither_controller_reads_beyond_its_decision_point() -> None:
    context = _context()
    poisoned = context.demand.copy()
    cut = len(poisoned) - SEASON * 2
    poisoned[cut:] = 50_000.0
    tampered = ControlContext(
        series=DemandSeries(
            workload_id=context.series.workload_id,
            source_id=context.series.source_id,
            resource_kind=context.series.resource_kind,
            unit=context.series.unit,
            step_seconds=context.series.step_seconds,
            timestamps=context.series.timestamps,
            values=poisoned,
            is_imputed=context.series.is_imputed,
            quality=context.series.quality,
        ),
        profile=context.profile,
        start_step=context.start_step,
    )
    for controller in (
        BackwardCommitmentController(window_steps=6, quantile=0.95),
        ForwardCommitmentController(
            window_steps=6, forecaster=SeasonalNaiveForecaster(SEASON), quantile=0.95
        ),
    ):
        before = controller.plan(context)
        after = controller.plan(tampered)
        assert np.array_equal(before[:cut], after[:cut]), controller.controller_id


def test_a_higher_quantile_never_commits_less_capacity() -> None:
    context = _context()
    previous = -1.0
    for quantile in (0.5, 0.8, 0.95, 0.99):
        plan = ForwardCommitmentController(
            window_steps=12, forecaster=SeasonalNaiveForecaster(SEASON), quantile=quantile
        ).plan(context)
        mean = float(np.mean(plan[SEASON * 4 :]))
        assert mean >= previous
        previous = mean


def test_controllers_reject_impossible_configuration() -> None:
    with pytest.raises(ValueError, match="commitment window"):
        BackwardCommitmentController(window_steps=0)
    with pytest.raises(ValueError, match="quantile"):
        ForwardCommitmentController(
            window_steps=6, forecaster=SeasonalNaiveForecaster(SEASON), quantile=1.5
        )


def test_plans_stay_inside_the_replica_band() -> None:
    series = generate_synthetic(
        "burst_known",
        periods=SEASON * 20,
        step_seconds=3600,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    profile = CapacityProfile(
        workload_id="w",
        step_seconds=3600,
        capacity_per_replica=10.0,
        startup_seconds=3600.0,
        utilisation_target=1.0,
        min_replicas=2,
        max_replicas=6,
        scale_to_zero=False,
    )
    context = ControlContext(series=series, profile=profile, start_step=SEASON * 2)
    for controller in (
        BackwardCommitmentController(window_steps=6, quantile=0.99),
        ForwardCommitmentController(
            window_steps=6, forecaster=SeasonalNaiveForecaster(SEASON), quantile=0.99
        ),
    ):
        plan = controller.plan(context)
        assert plan.min() >= 2 and plan.max() <= 6, controller.controller_id
