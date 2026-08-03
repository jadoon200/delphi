"""Budget pacing (C5) and the newsvendor controller (C6)."""

from datetime import UTC, datetime

import numpy as np
import pytest

from delphi.control.adaptive import BudgetPacedController, BudgetPacer, NewsvendorController
from delphi.control.controllers import ControlContext
from delphi.control.newsvendor import CostRatio
from delphi.control.simulator import CapacityProfile, simulate
from delphi.data.synthetic import generate_synthetic
from delphi.forecast.baselines import SeasonalNaiveForecaster

SEASON = 24


def _setup(kind: str = "clean_daily", periods: int = SEASON * 30) -> ControlContext:
    series = generate_synthetic(
        kind,  # type: ignore[arg-type]
        periods=periods,
        step_seconds=3600,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    profile = CapacityProfile(
        workload_id="w",
        step_seconds=3600,
        capacity_per_replica=8.0,
        startup_seconds=3600.0,
        teardown_seconds=3600.0,
        utilisation_target=1.0,
    )
    return ControlContext(series=series, profile=profile, start_step=SEASON * 8)


# --- the pacer in isolation -------------------------------------------------------


def test_pacer_becomes_conservative_when_over_budget() -> None:
    pacer = BudgetPacer(target_violation_rate=0.05)
    base = 0.95
    relaxed = pacer.effective_quantile(base)
    for _ in range(40):
        pacer.observe(violated=True)
    assert pacer.effective_quantile(base) > relaxed


def test_pacer_relaxes_when_comfortably_under_budget() -> None:
    pacer = BudgetPacer(target_violation_rate=0.20)
    for _ in range(60):
        pacer.observe(violated=False)
    assert pacer.effective_quantile(0.80) < 0.80


def test_pacer_respects_its_quantile_bounds() -> None:
    pacer = BudgetPacer(target_violation_rate=0.01, quantile_min=0.6, quantile_max=0.99)
    for _ in range(500):
        pacer.observe(violated=True)
    assert pacer.effective_quantile(0.9) <= 0.99
    calm = BudgetPacer(target_violation_rate=0.5, quantile_min=0.6, quantile_max=0.99)
    for _ in range(500):
        calm.observe(violated=False)
    assert calm.effective_quantile(0.9) >= 0.6


def test_pacer_integral_is_clamped_against_windup() -> None:
    """An unbounded integral pins the controller at maximum capacity forever."""
    pacer = BudgetPacer(target_violation_rate=0.05, integral_clamp=2.0)
    for _ in range(10_000):
        pacer.observe(violated=True)
    assert abs(pacer._integral) <= 2.0


def test_pacer_rejects_impossible_configuration() -> None:
    with pytest.raises(ValueError, match="target violation rate"):
        BudgetPacer(target_violation_rate=0.0)
    with pytest.raises(ValueError, match="quantile bounds"):
        BudgetPacer(target_violation_rate=0.05, quantile_min=0.9, quantile_max=0.5)


# --- the controllers --------------------------------------------------------------


def test_budget_paced_controller_tracks_its_compliance_target() -> None:
    context = _setup()
    controller = BudgetPacedController(
        forecaster=SeasonalNaiveForecaster(SEASON),
        target_violation_rate=0.10,
        refit_stride=SEASON,
        score_window=SEASON * 4,
    )
    plan = controller.plan(context)
    scored = slice(SEASON * 12, None)
    result = simulate(
        demand=context.demand[scored], requested=plan[scored], profile=context.profile
    )
    # a controller aiming at 10% should land far nearer 10% than an uncalibrated one
    assert result.violation_rate < 0.35


def test_a_costlier_breach_buys_strictly_more_capacity() -> None:
    """The newsvendor claim, end to end: price of failure -> provisioned capacity."""
    context = _setup()
    plans = {}
    for quantile in (0.5, 0.9, 0.99):
        controller = NewsvendorController(
            forecaster=SeasonalNaiveForecaster(SEASON),
            ratio=CostRatio.from_quantile(quantile),
            pace_budget=False,
            refit_stride=SEASON,
            score_window=SEASON * 4,
        )
        plans[quantile] = controller.plan(context)
    scored = slice(SEASON * 12, None)
    means = {q: float(np.mean(plan[scored])) for q, plan in plans.items()}
    assert means[0.5] < means[0.9] < means[0.99], means


def test_a_costlier_breach_lowers_the_violation_rate() -> None:
    context = _setup()
    scored = slice(SEASON * 12, None)
    rates = {}
    for quantile in (0.5, 0.95):
        plan = NewsvendorController(
            forecaster=SeasonalNaiveForecaster(SEASON),
            ratio=CostRatio.from_quantile(quantile),
            pace_budget=False,
            refit_stride=SEASON,
            score_window=SEASON * 4,
        ).plan(context)
        rates[quantile] = simulate(
            demand=context.demand[scored], requested=plan[scored], profile=context.profile
        ).violation_rate
    assert rates[0.95] < rates[0.5]


def test_controller_ids_expose_the_derived_quantile() -> None:
    controller = NewsvendorController(
        forecaster=SeasonalNaiveForecaster(SEASON),
        ratio=CostRatio(underage_per_unit=19.0, overage_per_unit=1.0),
    )
    assert "q*=0.9500" in controller.controller_id
    assert controller.target_quantile == pytest.approx(0.95)


def test_controllers_never_read_beyond_the_step_they_are_deciding() -> None:
    """A late spike must not change any earlier request."""
    context = _setup()
    controller = NewsvendorController(
        forecaster=SeasonalNaiveForecaster(SEASON),
        ratio=CostRatio.from_quantile(0.9),
        pace_budget=False,
        refit_stride=SEASON,
        score_window=SEASON * 4,
    )
    before = controller.plan(context)

    poisoned = context.demand.copy()
    cut = len(poisoned) - SEASON
    poisoned[cut:] = 10_000.0
    tampered = ControlContext(
        series=context.series.__class__(
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
    after = controller.plan(tampered)
    assert np.array_equal(before[:cut], after[:cut]), "future demand leaked into past plans"


def test_pacing_changes_the_plan_but_keeps_it_feasible() -> None:
    context = _setup("level_shift")
    common = {
        "forecaster": SeasonalNaiveForecaster(SEASON),
        "ratio": CostRatio.from_quantile(0.9),
        "refit_stride": SEASON,
        "score_window": SEASON * 4,
    }
    unpaced = NewsvendorController(pace_budget=False, **common).plan(context)  # type: ignore[arg-type]
    paced = NewsvendorController(pace_budget=True, **common).plan(context)  # type: ignore[arg-type]
    assert not np.array_equal(unpaced, paced)
    assert paced.min() >= context.profile.min_replicas
    assert paced.max() <= context.profile.max_replicas


def test_pacer_mirrors_the_clamped_plan_not_the_raw_request() -> None:
    """A bounded replica band must be visible to the controller's own feedback loop.

    If the internal actuation mirror runs on unclamped requests, the pacer counts
    violations that the platform's replica band made impossible, and then over-corrects
    for failures that never happened.
    """
    from delphi.control.simulator import apply_actuation_delay

    series = generate_synthetic(
        "clean_daily",
        periods=SEASON * 30,
        step_seconds=3600,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    profile = CapacityProfile(
        workload_id="w",
        step_seconds=3600,
        capacity_per_replica=8.0,
        startup_seconds=3600.0,
        teardown_seconds=3600.0,
        utilisation_target=1.0,
        min_replicas=4,
        max_replicas=6,
        scale_to_zero=False,
    )
    context = ControlContext(series=series, profile=profile, start_step=SEASON * 8)
    plan = NewsvendorController(
        forecaster=SeasonalNaiveForecaster(SEASON),
        ratio=CostRatio.from_quantile(0.99),
        pace_budget=True,
        refit_stride=SEASON,
        score_window=SEASON * 4,
    ).plan(context)

    assert plan.min() >= 4 and plan.max() <= 6
    # every emitted value must already be inside the band, so the final clamp is a no-op
    served = apply_actuation_delay(plan, profile)
    assert served.min() >= 4 and served.max() <= 6
