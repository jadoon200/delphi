"""Calibrated, budget-paced capacity controllers — C5 and C6.

C5 reproduces the published state of the art (BACC, arXiv 2606.20575): an online
conformally-calibrated forecast plus a PI loop that *paces* the consumption of a
fixed-period SLO violation budget. Hitting a monthly target in aggregate is not enough —
spending the whole month's error budget on day two satisfies nobody.

C6 is DELPHI's contribution and differs in exactly one place: the compliance target is not
an input. It is derived from the two prices via the newsvendor identity, so the controller
buys the quantile the economics imply rather than the quantile someone typed in. The delta
between C5 and C6 is therefore the whole claim, and M8 measures it.

Both share one engine so the comparison is not contaminated by incidental implementation
differences.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from delphi.control.controllers import ControlContext, PercentileRecommender, _replicas_for
from delphi.control.newsvendor import CostRatio, interpolate_quantile
from delphi.control.simulator import ActuationTracker, clamp_plan
from delphi.forecast.calibration import AdaptiveConformalCalibrator
from delphi.forecast.contracts import QuantileForecaster

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Levels forecast once per block; the effective quantile is interpolated between them so
#: the PI loop can move continuously without refitting the model every step.
DEFAULT_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.99)


@dataclass
class BudgetPacer:
    """PI controller over the cumulative SLO violation rate.

    ``e_t = v_hat_t - epsilon_budget`` drives a proportional-integral correction that
    shifts the effective quantile. Burning the budget too fast makes the controller more
    conservative; running under budget lets it relax and stop over-buying.
    """

    target_violation_rate: float
    kp: float = 1.0
    ki: float = 0.1
    quantile_min: float = 0.50
    quantile_max: float = 0.995
    integral_clamp: float = 5.0
    _integral: float = field(default=0.0, init=False)
    _observed: int = field(default=0, init=False)
    _violations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0 < self.target_violation_rate < 1:
            raise ValueError("target violation rate must lie strictly within (0, 1)")
        if self.kp < 0 or self.ki < 0:
            raise ValueError("PI gains cannot be negative")
        if not 0 < self.quantile_min < self.quantile_max < 1:
            raise ValueError("quantile bounds must satisfy 0 < min < max < 1")

    @property
    def observed_violation_rate(self) -> float:
        return self._violations / self._observed if self._observed else 0.0

    def observe(self, *, violated: bool) -> None:
        self._observed += 1
        self._violations += int(violated)
        error = self.observed_violation_rate - self.target_violation_rate
        # Anti-windup: an unbounded integral term pins the controller at max capacity and
        # never lets it come back down.
        self._integral = float(
            np.clip(self._integral + error, -self.integral_clamp, self.integral_clamp)
        )

    def effective_quantile(self, base_quantile: float) -> float:
        """Nudge the base quantile by the pacing correction, within bounds."""
        error = self.observed_violation_rate - self.target_violation_rate
        correction = self.kp * error + self.ki * self._integral
        # Positive error = over budget = buy a higher quantile.
        adjusted = base_quantile + correction * (1.0 - base_quantile)
        return float(np.clip(adjusted, self.quantile_min, self.quantile_max))


def _plan_calibrated(
    context: ControlContext,
    *,
    forecaster: QuantileForecaster,
    base_quantile_at: Callable[[int], float],
    pacer: BudgetPacer | None,
    levels: tuple[float, ...],
    refit_stride: int,
    gamma: float,
    score_window: int,
    aci_target_coverage: float,
) -> IntArray:
    """Walk forward emitting a plan, calibrating and pacing causally as we go.

    The controller tracks the consequences of its *own* past requests: it knows what it
    asked for and it knows the platform's lag, so it can reconstruct what was actually
    serving and compare that to demand it has since observed. Nothing here reads demand at
    or beyond the step being decided.
    """
    profile = context.profile
    demand = context.demand
    lead = profile.startup_steps
    total = len(demand)
    requested = np.empty(total, dtype=np.int64)

    def _clamped(replicas: int) -> int:
        """Clamp at assignment, not only at the end.

        The pacer reconstructs what its own past requests would actually have served, so
        it has to see the values the platform will honour. Mirroring an unclamped plan
        makes it count violations that a bounded replica band could never have produced,
        and it then over-corrects for them.
        """
        bounded = int(np.clip(replicas, profile.min_replicas, profile.max_replicas))
        return bounded if profile.scale_to_zero else max(bounded, 1)

    warmup = demand[: context.start_step]
    requested[:] = _clamped(_replicas_for(float(np.quantile(warmup, 0.95)), profile))

    calibrator: AdaptiveConformalCalibrator | None = None
    # raw forecast value emitted for each step, so its residual can be scored once the
    # true demand for that step becomes observable
    raw_for_step: dict[int, float] = {}
    # incremental mirror of the simulator's actuation delay, over our own plan. Shares the
    # simulator's implementation rather than restating it, so the two cannot drift.
    tracker = ActuationTracker(profile, initial=int(requested[0]))
    served: dict[int, int] = {}
    settled = context.start_step

    step = context.start_step
    while step < total:
        # --- settle everything now observable: demand[:step] is known ---------------
        while settled < step:
            if settled in raw_for_step and calibrator is not None:
                calibrator.observe(
                    timestamp=context.series.timestamps[settled],
                    raw_prediction=raw_for_step.pop(settled),
                    actual=float(demand[settled]),
                )
            if pacer is not None and settled in served:
                capacity = served[settled] * profile.capacity_per_replica
                capacity *= profile.utilisation_target
                pacer.observe(violated=bool(demand[settled] > capacity))
            settled += 1

        origin = context.observable_upto(step)
        if origin < forecaster.minimum_history:
            step += 1
            continue

        block = min(refit_stride, total - step)
        forecast = forecaster.forecast(
            context.series.slice(0, origin),
            horizon_steps=lead + block,
            quantile_levels=levels,
        )

        if calibrator is None:
            residuals = _warmup_residuals(context, forecaster, levels, aci_target_coverage)
            calibrator = AdaptiveConformalCalibrator(
                residuals,
                target_coverage=aci_target_coverage,
                gamma=gamma,
                score_window=score_window,
            )

        for offset in range(block):
            target_step = step + offset
            base = base_quantile_at(target_step)
            quantile = pacer.effective_quantile(base) if pacer is not None else base
            row = forecast.quantile_values[lead + offset]
            raw = interpolate_quantile(levels, row, quantile)
            correction = calibrator.current_correction()
            requested[target_step] = _clamped(_replicas_for(max(raw + correction, 0.0), profile))
            raw_for_step[target_step] = raw

            # advance our mirror of the platform's actuation lag one step
            served[target_step] = tracker.advance(target_step, int(requested[target_step]))

        step += block

    return clamp_plan(requested, profile)


def _warmup_residuals(
    context: ControlContext,
    forecaster: QuantileForecaster,
    levels: tuple[float, ...],
    coverage: float,
) -> FloatArray:
    """Seed ACI from one-step residuals over the warmup window only.

    Strictly pre-``start_step`` data, so the calibrator never sees an evaluated step.
    """
    demand = context.demand
    origin = max(forecaster.minimum_history, context.start_step // 2)
    residuals: list[float] = []
    for target in range(origin, context.start_step):
        forecast = forecaster.forecast(
            context.series.slice(0, target), horizon_steps=1, quantile_levels=levels
        )
        raw = interpolate_quantile(levels, forecast.quantile_values[0], coverage)
        residuals.append(float(demand[target]) - raw)
    if not residuals:
        return np.zeros(1, dtype=np.float64)
    return np.asarray(residuals, dtype=np.float64)


@dataclass(frozen=True)
class BudgetPacedController:
    """C5 — the published baseline: ACI-calibrated forecast plus PI budget pacing.

    The compliance target is an *input* here, exactly as in BACC. That is the thing C6
    replaces, and keeping C5 faithful is what makes the comparison mean anything.
    """

    forecaster: QuantileForecaster
    target_violation_rate: float = 0.05
    gamma: float = 0.01
    kp: float = 1.0
    ki: float = 0.1
    refit_stride: int = 60
    score_window: int = 1440
    levels: tuple[float, ...] = DEFAULT_LEVELS

    @property
    def controller_id(self) -> str:
        return (
            f"budget_paced:target={self.target_violation_rate:g}:"
            f"gamma={self.gamma:g}:kp={self.kp:g}:ki={self.ki:g}:{self.forecaster.model_id}"
        )

    @property
    def minimum_history(self) -> int:
        return self.forecaster.minimum_history

    def plan(self, context: ControlContext) -> IntArray:
        base = 1.0 - self.target_violation_rate
        pacer = BudgetPacer(
            target_violation_rate=self.target_violation_rate, kp=self.kp, ki=self.ki
        )
        return _plan_calibrated(
            context,
            forecaster=self.forecaster,
            base_quantile_at=lambda _: base,
            pacer=pacer,
            levels=self.levels,
            refit_stride=self.refit_stride,
            gamma=self.gamma,
            score_window=self.score_window,
            aci_target_coverage=base,
        )


@dataclass(frozen=True)
class NewsvendorPercentileController:
    """C7 — the incumbent's predictor, with the target derived from prices.

    Measured on the Azure LLM inference traces (2026-08-04): a sliding-window percentile
    recommender beat every explicit forecaster tried — seasonal-naive, persistence and
    drift — at all four cold-start settings on both traces. The reason is in the data:
    GPU-work autocorrelation is 0.989 at one minute and 0.730 at one day, so *recent load*
    carries almost all the signal and an explicit model mostly adds its own error on top.

    The honest conclusion is not that the newsvendor idea fails — it is that it was
    attached to the wrong predictor. **A percentile recommender is already a forecaster**,
    and a strong one; what it has never had is a principled way to choose *which*
    percentile. That choice is normally convention (VPA ships p95).

    So this keeps the winning predictor and replaces only the arbitrary part:
    ``q* = C_u / (C_u + C_o)`` sets the percentile, and ACI keeps it honest under drift.
    The claim under test narrows from "forecasting helps" to "deriving the target from
    prices helps" — which is the claim the project actually wanted to make.
    """

    ratio: CostRatio
    window_steps: int = 240
    half_life_steps: int = 120
    gamma: float = 0.01
    calibrate: bool = True
    margin: float = 0.0

    def __post_init__(self) -> None:
        if self.window_steps < 1 or self.half_life_steps < 1:
            raise ValueError("window and half-life must be positive")
        if self.margin < 0:
            raise ValueError("margin cannot be negative")

    @property
    def controller_id(self) -> str:
        suffix = "aci" if self.calibrate else "raw"
        return (
            f"newsvendor_percentile:q*={self.ratio.critical_ratio:.4f}:"
            f"w={self.window_steps}:{suffix}:v1"
        )

    @property
    def target_quantile(self) -> float:
        return self.ratio.critical_ratio

    def plan(self, context: ControlContext) -> IntArray:
        profile = context.profile
        demand = context.demand
        quantile = self.ratio.critical_ratio
        base = PercentileRecommender(
            window_steps=self.window_steps,
            percentile=quantile,
            half_life_steps=self.half_life_steps,
            margin=self.margin,
        ).plan(context)
        if not self.calibrate:
            return base

        # Online conformal correction on the recommender's own residuals: it is a predictor
        # like any other, so its coverage can drift and can be corrected.
        per_replica = profile.capacity_per_replica * profile.utilisation_target
        predicted = base.astype(np.float64) * per_replica
        warmup = slice(0, context.start_step)
        residuals = demand[warmup] - predicted[warmup]
        calibrator = AdaptiveConformalCalibrator(
            residuals if len(residuals) else np.zeros(1, dtype=np.float64),
            target_coverage=quantile,
            gamma=self.gamma,
            score_window=max(self.window_steps, 2),
        )
        requested = base.copy()
        for step in range(context.start_step, len(demand)):
            correction = calibrator.current_correction()
            requested[step] = _replicas_for(max(float(predicted[step]) + correction, 0.0), profile)
            # the outcome of the previous step is observable now, and only now
            calibrator.observe(
                timestamp=context.series.timestamps[step - 1],
                raw_prediction=float(predicted[step - 1]),
                actual=float(demand[step - 1]),
            )
        return clamp_plan(requested, profile)


@dataclass(frozen=True)
class NewsvendorController:
    """C6 — DELPHI: the compliance target is derived from prices, not chosen.

    ``q* = C_u / (C_u + C_o)``. Optionally paced by the same PI loop as C5, so the only
    difference that remains between the two is where the target came from.
    """

    forecaster: QuantileForecaster
    ratio: CostRatio
    gamma: float = 0.01
    kp: float = 1.0
    ki: float = 0.1
    pace_budget: bool = True
    refit_stride: int = 60
    score_window: int = 1440
    levels: tuple[float, ...] = DEFAULT_LEVELS

    @property
    def controller_id(self) -> str:
        paced = "paced" if self.pace_budget else "unpaced"
        return (
            f"newsvendor:q*={self.ratio.critical_ratio:.4f}:"
            f"{paced}:gamma={self.gamma:g}:{self.forecaster.model_id}"
        )

    @property
    def minimum_history(self) -> int:
        return self.forecaster.minimum_history

    @property
    def target_quantile(self) -> float:
        return self.ratio.critical_ratio

    def plan(self, context: ControlContext) -> IntArray:
        target = self.ratio.critical_ratio
        pacer = (
            BudgetPacer(target_violation_rate=1.0 - target, kp=self.kp, ki=self.ki)
            if self.pace_budget
            else None
        )
        return _plan_calibrated(
            context,
            forecaster=self.forecaster,
            base_quantile_at=lambda _: target,
            pacer=pacer,
            levels=self.levels,
            refit_stride=self.refit_stride,
            gamma=self.gamma,
            score_window=self.score_window,
            aci_target_coverage=target,
        )
