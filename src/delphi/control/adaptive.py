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

from delphi.control.controllers import ControlContext, _replicas_for
from delphi.control.newsvendor import CostRatio, interpolate_quantile
from delphi.control.simulator import CapacityProfile, clamp_plan
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

    warmup = demand[: context.start_step]
    requested[:] = _replicas_for(float(np.quantile(warmup, 0.95)), profile)

    calibrator: AdaptiveConformalCalibrator | None = None
    # raw forecast value emitted for each step, so its residual can be scored once the
    # true demand for that step becomes observable
    raw_for_step: dict[int, float] = {}
    # incremental mirror of the simulator's actuation delay, over our own plan
    current = int(requested[0])
    pending_target: int | None = None
    pending_at = 0
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
            requested[target_step] = _replicas_for(max(raw + correction, 0.0), profile)
            raw_for_step[target_step] = raw

            # advance our mirror of the platform's actuation lag one step
            asked = int(requested[target_step])
            in_flight = pending_target if pending_target is not None else current
            if asked != in_flight:
                lag = profile.startup_steps if asked > current else profile.teardown_steps
                pending_target = asked
                pending_at = target_step + max(lag, 0)
            if pending_target is not None and target_step >= pending_at:
                current = pending_target
                pending_target = None
            served[target_step] = current

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


def size_for_profile(demand: float, profile: CapacityProfile) -> int:
    """Public wrapper so callers outside this module need not import a private helper."""
    return _replicas_for(demand, profile)
