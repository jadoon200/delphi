"""Commitment-interval capacity control — size once, hold for a whole window.

This is a genuinely different control regime from autoscaling, and it is the one where
forecasting has something to contribute.

An autoscaler reacts continuously, so a trailing percentile is nearly optimal when demand
is strongly persistent (minute-scale autocorrelation ~0.98 on every workload measured here).
A **commitment** cannot react: reserved instances, cluster sizing and procurement fix a
capacity level for hours, so the decision must cover the *entire* coming window including
its peak. A backward-looking window can only assume the next window resembles the last one;
a forecast can know that a daily ramp falls inside it.

Both controllers below perform the *same* operation — take the ``q``-quantile of a window
of demand values — differing only in whether that window is observed history or a predicted
future. Keeping the operation identical is what makes the comparison about information
rather than about implementation.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from delphi.control.controllers import ControlContext, _replicas_for
from delphi.control.simulator import clamp_plan
from delphi.forecast.contracts import QuantileForecaster

IntArray = npt.NDArray[np.int64]


def commitment_boundaries(context: ControlContext, window_steps: int) -> range:
    """Decision points: the start of each commitment window inside the planned region."""
    if window_steps < 1:
        raise ValueError("commitment window must be at least one step")
    return range(context.start_step, len(context.series), window_steps)


@dataclass(frozen=True)
class BackwardCommitmentController:
    """Commit to the ``q``-quantile of the *trailing* window of equal length.

    The incumbent's logic transplanted into the commitment regime: assume the next window
    looks like the last one. Cheap, and correct whenever demand has no exploitable
    structure beyond its recent level.
    """

    window_steps: int
    quantile: float = 0.95

    def __post_init__(self) -> None:
        if self.window_steps < 1:
            raise ValueError("commitment window must be at least one step")
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must lie strictly within (0, 1)")

    @property
    def controller_id(self) -> str:
        return f"commit_backward:w={self.window_steps}:q{self.quantile:g}:v1"

    def plan(self, context: ControlContext) -> IntArray:
        demand = context.demand
        requested = np.empty(len(demand), dtype=np.int64)
        warmup = demand[: context.start_step]
        requested[:] = _replicas_for(float(np.quantile(warmup, self.quantile)), context.profile)
        for start in commitment_boundaries(context, self.window_steps):
            history = demand[max(0, start - self.window_steps) : start]
            if not len(history):
                continue
            level = float(np.quantile(history, self.quantile))
            replicas = _replicas_for(level, context.profile)
            requested[start : start + self.window_steps] = replicas
        return clamp_plan(requested, context.profile)


@dataclass(frozen=True)
class ForwardCommitmentController:
    """Commit to the ``q``-quantile of the *forecast* for the coming window.

    Identical arithmetic to the backward controller, applied to a predicted window instead
    of an observed one. The forecaster is asked for a median path over the window and the
    spatial quantile is taken across it, so what is being compared is purely the
    information in the window, not two different sizing rules.
    """

    window_steps: int
    forecaster: QuantileForecaster
    quantile: float = 0.95

    def __post_init__(self) -> None:
        if self.window_steps < 1:
            raise ValueError("commitment window must be at least one step")
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must lie strictly within (0, 1)")

    @property
    def controller_id(self) -> str:
        return f"commit_forward:w={self.window_steps}:q{self.quantile:g}:{self.forecaster.model_id}"

    def plan(self, context: ControlContext) -> IntArray:
        demand = context.demand
        profile = context.profile
        requested = np.empty(len(demand), dtype=np.int64)
        warmup = demand[: context.start_step]
        requested[:] = _replicas_for(float(np.quantile(warmup, self.quantile)), profile)

        for start in commitment_boundaries(context, self.window_steps):
            origin = context.observable_upto(start)
            if origin < self.forecaster.minimum_history:
                continue
            horizon = min(self.window_steps, len(demand) - start)
            if horizon < 1:
                continue
            forecast = self.forecaster.forecast(
                context.series.slice(0, origin),
                horizon_steps=horizon,
                quantile_levels=(0.5,),
            )
            path = forecast.quantile_values[:, 0]
            level = float(np.quantile(path, self.quantile))
            requested[start : start + horizon] = _replicas_for(level, profile)
        return clamp_plan(requested, profile)
