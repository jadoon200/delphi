"""Baseline capacity controllers, reimplemented so they can be tuned as hard as ours.

A controller comparison in which only the proposed method was tuned properly is worthless,
and it is the most common flaw in the autoscaling literature. Every controller here takes
the same context, emits the same trajectory type, and is fitted through the same parity
harness.

**Causality contract.** ``plan()[t]`` is what the controller *asks for* at step ``t``,
exactly as a real autoscaler emits a desired replica count and the platform takes time to
honour it. Applying the actuation delay is the simulator's job and the simulator's alone —
a controller that also shifts its own origin double-counts the lag, which silently wrecks
every long-lead result.

At step ``t`` a controller may read ``demand[:t]`` and nothing more: the current step is
still in flight when the decision is taken. A proactive controller therefore forecasts
``startup_steps`` ahead, because that is when the capacity it is requesting will land.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from delphi.control.simulator import CapacityProfile, clamp_plan
from delphi.data.series import DemandSeries
from delphi.forecast.contracts import QuantileForecaster

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ControlContext:
    """Everything a controller may look at, plus the boundary it may not cross."""

    series: DemandSeries
    profile: CapacityProfile
    #: First step the controller is asked to plan for. Steps before this are warmup and
    #: are excluded from scoring, so a controller is never judged on its cold start.
    start_step: int

    def __post_init__(self) -> None:
        if not 0 < self.start_step < len(self.series):
            raise ValueError("start_step must leave warmup history and lie inside the series")

    @property
    def demand(self) -> FloatArray:
        return self.series.values

    def observable_upto(self, step: int) -> int:
        """Exclusive bound on demand readable when deciding step ``step``.

        ``demand[:t]`` — the step being decided is still in flight, so its own demand is
        not yet measurable. The actuation delay is deliberately *not* subtracted here: the
        simulator applies it to the emitted plan, and subtracting it again would charge
        every controller for the same lag twice.
        """
        return max(0, min(step, len(self.series)))


class Controller(Protocol):
    @property
    def controller_id(self) -> str: ...

    def plan(self, context: ControlContext) -> IntArray: ...


def _replicas_for(demand: float, profile: CapacityProfile) -> int:
    """Smallest replica count whose serving rate covers ``demand``."""
    per_replica = profile.capacity_per_replica * profile.utilisation_target
    return int(np.ceil(demand / per_replica - 1e-9))


@dataclass(frozen=True)
class StaticController:
    """C0 — fixed capacity at a percentile of the warmup window.

    Trivial, and frequently competitive on noisy traces. It is in the table to stop anyone
    claiming a win that a constant would also have delivered.
    """

    percentile: float = 0.95

    def __post_init__(self) -> None:
        if not 0 < self.percentile <= 1:
            raise ValueError("percentile must lie within (0, 1]")

    @property
    def controller_id(self) -> str:
        return f"static:p{self.percentile:g}:v1"

    def plan(self, context: ControlContext) -> IntArray:
        warmup = context.demand[: context.start_step]
        level = float(np.quantile(warmup, self.percentile))
        replicas = _replicas_for(level, context.profile)
        return clamp_plan(np.full(len(context.demand), replicas, dtype=np.int64), context.profile)


@dataclass(frozen=True)
class ReactiveController:
    """C1 — Kubernetes HPA: scale on the most recent observation against a target.

    Implements the real control law, ``ceil(replicas * metric / target)``, which reduces to
    sizing for the last observed demand, plus the two behaviours that stop production HPA
    from thrashing: a tolerance band and a scale-down stabilisation window.
    """

    tolerance: float = 0.1
    stabilisation_steps: int = 5

    def __post_init__(self) -> None:
        if not 0 <= self.tolerance < 1:
            raise ValueError("tolerance must lie within [0, 1)")
        if self.stabilisation_steps < 0:
            raise ValueError("stabilisation window cannot be negative")

    @property
    def controller_id(self) -> str:
        return f"reactive_hpa:tol={self.tolerance:g}:stab={self.stabilisation_steps}:v1"

    def plan(self, context: ControlContext) -> IntArray:
        demand = context.demand
        profile = context.profile
        requested = np.empty(len(demand), dtype=np.int64)
        current = _replicas_for(float(np.quantile(demand[: context.start_step], 0.95)), profile)
        for step in range(len(demand)):
            observable = context.observable_upto(step)
            if observable == 0:
                requested[step] = current
                continue
            latest = float(demand[observable - 1])
            desired = _replicas_for(latest, profile)
            if current > 0:
                ratio = desired / current
                if abs(ratio - 1.0) <= self.tolerance:
                    desired = current
            if desired < current and self.stabilisation_steps:
                window = demand[max(0, observable - self.stabilisation_steps) : observable]
                desired = max(desired, _replicas_for(float(window.max()), profile))
            current = desired
            requested[step] = desired
        return clamp_plan(requested, profile)


@dataclass(frozen=True)
class PercentileRecommender:
    """C2 — the Borg Autopilot / Kubernetes VPA sliding-window percentile recommender.

    VPA's recommender takes a decaying-weight histogram percentile over a sliding window
    (CPU p95 upper bound, memory p90 base). This is the strongest *simple* baseline and the
    one a real cluster is most likely to already be running.
    """

    window_steps: int = 1440
    percentile: float = 0.95
    half_life_steps: int = 720
    margin: float = 0.15

    def __post_init__(self) -> None:
        if self.window_steps < 1 or self.half_life_steps < 1:
            raise ValueError("window and half-life must be positive")
        if not 0 < self.percentile <= 1 or self.margin < 0:
            raise ValueError("percentile must be in (0, 1] and margin non-negative")

    @property
    def controller_id(self) -> str:
        return (
            f"percentile_recommender:w={self.window_steps}:"
            f"p{self.percentile:g}:margin={self.margin:g}:v1"
        )

    def _weighted_percentile(self, sample: FloatArray) -> float:
        if not len(sample):
            return 0.0
        ages = np.arange(len(sample) - 1, -1, -1, dtype=np.float64)
        weights = np.exp2(-ages / self.half_life_steps)
        order = np.argsort(sample)
        ordered_values = sample[order]
        cumulative = np.cumsum(weights[order])
        cutoff = self.percentile * cumulative[-1]
        return float(ordered_values[int(np.searchsorted(cumulative, cutoff))])

    def plan(self, context: ControlContext) -> IntArray:
        demand = context.demand
        profile = context.profile
        requested = np.empty(len(demand), dtype=np.int64)
        for step in range(len(demand)):
            observable = context.observable_upto(step)
            window = demand[max(0, observable - self.window_steps) : observable]
            level = self._weighted_percentile(window) * (1.0 + self.margin)
            requested[step] = _replicas_for(level, profile)
        return clamp_plan(requested, profile)


@dataclass(frozen=True)
class ProactiveQuantileController:
    """C3 — forecast, take a fixed quantile, size for it.

    This isolates the value of *forecasting alone*, before calibration (M3) and before the
    newsvendor derivation of the quantile (M6). The gap between this and C1 is what
    prediction buys; the gap between this and C6 is what the rest of DELPHI buys.

    Capacity requested at step ``t`` lands at ``t + startup_steps``, so the forecast is
    read ``startup_steps`` ahead of the origin: the model is asked exactly the question the
    actuation delay poses — what will demand be once this capacity actually arrives?
    """

    forecaster: QuantileForecaster
    quantile: float = 0.95
    refit_stride: int = 60

    def __post_init__(self) -> None:
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must lie strictly within (0, 1)")
        if self.refit_stride < 1:
            raise ValueError("refit stride must be positive")

    @property
    def controller_id(self) -> str:
        return (
            f"proactive_quantile:q{self.quantile:g}:"
            f"stride={self.refit_stride}:{self.forecaster.model_id}"
        )

    def plan(self, context: ControlContext) -> IntArray:
        profile = context.profile
        demand = context.demand
        lead = profile.startup_steps
        levels = (self.quantile,)
        requested = np.empty(len(demand), dtype=np.int64)
        fallback = _replicas_for(
            float(np.quantile(demand[: context.start_step], self.quantile)), profile
        )
        requested[:] = fallback

        step = context.start_step
        while step < len(demand):
            origin = context.observable_upto(step)
            if origin < self.forecaster.minimum_history:
                step += 1
                continue
            block = min(self.refit_stride, len(demand) - step)
            forecast = self.forecaster.forecast(
                context.series.slice(0, origin),
                horizon_steps=lead + block,
                quantile_levels=levels,
            )
            # Forecast index i is demand index origin + i, and origin == step. Capacity
            # asked for at ``step + j`` serves ``step + j + lead``, which is index j + lead.
            band = forecast.quantile_values[lead : lead + block, 0]
            for offset, value in enumerate(band):
                requested[step + offset] = _replicas_for(float(value), profile)
            step += block
        return clamp_plan(requested, profile)
