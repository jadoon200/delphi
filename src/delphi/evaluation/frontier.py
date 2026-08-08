"""Cost-versus-violation Pareto frontiers, with tuning parity enforced mechanically.

Two rules from the evaluation doctrine are implemented here rather than left to discipline:

1. **No single-number headline where a frontier is the truth.** A controller is a *curve*,
   traced by sweeping its risk setting. Comparing single points taken from different curves
   is how this literature flatters itself, so the comparison object is the frontier and the
   scalars derived from it are stated with their construction.
2. **Every controller gets the same tuning budget.** ``ParitySpec`` records the number of
   configurations each arm was allowed, so a reader can see that ours was not hand-tuned
   while the baseline took defaults.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from delphi.control.controllers import ControlContext
from delphi.control.simulator import (
    CostModel,
    SimulationResult,
    expost_optimal_plan,
    simulate,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FrontierPoint:
    """One (cost, violation) outcome for one controller at one risk setting."""

    controller_id: str
    family: str
    setting: float
    violation_rate: float
    violation_sum: float
    cost: float
    capacity_hours: float
    scaling_actions: int
    mean_replicas: float
    regret: float


@dataclass(frozen=True)
class ParitySpec:
    """The tuning budget every arm was held to."""

    configurations_per_controller: int
    selection_metric: str
    validation_fraction: float

    def describe(self) -> str:
        return (
            f"{self.configurations_per_controller} configurations per controller, "
            f"selected on {self.selection_metric} over the first "
            f"{self.validation_fraction:.0%} of the scored window"
        )


def score_plan(
    *,
    controller_id: str,
    family: str,
    setting: float,
    plan: npt.NDArray[np.int64],
    context: ControlContext,
    cost_model: CostModel,
    scored: slice,
) -> FrontierPoint:
    """Simulate one plan and package the metrics the frontier needs."""
    demand = context.demand[scored]
    result: SimulationResult = simulate(
        demand=demand,
        requested=plan[scored],
        profile=context.profile,
        cost_model=cost_model,
    )
    optimal = expost_optimal_plan(demand, context.profile)
    optimal_hours = float(optimal.sum()) * context.profile.step_seconds / 3600.0
    optimal_cost = optimal_hours * cost_model.price_per_replica_hour
    return FrontierPoint(
        controller_id=controller_id,
        family=family,
        setting=setting,
        violation_rate=result.violation_rate,
        violation_sum=result.violation_sum,
        cost=result.cost,
        capacity_hours=result.capacity_hours,
        scaling_actions=result.scaling_actions,
        mean_replicas=float(np.mean(result.provisioned)),
        regret=result.cost - optimal_cost,
    )


def pareto_front(points: Sequence[FrontierPoint]) -> list[FrontierPoint]:
    """Non-dominated points, minimising both cost and violation rate.

    A point is dominated when another is no worse on both axes and strictly better on at
    least one. Ties are kept once, ordered by cost, so the frontier is deterministic.
    """
    ordered = sorted(points, key=lambda point: (point.cost, point.violation_rate))
    front: list[FrontierPoint] = []
    best_violation = float("inf")
    for point in ordered:
        if point.violation_rate < best_violation - 1e-12:
            front.append(point)
            best_violation = point.violation_rate
    return front


def dominated_hypervolume(
    points: Sequence[FrontierPoint],
    *,
    cost_reference: float,
    violation_reference: float = 1.0,
) -> float:
    """Area dominated by a frontier relative to a reference (worst-case) corner.

    A scalar summary for when one is genuinely needed. Larger is better. It is reported
    only alongside its reference point, because a hypervolume without its reference is
    meaningless and trivially manipulable.

    The staircase runs *forward* from each point: over the cost interval between one
    frontier point and the next, the best violation rate available is the **cheaper**
    point's, because the dearer one cannot be afforded yet. An earlier version credited each
    strip with the dearer point's rate and started the sweep at cost zero, which double
    counted a single-point frontier and could reverse the ranking of two curves.
    """
    front = [
        point
        for point in pareto_front(points)
        if point.cost < cost_reference and point.violation_rate < violation_reference
    ]
    if not front:
        return 0.0
    area = 0.0
    for current, following in zip(front, front[1:], strict=False):
        area += (following.cost - current.cost) * (violation_reference - current.violation_rate)
    last = front[-1]
    area += (cost_reference - last.cost) * (violation_reference - last.violation_rate)
    return float(max(area, 0.0))


def cost_at_matched_violation(
    points: Sequence[FrontierPoint],
    target_violation: float,
) -> float | None:
    """Cheapest point meeting a violation target, or ``None`` if the curve never does.

    Returning ``None`` rather than the closest point is deliberate: a controller that
    cannot reach the target should be reported as unable to, not silently compared at a
    level it never achieved.
    """
    feasible = [point for point in points if point.violation_rate <= target_violation + 1e-12]
    if not feasible:
        return None
    return min(point.cost for point in feasible)


def frontier_dynamic_range(points: Sequence[FrontierPoint]) -> float:
    """Spread between the best and worst cost on the curve.

    Guards against L5's saturated-benchmark trap: when the dynamic range collapses toward
    zero, every method is tied and the benchmark has no headroom to measure anything.
    """
    if not points:
        return 0.0
    costs = [point.cost for point in points]
    return float(max(costs) - min(costs))


def format_frontier_table(points: Sequence[FrontierPoint]) -> str:
    """Markdown table, ordered by violation rate then cost."""
    header = (
        "| controller | setting | violations | cost | regret | churn | mean replicas |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = [
        f"| `{point.controller_id[:52]}` | {point.setting:.3f} | "
        f"{point.violation_rate:.4f} | {point.cost:.1f} | {point.regret:.1f} | "
        f"{point.scaling_actions} | {point.mean_replicas:.2f} |"
        for point in sorted(points, key=lambda item: (item.violation_rate, item.cost))
    ]
    return "\n".join([header, *rows])
