"""Trace-replay capacity simulator.

This is the load-bearing component of the entire evaluation: every controller comparison,
every cost-versus-violation frontier, and every claim about proactive scaling is produced
here. If it is wrong, every number in the repo is wrong, so it is deterministic, its
assumptions are named rather than buried, and it is validated against closed-form
queueing theory in ``scripts/validate_simulator.py`` before any result is published.

Two deliberate design commitments:

* **Actuation delay is explicit.** A decision taken at step *t* yields capacity at
  ``t + ceil(startup_seconds / step_seconds)``. A simulator without this flatters every
  proactive method for free, because forecasting only pays for itself when acting takes
  time. This is the single parameter most often silently set to zero elsewhere.
* **Fidelity is a first-class field on every result.** The cheap utilisation model and the
  event-driven queue answer different questions and must never share a results table.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from delphi.control.queueing import QueueOutcome, simulate_multi_server_queue

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

Fidelity = Literal["utilisation", "queue"]


@dataclass(frozen=True)
class CapacityProfile:
    """How one workload's demand translates into provisioned replicas."""

    workload_id: str
    step_seconds: int
    #: Demand units one replica can serve within one step.
    capacity_per_replica: float
    #: Actuation delay. Pods are seconds; a GPU model load is minutes. This is what makes
    #: forecasting beat reaction, so it is never defaulted silently to zero.
    startup_seconds: float = 30.0
    teardown_seconds: float = 10.0
    min_replicas: int = 0
    max_replicas: int = 10_000
    scale_to_zero: bool = True
    #: Utilisation ceiling: a step is a violation when demand exceeds this fraction of the
    #: provisioned serving rate. Modelled, not measured — swept in the sensitivity gate.
    utilisation_target: float = 0.8

    def __post_init__(self) -> None:
        if self.step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        if self.capacity_per_replica <= 0:
            raise ValueError("capacity_per_replica must be positive")
        if self.startup_seconds < 0 or self.teardown_seconds < 0:
            raise ValueError("actuation delays cannot be negative")
        if not 0 < self.utilisation_target <= 1:
            raise ValueError("utilisation_target must lie within (0, 1]")
        if self.min_replicas < 0 or self.max_replicas < self.min_replicas:
            raise ValueError("replica bounds are inconsistent")
        if self.min_replicas == 0 and not self.scale_to_zero:
            raise ValueError("min_replicas=0 requires scale_to_zero")

    @property
    def startup_steps(self) -> int:
        """Steps before a scale-up lands. Rounded up: partial steps do not serve traffic."""
        return int(np.ceil(self.startup_seconds / self.step_seconds))

    @property
    def teardown_steps(self) -> int:
        return int(np.ceil(self.teardown_seconds / self.step_seconds))

    def serving_rate(self, replicas: IntArray) -> FloatArray:
        """Demand units per step that a replica count can absorb at the target ceiling."""
        return replicas.astype(np.float64) * self.capacity_per_replica * self.utilisation_target


@dataclass(frozen=True)
class CostModel:
    """What a capacity trajectory costs.

    ``price_per_replica_hour`` is intended to come from the live keyless Azure Retail
    Prices API so the cost side of the newsvendor ratio is a cited number rather than a
    constant. ``churn_cost_per_action`` prices thrashing, without which any
    quantile-tracking controller looks better than it is.
    """

    price_per_replica_hour: float = 0.10
    churn_cost_per_action: float = 0.0
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.price_per_replica_hour < 0 or self.churn_cost_per_action < 0:
            raise ValueError("prices cannot be negative")


@dataclass(frozen=True)
class SimulationResult:
    """One replay of one capacity trajectory against one demand trace."""

    workload_id: str
    fidelity: Fidelity
    demand: FloatArray
    #: What the controller asked for at each step.
    requested: IntArray
    #: What was actually serving at each step, after the actuation delay.
    provisioned: IntArray
    violations: BoolArray
    violation_magnitude: FloatArray
    capacity_hours: float
    scaling_actions: int
    cost: float
    #: Present only for the queue fidelity.
    queue: QueueOutcome | None = None

    @property
    def steps(self) -> int:
        return len(self.demand)

    @property
    def violation_rate(self) -> float:
        return float(np.mean(self.violations))

    @property
    def violation_sum(self) -> float:
        """Cumulative unmet demand. Ten small overshoots are not one large one."""
        return float(np.sum(self.violation_magnitude))

    @property
    def cold_start_steps(self) -> int:
        """Steps where a requested scale-up had not yet landed."""
        return int(np.sum(self.requested > self.provisioned))

    def summary(self) -> dict[str, float]:
        return {
            "violation_rate": self.violation_rate,
            "violation_sum": self.violation_sum,
            "capacity_hours": self.capacity_hours,
            "cost": self.cost,
            "scaling_actions": float(self.scaling_actions),
            "cold_start_steps": float(self.cold_start_steps),
            "mean_replicas": float(np.mean(self.provisioned)),
        }


def apply_actuation_delay(requested: IntArray, profile: CapacityProfile) -> IntArray:
    """Turn a requested trajectory into the capacity that is actually serving.

    ``requested[t]`` is the capacity a controller wants *serving* at step ``t``. A change
    lands at ``t + startup_steps`` when scaling up and ``t + teardown_steps`` when scaling
    down, so a profile with zero delay reproduces the plan exactly — anything else would
    impose a hidden one-step lag that no configuration could remove, and would make even
    the hindsight-optimal plan fail to meet its own demand.

    A later request supersedes one still in flight, which is what a real reconciliation
    loop does: it acts on the current desired state, not on a backlog of past intentions.
    **Superseding retargets the change in flight; it does not restart its clock.** Pods
    already starting do not begin again because the desired count moved, so a request that
    keeps moving still lands every ``lag`` steps at whatever is desired by then. Restarting
    the timer instead would freeze capacity for any continuously-varying plan — an
    unrequestable state that silently penalises exactly the smooth, forecast-driven
    controllers this project exists to evaluate.
    """
    steps = len(requested)
    provisioned = np.empty(steps, dtype=np.int64)
    current = int(requested[0])
    pending_target: int | None = None
    pending_at = 0

    for step in range(steps):
        target = int(requested[step])
        if pending_target is None:
            if target != current:
                lag = profile.startup_steps if target > current else profile.teardown_steps
                pending_target = target
                pending_at = step + max(lag, 0)
        elif target != pending_target:
            # Retarget the in-flight change, keeping its original landing step.
            pending_target = target
        if pending_target is not None and step >= pending_at:
            current = pending_target
            pending_target = None
        provisioned[step] = current
    return provisioned


def clamp_plan(requested: IntArray, profile: CapacityProfile) -> IntArray:
    """Clip a controller's request into the profile's feasible replica band."""
    floor = profile.min_replicas
    plan = np.clip(requested.astype(np.int64), floor, profile.max_replicas)
    if not profile.scale_to_zero:
        plan = np.maximum(plan, max(1, floor))
    return plan


def expost_optimal_plan(demand: FloatArray, profile: CapacityProfile) -> IntArray:
    """The cheapest trajectory that would have met every step, known only in hindsight.

    This is the reference for regret. It is not achievable — it uses the future — which is
    exactly why it is the honest denominator for "how much did not knowing cost us".
    """
    per_replica = profile.capacity_per_replica * profile.utilisation_target
    needed = np.ceil(demand / per_replica - 1e-9).astype(np.int64)
    return clamp_plan(needed, profile)


def _synthesize_arrivals(
    demand: FloatArray,
    profile: CapacityProfile,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Expand per-step demand counts into individual arrival and service times.

    Arrivals are placed uniformly at random inside their step, which reproduces a Poisson
    process conditioned on the per-step count. Service times are exponential with a mean
    implied by ``capacity_per_replica`` so that one replica serves exactly that many units
    per step on average — keeping the two fidelities describing the same physical system.
    """
    counts = np.rint(np.maximum(demand, 0.0)).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    step_index = np.repeat(np.arange(len(counts), dtype=np.float64), counts)
    offsets = rng.random(total) * profile.step_seconds
    arrivals = np.sort(step_index * profile.step_seconds + offsets)
    mean_service = profile.step_seconds / profile.capacity_per_replica
    service = rng.exponential(mean_service, size=total)
    return arrivals, service


def simulate(
    *,
    demand: FloatArray,
    requested: IntArray,
    profile: CapacityProfile,
    cost_model: CostModel | None = None,
    fidelity: Fidelity = "utilisation",
    seed: int = 20260803,
    wait_slo_seconds: float | None = None,
) -> SimulationResult:
    """Replay one capacity trajectory against one demand trace.

    Deterministic: the same inputs and seed produce byte-identical output.

    ``fidelity="utilisation"`` marks a step as violated when demand exceeds the
    provisioned serving rate — the cheap linear model most of the autoscaling literature
    uses, kept for comparability. ``fidelity="queue"`` runs an event-driven multi-server
    queue and marks a step as violated when a request that arrived in it waited longer
    than ``wait_slo_seconds``, which is closer to what an operator actually promises.
    """
    if demand.ndim != 1 or not len(demand):
        raise ValueError("demand must be a non-empty 1-D array")
    if requested.shape != demand.shape:
        raise ValueError("the requested plan must have one entry per demand step")
    if not np.isfinite(demand).all() or np.any(demand < 0):
        raise ValueError("demand must be finite and non-negative")

    costs = cost_model or CostModel()
    plan = clamp_plan(requested, profile)
    provisioned = apply_actuation_delay(plan, profile)
    serving = profile.serving_rate(provisioned)

    if fidelity == "utilisation":
        shortfall = np.maximum(demand - serving, 0.0)
        violations = shortfall > 0
        magnitude = shortfall
        queue: QueueOutcome | None = None
    elif fidelity == "queue":
        if wait_slo_seconds is None or wait_slo_seconds < 0:
            raise ValueError("the queue fidelity requires a non-negative wait_slo_seconds")
        rng = np.random.default_rng(seed)
        arrivals, service = _synthesize_arrivals(demand, profile, rng)
        queue = simulate_multi_server_queue(
            arrival_seconds=arrivals,
            service_seconds=service,
            servers_per_step=provisioned,
            step_seconds=float(profile.step_seconds),
        )
        breached = (queue.wait_seconds > wait_slo_seconds) | queue.dropped
        step_of = np.minimum((arrivals // profile.step_seconds).astype(np.int64), len(demand) - 1)
        magnitude = np.bincount(step_of[breached], minlength=len(demand)).astype(np.float64)
        violations = magnitude > 0
    else:  # pragma: no cover - Literal keeps this unreachable
        raise ValueError(f"unknown fidelity: {fidelity}")

    hours = float(provisioned.sum()) * profile.step_seconds / 3600.0
    actions = int(np.count_nonzero(np.diff(provisioned)))
    total_cost = hours * costs.price_per_replica_hour + actions * costs.churn_cost_per_action

    return SimulationResult(
        workload_id=profile.workload_id,
        fidelity=fidelity,
        demand=demand.copy(),
        requested=plan,
        provisioned=provisioned,
        violations=violations,
        violation_magnitude=magnitude,
        capacity_hours=hours,
        scaling_actions=actions,
        cost=total_cost,
        queue=queue,
    )


def regret(result: SimulationResult, profile: CapacityProfile, cost_model: CostModel) -> float:
    """Cost paid above the cheapest trajectory that would have met every step.

    Hindsight is free in replay, so this is the one honest scalar: it prices *not knowing*
    rather than rewarding a controller for buying its way out of uncertainty.
    """
    optimal = expost_optimal_plan(result.demand, profile)
    optimal_hours = float(optimal.sum()) * profile.step_seconds / 3600.0
    optimal_cost = optimal_hours * cost_model.price_per_replica_hour
    return result.cost - optimal_cost
