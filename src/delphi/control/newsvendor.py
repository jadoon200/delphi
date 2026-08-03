"""Capacity as a newsvendor decision.

The thesis of the project, in one identity. Capacity bought for a future interval is a
single-period stocking decision under uncertain demand:

* buy too little and pay ``C_u`` per unit of unmet demand — breached SLO, queued work;
* buy too much and pay ``C_o`` per unit of idle capacity — the bill.

Minimising ``C_o * E[(C - D)+] + C_u * E[(D - C)+]`` gives ``F(C*) = C_u / (C_u + C_o)``:

    **buy the q* = C_u / (C_u + C_o) quantile of the demand distribution.**

Three things follow, and they are the argument DELPHI is making:

1. **The compliance target is derived, not chosen.** Every fixed-target autoscaler is
   implicitly asserting a cost ratio it never states. Ours states it.
2. **``C_u`` is the honest unknown.** ``C_o`` is a live, citable, per-SKU price. ``C_u`` is
   a business judgement nobody can measure — so the deliverable is a *curve over ``C_u``*,
   not a single number, and the operator's own ratio picks the point.
3. **The frontier is the truthful object.** Sweeping the ratio traces the whole
   cost-versus-violation frontier, which is why controllers must be compared as curves
   rather than as single flattering points.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from delphi.control.simulator import CapacityProfile

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CostRatio:
    """The two prices that decide how much capacity to buy.

    ``underage_per_unit`` is the cost of one unit of demand going unserved for one step;
    ``overage_per_unit`` is the cost of one unit of idle serving capacity for one step.
    Only their ratio matters to the quantile, which is precisely why the absolute value of
    the unknown one can be swept instead of guessed.
    """

    underage_per_unit: float
    overage_per_unit: float

    def __post_init__(self) -> None:
        if self.underage_per_unit <= 0 or self.overage_per_unit <= 0:
            raise ValueError("both newsvendor costs must be strictly positive")

    @property
    def critical_ratio(self) -> float:
        """``q* = C_u / (C_u + C_o)`` — the quantile of demand worth provisioning for."""
        return self.underage_per_unit / (self.underage_per_unit + self.overage_per_unit)

    @classmethod
    def from_quantile(cls, quantile: float, *, overage_per_unit: float = 1.0) -> "CostRatio":
        """Invert the identity: what cost ratio does a chosen compliance target imply?

        Useful for reading a fixed-target autoscaler's *implicit* economics back out —
        a controller pinned at P95 is asserting that a unit of unmet demand costs 19x a
        unit of idle capacity, whether or not anyone ever decided that.
        """
        if not 0 < quantile < 1:
            raise ValueError("quantile must lie strictly within (0, 1)")
        return cls(
            underage_per_unit=overage_per_unit * quantile / (1 - quantile),
            overage_per_unit=overage_per_unit,
        )


def newsvendor_quantile(ratio: CostRatio) -> float:
    return ratio.critical_ratio


@dataclass(frozen=True)
class SizingDecision:
    """One capacity choice, with the reasoning that produced it kept attached."""

    replicas: int
    quantile: float
    demand_at_quantile: float
    underage_per_unit: float
    overage_per_unit: float

    def as_evidence(self) -> dict[str, float | int]:
        """Flat record for the decision ledger the agent layer will write."""
        return {
            "replicas": self.replicas,
            "q_star": self.quantile,
            "demand_at_q_star": self.demand_at_quantile,
            "underage_per_unit": self.underage_per_unit,
            "overage_per_unit": self.overage_per_unit,
        }


def interpolate_quantile(
    levels: tuple[float, ...],
    values: FloatArray,
    target: float,
) -> float:
    """Read demand at an arbitrary quantile from a forecast's discrete levels.

    ``q*`` almost never lands on one of the levels a forecaster emits, so it is
    interpolated — and, critically, **clamped rather than extrapolated** at the ends. A
    cost ratio implying p99.9 cannot be answered by a forecast whose highest level is p95,
    and inventing that tail by linear extrapolation would manufacture confidence the model
    never expressed.
    """
    if len(levels) != len(values) or not len(levels):
        raise ValueError("levels and values must be parallel and non-empty")
    if tuple(sorted(levels)) != tuple(levels):
        raise ValueError("quantile levels must be sorted")
    if not 0 < target < 1:
        raise ValueError("target quantile must lie strictly within (0, 1)")
    return float(np.interp(target, np.asarray(levels, dtype=np.float64), values))


def size_from_quantiles(
    *,
    levels: tuple[float, ...],
    values: FloatArray,
    ratio: CostRatio,
    profile: CapacityProfile,
) -> SizingDecision:
    """Turn a calibrated forecast plus two prices into a replica count."""
    quantile = ratio.critical_ratio
    demand = interpolate_quantile(levels, values, quantile)
    per_replica = profile.capacity_per_replica * profile.utilisation_target
    replicas = int(np.ceil(max(demand, 0.0) / per_replica - 1e-9))
    replicas = int(np.clip(replicas, profile.min_replicas, profile.max_replicas))
    if not profile.scale_to_zero:
        replicas = max(replicas, 1)
    return SizingDecision(
        replicas=replicas,
        quantile=quantile,
        demand_at_quantile=demand,
        underage_per_unit=ratio.underage_per_unit,
        overage_per_unit=ratio.overage_per_unit,
    )


def expected_newsvendor_cost(
    *,
    capacity: float,
    demand_samples: FloatArray,
    ratio: CostRatio,
) -> float:
    """Empirical expected cost of holding ``capacity`` against a demand distribution.

    Used to *verify* the identity rather than to assume it: sweeping capacity and taking
    the argmin must land on the critical-ratio quantile, and the tests assert exactly that.
    """
    if not len(demand_samples):
        raise ValueError("expected cost needs at least one demand sample")
    over = np.maximum(capacity - demand_samples, 0.0)
    under = np.maximum(demand_samples - capacity, 0.0)
    return float(ratio.overage_per_unit * np.mean(over) + ratio.underage_per_unit * np.mean(under))


def implied_ratio_of_controller(violation_rate: float) -> float:
    """The cost ratio a controller's realised violation rate implicitly asserts.

    A controller that violates 5% of steps is behaving as though it bought the p95 of
    demand, i.e. as though ``C_u / C_o = 19``. This is the lens that makes fixed-target
    baselines comparable to a newsvendor sizer on the same axis: every autoscaler has an
    implicit price of failure, and most never say what it is.
    """
    if not 0 <= violation_rate < 1:
        raise ValueError("violation rate must lie within [0, 1)")
    served = 1.0 - violation_rate
    if served >= 1.0:
        return float("inf")
    return served / violation_rate if violation_rate > 0 else float("inf")
