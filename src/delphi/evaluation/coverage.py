"""First-class empirical coverage records and recovery measurements."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class CoverageObservation:
    timestamp: datetime
    nominal_level: float
    predicted: float
    actual: float
    covered: bool


@dataclass(frozen=True)
class CoverageRecord:
    model_id: str
    calibrator_id: str
    nominal_level: float
    observations: tuple[CoverageObservation, ...]

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("coverage record must contain observations")
        if any(
            observation.nominal_level != self.nominal_level for observation in self.observations
        ):
            raise ValueError("coverage observation levels must match the record")

    @property
    def empirical_coverage(self) -> float:
        return float(np.mean([observation.covered for observation in self.observations]))

    @property
    def coverage_series(self) -> tuple[bool, ...]:
        return tuple(observation.covered for observation in self.observations)


def coverage_recovery_steps(
    coverage: tuple[bool, ...],
    *,
    target: float,
    window: int,
    tolerance: float = 0.05,
) -> int | None:
    """Return steps until trailing coverage first recovers to the target tolerance."""
    if not coverage or window <= 0 or window > len(coverage):
        raise ValueError("coverage and recovery window are incompatible")
    threshold = target - tolerance
    for end in range(window, len(coverage) + 1):
        if float(np.mean(coverage[end - window : end])) >= threshold:
            return end
    return None
