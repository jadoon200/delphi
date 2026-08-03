"""The quantile-only contract between forecasting and capacity control."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np
import numpy.typing as npt

from delphi.data.series import DemandSeries

FloatMatrix = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DemandForecast:
    """Predictive demand quantiles made using observations through ``origin_ts`` only."""

    workload_id: str
    origin_ts: datetime
    horizon_steps: int
    step_seconds: int
    quantile_levels: tuple[float, ...]
    quantile_values: FloatMatrix
    model_id: str
    calibrator_id: str = "uncalibrated"
    coverage_record_id: str = "unmeasured"

    def __post_init__(self) -> None:
        values = np.asarray(self.quantile_values, dtype=np.float64).copy()
        if self.origin_ts.tzinfo is None or self.origin_ts.utcoffset() is None:
            raise ValueError("forecast origin must be timezone-aware")
        if self.origin_ts.utcoffset() != timedelta(0):
            raise ValueError("forecast origin must be normalized to UTC")
        if self.horizon_steps <= 0 or self.step_seconds <= 0:
            raise ValueError("forecast horizon and cadence must be positive")
        if not self.workload_id or not self.model_id:
            raise ValueError("forecast workload_id and model_id must be non-empty")
        if (
            not self.quantile_levels
            or tuple(sorted(self.quantile_levels)) != self.quantile_levels
            or len(set(self.quantile_levels)) != len(self.quantile_levels)
            or any(level <= 0 or level >= 1 for level in self.quantile_levels)
        ):
            raise ValueError("quantile levels must be unique, sorted, and strictly within (0, 1)")
        expected_shape = (self.horizon_steps, len(self.quantile_levels))
        if values.shape != expected_shape:
            raise ValueError(f"quantile values must have shape {expected_shape}")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("forecast values must be finite and non-negative")
        if np.any(np.diff(values, axis=1) < 0):
            raise ValueError("forecast quantiles must not cross")
        values.setflags(write=False)
        object.__setattr__(self, "quantile_values", values)

    @property
    def timestamps(self) -> tuple[datetime, ...]:
        return tuple(
            self.origin_ts + timedelta(seconds=self.step_seconds * step)
            for step in range(1, self.horizon_steps + 1)
        )


class QuantileForecaster(Protocol):
    """A forecaster that can see only the history supplied by the evaluator."""

    @property
    def model_id(self) -> str: ...

    @property
    def minimum_history(self) -> int: ...

    def forecast(
        self,
        history: DemandSeries,
        *,
        horizon_steps: int,
        quantile_levels: tuple[float, ...],
    ) -> DemandForecast: ...


def build_forecast(
    history: DemandSeries,
    *,
    horizon_steps: int,
    quantile_levels: tuple[float, ...],
    quantile_values: FloatMatrix,
    model_id: str,
) -> DemandForecast:
    """Build and validate an uncalibrated forecast from a model implementation."""
    return DemandForecast(
        workload_id=history.workload_id,
        origin_ts=history.end,
        horizon_steps=horizon_steps,
        step_seconds=history.step_seconds,
        quantile_levels=quantile_levels,
        quantile_values=quantile_values,
        model_id=model_id,
    )
