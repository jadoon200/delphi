"""Causal lag, rolling, volatility, burst, and calendar features."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from delphi.data.series import DemandSeries

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class FeatureSpec:
    lags: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 60, 1440, 10080)
    windows: tuple[int, ...] = (15, 60, 1440)

    def __post_init__(self) -> None:
        if not self.lags or not self.windows:
            raise ValueError("feature lags and windows must be non-empty")
        if any(value <= 0 for value in (*self.lags, *self.windows)):
            raise ValueError("feature lags and windows must be positive")
        if tuple(sorted(set(self.lags))) != self.lags:
            raise ValueError("feature lags must be sorted and unique")
        if tuple(sorted(set(self.windows))) != self.windows:
            raise ValueError("feature windows must be sorted and unique")

    @property
    def lookback(self) -> int:
        return max(*self.lags, *self.windows)


@dataclass(frozen=True)
class CausalFeatureMatrix:
    values: FloatArray
    targets: FloatArray
    target_indices: IntArray
    feature_as_of_indices: IntArray


def assert_causal_features(matrix: CausalFeatureMatrix, *, fit_end_exclusive: int) -> None:
    """Reject target leakage and fitting beyond a declared chronological boundary."""
    rows = matrix.targets.shape[0]
    if matrix.values.ndim != 2 or matrix.values.shape[0] != rows:
        raise ValueError("feature matrix and targets must have equal rows")
    if matrix.target_indices.shape != (rows,) or matrix.feature_as_of_indices.shape != (rows,):
        raise ValueError("feature provenance indices must have one entry per row")
    if np.any(matrix.feature_as_of_indices >= matrix.target_indices):
        raise ValueError("feature leakage: a feature observes its own target or the future")
    if rows and int(matrix.target_indices.max()) >= fit_end_exclusive:
        raise ValueError("feature leakage: fit data crosses the declared split boundary")


def _calendar(timestamp: datetime) -> list[float]:
    minute = timestamp.minute + timestamp.second / 60
    hour = timestamp.hour + minute / 60
    weekday = timestamp.weekday()
    return [
        np.sin(2 * np.pi * minute / 60),
        np.cos(2 * np.pi * minute / 60),
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * weekday / 7),
        np.cos(2 * np.pi * weekday / 7),
        float(weekday >= 5),
    ]


def causal_feature_row(values: FloatArray, timestamp: datetime, spec: FeatureSpec) -> FloatArray:
    """Create one feature row using values strictly before ``timestamp``."""
    if len(values) < spec.lookback:
        raise ValueError(f"at least {spec.lookback} historical points are required")
    row = [float(values[-lag]) for lag in spec.lags]
    for window in spec.windows:
        sample = values[-window:]
        row.extend(
            [
                float(np.mean(sample)),
                float(np.std(sample)),
                float(np.min(sample)),
                float(np.max(sample)),
                float(np.quantile(sample, 0.5)),
                float(np.quantile(sample, 0.95)),
            ]
        )
    short_std = float(np.std(values[-spec.windows[0] :]))
    long_std = float(np.std(values[-spec.windows[-1] :]))
    row.append(short_std / max(long_std, 1e-12))
    burst_window = values[-spec.windows[-1] :]
    threshold = float(np.quantile(burst_window, 0.99))
    burst_positions = np.flatnonzero(burst_window > threshold)
    since_burst = (
        len(burst_window)
        if not len(burst_positions)
        else len(burst_window) - 1 - int(burst_positions[-1])
    )
    row.append(float(since_burst))
    row.extend(_calendar(timestamp))
    return np.asarray(row, dtype=np.float64)


def build_causal_training_matrix(
    series: DemandSeries,
    spec: FeatureSpec,
    *,
    fit_end_exclusive: int | None = None,
    max_rows: int | None = None,
) -> CausalFeatureMatrix:
    """Build supervised one-step rows; every row stops one step before its target."""
    end = len(series) if fit_end_exclusive is None else fit_end_exclusive
    if end > len(series) or end <= spec.lookback:
        raise ValueError("fit boundary must leave enough causal history")
    start = spec.lookback
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        start = max(start, end - max_rows)
    target_indices = np.arange(start, end, dtype=np.int64)
    rows = [
        causal_feature_row(series.values[:index], series.timestamps[index], spec)
        for index in target_indices
    ]
    matrix = CausalFeatureMatrix(
        values=np.vstack(rows),
        targets=series.values[target_indices].copy(),
        target_indices=target_indices,
        feature_as_of_indices=target_indices - 1,
    )
    assert_causal_features(matrix, fit_end_exclusive=end)
    return matrix
