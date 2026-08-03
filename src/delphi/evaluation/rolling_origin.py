"""Leak-resistant chronological rolling-origin evaluation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt

from delphi.data.series import DemandSeries
from delphi.evaluation.metrics import ForecastMetrics, summarize_forecasts
from delphi.forecast.contracts import DemandForecast, QuantileForecaster

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SplitBoundaries:
    """Exclusive chronological boundaries: train, validation, then test."""

    train_end: int
    validation_end: int
    test_end: int

    def validate(self, length: int) -> None:
        if not 0 < self.train_end < self.validation_end < self.test_end <= length:
            raise ValueError("splits must be strictly chronological and within the series")


@dataclass(frozen=True)
class ForecastRecord:
    split: Literal["validation", "test"]
    target_start: int
    target_end: int
    origin_ts: datetime
    actual: FloatArray
    forecast: DemandForecast


@dataclass(frozen=True)
class BacktestResult:
    model_id: str
    boundaries: SplitBoundaries
    quantile_levels: tuple[float, ...]
    records: tuple[ForecastRecord, ...]

    def arrays(self) -> tuple[FloatArray, FloatArray]:
        if not self.records:
            raise ValueError("backtest contains no forecast records")
        actual = np.concatenate([record.actual for record in self.records])
        predicted = np.concatenate(
            [record.forecast.quantile_values for record in self.records], axis=0
        )
        return actual, predicted


def rolling_origin_backtest(
    series: DemandSeries,
    forecaster: QuantileForecaster,
    boundaries: SplitBoundaries,
    *,
    horizon_steps: int,
    stride_steps: int,
    quantile_levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.99),
    split: Literal["validation", "test"] = "test",
) -> BacktestResult:
    """Walk forward while giving the model no observations after each origin."""
    boundaries.validate(len(series))
    if horizon_steps <= 0 or stride_steps <= 0:
        raise ValueError("horizon and stride must be positive")
    target_start = boundaries.train_end if split == "validation" else boundaries.validation_end
    target_stop = boundaries.validation_end if split == "validation" else boundaries.test_end
    if target_start < forecaster.minimum_history:
        raise ValueError("the selected split does not leave enough model history")
    records: list[ForecastRecord] = []
    for start in range(target_start, target_stop - horizon_steps + 1, stride_steps):
        history = series.slice(0, start)
        forecast = forecaster.forecast(
            history,
            horizon_steps=horizon_steps,
            quantile_levels=quantile_levels,
        )
        if forecast.origin_ts != history.end:
            raise ValueError(
                "forecast leakage: origin does not equal the last supplied observation"
            )
        expected_timestamps = series.timestamps[start : start + horizon_steps]
        if forecast.timestamps != expected_timestamps:
            raise ValueError("forecast timestamps do not match the unseen target interval")
        records.append(
            ForecastRecord(
                split=split,
                target_start=start,
                target_end=start + horizon_steps,
                origin_ts=history.end,
                actual=series.values[start : start + horizon_steps].copy(),
                forecast=forecast,
            )
        )
    if not records:
        raise ValueError("split is too short for one complete forecast horizon")
    return BacktestResult(
        model_id=forecaster.model_id,
        boundaries=boundaries,
        quantile_levels=quantile_levels,
        records=tuple(records),
    )


def score_backtest(
    result: BacktestResult,
    series: DemandSeries,
    *,
    seasonal_period: int,
) -> ForecastMetrics:
    actual, predicted = result.arrays()
    return summarize_forecasts(
        actual=actual,
        predicted=predicted,
        quantile_levels=result.quantile_levels,
        training=series.values[: result.boundaries.train_end],
        seasonal_period=seasonal_period,
    )
