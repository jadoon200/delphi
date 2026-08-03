"""Accuracy and calibration metrics for quantile forecasts."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def pinball_loss(actual: FloatArray, predicted: FloatArray, quantile: float) -> float:
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted arrays must have the same non-empty shape")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly within (0, 1)")
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def mase(
    actual: FloatArray,
    predicted_median: FloatArray,
    training: FloatArray,
    seasonal_period: int,
) -> float:
    """Mean absolute scaled error with a train-only seasonal-naive denominator."""
    if actual.shape != predicted_median.shape or actual.size == 0:
        raise ValueError("actual and predicted arrays must have the same non-empty shape")
    if seasonal_period <= 0 or len(training) <= seasonal_period:
        raise ValueError("training data must exceed the positive seasonal period")
    denominator = float(np.mean(np.abs(training[seasonal_period:] - training[:-seasonal_period])))
    if denominator <= 0:
        raise ValueError("MASE is undefined for a constant seasonal training series")
    return float(np.mean(np.abs(actual - predicted_median)) / denominator)


def weighted_quantile_loss(
    actual: FloatArray,
    predicted: FloatArray,
    quantile_levels: tuple[float, ...],
) -> float:
    """Mean normalized quantile loss across levels (lower is better)."""
    if predicted.shape != (len(actual), len(quantile_levels)) or actual.size == 0:
        raise ValueError("predictions must have shape [observations, quantile_levels]")
    denominator = float(np.sum(np.abs(actual)))
    if denominator <= 0:
        raise ValueError("weighted quantile loss is undefined when all actuals are zero")
    losses = [
        2
        * float(
            np.sum(
                np.maximum(
                    quantile * (actual - predicted[:, index]),
                    (quantile - 1) * (actual - predicted[:, index]),
                )
            )
        )
        / denominator
        for index, quantile in enumerate(quantile_levels)
    ]
    return float(np.mean(losses))


def empirical_coverage(actual: FloatArray, predicted: FloatArray) -> float:
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted arrays must have the same non-empty shape")
    return float(np.mean(actual <= predicted))


@dataclass(frozen=True)
class ForecastMetrics:
    mase: float
    weighted_quantile_loss: float
    mean_pinball: float
    coverage: tuple[tuple[float, float], ...]


def summarize_forecasts(
    *,
    actual: FloatArray,
    predicted: FloatArray,
    quantile_levels: tuple[float, ...],
    training: FloatArray,
    seasonal_period: int,
) -> ForecastMetrics:
    if 0.5 not in quantile_levels:
        raise ValueError("metrics require a p50 forecast")
    median_index = quantile_levels.index(0.5)
    pinball = tuple(
        pinball_loss(actual, predicted[:, index], quantile)
        for index, quantile in enumerate(quantile_levels)
    )
    coverage = tuple(
        (quantile, empirical_coverage(actual, predicted[:, index]))
        for index, quantile in enumerate(quantile_levels)
    )
    return ForecastMetrics(
        mase=mase(actual, predicted[:, median_index], training, seasonal_period),
        weighted_quantile_loss=weighted_quantile_loss(actual, predicted, quantile_levels),
        mean_pinball=float(np.mean(pinball)),
        coverage=coverage,
    )
