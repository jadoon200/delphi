"""Split conformal, CQR, and adaptive conformal calibration."""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from delphi.forecast.contracts import DemandForecast

FloatArray = npt.NDArray[np.float64]


def finite_sample_quantile(scores: FloatArray, level: float) -> float:
    """Conformal order statistic with the finite-sample ``(n+1)`` correction."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("conformal scores must be a non-empty finite 1-D array")
    if not 0 < level < 1:
        raise ValueError("conformal level must be strictly within (0, 1)")
    rank = min(math.ceil((len(values) + 1) * level), len(values))
    return float(np.partition(values, rank - 1)[rank - 1])


def _validate_calibration_arrays(
    actual: FloatArray,
    predicted: npt.NDArray[np.float64],
    quantile_levels: tuple[float, ...],
) -> None:
    if actual.ndim != 1 or not len(actual):
        raise ValueError("calibration actuals must be a non-empty 1-D array")
    if predicted.shape != (len(actual), len(quantile_levels)):
        raise ValueError("calibration predictions have the wrong shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("calibration arrays must be finite")


def _calibrated_forecast(
    forecast: DemandForecast,
    values: npt.NDArray[np.float64],
    *,
    calibrator_id: str,
    coverage_record_id: str,
) -> DemandForecast:
    non_crossing = np.maximum.accumulate(np.maximum(values, 0.0), axis=1)
    return DemandForecast(
        workload_id=forecast.workload_id,
        origin_ts=forecast.origin_ts,
        horizon_steps=forecast.horizon_steps,
        step_seconds=forecast.step_seconds,
        quantile_levels=forecast.quantile_levels,
        quantile_values=non_crossing,
        model_id=forecast.model_id,
        calibrator_id=calibrator_id,
        coverage_record_id=coverage_record_id,
    )


@dataclass(frozen=True)
class SplitConformalCalibrator:
    """One-sided residual calibration for every controller-facing demand quantile."""

    quantile_levels: tuple[float, ...]
    corrections: tuple[float, ...]
    calibration_size: int

    @classmethod
    def fit(
        cls,
        actual: FloatArray,
        predicted: npt.NDArray[np.float64],
        quantile_levels: tuple[float, ...],
    ) -> "SplitConformalCalibrator":
        _validate_calibration_arrays(actual, predicted, quantile_levels)
        corrections = tuple(
            finite_sample_quantile(actual - predicted[:, index], level)
            for index, level in enumerate(quantile_levels)
        )
        return cls(quantile_levels, corrections, len(actual))

    @property
    def calibrator_id(self) -> str:
        return f"split_conformal:n={self.calibration_size}:v1"

    def calibrate(self, forecast: DemandForecast) -> DemandForecast:
        if forecast.quantile_levels != self.quantile_levels:
            raise ValueError("forecast levels differ from fitted conformal levels")
        values = forecast.quantile_values + np.asarray(self.corrections)[None, :]
        return _calibrated_forecast(
            forecast,
            values,
            calibrator_id=self.calibrator_id,
            coverage_record_id=f"calibration-window:n={self.calibration_size}",
        )


@dataclass(frozen=True)
class CQRCalibrator:
    """Conformalized quantile regression for symmetric lower/upper quantile pairs."""

    quantile_levels: tuple[float, ...]
    adjustments: tuple[float, ...]
    calibration_size: int

    @classmethod
    def fit(
        cls,
        actual: FloatArray,
        predicted: npt.NDArray[np.float64],
        quantile_levels: tuple[float, ...],
    ) -> "CQRCalibrator":
        _validate_calibration_arrays(actual, predicted, quantile_levels)
        adjustments = np.zeros(len(quantile_levels), dtype=np.float64)
        for high_index, high in enumerate(quantile_levels):
            if high <= 0.5:
                continue
            low = 1 - high
            candidates = [
                index for index, level in enumerate(quantile_levels) if math.isclose(level, low)
            ]
            if not candidates:
                raise ValueError(f"CQR requires the symmetric lower level {low:g} for p{high:g}")
            low_index = candidates[0]
            scores = np.maximum(
                predicted[:, low_index] - actual,
                actual - predicted[:, high_index],
            )
            correction = finite_sample_quantile(scores, 2 * high - 1)
            adjustments[low_index] -= correction
            adjustments[high_index] += correction
        if 0.5 in quantile_levels:
            median_index = quantile_levels.index(0.5)
            adjustments[median_index] = finite_sample_quantile(
                actual - predicted[:, median_index], 0.5
            )
        return cls(quantile_levels, tuple(float(value) for value in adjustments), len(actual))

    @property
    def calibrator_id(self) -> str:
        return f"cqr:n={self.calibration_size}:v1"

    def calibrate(self, forecast: DemandForecast) -> DemandForecast:
        if forecast.quantile_levels != self.quantile_levels:
            raise ValueError("forecast levels differ from fitted CQR levels")
        values = forecast.quantile_values + np.asarray(self.adjustments)[None, :]
        return _calibrated_forecast(
            forecast,
            values,
            calibrator_id=self.calibrator_id,
            coverage_record_id=f"calibration-window:n={self.calibration_size}",
        )


@dataclass(frozen=True)
class AdaptiveCoveragePoint:
    timestamp: datetime
    raw_prediction: float
    calibrated_prediction: float
    actual: float
    correction: float
    alpha_before: float
    alpha_after: float
    covered: bool


class AdaptiveConformalCalibrator:
    """Online one-sided ACI over a rolling window of causal residual scores."""

    def __init__(
        self,
        initial_scores: FloatArray,
        *,
        target_coverage: float = 0.95,
        gamma: float = 0.01,
        alpha_min: float = 0.001,
        alpha_max: float = 0.5,
        score_window: int = 1440,
    ) -> None:
        if not 0.5 < target_coverage < 1:
            raise ValueError("ACI target coverage must be within (0.5, 1)")
        if gamma <= 0 or not 0 < alpha_min < 1 - target_coverage < alpha_max < 1:
            raise ValueError("ACI gamma and alpha bounds are invalid")
        if score_window <= 1:
            raise ValueError("ACI score window must exceed one")
        scores = np.asarray(initial_scores, dtype=np.float64)
        if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
            raise ValueError("ACI initial scores must be a non-empty finite 1-D array")
        self.target_coverage = target_coverage
        self.target_alpha = 1 - target_coverage
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.score_window = score_window
        self.alpha = self.target_alpha
        self._scores = [float(score) for score in scores[-score_window:]]

    @property
    def calibrator_id(self) -> str:
        return f"aci:coverage={self.target_coverage:g}:gamma={self.gamma:g}:v1"

    def observe(
        self,
        *,
        timestamp: datetime,
        raw_prediction: float,
        actual: float,
    ) -> AdaptiveCoveragePoint:
        if not np.isfinite(raw_prediction) or not np.isfinite(actual) or actual < 0:
            raise ValueError("ACI observations must be finite and actual demand non-negative")
        alpha_before = self.alpha
        correction = finite_sample_quantile(
            np.asarray(self._scores, dtype=np.float64), 1 - alpha_before
        )
        calibrated = max(0.0, raw_prediction + correction)
        covered = actual <= calibrated
        error = 0.0 if covered else 1.0
        self.alpha = float(
            np.clip(
                alpha_before + self.gamma * (self.target_alpha - error),
                self.alpha_min,
                self.alpha_max,
            )
        )
        self._scores.append(actual - raw_prediction)
        self._scores = self._scores[-self.score_window :]
        return AdaptiveCoveragePoint(
            timestamp=timestamp,
            raw_prediction=raw_prediction,
            calibrated_prediction=calibrated,
            actual=actual,
            correction=correction,
            alpha_before=alpha_before,
            alpha_after=self.alpha,
            covered=covered,
        )


@dataclass(frozen=True)
class GammaSelection:
    gamma: float
    empirical_coverage: float
    mean_pinball: float


def tune_aci_gamma(
    *,
    initial_scores: FloatArray,
    timestamps: tuple[datetime, ...],
    raw_predictions: FloatArray,
    actual: FloatArray,
    candidates: tuple[float, ...],
    target_coverage: float,
    score_window: int,
) -> GammaSelection:
    """Choose gamma once on validation by coverage gap, then pinball, then stability."""
    if not candidates or any(candidate <= 0 for candidate in candidates):
        raise ValueError("gamma candidates must be non-empty and positive")
    if len(timestamps) != len(raw_predictions) or actual.shape != raw_predictions.shape:
        raise ValueError("gamma validation arrays must have matching lengths")
    selections: list[GammaSelection] = []
    for gamma in candidates:
        calibrator = AdaptiveConformalCalibrator(
            initial_scores,
            target_coverage=target_coverage,
            gamma=gamma,
            score_window=score_window,
        )
        points = tuple(
            calibrator.observe(
                timestamp=timestamp,
                raw_prediction=float(raw),
                actual=float(observed),
            )
            for timestamp, raw, observed in zip(timestamps, raw_predictions, actual, strict=True)
        )
        coverage = float(np.mean([point.covered for point in points]))
        predictions = np.asarray(
            [point.calibrated_prediction for point in points], dtype=np.float64
        )
        error = actual - predictions
        pinball = float(
            np.mean(
                np.maximum(
                    target_coverage * error,
                    (target_coverage - 1) * error,
                )
            )
        )
        selections.append(GammaSelection(gamma, coverage, pinball))
    return min(
        selections,
        key=lambda result: (
            abs(result.empirical_coverage - target_coverage),
            result.mean_pinball,
            result.gamma,
        ),
    )
