"""Split conformal, CQR, and adaptive conformal calibration."""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from delphi.forecast.contracts import DemandForecast

FloatArray = npt.NDArray[np.float64]


def conformal_rank(sample_size: int, level: float) -> tuple[int, bool]:
    """Return the ``(rank, saturated)`` order statistic for a conformal level.

    ``saturated`` is True when ``ceil((n+1)*level)`` exceeds ``n``: the requested level is
    not representable from ``n`` scores, so the estimate silently falls back to the sample
    maximum. A window of 72 scores cannot express any level above 72/73 ≈ 0.986, which
    matters because it caps how conservative a calibrator can ever become.
    """
    if sample_size < 1:
        raise ValueError("conformal sample size must be positive")
    if not 0 < level < 1:
        raise ValueError("conformal level must be strictly within (0, 1)")
    raw = math.ceil((sample_size + 1) * level)
    return min(raw, sample_size), raw > sample_size


def finite_sample_quantile(scores: FloatArray, level: float) -> float:
    """Conformal order statistic with the finite-sample ``(n+1)`` correction."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("conformal scores must be a non-empty finite 1-D array")
    rank, _ = conformal_rank(len(values), level)
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


def _corrections_for(
    corrections: tuple[tuple[float, ...], ...],
    horizon_steps: int,
) -> npt.NDArray[np.float64]:
    """Broadcast pooled or per-horizon corrections onto ``[horizon, levels]``."""
    table = np.asarray(corrections, dtype=np.float64)
    if table.shape[0] == 1:
        return np.repeat(table, horizon_steps, axis=0)
    if table.shape[0] != horizon_steps:
        raise ValueError(
            f"calibrator was fitted for {table.shape[0]} horizon steps "
            f"but the forecast has {horizon_steps}"
        )
    return table


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
    """One-sided residual calibration for every controller-facing demand quantile.

    ``corrections`` is ``[rows, levels]`` where ``rows`` is 1 for a pooled fit or
    ``horizon_steps`` for a per-horizon fit. Prefer :meth:`fit_by_horizon` whenever the
    horizon exceeds one step: forecast error grows with lead time, so a single pooled
    correction over-covers step 1 and under-covers the last step — and the last step is
    precisely the one capacity control acts on.
    """

    quantile_levels: tuple[float, ...]
    corrections: tuple[tuple[float, ...], ...]
    calibration_size: int
    horizon_steps: int = 1

    @classmethod
    def fit(
        cls,
        actual: FloatArray,
        predicted: npt.NDArray[np.float64],
        quantile_levels: tuple[float, ...],
    ) -> "SplitConformalCalibrator":
        """Fit one pooled correction per level. Correct only for a one-step horizon."""
        _validate_calibration_arrays(actual, predicted, quantile_levels)
        corrections = tuple(
            finite_sample_quantile(actual - predicted[:, index], level)
            for index, level in enumerate(quantile_levels)
        )
        return cls(quantile_levels, (corrections,), len(actual))

    @classmethod
    def fit_by_horizon(
        cls,
        actual: npt.NDArray[np.float64],
        predicted: npt.NDArray[np.float64],
        quantile_levels: tuple[float, ...],
    ) -> "SplitConformalCalibrator":
        """Fit one correction per horizon step from ``arrays_by_horizon`` output."""
        if actual.ndim != 2 or predicted.shape != (*actual.shape, len(quantile_levels)):
            raise ValueError(
                "per-horizon calibration needs actual[origins, horizon] and "
                "predicted[origins, horizon, levels]"
            )
        origins, horizon = actual.shape
        if origins < 1:
            raise ValueError("per-horizon calibration needs at least one origin")
        corrections = tuple(
            tuple(
                finite_sample_quantile(actual[:, step] - predicted[:, step, index], level)
                for index, level in enumerate(quantile_levels)
            )
            for step in range(horizon)
        )
        return cls(quantile_levels, corrections, origins, horizon)

    @property
    def calibrator_id(self) -> str:
        scope = "pooled" if len(self.corrections) == 1 else f"h={self.horizon_steps}"
        return f"split_conformal:{scope}:n={self.calibration_size}:v1"

    def calibrate(self, forecast: DemandForecast) -> DemandForecast:
        if forecast.quantile_levels != self.quantile_levels:
            raise ValueError("forecast levels differ from fitted conformal levels")
        values = forecast.quantile_values + _corrections_for(
            self.corrections, forecast.horizon_steps
        )
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
    adjustments: tuple[tuple[float, ...], ...]
    calibration_size: int

    @classmethod
    def fit(
        cls,
        actual: FloatArray,
        predicted: npt.NDArray[np.float64],
        quantile_levels: tuple[float, ...],
    ) -> "CQRCalibrator":
        """Fit one pooled adjustment per level. Correct only for a one-step horizon."""
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
        return cls(quantile_levels, (tuple(float(value) for value in adjustments),), len(actual))

    @property
    def calibrator_id(self) -> str:
        return f"cqr:n={self.calibration_size}:v1"

    def calibrate(self, forecast: DemandForecast) -> DemandForecast:
        if forecast.quantile_levels != self.quantile_levels:
            raise ValueError("forecast levels differ from fitted CQR levels")
        values = forecast.quantile_values + _corrections_for(
            self.adjustments, forecast.horizon_steps
        )
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
    #: True when ``1 - alpha`` exceeded what the score window can express, so the
    #: correction fell back to the window maximum and the calibrator cannot become any
    #: more conservative no matter how many misses follow.
    saturated: bool = False


class AdaptiveConformalCalibrator:
    """Online one-sided ACI over a rolling window of causal residual scores."""

    def __init__(
        self,
        initial_scores: FloatArray,
        *,
        target_coverage: float = 0.95,
        gamma: float = 0.01,
        alpha_min: float | None = None,
        alpha_max: float | None = None,
        score_window: int = 1440,
    ) -> None:
        """Bounds default *relative to the target* rather than to fixed constants.

        A newsvendor ``q*`` is a price ratio, so it can legitimately sit at or below 0.5
        when idle capacity costs more than a breach — opportunistic and spot-tier work is
        exactly that case. Hard-coding a floor of 0.5 would have made those targets
        unrepresentable for no mathematical reason.
        """
        if not 0 < target_coverage < 1:
            raise ValueError("ACI target coverage must be strictly within (0, 1)")
        target_alpha = 1 - target_coverage
        if alpha_min is None:
            alpha_min = max(1e-4, target_alpha * 0.1)
        if alpha_max is None:
            alpha_max = min(1 - 1e-4, target_alpha + (1 - target_alpha) * 0.5)
        if gamma <= 0 or not 0 < alpha_min < target_alpha < alpha_max < 1:
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

    def current_correction(self) -> float:
        """The additive correction the calibrator would apply right now.

        A controller has to size capacity *before* the outcome exists, so it needs to read
        the correction without consuming an observation. ``observe`` remains the only way
        to advance the calibrator's state.
        """
        return finite_sample_quantile(np.asarray(self._scores, dtype=np.float64), 1 - self.alpha)

    def is_saturated(self) -> bool:
        """Whether the score window can still express the current confidence level."""
        _, saturated = conformal_rank(len(self._scores), 1 - self.alpha)
        return saturated

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
        _, saturated = conformal_rank(len(self._scores), 1 - alpha_before)
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
            saturated=saturated,
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
