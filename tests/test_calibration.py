from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from delphi.forecast.calibration import (
    AdaptiveConformalCalibrator,
    CQRCalibrator,
    SplitConformalCalibrator,
    finite_sample_quantile,
    tune_aci_gamma,
)
from delphi.forecast.contracts import DemandForecast


def _forecast(levels: tuple[float, ...], values: list[list[float]]) -> DemandForecast:
    return DemandForecast(
        workload_id="w",
        origin_ts=datetime(2026, 1, 1, tzinfo=UTC),
        horizon_steps=len(values),
        step_seconds=60,
        quantile_levels=levels,
        quantile_values=np.asarray(values, dtype=np.float64),
        model_id="raw:v1",
    )


def test_finite_sample_quantile_uses_conservative_order_statistic() -> None:
    scores = np.arange(1, 11, dtype=np.float64)
    assert finite_sample_quantile(scores, 0.8) == 9
    assert finite_sample_quantile(scores, 0.95) == 10


def test_split_conformal_corrects_each_quantile_from_past_residuals() -> None:
    levels = (0.5, 0.9)
    predicted = np.asarray([[5, 6], [6, 7], [7, 8], [8, 9]], dtype=np.float64)
    actual = predicted[:, 0] + 2
    calibrator = SplitConformalCalibrator.fit(actual, predicted, levels)
    calibrated = calibrator.calibrate(_forecast(levels, [[10, 11]]))
    assert calibrator.corrections == ((2.0, 1.0),)
    assert calibrated.quantile_values.tolist() == [[12.0, 12.0]]
    assert calibrated.calibrator_id.startswith("split_conformal")


def test_per_horizon_conformal_tracks_error_that_grows_with_lead_time() -> None:
    """Pooling one correction across the horizon is wrong in both directions.

    Forecast error grows with lead time, so a single pooled correction over-covers the
    near steps (wasted capacity) and under-covers the far ones (SLO breach) — and the far
    step is precisely the one capacity control acts on, because the decision has to be
    made a whole ``startup_seconds`` ahead.
    """
    levels = (0.9,)
    origins, horizon = 400, 10
    rng = np.random.default_rng(20260803)
    # residual scale grows linearly with lead time
    scales = np.arange(1, horizon + 1, dtype=np.float64)
    actual = rng.normal(0.0, 1.0, size=(origins, horizon)) * scales
    actual = np.abs(actual)
    predicted = np.zeros((origins, horizon, 1), dtype=np.float64)

    per_horizon = SplitConformalCalibrator.fit_by_horizon(actual, predicted, levels)
    pooled = SplitConformalCalibrator.fit(actual.reshape(-1), predicted.reshape(-1, 1), levels)

    near = per_horizon.corrections[0][0]
    far = per_horizon.corrections[-1][0]
    pooled_correction = pooled.corrections[0][0]
    assert near < pooled_correction < far

    forecast = _forecast(levels, [[0.0]] * horizon)
    pooled_values = pooled.calibrate(forecast).quantile_values[:, 0]
    exact_values = per_horizon.calibrate(forecast).quantile_values[:, 0]

    # realised one-sided coverage per step: pooled misses badly at the far horizon
    pooled_far = float(np.mean(actual[:, -1] <= pooled_values[-1]))
    exact_far = float(np.mean(actual[:, -1] <= exact_values[-1]))
    assert pooled_far < 0.75, "pooled correction should visibly under-cover the far step"
    assert exact_far >= 0.88, "per-horizon correction should hold ~90% at the far step"

    pooled_near = float(np.mean(actual[:, 0] <= pooled_values[0]))
    assert pooled_near > 0.99, "pooled correction should visibly over-cover the near step"


def test_per_horizon_calibrator_rejects_a_mismatched_horizon() -> None:
    levels = (0.9,)
    actual = np.tile(np.asarray([1.0, 2.0]), (5, 1))
    predicted = np.zeros((5, 2, 1), dtype=np.float64)
    calibrator = SplitConformalCalibrator.fit_by_horizon(actual, predicted, levels)
    with pytest.raises(ValueError, match="fitted for 2 horizon steps"):
        calibrator.calibrate(_forecast(levels, [[0.0], [0.0], [0.0]]))


def test_aci_reports_when_its_score_window_cannot_express_the_level() -> None:
    """A short window silently caps how conservative ACI can get; that must be visible."""
    scores = np.arange(1.0, 11.0)  # n=10 -> no level above 10/11 is representable
    calibrator = AdaptiveConformalCalibrator(
        scores, target_coverage=0.95, gamma=0.01, alpha_min=0.001, score_window=10
    )
    calibrator.alpha = 0.001  # the most conservative state the bounds allow
    point = calibrator.observe(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC), raw_prediction=0.0, actual=0.0
    )
    assert point.saturated
    assert point.correction == 10.0  # fell back to the window maximum


def test_cqr_widens_symmetric_interval_without_crossing() -> None:
    levels = (0.05, 0.5, 0.95)
    predicted = np.asarray([[4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]], dtype=np.float64)
    actual = np.asarray([5, 6, 7, 8], dtype=np.float64)
    calibrator = CQRCalibrator.fit(actual, predicted, levels)
    calibrated = calibrator.calibrate(_forecast(levels, [[4, 5, 6]]))
    assert calibrated.quantile_values[0, 0] <= 4
    assert calibrated.quantile_values[0, 2] >= 8
    assert np.all(np.diff(calibrated.quantile_values, axis=1) >= 0)


def test_cqr_requires_symmetric_quantile_pairs() -> None:
    with pytest.raises(ValueError, match="symmetric lower"):
        CQRCalibrator.fit(
            np.asarray([1, 2], dtype=np.float64),
            np.asarray([[1, 2], [2, 3]], dtype=np.float64),
            (0.5, 0.95),
        )


def test_aci_becomes_more_conservative_after_a_miss() -> None:
    calibrator = AdaptiveConformalCalibrator(
        np.zeros(20, dtype=np.float64),
        target_coverage=0.95,
        gamma=0.01,
        score_window=20,
    )
    first = calibrator.observe(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        raw_prediction=0,
        actual=10,
    )
    second = calibrator.observe(
        timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        raw_prediction=0,
        actual=10,
    )
    assert not first.covered
    assert first.alpha_after < first.alpha_before
    assert second.correction == 10
    assert second.covered


def test_gamma_is_selected_once_from_validation_stream() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    actual = np.asarray([0, 0, 3, 3, 3, 3], dtype=np.float64)
    selection = tune_aci_gamma(
        initial_scores=np.zeros(10, dtype=np.float64),
        timestamps=tuple(start + timedelta(minutes=index) for index in range(len(actual))),
        raw_predictions=np.zeros(len(actual), dtype=np.float64),
        actual=actual,
        candidates=(0.001, 0.01),
        target_coverage=0.9,
        score_window=10,
    )
    assert selection.gamma in (0.001, 0.01)
    assert 0 <= selection.empirical_coverage <= 1
