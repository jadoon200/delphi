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
    assert calibrator.corrections == (2.0, 1.0)
    assert calibrated.quantile_values.tolist() == [[12.0, 12.0]]
    assert calibrated.calibrator_id.startswith("split_conformal")


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
