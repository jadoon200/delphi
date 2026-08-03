from datetime import UTC, datetime

import numpy as np
import pytest

from delphi.data.synthetic import generate_synthetic
from delphi.evaluation.metrics import mase, pinball_loss, weighted_quantile_loss
from delphi.evaluation.rolling_origin import (
    SplitBoundaries,
    rolling_origin_backtest,
    score_backtest,
)
from delphi.forecast.baselines import SeasonalNaiveForecaster


def test_metrics_have_known_values_and_train_only_scale() -> None:
    actual = np.asarray([2.0, 4.0], dtype=np.float64)
    predicted = np.asarray([1.0, 5.0], dtype=np.float64)
    training = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert pinball_loss(actual, predicted, 0.5) == 0.5
    assert mase(actual, predicted, training, seasonal_period=1) == 1.0
    matrix = np.column_stack([predicted, predicted])
    assert weighted_quantile_loss(actual, matrix, (0.5, 0.9)) > 0


def test_rolling_origin_records_boundaries_and_scores_quantiles() -> None:
    series = generate_synthetic(
        "clean_daily",
        periods=120,
        step_seconds=3600,
        seed=13,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    boundaries = SplitBoundaries(train_end=60, validation_end=84, test_end=120)
    result = rolling_origin_backtest(
        series,
        SeasonalNaiveForecaster(period_steps=24),
        boundaries,
        horizon_steps=6,
        stride_steps=6,
        quantile_levels=(0.5, 0.9),
    )
    assert len(result.records) == 6
    assert result.records[0].target_start == 84
    assert result.records[0].origin_ts == series.timestamps[83]
    metrics = score_backtest(result, series, seasonal_period=24)
    assert metrics.mase >= 0
    assert tuple(level for level, _ in metrics.coverage) == (0.5, 0.9)


def test_rolling_origin_rejects_nonchronological_splits() -> None:
    series = generate_synthetic("clean_daily", periods=100, step_seconds=3600)
    with pytest.raises(ValueError, match="strictly chronological"):
        rolling_origin_backtest(
            series,
            SeasonalNaiveForecaster(period_steps=24),
            SplitBoundaries(train_end=60, validation_end=60, test_end=100),
            horizon_steps=4,
            stride_steps=4,
        )
