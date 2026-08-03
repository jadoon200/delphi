from datetime import UTC, datetime, timedelta

import numpy as np

from delphi.data.series import DemandSeries
from delphi.data.synthetic import generate_synthetic
from delphi.forecast.baselines import (
    DriftNaiveForecaster,
    LightGBMQuantileForecaster,
    RollingPercentileForecaster,
    SeasonalNaiveForecaster,
    StatsForecastForecaster,
)
from delphi.forecast.features import FeatureSpec

LEVELS = (0.1, 0.5, 0.9)


def _periodic_series(periods: int = 100, period: int = 10) -> DemandSeries:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    values = np.tile(np.arange(1, period + 1, dtype=np.float64), periods // period)
    return DemandSeries(
        workload_id="periodic",
        source_id="test",
        resource_kind="invocations",
        unit="requests/hour",
        step_seconds=3600,
        timestamps=tuple(start + timedelta(hours=index) for index in range(periods)),
        values=values,
        is_imputed=np.zeros(periods, dtype=np.bool_),
        quality=np.ones(periods, dtype=np.float64),
    )


def test_honesty_baselines_emit_non_crossing_quantiles() -> None:
    series = _periodic_series()
    models = (
        SeasonalNaiveForecaster(period_steps=10),
        DriftNaiveForecaster(residual_window=50),
        RollingPercentileForecaster(window_steps=20),
    )
    for model in models:
        forecast = model.forecast(series, horizon_steps=5, quantile_levels=LEVELS)
        assert forecast.quantile_values.shape == (5, 3)
        assert np.all(np.diff(forecast.quantile_values, axis=1) >= 0)
    seasonal = models[0].forecast(series, horizon_steps=5, quantile_levels=LEVELS)
    assert seasonal.quantile_values[:, 1].tolist() == [1, 2, 3, 4, 5]


def test_statsforecast_classical_models_emit_quantiles() -> None:
    series = generate_synthetic("clean_daily", periods=96, step_seconds=3600, seed=7)
    for kind in ("arima", "auto_arima", "auto_ets"):
        model = StatsForecastForecaster(kind=kind, season_length=24, max_history=96)
        forecast = model.forecast(series, horizon_steps=3, quantile_levels=LEVELS)
        assert forecast.quantile_values.shape == (3, 3)
        assert np.all(np.diff(forecast.quantile_values, axis=1) >= 0)


def test_lightgbm_quantile_model_trains_on_causal_features() -> None:
    series = generate_synthetic("clean_daily", periods=120, step_seconds=3600, seed=11)
    model = LightGBMQuantileForecaster(
        feature_spec=FeatureSpec(lags=(1, 2, 6, 12), windows=(3, 12)),
        n_estimators=20,
        num_leaves=7,
        max_training_rows=80,
        seed=11,
    )
    forecast = model.forecast(series, horizon_steps=3, quantile_levels=LEVELS)
    assert forecast.quantile_values.shape == (3, 3)
    assert np.all(np.diff(forecast.quantile_values, axis=1) >= 0)
