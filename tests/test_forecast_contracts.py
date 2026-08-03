from datetime import UTC, datetime

import numpy as np
import pytest

from delphi.forecast.contracts import DemandForecast


def _forecast(values: np.ndarray) -> DemandForecast:
    return DemandForecast(
        workload_id="w",
        origin_ts=datetime(2026, 1, 1, tzinfo=UTC),
        horizon_steps=2,
        step_seconds=60,
        quantile_levels=(0.5, 0.9),
        quantile_values=values,
        model_id="model:v1",
    )


def test_forecast_contract_is_quantile_only_and_immutable() -> None:
    forecast = _forecast(np.asarray([[1, 2], [3, 4]], dtype=np.float64))
    assert not forecast.quantile_values.flags.writeable
    assert forecast.timestamps[0] == datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        forecast.quantile_values[0, 0] = 7


def test_forecast_contract_rejects_crossing_quantiles() -> None:
    with pytest.raises(ValueError, match="must not cross"):
        _forecast(np.asarray([[2, 1], [3, 4]], dtype=np.float64))
