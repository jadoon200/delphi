"""M17's optional foundation forecaster, including the quantile ceiling that disqualifies it."""

import numpy as np
import pytest

from delphi.data.synthetic import generate_synthetic
from delphi.forecast.foundation import (
    CHRONOS_TRAINED_RANGE,
    ChronosForecaster,
    chronos_available,
)

pytestmark = [
    pytest.mark.foundation,
    pytest.mark.skipif(not chronos_available(), reason="chronos-forecasting is an optional extra"),
]


def test_clamped_levels_names_exactly_what_the_model_cannot_express() -> None:
    """This is the finding, and it must not depend on loading any weights."""
    low, high = CHRONOS_TRAINED_RANGE
    assert ChronosForecaster.clamped_levels((0.5, 0.8, 0.9)) == ()
    assert ChronosForecaster.clamped_levels((0.5, 0.95, 0.99)) == (0.95, 0.99)
    assert ChronosForecaster.clamped_levels((0.05, 0.5)) == (0.05,)
    assert low == 0.1 and high == 0.9


def test_forecast_satisfies_the_project_contract() -> None:
    series = generate_synthetic("clean_daily", periods=600, step_seconds=3600)
    forecaster = ChronosForecaster()
    levels = (0.5, 0.8, 0.9)
    forecast = forecaster.forecast(series, horizon_steps=24, quantile_levels=levels)

    assert forecast.quantile_values.shape == (24, len(levels))
    assert np.isfinite(forecast.quantile_values).all()
    assert (forecast.quantile_values >= 0).all(), "demand cannot be negative"
    assert (np.diff(forecast.quantile_values, axis=1) >= 0).all(), "quantiles must not cross"
    assert forecast.model_id.startswith("chronos:")


def test_upper_quantiles_collapse_onto_p90() -> None:
    """Asked for p95 and p99, the model returns p90 three times over.

    Not a bug in this wrapper — Chronos-Bolt is trained on 0.1-0.9 and clamps outside it.
    It is recorded as a test because a capacity controller consumes exactly those upper
    levels, and a silently flattened band would size for the wrong quantile while looking
    perfectly well-formed.
    """
    series = generate_synthetic("clean_daily", periods=600, step_seconds=3600)
    forecast = ChronosForecaster().forecast(
        series, horizon_steps=24, quantile_levels=(0.9, 0.95, 0.99)
    )
    p90, p95, p99 = (forecast.quantile_values[:, i] for i in range(3))
    assert np.allclose(p90, p95), "p95 should be indistinguishable from p90"
    assert np.allclose(p90, p99), "p99 should be indistinguishable from p90"
