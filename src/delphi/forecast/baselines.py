"""Honest probabilistic baselines that precede calibration and capacity control."""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from delphi.data.series import DemandSeries
from delphi.forecast.contracts import DemandForecast, build_forecast
from delphi.forecast.features import FeatureSpec, build_causal_training_matrix, causal_feature_row

FloatArray = npt.NDArray[np.float64]


def _validate_request(
    history: DemandSeries,
    horizon_steps: int,
    quantile_levels: tuple[float, ...],
    minimum_history: int,
) -> None:
    if len(history) < minimum_history:
        raise ValueError(f"model requires at least {minimum_history} historical points")
    if horizon_steps <= 0:
        raise ValueError("forecast horizon must be positive")
    if tuple(sorted(set(quantile_levels))) != quantile_levels or any(
        level <= 0 or level >= 1 for level in quantile_levels
    ):
        raise ValueError("quantile levels must be unique, sorted, and within (0, 1)")


def _residual_forecast(
    point: FloatArray,
    residuals: FloatArray,
    quantile_levels: tuple[float, ...],
    scale: FloatArray | None = None,
) -> FloatArray:
    offsets = np.quantile(residuals, quantile_levels)
    multiplier = np.ones(len(point), dtype=np.float64) if scale is None else scale
    values = point[:, None] + multiplier[:, None] * offsets[None, :]
    return np.maximum(values, 0.0)


@dataclass(frozen=True)
class SeasonalNaiveForecaster:
    period_steps: int
    name: str = "seasonal_naive"

    def __post_init__(self) -> None:
        if self.period_steps <= 0:
            raise ValueError("seasonal period must be positive")

    @property
    def model_id(self) -> str:
        return f"{self.name}:period={self.period_steps}:v1"

    @property
    def minimum_history(self) -> int:
        return self.period_steps + 2

    def forecast(
        self,
        history: DemandSeries,
        *,
        horizon_steps: int,
        quantile_levels: tuple[float, ...],
    ) -> DemandForecast:
        _validate_request(history, horizon_steps, quantile_levels, self.minimum_history)
        pattern = history.values[-self.period_steps :]
        point = np.resize(pattern, horizon_steps)
        residuals = history.values[self.period_steps :] - history.values[: -self.period_steps]
        values = _residual_forecast(point, residuals, quantile_levels)
        return build_forecast(
            history,
            horizon_steps=horizon_steps,
            quantile_levels=quantile_levels,
            quantile_values=values,
            model_id=self.model_id,
        )


@dataclass(frozen=True)
class DriftNaiveForecaster:
    residual_window: int = 1440

    @property
    def model_id(self) -> str:
        return f"drift_naive:residual_window={self.residual_window}:v1"

    @property
    def minimum_history(self) -> int:
        return 3

    def forecast(
        self,
        history: DemandSeries,
        *,
        horizon_steps: int,
        quantile_levels: tuple[float, ...],
    ) -> DemandForecast:
        _validate_request(history, horizon_steps, quantile_levels, self.minimum_history)
        sample = history.values[-min(len(history), self.residual_window) :]
        slope = float((sample[-1] - sample[0]) / (len(sample) - 1))
        steps = np.arange(1, horizon_steps + 1, dtype=np.float64)
        point = sample[-1] + slope * steps
        residuals = np.diff(sample) - slope
        values = _residual_forecast(point, residuals, quantile_levels, np.sqrt(steps))
        return build_forecast(
            history,
            horizon_steps=horizon_steps,
            quantile_levels=quantile_levels,
            quantile_values=values,
            model_id=self.model_id,
        )


@dataclass(frozen=True)
class RollingPercentileForecaster:
    window_steps: int = 1440

    def __post_init__(self) -> None:
        if self.window_steps <= 1:
            raise ValueError("rolling window must exceed one point")

    @property
    def model_id(self) -> str:
        return f"rolling_percentile:window={self.window_steps}:v1"

    @property
    def minimum_history(self) -> int:
        return self.window_steps

    def forecast(
        self,
        history: DemandSeries,
        *,
        horizon_steps: int,
        quantile_levels: tuple[float, ...],
    ) -> DemandForecast:
        _validate_request(history, horizon_steps, quantile_levels, self.minimum_history)
        quantiles = np.quantile(history.values[-self.window_steps :], quantile_levels)
        values = np.repeat(quantiles[None, :], horizon_steps, axis=0)
        return build_forecast(
            history,
            horizon_steps=horizon_steps,
            quantile_levels=quantile_levels,
            quantile_values=values,
            model_id=self.model_id,
        )


@dataclass(frozen=True)
class StatsForecastForecaster:
    kind: Literal["arima", "auto_arima", "auto_ets"]
    season_length: int
    max_history: int | None = None

    def __post_init__(self) -> None:
        if self.season_length <= 0:
            raise ValueError("season length must be positive")

    @property
    def model_id(self) -> str:
        return f"statsforecast:{self.kind}:season={self.season_length}:v1"

    @property
    def minimum_history(self) -> int:
        return max(20, 2 * self.season_length)

    def forecast(
        self,
        history: DemandSeries,
        *,
        horizon_steps: int,
        quantile_levels: tuple[float, ...],
    ) -> DemandForecast:
        _validate_request(history, horizon_steps, quantile_levels, self.minimum_history)
        from statsforecast.models import ARIMA, AutoARIMA, AutoETS

        model: ARIMA | AutoARIMA | AutoETS
        if self.kind == "arima":
            model = ARIMA(
                order=(2, 0, 2),
                seasonal_order=(1, 1, 1),
                season_length=self.season_length,
            )
        elif self.kind == "auto_arima":
            model = AutoARIMA(
                season_length=self.season_length,
                max_p=3,
                max_q=3,
                max_P=1,
                max_Q=1,
                max_order=4,
                nmodels=20,
            )
        else:
            model = AutoETS(season_length=self.season_length)
        levels = sorted(
            {round(abs(2 * quantile - 1) * 100) for quantile in quantile_levels if quantile != 0.5}
        )
        sample = history.values if self.max_history is None else history.values[-self.max_history :]
        raw = cast(
            dict[str, npt.ArrayLike],
            model.forecast(y=np.asarray(sample, dtype=np.float64), h=horizon_steps, level=levels),
        )
        columns: list[FloatArray] = []
        for quantile in quantile_levels:
            if quantile == 0.5:
                key = "mean"
            else:
                level = round(abs(2 * quantile - 1) * 100)
                key = f"lo-{level}" if quantile < 0.5 else f"hi-{level}"
            columns.append(np.asarray(raw[key], dtype=np.float64))
        values = np.maximum(np.sort(np.column_stack(columns), axis=1), 0.0)
        return build_forecast(
            history,
            horizon_steps=horizon_steps,
            quantile_levels=quantile_levels,
            quantile_values=values,
            model_id=self.model_id,
        )


@dataclass(frozen=True)
class LightGBMQuantileForecaster:
    """Per-quantile one-step models rolled forward recursively.

    Known limitation, measured 2026-08-03: because each quantile model is a *one-step*
    model applied recursively along its own path, the emitted band does not widen with
    lead time (p99-p50 measured at 31.95 at step 1 and 30.12 at step 48 on a clean daily
    trace). Horizon-h quantiles therefore under-state horizon-h uncertainty, which for a
    capacity controller means systematic under-provisioning at long lead times.

    Mitigation in place: ``SplitConformalCalibrator.fit_by_horizon`` fits one correction
    per horizon step, which restores the widening empirically. Do not consume this
    forecaster's raw quantiles at horizons beyond one step without per-horizon
    calibration. A direct multi-horizon model (one model per step) is the proper fix and
    is deferred, not solved.
    """

    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)
    n_estimators: int = 120
    learning_rate: float = 0.05
    num_leaves: int = 15
    max_training_rows: int = 50_000
    seed: int = 20260802

    @property
    def model_id(self) -> str:
        return (
            "lightgbm_quantile:"
            f"trees={self.n_estimators}:leaves={self.num_leaves}:seed={self.seed}:v1"
        )

    @property
    def minimum_history(self) -> int:
        return self.feature_spec.lookback + 1

    def forecast(
        self,
        history: DemandSeries,
        *,
        horizon_steps: int,
        quantile_levels: tuple[float, ...],
    ) -> DemandForecast:
        _validate_request(history, horizon_steps, quantile_levels, self.minimum_history)
        from lightgbm import LGBMRegressor

        training = build_causal_training_matrix(
            history,
            self.feature_spec,
            fit_end_exclusive=len(history),
            max_rows=self.max_training_rows,
        )
        models = []
        for quantile in quantile_levels:
            model = LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=10,
                random_state=self.seed,
                n_jobs=1,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )
            model.fit(training.values, training.targets)
            models.append(model)

        paths = [history.values.copy() for _ in quantile_levels]
        values = np.empty((horizon_steps, len(quantile_levels)), dtype=np.float64)
        for step in range(horizon_steps):
            timestamp = history.end + timedelta(seconds=history.step_seconds * (step + 1))
            raw_predictions = np.asarray(
                [
                    float(
                        np.asarray(
                            model.predict(
                                causal_feature_row(path, timestamp, self.feature_spec)[None, :]
                            )
                        )[0]
                    )
                    for model, path in zip(models, paths, strict=True)
                ],
                dtype=np.float64,
            )
            ordered = np.maximum(np.sort(raw_predictions), 0.0)
            values[step] = ordered
            for index, prediction in enumerate(ordered):
                paths[index] = np.append(paths[index], prediction)
        return build_forecast(
            history,
            horizon_steps=horizon_steps,
            quantile_levels=quantile_levels,
            quantile_values=values,
            model_id=self.model_id,
        )
