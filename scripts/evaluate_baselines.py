"""Print a deterministic rolling-origin baseline table on labelled synthetic regimes."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from delphi.data.synthetic import SyntheticKind, generate_synthetic
from delphi.evaluation.metrics import ForecastMetrics
from delphi.evaluation.rolling_origin import (
    SplitBoundaries,
    rolling_origin_backtest,
    score_backtest,
)
from delphi.forecast.baselines import (
    DriftNaiveForecaster,
    LightGBMQuantileForecaster,
    RollingPercentileForecaster,
    SeasonalNaiveForecaster,
    StatsForecastForecaster,
)
from delphi.forecast.contracts import QuantileForecaster
from delphi.forecast.features import FeatureSpec

LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
REGIMES: tuple[SyntheticKind, ...] = (
    "clean_daily",
    "burst_known",
    "level_shift",
    "near_noise",
)


@dataclass(frozen=True)
class EvaluationRow:
    regime: str
    model: str
    metrics: ForecastMetrics


def models() -> tuple[QuantileForecaster, ...]:
    features = FeatureSpec(lags=(1, 2, 3, 6, 12, 24, 168), windows=(6, 24, 168))
    return (
        SeasonalNaiveForecaster(24, name="seasonal_naive_d"),
        SeasonalNaiveForecaster(168, name="seasonal_naive_w"),
        DriftNaiveForecaster(residual_window=168),
        RollingPercentileForecaster(window_steps=168),
        StatsForecastForecaster("auto_ets", season_length=24, max_history=336),
        StatsForecastForecaster("arima", season_length=24, max_history=336),
        LightGBMQuantileForecaster(
            feature_spec=features,
            n_estimators=80,
            num_leaves=15,
            max_training_rows=336,
        ),
    )


def evaluate() -> tuple[EvaluationRow, ...]:
    boundaries = SplitBoundaries(train_end=14 * 24, validation_end=17 * 24, test_end=21 * 24)
    rows: list[EvaluationRow] = []
    for regime in REGIMES:
        series = generate_synthetic(
            regime,
            periods=21 * 24,
            step_seconds=3600,
            seed=20260802,
            start=datetime(2026, 1, 1, tzinfo=UTC),
        )
        for model in models():
            result = rolling_origin_backtest(
                series,
                model,
                boundaries,
                horizon_steps=24,
                stride_steps=24,
                quantile_levels=LEVELS,
            )
            rows.append(
                EvaluationRow(
                    regime=regime,
                    model=model.model_id,
                    metrics=score_backtest(result, series, seasonal_period=24),
                )
            )
    return tuple(rows)


def markdown(rows: Iterable[EvaluationRow]) -> str:
    lines = [
        "| Regime | Model | MASE | WQL | Mean pinball | p95 coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        coverage = dict(row.metrics.coverage)[0.95]
        lines.append(
            f"| {row.regime} | `{row.model}` | {row.metrics.mase:.3f} | "
            f"{row.metrics.weighted_quantile_loss:.3f} | {row.metrics.mean_pinball:.3f} | "
            f"{coverage:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown(evaluate()))
