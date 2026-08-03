"""Evaluate the baseline ladder on one reproducibly selected real Azure workload."""

from pathlib import Path

from evaluate_baselines import LEVELS, EvaluationRow, markdown, models

from delphi.data.azure_functions import load_archive_cohort, select_cohort
from delphi.data.series import aggregate_series
from delphi.evaluation.rolling_origin import (
    SplitBoundaries,
    rolling_origin_backtest,
    score_backtest,
)


def main() -> None:
    archive = Path("data/raw/azure-functions-2019.tar.xz")
    selection = select_cohort(archive, top_n=1, per_decile=0, seed=20260802)
    minute_series = load_archive_cohort(archive, selection)[0]
    series = aggregate_series(minute_series, 60, reducer="mean")
    boundaries = SplitBoundaries(train_end=10 * 24, validation_end=12 * 24, test_end=14 * 24)
    rows: list[EvaluationRow] = []
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
                regime="azure_top1_hourly",
                model=model.model_id,
                metrics=score_backtest(result, series, seasonal_period=24),
            )
        )
    print(markdown(rows))


if __name__ == "__main__":
    main()
