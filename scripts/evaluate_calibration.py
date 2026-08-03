"""Measure raw, static conformal, and ACI p95 coverage through a level shift."""

from datetime import UTC, datetime

import numpy as np

from delphi.data.synthetic import generate_synthetic
from delphi.evaluation.coverage import coverage_recovery_steps
from delphi.evaluation.rolling_origin import (
    SplitBoundaries,
    rolling_origin_backtest,
)
from delphi.forecast.baselines import SeasonalNaiveForecaster
from delphi.forecast.calibration import (
    AdaptiveConformalCalibrator,
    SplitConformalCalibrator,
    tune_aci_gamma,
)


def main() -> None:
    series = generate_synthetic(
        "level_shift",
        periods=30 * 24,
        step_seconds=3600,
        seed=20260802,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    boundaries = SplitBoundaries(train_end=14 * 24, validation_end=18 * 24, test_end=30 * 24)
    model = SeasonalNaiveForecaster(24, name="seasonal_naive_d")
    levels = (0.5, 0.95)
    validation = rolling_origin_backtest(
        series,
        model,
        boundaries,
        horizon_steps=1,
        stride_steps=1,
        quantile_levels=levels,
        split="validation",
    )
    test = rolling_origin_backtest(
        series,
        model,
        boundaries,
        horizon_steps=1,
        stride_steps=1,
        quantile_levels=levels,
    )
    validation_actual, validation_predicted = validation.arrays()
    test_actual, test_predicted = test.arrays()
    p95_index = levels.index(0.95)
    validation_scores = validation_actual - validation_predicted[:, p95_index]
    midpoint = len(validation_scores) // 2
    gamma = tune_aci_gamma(
        initial_scores=validation_scores[:midpoint],
        timestamps=tuple(record.forecast.timestamps[0] for record in validation.records[midpoint:]),
        raw_predictions=validation_predicted[midpoint:, p95_index],
        actual=validation_actual[midpoint:],
        candidates=(0.001, 0.005, 0.01, 0.02, 0.05),
        target_coverage=0.95,
        score_window=72,
    )

    static = SplitConformalCalibrator.fit(validation_actual, validation_predicted, levels)
    static_predictions = np.asarray(
        [
            static.calibrate(record.forecast).quantile_values[0, p95_index]
            for record in test.records
        ],
        dtype=np.float64,
    )
    adaptive = AdaptiveConformalCalibrator(
        validation_scores,
        target_coverage=0.95,
        gamma=gamma.gamma,
        score_window=72,
    )
    adaptive_points = tuple(
        adaptive.observe(
            timestamp=record.forecast.timestamps[0],
            raw_prediction=float(raw),
            actual=float(actual),
        )
        for record, raw, actual in zip(
            test.records,
            test_predicted[:, p95_index],
            test_actual,
            strict=True,
        )
    )
    methods = (
        ("raw", test_predicted[:, p95_index], tuple(test_actual <= test_predicted[:, p95_index])),
        ("split_conformal", static_predictions, tuple(test_actual <= static_predictions)),
        (
            f"aci_gamma_{gamma.gamma:g}",
            np.asarray(
                [point.calibrated_prediction for point in adaptive_points], dtype=np.float64
            ),
            tuple(point.covered for point in adaptive_points),
        ),
    )
    print("| Method | p95 coverage | Mean p95 | Recovery steps (24 h window) |")
    print("|---|---:|---:|---:|")
    for name, predictions, coverage in methods:
        recovery = coverage_recovery_steps(coverage, target=0.95, window=24, tolerance=0.05)
        label = "not recovered" if recovery is None else str(recovery)
        print(f"| `{name}` | {np.mean(coverage):.3f} | {np.mean(predictions):.3f} | {label} |")


if __name__ == "__main__":
    main()
