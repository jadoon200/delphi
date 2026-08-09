"""Measure raw, fixed-margin, static conformal, and ACI p95 coverage through a level shift.

This answers **Q2** — *does conformal calibration beat a fixed safety margin?* The fixed
margin is the practitioner's alternative and the honest comparator: multiply the raw
forecast by ``1 + m`` and pick ``m`` as the smallest value hitting nominal coverage **on
validation only**, the same budget and the same discipline ACI's gamma gets. Comparing
conformal against an uncalibrated forecast alone would have been a strawman, and for some
time that is all this experiment did.
"""

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
    # --- Q2's comparator: a fixed safety margin, tuned on validation only ---------------
    validation_raw = validation_predicted[:, p95_index]
    margin = 1.0
    for candidate in np.linspace(0.0, 1.0, 101):
        if float(np.mean(validation_actual <= validation_raw * (1.0 + candidate))) >= 0.95:
            margin = float(candidate)
            break
    fixed_margin_predictions = test_predicted[:, p95_index] * (1.0 + margin)
    # The conventional margin as well as the tuned one. Kubernetes VPA ships a headroom
    # multiplier and this repo's PercentileRecommender defaults to 0.15, so a comparator
    # that only reports the tuned value would miss what an operator actually runs.
    conventional = 0.15
    conventional_predictions = test_predicted[:, p95_index] * (1.0 + conventional)
    validation_raw_coverage = float(np.mean(validation_actual <= validation_raw))
    print(
        f"Raw p95 already covers {validation_raw_coverage:.1%} on validation, so the "
        f"smallest margin reaching the 95% target is **+{margin:.0%}** — a margin tuned "
        f"honestly on stationary data adds no headroom at all. The conventional "
        f"+{conventional:.0%} is reported alongside it.\n"
    )

    methods = (
        ("raw", test_predicted[:, p95_index], tuple(test_actual <= test_predicted[:, p95_index])),
        (
            f"fixed_margin_tuned_{margin:+.0%}",
            fixed_margin_predictions,
            tuple(test_actual <= fixed_margin_predictions),
        ),
        (
            f"fixed_margin_conventional_{conventional:+.0%}",
            conventional_predictions,
            tuple(test_actual <= conventional_predictions),
        ),
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
