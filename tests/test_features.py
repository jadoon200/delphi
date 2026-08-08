from datetime import UTC, datetime

import numpy as np
import pytest

from delphi.data.synthetic import generate_synthetic
from delphi.forecast import features
from delphi.forecast.features import (
    CausalFeatureMatrix,
    FeatureSpec,
    assert_causal_features,
    assert_no_target_leak,
    build_causal_training_matrix,
)


def test_training_features_stop_before_each_target() -> None:
    series = generate_synthetic(
        "clean_daily",
        periods=120,
        step_seconds=3600,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    matrix = build_causal_training_matrix(
        series,
        FeatureSpec(lags=(1, 2, 12), windows=(3, 12)),
        fit_end_exclusive=100,
    )
    assert np.all(matrix.feature_as_of_indices == matrix.target_indices - 1)
    assert int(matrix.target_indices.max()) == 99


def test_deliberate_target_leak_is_rejected() -> None:
    leaked = CausalFeatureMatrix(
        values=np.asarray([[123.0]], dtype=np.float64),
        targets=np.asarray([123.0], dtype=np.float64),
        target_indices=np.asarray([10], dtype=np.int64),
        feature_as_of_indices=np.asarray([10], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="observes its own target"):
        assert_causal_features(leaked, fit_end_exclusive=11)


def test_leak_injected_through_the_real_builder_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must catch a leak in the feature *values*, not just in the bookkeeping.

    A builder that closes over the full series still reports honest provenance indices, so
    the index check passes unaltered. This is the exact failure the leakage discipline
    exists to prevent, and it is only detectable by inspecting the realised matrix.
    """
    series = generate_synthetic(
        "clean_daily",
        periods=200,
        step_seconds=3600,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    spec = FeatureSpec(lags=(1, 2, 12), windows=(3, 12))
    honest = features.causal_feature_row

    def leaking_row(values: np.ndarray, timestamp: datetime, spec: FeatureSpec) -> np.ndarray:
        row = honest(values, timestamp, spec)
        index = len(values)
        peek = float(series.values[index]) if index < len(series) else 0.0
        return np.append(row, peek)

    monkeypatch.setattr(features, "causal_feature_row", leaking_row)
    with pytest.raises(ValueError, match="equals its own target"):
        features.build_causal_training_matrix(series, spec, fit_end_exclusive=150)


def test_near_perfect_correlation_is_also_treated_as_a_leak() -> None:
    targets = np.linspace(10.0, 90.0, 60)
    matrix = CausalFeatureMatrix(
        values=np.column_stack([targets * 1.0001 + 0.5, np.zeros(60)]),
        targets=targets,
        target_indices=np.arange(40, 100, dtype=np.int64),
        feature_as_of_indices=np.arange(39, 99, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="correlates"):
        assert_no_target_leak(matrix)


def test_legitimate_causal_features_are_not_flagged() -> None:
    series = generate_synthetic(
        "burst_known",
        periods=400,
        step_seconds=3600,
        start=datetime(2026, 1, 1, tzinfo=UTC),
    )
    matrix = build_causal_training_matrix(
        series,
        FeatureSpec(lags=(1, 2, 24), windows=(3, 24)),
        fit_end_exclusive=300,
    )
    assert_no_target_leak(matrix)


def test_feature_fit_cannot_cross_split_boundary() -> None:
    matrix = CausalFeatureMatrix(
        values=np.asarray([[1.0]], dtype=np.float64),
        targets=np.asarray([2.0], dtype=np.float64),
        target_indices=np.asarray([10], dtype=np.int64),
        feature_as_of_indices=np.asarray([9], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="split boundary"):
        assert_causal_features(matrix, fit_end_exclusive=10)
