from datetime import UTC, datetime

import numpy as np
import pytest

from delphi.data.synthetic import generate_synthetic
from delphi.forecast.features import (
    CausalFeatureMatrix,
    FeatureSpec,
    assert_causal_features,
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


def test_feature_fit_cannot_cross_split_boundary() -> None:
    matrix = CausalFeatureMatrix(
        values=np.asarray([[1.0]], dtype=np.float64),
        targets=np.asarray([2.0], dtype=np.float64),
        target_indices=np.asarray([10], dtype=np.int64),
        feature_as_of_indices=np.asarray([9], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="split boundary"):
        assert_causal_features(matrix, fit_end_exclusive=10)
