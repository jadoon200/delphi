from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from delphi.data.series import DemandSeries, SeriesAnnotation, concatenate_series


def _series(start: datetime, values: list[float]) -> DemandSeries:
    return DemandSeries(
        workload_id="w",
        source_id="s",
        resource_kind="invocations",
        unit="requests/minute",
        step_seconds=60,
        timestamps=tuple(start + timedelta(minutes=index) for index in range(len(values))),
        values=np.asarray(values, dtype=np.float64),
        is_imputed=np.zeros(len(values), dtype=np.bool_),
        quality=np.ones(len(values), dtype=np.float64),
        annotations=(SeriesAnnotation("event", 1, len(values)),),
    )


def test_series_is_regular_utc_and_read_only() -> None:
    series = _series(datetime(2026, 1, 1, tzinfo=UTC), [1, 2, 3])
    assert len(series) == 3 and not series.values.flags.writeable
    with pytest.raises(ValueError):
        series.values[0] = 9


def test_series_rejects_irregular_or_naive_time() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="strictly regular"):
        DemandSeries(
            workload_id="w",
            source_id="s",
            resource_kind="cpu",
            unit="percent",
            step_seconds=60,
            timestamps=(base, base + timedelta(seconds=61)),
            values=np.asarray([1, 2], dtype=np.float64),
            is_imputed=np.zeros(2, dtype=np.bool_),
            quality=np.ones(2, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        _series(datetime(2026, 1, 1), [1, 2])
    with pytest.raises(ValueError, match="normalized to UTC"):
        _series(datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=8))), [1, 2])


def test_slice_clips_annotations_and_concatenate_restores_coordinates() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    whole = _series(base, [1, 2, 3, 4])
    left, right = whole.slice(0, 2), whole.slice(2, 4)
    joined = concatenate_series([left, right])
    assert joined.values.tolist() == whole.values.tolist()
    assert joined.annotations == (
        SeriesAnnotation("event", 1, 2),
        SeriesAnnotation("event", 2, 4),
    )
