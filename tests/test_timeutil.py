from datetime import UTC, datetime, timedelta, timezone

import pytest

from delphi.timeutil import ensure_utc, floor_time, utc_now


def test_utc_now_is_aware() -> None:
    assert utc_now().tzinfo is UTC


def test_ensure_utc_rejects_naive_and_normalises_offsets() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ensure_utc(datetime(2026, 8, 3, 12, 0))
    local = datetime(2026, 8, 3, 20, 0, tzinfo=timezone(timedelta(hours=8)))
    assert ensure_utc(local) == datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def test_floor_time_is_epoch_aligned() -> None:
    value = datetime(2026, 8, 3, 12, 4, 59, tzinfo=UTC)
    assert floor_time(value, 300) == datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="positive"):
        floor_time(value, 0)
