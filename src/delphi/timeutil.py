"""UTC-only time helpers.

Trace timestamps are unusually easy to mis-anchor: some sources use minute columns, some
relative seconds, and some local timestamps. Every boundary entering DELPHI must be explicit.
"""

from datetime import UTC, datetime, timedelta


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Return an aware UTC datetime, rejecting silent assumptions about naive values."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def floor_time(value: datetime, step_seconds: int) -> datetime:
    """Floor an aware timestamp to a positive, epoch-aligned interval."""
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    value = ensure_utc(value)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed = int((value - epoch).total_seconds())
    return epoch + timedelta(seconds=(elapsed // step_seconds) * step_seconds)
