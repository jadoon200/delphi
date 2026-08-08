from datetime import UTC, date, datetime

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from delphi.data.series import DemandSeries
from delphi.db.base import Base
from delphi.db.models import DemandPoint, DemandSource, WorkloadProfile


def _source() -> DemandSource:
    return DemandSource(
        source_id="synthetic-gold",
        vendor="DELPHI",
        licence="MIT",
        licence_verified_on=date(2026, 8, 3),
        citation=None,
        url="https://github.com/jadoon200/delphi",
        sha256=None,
        retrieved_at=datetime(2026, 8, 3, tzinfo=UTC),
        terms_note="Generated locally; no external data.",
        epoch_assumption="Absolute UTC timestamps generated from a declared origin.",
    )


def _profile() -> WorkloadProfile:
    return WorkloadProfile(
        workload_id="synthetic:clean_daily",
        source_id="synthetic-gold",
        resource_kind="invocations",
        unit="requests/minute",
        capacity_per_replica=100.0,
        startup_seconds=30.0,
        teardown_seconds=10.0,
        min_replicas=0,
        max_replicas=100,
        scale_to_zero=True,
        tier="standard",
        deadline_slack_seconds=None,
    )


def test_canonical_demand_round_trip() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    point = DemandPoint(
        workload_id="synthetic:clean_daily",
        ts=datetime(2026, 8, 3, tzinfo=UTC),
        value=42.0,
        is_imputed=False,
        quality=1.0,
    )
    with Session(engine) as session:
        session.add_all([_source(), _profile(), point])
        session.commit()
        stored = session.get(DemandPoint, (point.workload_id, point.ts))
        assert stored is not None
        assert stored.value == 42.0 and stored.quality == 1.0


def test_demand_quality_constraint_rejects_false_precision() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                _source(),
                _profile(),
                DemandPoint(
                    workload_id="synthetic:clean_daily",
                    ts=datetime(2026, 8, 3, tzinfo=UTC),
                    value=1.0,
                    is_imputed=True,
                    quality=1.2,
                ),
            ]
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_timestamps_round_trip_as_aware_utc_on_sqlite() -> None:
    """SQLite has no timezone type; a naive read would break every DemandSeries load.

    Postgres returns aware datetimes and SQLite does not, so without a coercing column
    type the two dialects disagree about whether persisted data is loadable at all — and
    the test dialect is the one that lies.
    """
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    stored = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
    with Session(engine) as session:
        session.add(
            DemandSource(
                source_id="s",
                vendor="v",
                licence="CC-BY-4.0",
                licence_verified_on=date(2026, 8, 3),
                citation=None,
                url="https://example.invalid",
                sha256=None,
                retrieved_at=stored,
                terms_note="",
                epoch_assumption="test",
            )
        )
        session.add(
            WorkloadProfile(
                workload_id="w",
                source_id="s",
                resource_kind="invocations",
                unit="requests/minute",
                capacity_per_replica=1.0,
                startup_seconds=0.0,
                teardown_seconds=0.0,
                min_replicas=0,
                max_replicas=1,
                scale_to_zero=True,
                tier="standard",
                deadline_slack_seconds=None,
            )
        )
        session.add(DemandPoint(workload_id="w", ts=stored, value=1.0))
        session.commit()

    with Session(engine) as session:
        point = session.scalars(select(DemandPoint)).one()
        assert point.ts.tzinfo is not None
        assert point.ts == stored
        # the reloaded timestamp must satisfy the series contract, not merely exist
        DemandSeries(
            workload_id="w",
            source_id="s",
            resource_kind="invocations",
            unit="requests/minute",
            step_seconds=60,
            timestamps=(point.ts,),
            values=np.asarray([1.0]),
            is_imputed=np.asarray([False]),
            quality=np.asarray([1.0]),
        )


def test_naive_timestamps_are_refused_at_the_boundary() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session, pytest.raises((ValueError, StatementError)):
        session.add(DemandPoint(workload_id="w", ts=datetime(2026, 1, 1, 12, 30), value=1.0))
        session.flush()
