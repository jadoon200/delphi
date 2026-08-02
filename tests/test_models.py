from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

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
