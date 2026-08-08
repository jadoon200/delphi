"""SQLAlchemy base, engine, and transactional session helpers."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Dialect, Engine, create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator

from delphi.config import get_settings

JsonType = JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True), "postgresql")


class UTCDateTime(TypeDecorator[datetime]):
    """A timestamp that is always stored and returned as an aware UTC value.

    SQLite has no native timezone type, so a plain ``DateTime(timezone=True)`` column
    silently returns *naive* datetimes there while Postgres returns aware ones. Every
    consumer in DELPHI (``DemandSeries``, ``DemandForecast``) rejects naive timestamps by
    design, so without this the test dialect and the production dialect disagree about
    whether the data is loadable at all — the worst kind of parity bug, because the suite
    stays green until the first real read.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetimes are never persisted; normalize to UTC first")
        return value.astimezone(UTC)

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        stored: datetime = value
        if stored.tzinfo is None or stored.utcoffset() is None:
            return stored.replace(tzinfo=UTC)
        return stored.astimezone(UTC)


class Base(DeclarativeBase):
    pass


def make_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings().database_url, pool_pre_ping=True)


def make_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or make_engine(), expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    session = (factory or make_session_factory())()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
