"""SQLAlchemy base, engine, and transactional session helpers."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import JSON, Engine, create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from delphi.config import get_settings

JsonType = JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True), "postgresql")


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
