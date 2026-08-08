"""Fail CI when the hand-written migration and ORM schema drift apart."""

import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Engine, create_engine, inspect

from delphi.db import models  # noqa: F401
from delphi.db.base import Base

_VERSIONS_DIR = Path(__file__).resolve().parent.parent / "migrations" / "versions"


def _schema(engine: Engine) -> dict[str, dict[str, tuple[str, bool]]]:
    """Capture column type *and* nullability.

    Comparing names and nullability alone lets a type drift through — a migration column
    declared TEXT against an ORM column declared TIMESTAMP would pass silently, and the
    first real deployment would be the thing that noticed.
    """
    inspector = inspect(engine)
    return {
        table: {
            column["name"]: (str(column["type"]), bool(column["nullable"]))
            for column in inspector.get_columns(table)
        }
        for table in inspector.get_table_names()
        if table != "alembic_version"
    }


def _run_migrations(url: str) -> None:
    engine = create_engine(url)
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            paths = sorted(_VERSIONS_DIR.glob("[0-9]*.py"))
            assert paths, "no migrations found"
            for path in paths:
                spec = importlib.util.spec_from_file_location(f"migration_{path.stem}", path)
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.upgrade()
        connection.commit()


def test_migration_schema_matches_models(tmp_path: Path) -> None:
    migration_url = f"sqlite:///{tmp_path / 'migration.db'}"
    _run_migrations(migration_url)
    migration_schema = _schema(create_engine(migration_url))

    model_url = f"sqlite:///{tmp_path / 'models.db'}"
    Base.metadata.create_all(create_engine(model_url))
    model_schema = _schema(create_engine(model_url))

    assert migration_schema == model_schema
