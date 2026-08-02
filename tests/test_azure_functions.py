import csv
import io
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from delphi.data.azure_functions import (
    CohortSelection,
    azure_source,
    load_archive_cohort,
    parse_invocation_csv,
    select_cohort,
    upsert_series,
)
from delphi.db.base import Base
from delphi.db.models import DemandPoint, WorkloadProfile

FIXTURE = Path(__file__).parent / "fixtures" / "azure_invocations_d01.csv"


def test_wide_azure_csv_parses_to_regular_series_and_flags_missing_values() -> None:
    with FIXTURE.open() as stream:
        series = parse_invocation_csv(stream, day=1)
    assert len(series) == 3
    timer = next(item for item in series if item.workload_id.endswith("function-b"))
    assert timer.values.tolist() == [5.0, 4.0, 0.0, 2.0, 1.0]
    assert timer.is_imputed.tolist() == [False, False, True, False, False]
    assert timer.quality.tolist() == [1.0, 1.0, 0.0, 1.0, 1.0]
    assert timer.timestamps[0] == datetime(2019, 7, 15, tzinfo=UTC)


def test_selected_workload_filter_is_exact() -> None:
    with FIXTURE.open() as stream:
        all_series = parse_invocation_csv(stream, day=1)
    selected = frozenset({all_series[0].workload_id})
    with FIXTURE.open() as stream:
        filtered = parse_invocation_csv(stream, day=1, selected=selected)
    assert [item.workload_id for item in filtered] == [all_series[0].workload_id]


def test_upsert_is_idempotent() -> None:
    with FIXTURE.open() as stream:
        series = parse_invocation_csv(stream, day=1)[:1]
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    source = azure_source(retrieved_at=datetime(2026, 8, 3, tzinfo=UTC))
    with Session(engine) as session:
        assert upsert_series(session, source=source, series=series) == 5
        session.commit()
        assert upsert_series(session, source=source, series=series) == 5
        session.commit()
        assert session.scalar(select(func.count()).select_from(DemandPoint)) == 5
        assert session.scalar(select(func.count()).select_from(WorkloadProfile)) == 1


def _write_archive(path: Path) -> None:
    fieldnames = ["HashApp", "HashFunction", *[str(index) for index in range(1, 1441)]]
    with tarfile.open(path, mode="w:xz") as archive:
        for day in range(1, 15):
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            if day != 2:
                writer.writerow(
                    {
                        "HashApp": "app-a",
                        "HashFunction": "function-a",
                        **{str(index): str(day + index) for index in range(1, 1441)},
                    }
                )
            payload = stream.getvalue().encode()
            member = tarfile.TarInfo(f"invocations_per_function_md.anon.d{day:02d}.csv")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_archive_cohort_is_reproducible_and_missing_days_are_explicit(tmp_path: Path) -> None:
    archive_path = tmp_path / "azure.tar.xz"
    _write_archive(archive_path)
    first = select_cohort(archive_path, top_n=1, per_decile=0, seed=7)
    second = select_cohort(archive_path, top_n=1, per_decile=0, seed=7)
    assert first == second
    assert first == CohortSelection(
        frozenset({"azure-functions-2019:app-a:function-a"}), 1, 0, 7, 1
    )

    series = load_archive_cohort(archive_path, first)
    assert len(series) == 1
    assert len(series[0]) == 14 * 1440
    assert np.all(series[0].is_imputed[1440:2880])
    assert np.all(series[0].quality[1440:2880] == 0)
