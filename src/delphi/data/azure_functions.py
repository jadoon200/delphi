"""Streaming Azure Functions 2019 invocation parser and idempotent persistence."""

import csv
import io
import random
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TextIO

import numpy as np
from sqlalchemy.orm import Session

from delphi.data.series import DemandSeries, concatenate_series
from delphi.db.models import DemandPoint, DemandSource, WorkloadProfile

SOURCE_ID = "azure-functions-2019"
ARCHIVE_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/"
    "dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"
)
ARCHIVE_SHA256 = "aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89"
CITATION = (
    "Shahrad et al., Serverless in the Wild: Characterizing and Optimizing the Serverless "
    "Workload at a Large Cloud Provider, USENIX ATC 2020."
)
TRACE_START = date(2019, 7, 15)


@dataclass(frozen=True)
class CohortSelection:
    workload_ids: frozenset[str]
    top_n: int
    per_decile: int
    seed: int
    selection_day: int


def workload_id(row: dict[str, str]) -> str:
    app = row.get("HashApp", "").strip()
    function = row.get("HashFunction", "").strip()
    if not app or not function:
        raise ValueError("Azure row is missing HashApp or HashFunction")
    return f"azure-functions-2019:{app}:{function}"


def _minute_columns(fieldnames: Sequence[str] | None) -> list[str]:
    if fieldnames is None:
        raise ValueError("Azure invocation CSV has no header")
    minutes = sorted((name for name in fieldnames if name.isdigit()), key=int)
    if not minutes or [int(name) for name in minutes] != list(range(1, len(minutes) + 1)):
        raise ValueError("minute columns must be contiguous and one-indexed")
    return minutes


def parse_invocation_csv(
    stream: TextIO,
    *,
    day: int,
    selected: frozenset[str] | None = None,
) -> list[DemandSeries]:
    """Parse one wide day CSV into per-function regular demand series.

    A blank value is represented explicitly as an imputed zero with quality 0; it is never
    silently forward-filled or dropped.
    """
    if not 1 <= day <= 14:
        raise ValueError("Azure Functions day must be within 1..14")
    reader = csv.DictReader(stream)
    minutes = _minute_columns(reader.fieldnames)
    origin = datetime.combine(TRACE_START + timedelta(days=day - 1), datetime.min.time(), UTC)
    timestamps = tuple(origin + timedelta(minutes=int(name) - 1) for name in minutes)
    series: list[DemandSeries] = []
    for row in reader:
        identity = workload_id(row)
        if selected is not None and identity not in selected:
            continue
        raw = [row.get(name, "").strip() for name in minutes]
        imputed = np.asarray([not value for value in raw], dtype=np.bool_)
        try:
            values = np.asarray([float(value) if value else 0.0 for value in raw], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"non-numeric invocation value for {identity}") from exc
        series.append(
            DemandSeries(
                workload_id=identity,
                source_id=SOURCE_ID,
                resource_kind="invocations",
                unit="requests/minute",
                step_seconds=60,
                timestamps=timestamps,
                values=values,
                is_imputed=imputed,
                quality=np.where(imputed, 0.0, 1.0),
            )
        )
    return series


def _open_member(archive: tarfile.TarFile, day: int) -> TextIO:
    name = f"invocations_per_function_md.anon.d{day:02d}.csv"
    member = archive.extractfile(name)
    if member is None:
        raise FileNotFoundError(f"archive member missing: {name}")
    return io.TextIOWrapper(member, encoding="utf-8", newline="")


def select_cohort(
    archive_path: Path,
    *,
    top_n: int = 20,
    per_decile: int = 2,
    seed: int = 20260802,
    selection_day: int = 1,
) -> CohortSelection:
    """Select a recorded top-volume plus decile-stratified cohort from one early day."""
    if top_n < 0 or per_decile < 0 or top_n + per_decile == 0:
        raise ValueError("cohort sizes must be non-negative and not both zero")
    with (
        tarfile.open(archive_path, mode="r:xz") as archive,
        _open_member(archive, selection_day) as stream,
    ):
        reader = csv.DictReader(stream)
        minutes = _minute_columns(reader.fieldnames)
        totals = [
            (workload_id(row), sum(float(row.get(name, "") or 0.0) for name in minutes))
            for row in reader
        ]
    if not totals:
        raise ValueError("cohort source day contains no functions")
    ranked = sorted(totals, key=lambda item: (-item[1], item[0]))
    chosen = {identity for identity, _ in ranked[:top_n]}
    remainder = sorted(ranked[top_n:], key=lambda item: (item[1], item[0]))
    rng = random.Random(seed)
    for decile in range(10):
        left = decile * len(remainder) // 10
        right = (decile + 1) * len(remainder) // 10
        bucket = remainder[left:right]
        sample_size = min(per_decile, len(bucket))
        chosen.update(identity for identity, _ in rng.sample(bucket, sample_size))
    return CohortSelection(frozenset(chosen), top_n, per_decile, seed, selection_day)


def load_archive_cohort(archive_path: Path, selection: CohortSelection) -> list[DemandSeries]:
    """Load all 14 days for the selected cohort, explicitly flagging absent function-days."""
    parts: dict[str, list[DemandSeries]] = {identity: [] for identity in selection.workload_ids}
    with tarfile.open(archive_path, mode="r:xz") as archive:
        for day in range(1, 15):
            with _open_member(archive, day) as stream:
                found = {
                    series.workload_id: series
                    for series in parse_invocation_csv(
                        stream,
                        day=day,
                        selected=selection.workload_ids,
                    )
                }
            origin = datetime.combine(
                TRACE_START + timedelta(days=day - 1), datetime.min.time(), UTC
            )
            for identity in selection.workload_ids:
                part = found.get(identity)
                if part is None:
                    timestamps = tuple(origin + timedelta(minutes=index) for index in range(1440))
                    part = DemandSeries(
                        workload_id=identity,
                        source_id=SOURCE_ID,
                        resource_kind="invocations",
                        unit="requests/minute",
                        step_seconds=60,
                        timestamps=timestamps,
                        values=np.zeros(1440, dtype=np.float64),
                        is_imputed=np.ones(1440, dtype=np.bool_),
                        quality=np.zeros(1440, dtype=np.float64),
                    )
                parts[identity].append(part)
    return [concatenate_series(parts[identity]) for identity in sorted(parts)]


def azure_source(*, retrieved_at: datetime, sha256: str = ARCHIVE_SHA256) -> DemandSource:
    return DemandSource(
        source_id=SOURCE_ID,
        vendor="Microsoft Azure",
        licence="CC-BY-4.0",
        licence_verified_on=date(2026, 8, 3),
        citation=CITATION,
        url=ARCHIVE_URL,
        sha256=sha256,
        retrieved_at=retrieved_at,
        terms_note="Repo-wide CC-BY 4.0; attribution required; raw trace is not redistributed.",
        epoch_assumption=(
            "Day d01 is anchored to 2019-07-15 00:00 UTC; columns 1..1440 are minutes."
        ),
    )


def upsert_series(
    session: Session,
    *,
    source: DemandSource,
    series: list[DemandSeries],
    capacity_per_replica: float = 100.0,
    startup_seconds: float = 30.0,
) -> int:
    """Idempotently persist one source-scoped cohort. Returns points processed."""
    session.merge(source)
    rows = 0
    for item in series:
        session.merge(
            WorkloadProfile(
                workload_id=item.workload_id,
                source_id=item.source_id,
                resource_kind=item.resource_kind,
                unit=item.unit,
                capacity_per_replica=capacity_per_replica,
                startup_seconds=startup_seconds,
                teardown_seconds=10.0,
                min_replicas=0,
                max_replicas=10_000,
                scale_to_zero=True,
                tier="standard",
                deadline_slack_seconds=None,
            )
        )
        for index, timestamp in enumerate(item.timestamps):
            session.merge(
                DemandPoint(
                    workload_id=item.workload_id,
                    ts=timestamp,
                    value=float(item.values[index]),
                    is_imputed=bool(item.is_imputed[index]),
                    quality=float(item.quality[index]),
                )
            )
            rows += 1
    session.flush()
    return rows
