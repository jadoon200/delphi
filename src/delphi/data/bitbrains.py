"""Bitbrains GWA-T-12 — enterprise VM traces long enough to test weekly structure.

The Azure traces in this project are seven days long, which makes week-ahead forecasting
literally untestable: there is never a held-out week, and weekly seasonality cannot be
learned from a single instance of it. Bitbrains is the cheapest licence-usable trace that
fixes that — the ``rnd`` set covers **three consecutive months** (2013-06-30 to 2013-09-29,
~91 days) for the same 500 VMs at 5-minute resolution, in 284 MB.

Provenance caveat, carried in the source record rather than hidden: the canonical Grid
Workloads Archive host (``gwa.ewi.tudelft.nl``) has been unreachable since at least
2026-08-02, so the data comes from the @Large Research mirror and **the original terms-of-use
page could not be read**. Checksums are recorded so the exact bytes used are reproducible,
and the data is never redistributed.

Fleet-level aggregation is the point here. A capacity-planning question is "how much CPU
does the estate need next week", not "what will VM 417 do", so per-VM series are summed
into one demand signal.
"""

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt

from delphi.data.series import DemandSeries

FloatArray = npt.NDArray[np.float64]

SOURCE_ID = "bitbrains-gwa-t-12"
LICENCE = "unverified-host-unreachable"
CITATION = (
    "Siqi Shen, Vincent van Beek, Alexandru Iosup. Statistical Characterization of "
    "Business-Critical Workloads Hosted in Cloud Datacenters, CCGrid 2015. "
    "Grid Workloads Archive GWA-T-12."
)
MIRROR_BASE = "https://atlarge-research.com/gwa-traces/"
ARCHIVES: dict[str, str] = {
    "fastStorage": "gwa_t_12_fastStorage.zip",
    "rnd": "gwa_t_12_rnd.zip",
}
SHA256: dict[str, str] = {
    "fastStorage": "11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671",
    "rnd": "d3d9ddebb689c0b5463f2e4cfd8956e84bdcdf138b4476320855393e2b229a06",
}
#: `rnd` ships as three consecutive month directories for the same VM population.
RND_MONTHS: tuple[str, ...] = ("2013-7", "2013-8", "2013-9")

#: Column index of CPU usage in MHz, after the semicolon split.
CPU_USAGE_MHZ = 3
TIMESTAMP = 0


def _rows(path: Path) -> Iterator[tuple[int, float]]:
    """Yield ``(unix_seconds, cpu_mhz)``. Malformed lines are skipped, not guessed at."""
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        next(reader, None)  # header
        for row in reader:
            if len(row) <= CPU_USAGE_MHZ:
                continue
            try:
                timestamp = int(float(row[TIMESTAMP].strip()))
                usage = float(row[CPU_USAGE_MHZ].strip())
            except ValueError:
                continue
            if usage < 0:
                continue
            yield timestamp, usage


def vm_files(root: Path, trace: str) -> list[Path]:
    """All per-VM CSVs for a trace, in a deterministic order."""
    if trace == "fastStorage":
        return sorted((root / "fastStorage").rglob("*.csv"))
    return sorted(
        (path for month in RND_MONTHS for path in (root / "rnd" / month).glob("*.csv")),
        key=lambda p: (p.parent.name, int(p.stem)),
    )


@dataclass(frozen=True)
class FleetDemand:
    """Fleet-wide CPU demand, binned and aligned to an absolute UTC grid."""

    start: datetime
    bin_seconds: int
    cpu_mhz: FloatArray
    vm_count: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.cpu_mhz.shape != self.vm_count.shape:
            raise ValueError("fleet arrays must be parallel")
        if self.start.tzinfo is None or self.start.utcoffset() != timedelta(0):
            raise ValueError("fleet start must be UTC")

    def __len__(self) -> int:
        return len(self.cpu_mhz)

    @property
    def days(self) -> float:
        return len(self) * self.bin_seconds / 86400.0


def aggregate_fleet(root: Path, trace: str, *, bin_seconds: int = 300) -> FleetDemand:
    """Sum CPU demand across every VM onto one absolute time grid.

    Per-VM timestamps do not align exactly, so each reading is dropped into a fixed bin.
    ``vm_count`` travels alongside because a bin covering fewer VMs is a *different
    question*, not simply a quieter one — the estate changes size as VMs are created and
    deleted, and a demand drop caused by that must stay distinguishable.
    """
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")
    files = vm_files(root, trace)
    if not files:
        raise ValueError(f"no VM files found for {trace!r} under {root}")

    lowest = highest = None
    for path in files:
        for timestamp, _ in _rows(path):
            lowest = timestamp if lowest is None else min(lowest, timestamp)
            highest = timestamp if highest is None else max(highest, timestamp)
    if lowest is None or highest is None:
        raise ValueError(f"no parseable rows for {trace!r}")

    origin = lowest - (lowest % bin_seconds)
    bins = (highest - origin) // bin_seconds + 1
    total = np.zeros(bins, dtype=np.float64)
    seen = np.zeros(bins, dtype=np.int64)
    for path in files:
        for timestamp, usage in _rows(path):
            index = (timestamp - origin) // bin_seconds
            total[index] += usage
            seen[index] += 1

    return FleetDemand(
        start=datetime.fromtimestamp(origin, tz=UTC),
        bin_seconds=bin_seconds,
        cpu_mhz=total,
        vm_count=seen,
    )


def cache_path(trace: str, bin_seconds: int, directory: Path) -> Path:
    return directory / f"bitbrains-{trace}-{bin_seconds}s.npz"


def load_fleet(
    trace: str = "rnd",
    *,
    root: Path = Path("data/raw/bitbrains"),
    cache_dir: Path = Path("data/cache"),
    bin_seconds: int = 300,
    refresh: bool = False,
) -> FleetDemand:
    """Load fleet demand, parsing the 1,500 per-VM CSVs once and caching the result."""
    if trace not in ARCHIVES:
        raise ValueError(f"unknown Bitbrains trace {trace!r}")
    cache = cache_path(trace, bin_seconds, cache_dir)
    if cache.exists() and not refresh:
        payload = np.load(cache, allow_pickle=False)
        return FleetDemand(
            start=datetime.fromtimestamp(float(payload["start"]), tz=UTC),
            bin_seconds=int(payload["bin_seconds"]),
            cpu_mhz=payload["cpu_mhz"],
            vm_count=payload["vm_count"],
        )
    fleet = aggregate_fleet(root, trace, bin_seconds=bin_seconds)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        start=fleet.start.timestamp(),
        bin_seconds=fleet.bin_seconds,
        cpu_mhz=fleet.cpu_mhz,
        vm_count=fleet.vm_count,
    )
    return fleet


def to_demand_series(fleet: FleetDemand, *, trace: str) -> DemandSeries:
    """Project fleet CPU demand into the canonical contract.

    Bins whose VM count collapses are marked ``is_imputed`` with zero quality rather than
    silently treated as low demand: an estate that shrank is not an estate that idled, and
    the evaluation must be able to exclude those windows.
    """
    counts = fleet.vm_count
    typical = float(np.median(counts[counts > 0])) if (counts > 0).any() else 0.0
    sparse = counts < max(typical * 0.5, 1.0)
    return DemandSeries(
        workload_id=f"{SOURCE_ID}:{trace}:fleet_cpu",
        source_id=SOURCE_ID,
        resource_kind="cpu",
        unit="MHz",
        step_seconds=fleet.bin_seconds,
        timestamps=tuple(
            fleet.start + timedelta(seconds=fleet.bin_seconds * index)
            for index in range(len(fleet))
        ),
        values=np.maximum(fleet.cpu_mhz, 0.0),
        is_imputed=sparse,
        quality=np.where(sparse, 0.0, 1.0),
    )
