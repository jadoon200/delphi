"""Materna GWA-T-13 — three more month-long enterprise fleets.

The diagnostic claim of this project is that a workload's **daily autocorrelation** predicts
whether forecasting will pay for it. That claim was resting on three workloads, which is
too thin to publish. Materna adds three independent one-month fleets from a different
provider, taking the sample to seven.

| trace | VMs | period | span |
|---|---:|---|---:|
| Materna-Trace-1 | 519 | 2015-11-05 → 2015-12-03 | 29 days |
| Materna-Trace-2 | 527 | 2015-12-04 → 2016-01-03 | 31 days |
| Materna-Trace-3 | 547 | 2016-01-04 → 2016-02-08 | 36 days |

Trace-2 spans Christmas and New Year, which makes it a natural level-shift case rather than
a synthetic one.

**Format traps, handled explicitly.** Unlike Bitbrains, Materna is written in a German
locale: timestamps are ``DD.MM.YYYY HH:MM:SS`` and decimals use a **comma** (``3,4`` means
3.4). Parsing this with a naive ``float()`` silently truncates every fractional value to the
integer part, which would quietly deflate every demand figure. The parser rejects rather
than guesses.

The three traces are consecutive in time but cover *different VM populations* (519 / 527 /
547), so they are treated as three separate fleets rather than concatenated into one series.
"""

import csv
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from delphi.data.bitbrains import FleetDemand
from delphi.data.series import DemandSeries

SOURCE_ID = "materna-gwa-t-13"
LICENCE = "unverified-host-unreachable"
CITATION = (
    "Andreas Kohne, Marc Spohr, Lars Nagel, Olaf Spinczyk. FederatedCloudSim: A SLA-aware "
    "federated cloud simulation framework, CCB 2014. Grid Workloads Archive GWA-T-13."
)
MIRROR_URL = "https://atlarge-research.com/gwa-traces/gwa_t_13_materna.zip"
SHA256 = "1380879f0de17cb57619e55c312b41f26ef95743a382f41692db72168fd9afb4"
ROOT_DIR = "GWA-T-13_Materna-Workload-Traces"
TRACES: tuple[str, ...] = ("Materna-Trace-1", "Materna-Trace-2", "Materna-Trace-3")

TIMESTAMP = 0
CPU_USAGE_MHZ = 3
_TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M:%S"


def parse_german_decimal(raw: str) -> float:
    """Parse ``'3,4'`` as 3.4.

    A bare ``float('3,4')`` raises, but ``float('3')`` after a naive split would silently
    drop the fraction. Both failure modes deflate demand, so the conversion is explicit and
    a value that is neither form is rejected.
    """
    text = raw.strip().strip('"')
    if not text:
        raise ValueError("empty numeric field")
    if "," in text:
        if "." in text:
            raise ValueError(f"ambiguous mixed separators: {raw!r}")
        text = text.replace(",", ".")
    return float(text)


def _rows(path: Path) -> Iterator[tuple[int, float]]:
    """Yield ``(unix_seconds, cpu_mhz)``; malformed lines are skipped, never guessed."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=";", quotechar='"')
        next(reader, None)  # header
        for row in reader:
            if len(row) <= CPU_USAGE_MHZ:
                continue
            try:
                stamp = datetime.strptime(
                    row[TIMESTAMP].strip().strip('"'), _TIMESTAMP_FORMAT
                ).replace(tzinfo=UTC)
                usage = parse_german_decimal(row[CPU_USAGE_MHZ])
            except ValueError:
                continue
            if usage < 0:
                continue
            yield int(stamp.timestamp()), usage


def vm_files(root: Path, trace: str) -> list[Path]:
    if trace not in TRACES:
        raise ValueError(f"unknown Materna trace {trace!r}; expected one of {list(TRACES)}")
    return sorted((root / ROOT_DIR / trace).glob("*.csv"))


def aggregate_fleet(root: Path, trace: str, *, bin_seconds: int = 300) -> FleetDemand:
    """Sum CPU demand across every VM in one trace onto an absolute time grid."""
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")
    files = vm_files(root, trace)
    if not files:
        raise ValueError(f"no VM files found for {trace!r} under {root}")

    lowest = highest = None
    for path in files:
        for stamp, _ in _rows(path):
            lowest = stamp if lowest is None else min(lowest, stamp)
            highest = stamp if highest is None else max(highest, stamp)
    if lowest is None or highest is None:
        raise ValueError(f"no parseable rows for {trace!r}")

    origin = lowest - (lowest % bin_seconds)
    bins = (highest - origin) // bin_seconds + 1
    total = np.zeros(bins, dtype=np.float64)
    seen = np.zeros(bins, dtype=np.int64)
    for path in files:
        for stamp, usage in _rows(path):
            index = (stamp - origin) // bin_seconds
            total[index] += usage
            seen[index] += 1

    return FleetDemand(
        start=datetime.fromtimestamp(origin, tz=UTC),
        bin_seconds=bin_seconds,
        cpu_mhz=total,
        vm_count=seen,
    )


def cache_path(trace: str, bin_seconds: int, directory: Path) -> Path:
    return directory / f"materna-{trace}-{bin_seconds}s.npz"


def load_fleet(
    trace: str = "Materna-Trace-1",
    *,
    root: Path = Path("data/raw/materna"),
    cache_dir: Path = Path("data/cache"),
    bin_seconds: int = 300,
    refresh: bool = False,
) -> FleetDemand:
    """Load fleet demand for one Materna trace, caching the parsed aggregate."""
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
    """Project fleet CPU demand into the canonical contract, flagging sparse bins."""
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
