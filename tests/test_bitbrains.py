"""Bitbrains fleet aggregation — the only trace long enough to test weekly structure."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from delphi.data.bitbrains import (
    FleetDemand,
    aggregate_fleet,
    to_demand_series,
    vm_files,
)

HEADER = (
    "Timestamp [ms];\tCPU cores;\tCPU capacity provisioned [MHZ];\tCPU usage [MHZ];\t"
    "CPU usage [%];\tMemory capacity provisioned [KB];\tMemory usage [KB];\t"
    "Disk read throughput [KB/s];\tDisk write throughput [KB/s];\t"
    "Network received throughput [KB/s];\tNetwork transmitted throughput [KB/s]\n"
)


def _vm(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{t};\t4;\t1000.0;\t{cpu};\t50.0;\t1;\t1;\t0;\t0;\t0;\t0\n" for t, cpu in rows)
    path.write_text(HEADER + body)


def test_fleet_sums_concurrent_vms_into_one_signal(tmp_path: Path) -> None:
    _vm(tmp_path / "fastStorage" / "2013-8" / "1.csv", [(0, 100.0), (300, 200.0)])
    _vm(tmp_path / "fastStorage" / "2013-8" / "2.csv", [(0, 50.0), (300, 25.0)])
    fleet = aggregate_fleet(tmp_path, "fastStorage", bin_seconds=300)
    assert fleet.cpu_mhz.tolist() == [150.0, 225.0]
    assert fleet.vm_count.tolist() == [2, 2]


def test_rnd_months_are_concatenated_in_chronological_order(tmp_path: Path) -> None:
    """The three month directories are one continuous 13-week series, not three traces."""
    for month, base in (("2013-7", 0), ("2013-8", 3000), ("2013-9", 6000)):
        _vm(tmp_path / "rnd" / month / "1.csv", [(base, 10.0)])
    files = vm_files(tmp_path, "rnd")
    assert [p.parent.name for p in files] == ["2013-7", "2013-8", "2013-9"]
    fleet = aggregate_fleet(tmp_path, "rnd", bin_seconds=300)
    assert fleet.cpu_mhz[0] == 10.0
    assert fleet.cpu_mhz[-1] == 10.0
    assert len(fleet) == 6000 // 300 + 1


def test_malformed_rows_are_skipped_not_guessed(tmp_path: Path) -> None:
    path = tmp_path / "fastStorage" / "2013-8" / "1.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        HEADER + "0;\t4;\t1000;\t100.0;\t50;\t1;\t1;\t0;\t0;\t0;\t0\n"
        "bad;\tx\n300;\t4;\t1000;\tNaNish;\t50;\t1;\t1;\t0;\t0;\t0;\t0\n"
    )
    fleet = aggregate_fleet(tmp_path, "fastStorage", bin_seconds=300)
    assert fleet.cpu_mhz[0] == 100.0
    assert fleet.vm_count[0] == 1


def test_a_shrinking_estate_is_flagged_not_read_as_idle(tmp_path: Path) -> None:
    """A demand drop caused by fewer VMs is a different question from a quieter fleet."""
    for index in range(10):
        _vm(tmp_path / "fastStorage" / "2013-8" / f"{index}.csv", [(0, 100.0)])
    _vm(tmp_path / "fastStorage" / "2013-8" / "solo.csv", [(0, 100.0), (300, 100.0)])
    fleet = aggregate_fleet(tmp_path, "fastStorage", bin_seconds=300)
    series = to_demand_series(fleet, trace="fastStorage")
    assert not series.is_imputed[0]
    assert series.is_imputed[1], "a bin covering one VM of eleven must be flagged"
    assert series.quality[1] == 0.0


def test_fleet_requires_utc_start() -> None:
    with pytest.raises(ValueError, match="UTC"):
        FleetDemand(
            start=datetime(2013, 7, 1),
            bin_seconds=300,
            cpu_mhz=np.zeros(2),
            vm_count=np.zeros(2, dtype=np.int64),
        )


def test_empty_directory_raises_rather_than_returning_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no VM files"):
        aggregate_fleet(tmp_path, "fastStorage", bin_seconds=300)


def test_days_property_reports_the_span() -> None:
    fleet = FleetDemand(
        start=datetime(2013, 7, 1, tzinfo=UTC),
        bin_seconds=300,
        cpu_mhz=np.zeros(288),
        vm_count=np.ones(288, dtype=np.int64),
    )
    assert fleet.days == pytest.approx(1.0)
