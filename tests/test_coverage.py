from datetime import UTC, datetime, timedelta

import pytest

from delphi.evaluation.coverage import (
    CoverageObservation,
    CoverageRecord,
    coverage_recovery_steps,
)


def test_coverage_record_keeps_the_realized_time_series() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    observations = tuple(
        CoverageObservation(
            timestamp=start + timedelta(minutes=index),
            nominal_level=0.95,
            predicted=10,
            actual=9 if covered else 11,
            covered=covered,
        )
        for index, covered in enumerate((True, False, True, True))
    )
    record = CoverageRecord("model", "calibrator", 0.95, observations)
    assert record.empirical_coverage == 0.75
    assert record.coverage_series == (True, False, True, True)


def test_coverage_recovery_reports_steps_not_just_average() -> None:
    coverage = (False, False, True, True, True, True)
    assert coverage_recovery_steps(coverage, target=0.75, window=4, tolerance=0) == 5
    with pytest.raises(ValueError):
        coverage_recovery_steps(coverage, target=0.95, window=7)
