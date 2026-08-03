"""Validated in-memory contract shared by every trace and forecaster."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class SeriesAnnotation:
    """A labelled half-open interval ``[start, end)`` in a demand series."""

    label: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("annotation must be a non-empty half-open interval")


@dataclass(frozen=True)
class DemandSeries:
    """One regularly spaced, provenance-stamped workload demand series.

    Arrays are copied and made read-only so a forecast cannot accidentally mutate the trace
    that later evaluation treats as ground truth.
    """

    workload_id: str
    source_id: str
    resource_kind: str
    unit: str
    step_seconds: int
    timestamps: tuple[datetime, ...]
    values: FloatArray
    is_imputed: BoolArray
    quality: FloatArray
    annotations: tuple[SeriesAnnotation, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).copy()
        imputed = np.asarray(self.is_imputed, dtype=np.bool_).copy()
        quality = np.asarray(self.quality, dtype=np.float64).copy()
        size = len(self.timestamps)
        if self.step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        if not self.workload_id or not self.source_id or not self.resource_kind or not self.unit:
            raise ValueError("series identity and units must be non-empty")
        if (
            size == 0
            or values.shape != (size,)
            or imputed.shape != (size,)
            or quality.shape != (size,)
        ):
            raise ValueError(
                "timestamps, values, imputation flags, and quality must be equal 1-D arrays"
            )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("demand values must be finite and non-negative")
        if not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
            raise ValueError("quality must be finite and within [0, 1]")
        for timestamp in self.timestamps:
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("all timestamps must be timezone-aware")
            if timestamp.utcoffset() != timedelta(0):
                raise ValueError("all timestamps must be normalized to UTC")
        expected_step = float(self.step_seconds)
        if any(
            (right - left).total_seconds() != expected_step
            for left, right in zip(self.timestamps, self.timestamps[1:], strict=False)
        ):
            raise ValueError("timestamps must be strictly regular at step_seconds")
        if any(annotation.end > size for annotation in self.annotations):
            raise ValueError("annotation extends beyond the series")
        values.setflags(write=False)
        imputed.setflags(write=False)
        quality.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "is_imputed", imputed)
        object.__setattr__(self, "quality", quality)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def start(self) -> datetime:
        return self.timestamps[0]

    @property
    def end(self) -> datetime:
        return self.timestamps[-1]

    def slice(self, start: int, end: int) -> "DemandSeries":
        """Return a validated slice, clipping annotations to the new coordinate frame."""
        if start < 0 or end > len(self) or end <= start:
            raise ValueError("slice must be a non-empty interval within the series")
        annotations: list[SeriesAnnotation] = []
        for annotation in self.annotations:
            clipped_start = max(annotation.start, start)
            clipped_end = min(annotation.end, end)
            if clipped_end > clipped_start:
                annotations.append(
                    SeriesAnnotation(
                        annotation.label,
                        clipped_start - start,
                        clipped_end - start,
                    )
                )
        return DemandSeries(
            workload_id=self.workload_id,
            source_id=self.source_id,
            resource_kind=self.resource_kind,
            unit=self.unit,
            step_seconds=self.step_seconds,
            timestamps=self.timestamps[start:end],
            values=self.values[start:end],
            is_imputed=self.is_imputed[start:end],
            quality=self.quality[start:end],
            annotations=tuple(annotations),
        )


def concatenate_series(parts: list[DemandSeries]) -> DemandSeries:
    """Join adjacent parts of the same workload while preserving annotations."""
    if not parts:
        raise ValueError("at least one series part is required")
    first = parts[0]
    identity = (
        first.workload_id,
        first.source_id,
        first.resource_kind,
        first.unit,
        first.step_seconds,
    )
    if any(
        (part.workload_id, part.source_id, part.resource_kind, part.unit, part.step_seconds)
        != identity
        for part in parts[1:]
    ):
        raise ValueError("all series parts must describe the same workload and cadence")
    offset = 0
    annotations: list[SeriesAnnotation] = []
    for part in parts:
        annotations.extend(
            SeriesAnnotation(annotation.label, annotation.start + offset, annotation.end + offset)
            for annotation in part.annotations
        )
        offset += len(part)
    return DemandSeries(
        workload_id=first.workload_id,
        source_id=first.source_id,
        resource_kind=first.resource_kind,
        unit=first.unit,
        step_seconds=first.step_seconds,
        timestamps=tuple(timestamp for part in parts for timestamp in part.timestamps),
        values=np.concatenate([part.values for part in parts]),
        is_imputed=np.concatenate([part.is_imputed for part in parts]),
        quality=np.concatenate([part.quality for part in parts]),
        annotations=tuple(annotations),
    )


def aggregate_series(
    series: DemandSeries,
    factor: int,
    *,
    reducer: Literal["mean", "sum", "max"] = "mean",
) -> DemandSeries:
    """Aggregate fixed-size bins while retaining quality and annotation provenance."""
    if factor <= 1 or len(series) % factor:
        raise ValueError("aggregation factor must exceed one and divide the series length")
    matrix = series.values.reshape(-1, factor)
    if reducer == "mean":
        values = np.mean(matrix, axis=1)
    elif reducer == "sum":
        values = np.sum(matrix, axis=1)
    else:
        values = np.max(matrix, axis=1)
    imputed = np.any(series.is_imputed.reshape(-1, factor), axis=1)
    quality = np.mean(series.quality.reshape(-1, factor), axis=1)
    annotations = tuple(
        SeriesAnnotation(
            annotation.label,
            annotation.start // factor,
            min(len(values), (annotation.end + factor - 1) // factor),
        )
        for annotation in series.annotations
    )
    return DemandSeries(
        workload_id=series.workload_id,
        source_id=series.source_id,
        resource_kind=series.resource_kind,
        unit=series.unit,
        step_seconds=series.step_seconds * factor,
        timestamps=tuple(
            series.timestamps[index + factor - 1] for index in range(0, len(series), factor)
        ),
        values=values,
        is_imputed=imputed,
        quality=quality,
        annotations=annotations,
    )
