"""Deterministic labelled traces for sanity checks, drift, and negative controls."""

from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np

from delphi.data.series import DemandSeries, SeriesAnnotation

SyntheticKind = Literal[
    "clean_daily",
    "burst_known",
    "level_shift",
    "cold_start",
    "near_noise",
    "benign_confounder",
]


def _timestamps(start: datetime, periods: int, step_seconds: int) -> tuple[datetime, ...]:
    if start.tzinfo is None or start.utcoffset() is None or start.utcoffset() != timedelta(0):
        raise ValueError("synthetic start must be normalized to UTC")
    return tuple(start + timedelta(seconds=index * step_seconds) for index in range(periods))


def generate_synthetic(
    kind: SyntheticKind,
    *,
    periods: int = 14 * 24 * 60,
    step_seconds: int = 60,
    seed: int = 20260802,
    start: datetime = datetime(2026, 1, 1, tzinfo=UTC),
) -> DemandSeries:
    """Generate one labelled demand regime without using future information."""
    if periods < 48:
        raise ValueError("synthetic traces require at least 48 points")
    rng = np.random.default_rng(seed)
    index = np.arange(periods, dtype=np.float64)
    steps_per_day = max(2, round(86400 / step_seconds))
    daily = np.sin(2 * np.pi * index / steps_per_day - np.pi / 2)
    weekly = np.sin(2 * np.pi * index / (7 * steps_per_day))
    base = 55.0 + 22.0 * daily + 6.0 * weekly
    annotations: list[SeriesAnnotation] = []

    if kind == "near_noise":
        values = rng.poisson(55.0, size=periods).astype(np.float64)
        annotations.append(SeriesAnnotation("near_noise", 0, periods))
    else:
        values = base + rng.normal(0.0, 2.5, size=periods)
        if kind == "clean_daily":
            annotations.append(SeriesAnnotation("stationary_periodic", 0, periods))
        elif kind == "burst_known":
            width = max(2, periods // 48)
            for center in (periods // 3, 2 * periods // 3):
                left, right = center, min(periods, center + width)
                values[left:right] += np.linspace(45.0, 20.0, right - left)
                annotations.append(SeriesAnnotation("burst", left, right))
        elif kind == "level_shift":
            shift = 3 * periods // 5
            values[shift:] += 35.0
            annotations.append(SeriesAnnotation("level_shift", shift, periods))
        elif kind == "cold_start":
            annotations.append(SeriesAnnotation("cold_start", 0, periods))
        elif kind == "benign_confounder":
            width = max(2, periods // 96)
            every = max(width + 1, periods // 7)
            for left in range(every, periods, every):
                right = min(periods, left + width)
                values[left:right] += 30.0
                annotations.append(SeriesAnnotation("scheduled_batch", left, right))
        else:  # pragma: no cover - Literal plus callers keep this unreachable
            raise ValueError(f"unknown synthetic kind: {kind}")

    values = np.maximum(values, 0.0)
    return DemandSeries(
        workload_id=f"synthetic:{kind}",
        source_id="synthetic-gold",
        resource_kind="invocations",
        unit="requests/minute",
        step_seconds=step_seconds,
        timestamps=_timestamps(start, periods, step_seconds),
        values=values,
        is_imputed=np.zeros(periods, dtype=np.bool_),
        quality=np.ones(periods, dtype=np.float64),
        annotations=tuple(annotations),
    )


def generate_multi_resource(
    *,
    periods: int = 14 * 24 * 12,
    step_seconds: int = 300,
    seed: int = 20260802,
    start: datetime = datetime(2026, 1, 1, tzinfo=UTC),
) -> tuple[DemandSeries, DemandSeries]:
    """Correlated CPU/memory demand with deliberately decorrelated peak intervals."""
    if periods < 96:
        raise ValueError("multi-resource traces require at least 96 points")
    rng = np.random.default_rng(seed)
    index = np.arange(periods, dtype=np.float64)
    steps_per_day = max(2, round(86400 / step_seconds))
    shared = 45.0 + 15.0 * np.sin(2 * np.pi * index / steps_per_day)
    cpu = shared + rng.normal(0, 2.0, periods)
    memory = 0.8 * shared + 12.0 + rng.normal(0, 1.5, periods)
    width = max(3, periods // 64)
    cpu_peak = (periods // 3, periods // 3 + width)
    memory_peak = (2 * periods // 3, 2 * periods // 3 + width)
    cpu[slice(*cpu_peak)] += 40.0
    memory[slice(*memory_peak)] += 35.0
    timestamps = _timestamps(start, periods, step_seconds)

    def make(resource: str, values: np.ndarray, peak: tuple[int, int]) -> DemandSeries:
        return DemandSeries(
            workload_id=f"synthetic:multi_resource:{resource}",
            source_id="synthetic-gold",
            resource_kind=resource,
            unit="percent",
            step_seconds=step_seconds,
            timestamps=timestamps,
            values=np.maximum(values, 0.0),
            is_imputed=np.zeros(periods, dtype=np.bool_),
            quality=np.ones(periods, dtype=np.float64),
            annotations=(SeriesAnnotation(f"{resource}_peak", *peak),),
        )

    return make("cpu", cpu, cpu_peak), make("memory", memory, memory_peak)
