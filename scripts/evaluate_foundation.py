"""M17 / Q3 — does a foundation model beat the tuned classical baseline?

Registered expectation: *"On accuracy, marginally at best. On calibration, quite possibly
worse. The real win is cold start with zero per-workload training."*

Chronos-Bolt is the Apache-2.0 default from the licence audit. It is an **optional extra** —
`pip install chronos-forecasting` — because torch has no place in a deploy image that serves
a precomputed snapshot.

Two questions, and the project's whole thesis is that the second one is the one that matters:

1. **Accuracy** — MASE against the classical ladder on held-out time. This is the comparison
   the forecasting literature runs.
2. **The decision** — does it win commitment cells that seasonal-naive, ETS and LightGBM did
   not? A capacity controller consumes a quantile, not a point estimate, so a model that
   forecasts beautifully and cannot express p95 is useless to it however good its MASE is.

Run: `python scripts/evaluate_foundation.py`
"""

from __future__ import annotations

import os

# torch and lightgbm each load an OpenMP runtime and on macOS the pair segfaults the
# interpreter mid-run. Pinning both to a single thread removes the contention. This is a
# real mitigation rather than KMP_DUPLICATE_LIB_OK, which suppresses the duplicate-runtime
# check and can corrupt memory silently. Set before either library is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse

from evaluate_diagnostic import (  # type: ignore[import-not-found]
    QUANTILES,
    build_workloads,
    forecasters,
)
from evaluate_threshold import strict_dominance  # type: ignore[import-not-found]

from delphi.evaluation.metrics import mase
from delphi.forecast.foundation import (
    CHRONOS_TRAINED_RANGE,
    ChronosForecaster,
    chronos_available,
)

HORIZON = 24

#: Best classical result per workload at a 12-hour commitment, from the 2026-08-07 study in
#: `docs/EVAL.md`. Quoted rather than recomputed; see the note above that table.
PUBLISHED_CLASSICAL = {
    "materna-1": "0/4",
    "materna-2": "2/4 (seasonal-naive)",
    "materna-3": "0/4",
    "bitbrains-rnd": "0/4",
    "bitbrains-fastStorage": "0/4",
}


def accuracy_table(workloads: list, chronos: ChronosForecaster) -> None:
    print("## Accuracy — MASE at a 24-step horizon, held-out\n")
    print("| workload | seasonal-naive | ets | lightgbm | **chronos-bolt** |")
    print("|---|---:|---:|---:|---:|")
    for workload in workloads:
        series = workload.series
        origin = int(len(series) * 0.75)
        history = series.slice(0, origin)
        actual = series.values[origin : origin + HORIZON]
        if len(actual) < HORIZON:
            continue
        train = series.values[:origin]
        season = workload.day_steps

        def score(predicted_median, train=train, season=season, actual=actual) -> str:
            return f"{mase(actual, predicted_median, train, season):.3f}"

        cells = []
        candidates = forecasters(workload.day_steps)
        for name in ("seasonal-naive", "ets", "lightgbm"):
            try:
                predicted = candidates[name].forecast(
                    history, horizon_steps=HORIZON, quantile_levels=(0.5,)
                )
                cells.append(score(predicted.quantile_values[:, 0]))
            except (ValueError, RuntimeError):
                cells.append("—")
        predicted = chronos.forecast(history, horizon_steps=HORIZON, quantile_levels=(0.5,))
        cells.append(f"**{score(predicted.quantile_values[:, 0])}**")
        print(f"| `{workload.name}` | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int, default=[12])
    args = parser.parse_args()

    print("# M17 / Q3 — Chronos-Bolt against the tuned classical ladder\n")
    if not chronos_available():
        print(
            "**Not run.** `chronos-forecasting` is an optional extra and is not installed. "
            "This is not a result; install it and re-run.\n"
        )
        return

    chronos = ChronosForecaster()
    low, high = CHRONOS_TRAINED_RANGE
    clamped = chronos.clamped_levels(QUANTILES)

    print("## The ceiling, before any number below is read\n")
    print(
        f"Chronos-Bolt is trained on quantile levels {low} to {high}. Of the levels this "
        f"project sizes capacity at ({', '.join(f'{q:g}' for q in QUANTILES)}) it cannot "
        f"express **{', '.join(f'{q:g}' for q in clamped)}**. It does not extrapolate or "
        "refuse; it returns p90 and warns. A newsvendor sizer asking for p95 receives p90 "
        "wearing a p95 label, which is worse than an error because it looks like an answer.\n"
    )

    workloads = build_workloads(include_gpu=False)
    accuracy_table(workloads, chronos)

    print("\n## The decision — commitment cells won, out of 4 quantiles\n")
    print(
        "Forward-commitment dominance on total economic cost at a 12-hour window, the same "
        "test the 2026-08-07 study ran. **Only Chronos is recomputed here.** The classical "
        "columns are quoted from that study rather than re-run: the test refits at every "
        "window boundary, which is roughly 4,600 fits across four forecasters, and ETS alone "
        "would take about four and a half hours single-threaded. Chronos is scored at every "
        "quantile including the two it cannot express, because that is what an operator "
        "following its API would actually get.\n"
    )
    print("| workload | daily autocorr | classical best (2026-08-07) | **chronos-bolt** |")
    print("|---|---:|---:|---:|")
    for window_hours in args.windows:
        for workload in workloads:
            window_steps = window_hours * 3600 // workload.series.step_seconds
            wins, ties = strict_dominance(workload, window_steps, chronos)
            best = PUBLISHED_CLASSICAL.get(workload.name, "—")
            print(
                f"| `{workload.name}` | {workload.daily_autocorrelation:+.3f} | {best} | "
                f"**{wins}**{' (' + str(ties) + ' tied)' if ties else ''} |"
            )


if __name__ == "__main__":
    main()
