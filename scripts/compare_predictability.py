"""Which training-free measure actually predicts whether forecasting pays?

Q13 established that the shipped diagnostic — daily autocorrelation — does not generalise
past the fleet-aggregate traces it was derived on. The forecastability literature says why:
a single lagged correlation reads one frequency, while spectral entropy reads all of them,
and spectral measures are the established indicator for exactly this question.

This races three candidates against the same target on the same workloads:

* ``daily_autocorrelation`` — what shipped;
* ``spectral_predictability`` (1 - spectral entropy) — the field's standard;
* ``low_frequency_power_fraction`` — the share of power at periods longer than the
  commitment window, motivated by this project's own Q1 result that what a fixed-for-N-hours
  decision can exploit is structure slower than N hours.

**Comparison is threshold-free.** Reporting each measure at its own best cutoff would let a
measure win by overfitting a cutoff to 43 points. AUC — the probability that a randomly
chosen paying workload scores above a randomly chosen non-paying one — needs no cutoff and
is the honest way to ask which measure carries more signal. Best-threshold accuracy is
printed alongside as an optimistic upper bound, clearly labelled as one.

Run: `python scripts/compare_predictability.py --cohort 400 --sample 60`
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from evaluate_diagnostic import build_workloads  # type: ignore[import-not-found]
from evaluate_threshold import (  # type: ignore[import-not-found]
    BIN_SECONDS,
    build_candidates,
    stratify,
    strict_dominance,
)

from delphi.forecast.predictability import (
    daily_autocorrelation,
    low_frequency_power_fraction,
    spectral_predictability,
)


@dataclass(frozen=True)
class Scored:
    name: str
    population: str
    measures: dict[str, float]
    paid: bool


def auc(scores: list[float], labels: list[bool]) -> float:
    """Mann-Whitney U / ROC AUC. 0.5 is a coin flip; below 0.5 is anti-predictive."""
    positive = [s for s, y in zip(scores, labels, strict=True) if y]
    negative = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not positive or not negative:
        return float("nan")
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks within ties so a constant measure scores exactly 0.5, not 1.0.
    values = np.asarray(scores, dtype=np.float64)
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    positive_rank_sum = float(ranks[np.asarray(labels, dtype=bool)].sum())
    n_pos, n_neg = len(positive), len(negative)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def best_threshold_accuracy(scores: list[float], labels: list[bool]) -> tuple[float, float]:
    """Optimistic upper bound: the best cutoff chosen *on this same data*."""
    best = (float("nan"), 0.0)
    for cut in sorted(set(scores)):
        predicted = [s >= cut for s in scores]
        acc = sum(p == y for p, y in zip(predicted, labels, strict=True)) / len(labels)
        if acc > best[1]:
            best = (cut, acc)
    return best


def measures_for(values: np.ndarray, day_steps: int, window_steps: int) -> dict[str, float]:
    return {
        "daily autocorr": daily_autocorrelation(values, day_steps),
        "spectral predictability": spectral_predictability(values),
        "low-freq power": low_frequency_power_fraction(values, window_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=int, default=400)
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--windows", nargs="+", type=int, default=[6, 12])
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    print("# Which predictability measure actually works?\n")
    print(
        "Threshold-free comparison by AUC — the probability a paying workload scores above "
        "a non-paying one. 0.50 is a coin flip. Best-threshold accuracy is an optimistic "
        "upper bound, since the cutoff is fitted on the same data it is scored on.\n"
    )

    serverless = stratify(build_candidates(args.cohort, args.seed), args.sample, args.seed)
    fleet = build_workloads(include_gpu=False)

    for window_hours in args.windows:
        rows: list[Scored] = []

        window_steps = window_hours * 3600 // BIN_SECONDS
        for candidate in serverless:
            workload = candidate.workload
            wins, _ = strict_dominance(workload, window_steps)
            rows.append(
                Scored(
                    workload.name,
                    "serverless",
                    measures_for(workload.series.values, workload.day_steps, window_steps),
                    wins > 0,
                )
            )

        for workload in fleet:
            steps = window_hours * 3600 // workload.series.step_seconds
            wins, _ = strict_dominance(workload, steps)
            rows.append(
                Scored(
                    workload.name,
                    "fleet",
                    measures_for(workload.series.values, workload.day_steps, steps),
                    wins > 0,
                )
            )

        print(f"\n## {window_hours}-hour commitment\n")
        for population in ("serverless", "fleet", "both"):
            subset = [r for r in rows if population in (r.population, "both")]
            labels = [r.paid for r in subset]
            if len(set(labels)) < 2:
                print(
                    f"### {population}: all outcomes identical, AUC undefined (n={len(subset)})\n"
                )
                continue
            print(
                f"### {population} (n={len(subset)}, "
                f"{sum(labels)} paid / {len(labels) - sum(labels)} did not)\n"
            )
            print("| measure | AUC | best-threshold accuracy (optimistic) |")
            print("|---|---:|---:|")
            for key in ("daily autocorr", "spectral predictability", "low-freq power"):
                scores = [r.measures[key] for r in subset]
                area = auc(scores, labels)
                cut, acc = best_threshold_accuracy(scores, labels)
                print(f"| {key} | {area:.3f} | {acc:.0%} at >= {cut:.3f} |")
            baseline = max(sum(labels), len(labels) - sum(labels)) / len(labels)
            print(f"\nAlways-guess-the-majority baseline: {baseline:.0%}.\n")


if __name__ == "__main__":
    main()
