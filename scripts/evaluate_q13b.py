"""Q13b — the confirmatory test pre-registered in `docs/PREREGISTRATION-Q13B.md`.

One measure, one horizon, one test, on a cohort disjoint from the one that produced the
lead. This script is committed together with the pre-registration and before the
confirmatory data existed; the git history is what makes that claim checkable.

Read the pre-registration first. Nothing here may be changed after a result is seen without
saying so loudly in `docs/EVAL.md`.

Run: `python scripts/evaluate_q13b.py`
"""

from __future__ import annotations

import argparse

import numpy as np
from compare_predictability import auc, measures_for  # type: ignore[import-not-found]
from evaluate_threshold import (  # type: ignore[import-not-found]
    BIN_SECONDS,
    build_candidates,
    stratify,
    strict_dominance,
)

# --- everything below is pre-registered; see docs/PREREGISTRATION-Q13B.md -------------
MEASURE = "low-freq power"
WINDOW_HOURS = 12
SAMPLE_SIZE = 80
ALPHA = 0.05
PERMUTATIONS = 20_000
CONFIRMATORY_SEED = 20260809

#: The exploratory run these must not overlap with.
EXPLORATORY_COHORT = 400
EXPLORATORY_SAMPLE = 60
EXPLORATORY_SEED = 20260808

SECONDARY = ("daily autocorr", "spectral predictability")


def exploratory_names() -> set[str]:
    """Exactly the workloads the lead was found on, so they can be excluded."""
    seen = stratify(
        build_candidates(EXPLORATORY_COHORT, EXPLORATORY_SEED),
        EXPLORATORY_SAMPLE,
        EXPLORATORY_SEED,
    )
    return {candidate.workload.name for candidate in seen}


def one_sided_permutation_p(
    scores: list[float], labels: list[bool], observed: float, seed: int
) -> float:
    """P(AUC >= observed | labels shuffled). One-sided: H1 predicts a direction."""
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(labels, dtype=bool)
    hits = 0
    for _ in range(PERMUTATIONS):
        if auc(scores, list(rng.permutation(shuffled))) >= observed:
            hits += 1
    return (hits + 1) / (PERMUTATIONS + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=int, default=900, help="workloads to load and filter")
    args = parser.parse_args()

    window_steps = WINDOW_HOURS * 3600 // BIN_SECONDS

    print("# Q13b — confirmatory test\n")
    print(
        f"Pre-registered in `docs/PREREGISTRATION-Q13B.md`. Measure: **{MEASURE}**. "
        f"Horizon: **{WINDOW_HOURS} h**. One-sided permutation test, "
        f"alpha = {ALPHA}, {PERMUTATIONS:,} draws. Target n = {SAMPLE_SIZE}.\n"
    )

    excluded = exploratory_names()
    fresh = [
        candidate
        for candidate in build_candidates(args.cohort, CONFIRMATORY_SEED)
        if candidate.workload.name not in excluded
    ]
    rng = np.random.default_rng(CONFIRMATORY_SEED)
    if len(fresh) > SAMPLE_SIZE:
        index = rng.choice(len(fresh), size=SAMPLE_SIZE, replace=False)
        fresh = [fresh[int(i)] for i in index]
    print(
        f"Excluded {len(excluded)} exploratory workloads. "
        f"Confirmatory cohort: {len(fresh)} workloads, sampled uniformly at random "
        "with no stratification.\n"
    )

    rows: list[tuple[str, dict[str, float], bool]] = []
    all_tie = 0
    for candidate in fresh:
        workload = candidate.workload
        wins, ties = strict_dominance(workload, window_steps)
        if ties == 4:
            all_tie += 1
            continue
        rows.append(
            (
                workload.name,
                measures_for(workload.series.values, workload.day_steps, window_steps),
                wins > 0,
            )
        )

    labels = [paid for _, _, paid in rows]
    print(
        f"Measurable: {len(rows)} ({sum(labels)} where forecasting paid, "
        f"{len(labels) - sum(labels)} where it did not). "
        f"Dropped as all-tie non-measurements: {all_tie}.\n"
    )
    if len(set(labels)) < 2 or len(rows) < 20:
        print("**Test not evaluable** — the cohort is degenerate. Reported, not worked around.")
        return

    scores = [measures[MEASURE] for _, measures, _ in rows]
    observed = auc(scores, labels)
    p_value = one_sided_permutation_p(scores, labels, observed, CONFIRMATORY_SEED)
    confirmed = observed > 0.5 and p_value < ALPHA

    print("## The pre-registered test\n")
    print("| measure | AUC | one-sided p | alpha | verdict |")
    print("|---|---:|---:|---:|:--:|")
    print(
        f"| {MEASURE} | {observed:.3f} | {p_value:.4f} | {ALPHA} | "
        f"**{'CONFIRMED' if confirmed else 'NOT CONFIRMED'}** |"
    )
    print()
    if confirmed:
        print(
            f"The lead replicates on data that did not generate it. AUC {observed:.3f} at a "
            f"{WINDOW_HOURS}-hour commitment, p = {p_value:.4f} against a single "
            "pre-registered hypothesis."
        )
    elif observed <= 0.5:
        print(
            f"**Refuted.** AUC {observed:.3f} is at or below chance, so the measure carries "
            "no usable signal on a fresh cohort regardless of the p-value."
        )
    else:
        print(
            f"**Not confirmed.** AUC {observed:.3f} points the right way but p = {p_value:.4f} "
            f"does not clear {ALPHA}. On the pre-registered rule this is a null result and the "
            "shipped diagnostic does not change."
        )

    print("\n## Secondary — descriptive, not the test\n")
    print("| measure | AUC on the same fresh cohort |")
    print("|---|---:|")
    for key in SECONDARY:
        print(f"| {key} | {auc([m[key] for _, m, _ in rows], labels):.3f} |")
    print(
        "\nThese carry no alpha and cannot confirm anything. They are here so the three "
        "measures can be compared on data none of them has seen."
    )


if __name__ == "__main__":
    main()
