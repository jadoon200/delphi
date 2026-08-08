"""Q13: is 0.50 a real boundary, or did seven workloads flatter a round number?

The diagnostic's threshold rests on 7 workloads and 28 cells, with one exception that runs
against the rule — `materna-2` at r = 0.450 beat a trailing percentile at a twelve-hour
commitment while `materna-1` at r = 0.494 did not. Two workloads either side of the cutoff
cannot settle whether the cutoff exists.

This widens the sample using data already on disk. Azure Functions 2019 is CC-BY, already
fetched and checksum-verified, and carries tens of thousands of workloads at one-minute
resolution across the full range of daily structure. Nothing new is downloaded and no
licence question is opened.

**The sampling is deliberately hostile to the rule.** A cohort drawn by volume would cluster
in the easy extremes and report a flattering accuracy. This stratifies by the diagnostic
itself and *oversamples the borderline band*, so most of the evidence comes from precisely
the region where the threshold is least defensible. Reported accuracy is therefore a lower
bound on what a volume-weighted cohort would show, and that is the honest direction to err.

Run: `python scripts/evaluate_threshold.py --cohort 240 --sample 60`
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from evaluate_diagnostic import (  # type: ignore[import-not-found]
    QUANTILES,
    Workload,
    _fleet_workload,
)

from delphi.api.snapshot import (
    FORECASTABLE_AUTOCORRELATION,
    INDETERMINATE_BAND,
    classify_band,
)
from delphi.control.commitment import (
    BackwardCommitmentController,
    ForwardCommitmentController,
)
from delphi.control.controllers import ControlContext
from delphi.control.simulator import simulate
from delphi.data.azure_functions import load_archive_cohort, select_cohort
from delphi.data.series import DemandSeries, aggregate_series
from delphi.forecast.baselines import SeasonalNaiveForecaster

ARCHIVE = Path("data/raw/azure-functions-2019.tar.xz")
BIN_SECONDS = 300
DAY_STEPS = 86400 // BIN_SECONDS

#: Azure Functions is dominated by functions invoked a handful of times a day. A series that
#: is mostly zero has an autocorrelation dominated by its zeros and a commitment experiment
#: on it measures nothing, so a floor is applied and recorded rather than left implicit.
MIN_MEAN_INVOCATIONS = 1.0
MIN_NONZERO_FRACTION = 0.20


@dataclass(frozen=True)
class Candidate:
    workload: Workload
    daily: float

    @property
    def band(self) -> str:
        return classify_band(self.daily)


def usable(series: DemandSeries) -> bool:
    values = series.values
    if len(values) < DAY_STEPS * 5:
        return False
    if float(values.mean()) < MIN_MEAN_INVOCATIONS:
        return False
    if float(np.mean(values > 0)) < MIN_NONZERO_FRACTION:
        return False
    if float(values.std()) <= 0:
        return False
    # Both halves of the lagged pair must vary, or the correlation is NaN rather than low.
    return bool(values[:-DAY_STEPS].std() > 0 and values[DAY_STEPS:].std() > 0)


def strict_dominance(workload: Workload, window_steps: int) -> tuple[int, int]:
    """Return (strict wins, ties) out of ``len(QUANTILES)``.

    ``evaluate_diagnostic.dominance`` counts a cell when forward is no worse on both axes,
    which silently counts a *tie* as a win. On the fleet traces that never fired — checked:
    every published win is strict and no plan pair was identical — but Azure Functions
    carries many low-volume workloads where both controllers emit the same plan, and there
    a tie-as-win would manufacture agreement with the threshold out of nothing.
    """
    context = ControlContext(
        series=workload.series, profile=workload.profile, start_step=workload.day_steps * 2
    )
    scored = slice(workload.day_steps * 3, None)
    forecaster = SeasonalNaiveForecaster(workload.day_steps)
    wins = ties = 0
    for quantile in QUANTILES:
        backward_plan = BackwardCommitmentController(
            window_steps=window_steps, quantile=quantile
        ).plan(context)
        forward_plan = ForwardCommitmentController(
            window_steps=window_steps, forecaster=forecaster, quantile=quantile
        ).plan(context)
        if np.array_equal(backward_plan[scored], forward_plan[scored]):
            ties += 1
            continue
        backward = simulate(
            demand=workload.series.values[scored],
            requested=backward_plan[scored],
            profile=workload.profile,
            cost_model=workload.costs,
        )
        forward = simulate(
            demand=workload.series.values[scored],
            requested=forward_plan[scored],
            profile=workload.profile,
            cost_model=workload.costs,
        )
        cheaper = forward.cost < backward.cost
        safer = forward.violation_rate < backward.violation_rate
        no_worse_cost = forward.cost <= backward.cost
        no_worse_viol = forward.violation_rate <= backward.violation_rate
        if (cheaper and no_worse_viol) or (safer and no_worse_cost):
            wins += 1
        elif no_worse_cost and no_worse_viol:
            ties += 1
    return wins, ties


def build_candidates(cohort_size: int, seed: int) -> list[Candidate]:
    selection = select_cohort(
        ARCHIVE, top_n=cohort_size // 4, per_decile=cohort_size // 10, seed=seed
    )
    candidates: list[Candidate] = []
    for minute_series in load_archive_cohort(ARCHIVE, selection):
        usable_length = len(minute_series) - len(minute_series) % (BIN_SECONDS // 60)
        trimmed = minute_series.slice(0, usable_length)
        series = aggregate_series(trimmed, BIN_SECONDS // 60, reducer="sum")
        if not usable(series):
            continue
        workload = _fleet_workload(series.workload_id.split(":")[-1][:12], series, 0.0416)
        daily = workload.daily_autocorrelation
        if not np.isfinite(daily) or abs(daily) >= 0.999:
            # |r| = 1.000 is an instrument artefact on these sparse traces, not structure.
            continue
        candidates.append(Candidate(workload, daily))
    return candidates


def stratify(candidates: list[Candidate], sample: int, seed: int) -> list[Candidate]:
    """Oversample the borderline band; the extremes need far less evidence."""
    low, high = INDETERMINATE_BAND
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[Candidate]] = {"borderline": [], "strong": [], "weak": [], "none": []}
    for candidate in candidates:
        buckets[candidate.band].append(candidate)
    quota = {
        "borderline": sample // 2,
        "strong": sample // 6,
        "weak": sample // 6,
        "none": sample - sample // 2 - 2 * (sample // 6),
    }
    chosen: list[Candidate] = []
    for band, want in quota.items():
        pool = buckets[band]
        if not pool:
            continue
        take = min(want, len(pool))
        index = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[int(i)] for i in index)
    print(
        f"Cohort: {len(candidates)} usable workloads "
        f"(borderline band {low:.2f}-{high:.2f}: {len(buckets['borderline'])} available). "
        f"Sampled {len(chosen)}."
    )
    return sorted(chosen, key=lambda c: -c.daily)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=int, default=240, help="workloads to load")
    parser.add_argument("--sample", type=int, default=60, help="workloads to simulate")
    parser.add_argument("--windows", nargs="+", type=int, default=[6, 12])
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    print("# Q13 — does daily autocorrelation actually mark a boundary?\n")
    print(
        "Azure Functions 2019 (CC-BY, already fetched), aggregated to 5-minute bins, "
        f"stratified by the diagnostic and oversampled in the {INDETERMINATE_BAND[0]:.2f}"
        f"-{INDETERMINATE_BAND[1]:.2f} borderline band. Forecaster: seasonal-naive. "
        f"Quantiles: {', '.join(f'{q:g}' for q in QUANTILES)}.\n"
    )

    candidates = stratify(build_candidates(args.cohort, args.seed), args.sample, args.seed)
    if not candidates:
        print("\n**No usable workloads.** Check that the archive is fetched.")
        return

    for window_hours in args.windows:
        window_steps = window_hours * 3600 // BIN_SECONDS
        print(f"\n## {window_hours}-hour commitment\n")
        print("| workload | daily autocorr | band | predicted | strict wins /4 | ties | correct |")
        print("|---|---:|---|---|---:|---:|:--:|")
        tally: Counter[str] = Counter()
        confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for candidate in candidates:
            try:
                wins, ties = strict_dominance(candidate.workload, window_steps)
            except (ValueError, RuntimeError) as exc:  # a degenerate trace, recorded not hidden
                print(
                    f"| `{candidate.workload.name}` | {candidate.daily:+.3f} | "
                    f"{candidate.band} | — | skipped: {type(exc).__name__} | — |"
                )
                tally["skipped"] += 1
                continue
            predicted = candidate.daily >= FORECASTABLE_AUTOCORRELATION
            actual = wins > 0
            correct = predicted == actual
            confusion[
                "tp" if predicted and actual else "fp" if predicted else "fn" if actual else "tn"
            ] += 1
            tally["correct" if correct else "wrong"] += 1
            print(
                f"| `{candidate.workload.name}` | {candidate.daily:+.3f} | {candidate.band} | "
                f"{'wins' if predicted else 'loses'} | {wins} | {ties} | "
                f"{'yes' if correct else '**NO**'} |"
            )

        judged = tally["correct"] + tally["wrong"]
        if judged:
            print(
                f"\n**Threshold accuracy at {window_hours}h: {tally['correct']}/{judged} "
                f"({tally['correct'] / judged:.0%})**, on a sample deliberately weighted "
                f"toward the borderline band. Skipped: {tally['skipped']}."
            )
            print(
                f"- Predicted wins, did win: {confusion['tp']} · "
                f"predicted wins, did not: {confusion['fp']} · "
                f"predicted loses, did win: {confusion['fn']} · "
                f"predicted loses, did not: {confusion['tn']}"
            )
            if confusion["fn"]:
                print(
                    f"- **{confusion['fn']} workloads below the cutoff won anyway** — the same "
                    "direction as the `materna-2` exception, so it is not a one-off."
                )


if __name__ == "__main__":
    main()
