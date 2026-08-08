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
from delphi.control.simulator import CapacityProfile, CostModel, simulate
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


def _function_workload(name: str, series: DemandSeries) -> Workload:
    """Size a replica off the p95, not the median.

    ``_fleet_workload`` scales capacity by ``median/8``, which is right for a fleet
    aggregate but degenerates on serverless traces: `2b373145c4fa` is 65% zeros, so its
    median is 0, the floor of 1.0 applies, and the run reports a mean of 17,413 replicas.
    Anchoring on the p95 keeps the replica count in a sane band whatever the sparsity, and
    is applied identically to both controllers so it cannot favour either.
    """
    scale = max(float(np.quantile(series.values, 0.95)) / 10.0, 1.0)
    profile = CapacityProfile(
        workload_id=series.workload_id,
        step_seconds=series.step_seconds,
        capacity_per_replica=scale,
        startup_seconds=600.0,
        teardown_seconds=300.0,
        utilisation_target=0.85,
        min_replicas=0,
        max_replicas=100_000,
        scale_to_zero=True,
    )
    costs = CostModel(price_per_replica_hour=0.0416, churn_cost_per_action=0.0416 / 6.0)
    return Workload(name, series, profile, costs, DAY_STEPS)


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


def economic_cost(capacity_cost: float, unmet: float, workload: Workload, quantile: float) -> float:
    """Capacity bill plus unmet demand priced at the ratio the quantile implies.

    Pareto dominance is the wrong scorer here, and measurably so. On `762d22c5a3d7`
    (r = 0.892) forecasting cut violations from 0.193 to 0.072 — a factor of 2.7 — while
    costing 6% more, which strict dominance records as *not a win*. Scored that way the
    diagnostic looks near-random, but what is being measured is "forecasting must be free",
    not "forecasting pays".

    The newsvendor objective is the honest scorer and this project already uses it for Q4:
    a target quantile ``q`` asserts ``C_u/C_o = q/(1-q)``, so unmet demand is priced at that
    ratio and the two controllers are compared on a single number.
    """
    per_unit_hour = workload.costs.price_per_replica_hour / workload.profile.capacity_per_replica
    return capacity_cost + per_unit_hour * (quantile / (1.0 - quantile)) * unmet


def strict_dominance(workload: Workload, window_steps: int) -> tuple[int, int]:
    """Return (economic wins, ties) out of ``len(QUANTILES)``.

    A cell is a win when forward's total economic cost is strictly lower. Ties — both
    controllers emitting an identical plan, so nothing is being compared — are counted and
    excluded rather than scored as failures. ``evaluate_diagnostic.dominance`` would count
    those as wins; that never fired on the fleet traces (checked: every published win is
    strict and no plan pair identical, including materna-2's 2 of 4) but fires often on
    Azure Functions' low-volume workloads.
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
        back_total = economic_cost(backward.cost, backward.violation_sum, workload, quantile)
        fwd_total = economic_cost(forward.cost, forward.violation_sum, workload, quantile)
        if fwd_total < back_total:
            wins += 1
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
        workload = _function_workload(series.workload_id.split(":")[-1][:12], series)
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
        print(
            "| workload | daily autocorr | band | predicted | economic wins /4 | ties | correct |"
        )
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
