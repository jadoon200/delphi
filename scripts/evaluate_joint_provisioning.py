"""Q9 — do independent per-resource forecasts misprovision the joint plan?

Registered in `.claude/L5-EVAL.md` with the expectation "yes; the interesting part is by how
much". It was never measured. `generate_multi_resource` existed and was unit-tested, but no
experiment used it, while `docs/ROADMAP.md` claimed the synthetic regime "carries the
joint-provisioning test". It did not. This is that test.

**The model.** A replica supplies a fixed amount of each resource, so the replicas a workload
actually needs at time *t* is set by whichever resource is tightest:

    r_t = max(cpu_t / cpu_per_replica, memory_t / memory_per_replica)

**Two ways to size for it.** Both pick a level on a training half and are scored on a held-out
half, so neither sees the window it is judged on.

* **Joint** — take the ``q``-quantile of ``r_t`` directly.
* **Independent** — forecast each resource on its own, take each one's ``q``-quantile, convert
  both to replicas, and provision the larger. This is what a team with one dashboard per
  resource actually does.

**The gap only exists in a window of compliance targets, and finding that window is the
result.** Since ``max(a, b) >= a``, the joint quantile is never below the independent one, so
independent sizing can only under-provision. But *when* it does depends on how much of the
time each resource is individually elevated. If each is high for a fraction ``w`` of the
period and the two are disjoint, their maximum is high for ``2w``. Above ``q = 1 - w`` both
methods see the peak and agree; below ``q = 1 - 2w`` neither does, and they agree again. The
mismatch lives strictly in between.

Two earlier designs returned exact nulls for precisely this reason — one used spikes narrower
than the tail the quantile discards, the other used a target above ``1 - w``. Both are kept in
the sweep as controls rather than deleted, because a null from an instrument that cannot
detect the effect is not evidence of absence, and the boundary is the finding.

Separation is swept because the claim under test is that *decorrelation* drives the effect: at
separation 0 the resources peak together and there should be no gap at any target.

Run: `python scripts/evaluate_joint_provisioning.py`
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from delphi.data.synthetic import generate_multi_resource

#: Compliance targets spanning the range an operator would actually pick.
QUANTILES = (0.90, 0.95, 0.99)


@dataclass(frozen=True)
class Sizing:
    replicas: int
    coverage: float
    marginal_quantile: float


def _required_replicas(cpu: np.ndarray, memory: np.ndarray, per: tuple[float, float]) -> np.ndarray:
    return np.maximum(cpu / per[0], memory / per[1])


def _coverage(replicas: int, required: np.ndarray) -> float:
    return float(np.mean(replicas >= required))


def evaluate(
    offset: float, periods: int, seed: int, quantile: float
) -> tuple[int, int, float, float, float]:
    """Return (joint repl., independent repl., joint cov., indep. cov., correlation)."""
    cpu_series, memory_series = generate_multi_resource(
        periods=periods, seed=seed, diurnal_phase_offset=offset
    )
    cpu, memory = cpu_series.values, memory_series.values
    correlation = float(np.corrcoef(cpu, memory)[0, 1])
    midpoint = len(cpu) // 2
    train, test = slice(0, midpoint), slice(midpoint, None)
    # Size a replica so that **each resource alone** needs the same 8 replicas at its own
    # target quantile, computed on train only. Normalising on the median instead equalises
    # the two medians but not their ranges, and the wider-swinging resource then dominates
    # the maximum at every step — which produced an exact null at every phase offset, because
    # the second resource was never the binding one and decorrelation had nothing to act on.
    per = (
        float(np.quantile(cpu[train], quantile)) / 8.0,
        float(np.quantile(memory[train], quantile)) / 8.0,
    )

    required_train = _required_replicas(cpu[train], memory[train], per)
    required_test = _required_replicas(cpu[test], memory[test], per)

    joint = int(np.ceil(np.quantile(required_train, quantile)))
    independent = max(
        int(np.ceil(np.quantile(cpu[train], quantile) / per[0])),
        int(np.ceil(np.quantile(memory[train], quantile) / per[1])),
    )
    return (
        joint,
        independent,
        _coverage(joint, required_test),
        _coverage(independent, required_test),
        correlation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=14 * 24 * 12)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    print("# Q9 — independent per-resource sizing versus a joint plan\n")
    print(
        "CPU and memory share a daily cycle; `offset` lags memory's cycle by a fraction of a "
        "day, so at 0 they rise together and at 0.5 one peaks while the other troughs. The "
        "decorrelation recurs every day, which is what lets a train/test split measure it. "
        "Sized on the first half of each trace, scored on the second.\n"
    )
    print(
        "**Registered expectation: independent forecasts over-provision the joint plan.** "
        "The arithmetic says the opposite is possible — `max(a, b) >= a` means the joint "
        "quantile can only be the larger — so what is measured here is the shortfall and "
        "the coverage it costs.\n"
    )

    for quantile in QUANTILES:
        print(f"\n## Target q = {quantile:g}\n")
        print(
            "| offset | corr | joint repl. | indep. repl. | shortfall | joint cov. | indep. cov. |"
        )
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for offset in (0.0, 0.125, 0.25, 0.375, 0.5):
            joint, independent, joint_cov, indep_cov, corr = evaluate(
                offset, args.periods, args.seed, quantile
            )
            print(
                f"| {offset:.3f} | {corr:+.3f} | {joint} | {independent} | "
                f"{joint - independent:+d} | {joint_cov:.4f} | {indep_cov:.4f} |"
            )


if __name__ == "__main__":
    main()
