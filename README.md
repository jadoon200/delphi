# DELPHI

**Capacity planning that tells you whether prediction can help before it sells you a
predictor — then sizes capacity from the price of failure rather than from convention.**

DELPHI set out to show that a calibrated forecast beats conventional autoscaling. Measured
against the recommender Kubernetes actually ships, it mostly does not. The useful result is
the boundary: forecasting pays when capacity is committed for hours at a time *and* demand
has real daily structure, and a single number you can compute in seconds tells you which
side of that boundary you are on.

## The three findings

1. **In the autoscaling regime, forecasting loses to a trailing percentile.** Demand is
   ~0.98 autocorrelated at one minute on every workload measured here, so recent load
   already carries nearly all the signal. An explicit model mostly adds its own error.
2. **In the commitment regime it wins.** When capacity is fixed for six or twelve hours —
   reserved instances, cluster sizing, procurement — reaction is structurally unavailable
   and a forecast that covers the coming peak is worth having. On the one workload with
   strong daily structure, forecasting cut violations *while costing less*.
3. **Daily autocorrelation predicts which regime you are in — on aggregated demand.** Across
   7 fleet-aggregate workloads x 4 forecasters x 4 quantiles it calls 27 of 28 outcomes
   correctly at a six-hour commitment.
4. **And it does not generalise past that, which we went looking for and found.** On 43
   individual Azure Functions workloads the same cutoff is a coin flip — 51% at six hours,
   worse than ignoring it — and the relationship is not even monotone. What transfers is
   only the bottom of the range: below ~0.20, forecasting failed to pay on every population
   tested. The diagnostic is scoped to aggregates and says so on its own results page.

Capacity itself is sized as a newsvendor decision: the cost of unmet demand and the cost of
idle capacity set the demand quantile to buy, `q* = C_u / (C_u + C_o)`. Every fixed-target
autoscaler is asserting a cost ratio it never states; a controller pinned at p95 is claiming
that a unit of unmet demand costs 19x a unit of idle capacity. DELPHI states it.

## What is in here

- A **replay simulator** with an explicit actuation delay, validated against the Erlang-C
  closed form to within 2.2% and gated before any result is published.
- **Baseline controllers** — static, reactive HPA, the VPA/Autopilot percentile
  recommender, proactive quantile, budget-paced PI — tuned on an equal budget.
- **Calibrated quantile forecasting**: split conformal, CQR and adaptive conformal
  inference, with coverage measured as a time series rather than assumed.
- A **GPU inference lane** on Azure's CC-BY request-level traces, where prefill and decode
  contend for one device and the capacity currency is GPU-seconds, not CPU.
- A **read-only API and dashboard** serving a snapshot baked at image build.

## Read the evaluation

[`docs/EVAL.md`](docs/EVAL.md) is the real artifact. It carries the Pareto frontiers, the
tuning-parity protocol, sample sizes, the open-loop caveat, the pre-registered questions
answered whichever way they fell, and a negatives ledger — including a simulator bug that
reversed one of the headline answers after it was found.

## Principles

- Free and licence-clean by default; no paid key or hosted model is required.
- Quantile calibration is measured on held-out time before a forecast may drive capacity.
- Controller comparisons state the open-loop replay assumption and report Pareto frontiers,
  never a single flattering number.
- An implausibly clean number is an instrument artefact until proven otherwise.
- DELPHI recommends plans for human review; it never applies changes to a real cluster.

## Development

```bash
make env
conda activate delphi
make install
make check
```

Then, to reproduce the data lane and the results:

```bash
make fetch-azure          # 136 MB CC-BY trace, checksum-verified into ignored data/
make ingest-azure         # seeded top-volume + decile-stratified cohort into Postgres
make validate-simulator   # the M4 gate — must be GREEN before trusting any result
make evaluate-commitment  # the headline experiment
```

Postgres uses host port `5436`, keeping it separate from the sibling portfolio services:

```bash
make up
make down
```

## Architecture

```text
public trace / synthetic demand
              │
       canonical DemandSeries
              │
   predictability diagnostic  ──→  "is a forecaster worth building here?"
              │
      calibrated quantiles
              │
  newsvendor sizing  ×  commitment or autoscaling regime
              │
     replay simulator (validated)
              │
   cost-vs-violation frontier  →  human-reviewed plan
```

## Responsible scope

DELPHI is a replay and decision-support system. It does not control Kubernetes, cloud
accounts, or production infrastructure. Simulated savings are directional under an explicit
open-loop assumption; they are never presented as measured production savings. Retail list
prices are not what an enterprise pays — the shape of a frontier is the result, not the
absolute dollars.
