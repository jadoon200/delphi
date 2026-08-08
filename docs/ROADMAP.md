# Roadmap

Milestones are marked complete only after their tests and documentation are green. Where a
milestone was cut, it says so and says why — a roadmap that only records successes is a
marketing document.

## Built

| Milestone | Status | Deliverable |
|---|---|---|
| M0 — system spine | ✅ | Config, UTC utilities, canonical DB schema, Alembic parity, zero-cost test, CI |
| M1 — demand data | ✅ | Validated `DemandSeries`, deterministic labelled synthetic regimes, checksum-verified Azure Functions 2019 cohort ingest, Bitbrains and Materna fleet traces |
| M2 — forecast baselines | ✅ | Leak-guarded rolling-origin seasonal-naive, percentile, ARIMA/ETS, drift and LightGBM quantile baselines with MASE/WQL tables |
| M3 — calibration | ✅ | Split conformal, CQR and adaptive conformal inference with measured coverage and level-shift recovery |
| M4 — replay simulator | ✅ | Event-driven and utilisation fidelities, explicit actuation delay, seeded determinism; validated against Erlang-C within 2.2% and gated by `make validate-simulator` |
| M5 — baseline controllers | ✅ | Static, reactive HPA, VPA/Autopilot percentile recommender, proactive quantile — equal tuning budget, logged |
| M6 — newsvendor sizing | ✅ | `q* = C_u/(C_u+C_o)` driving the compliance target, priced from the keyless Azure Retail Prices API |
| M7 — budget-paced control | ✅ | PI controller over the SLO violation budget on the conformal forecast |
| M8 — honest evaluation | ✅ | Cost-versus-violation Pareto frontiers on real demand, tuning parity, open-loop caveat, pre-registered questions answered including the negatives |
| M14 — inference ingest | ✅ | Azure LLM/LMM traces (CC-BY, request-level, 44M requests) in the canonical schema |
| M15 — serving model | ✅ | Prefill and decode contending for one device in GPU-seconds, with a prewarm penalty |
| M16 — the structural claim | ✅ | Request-rate versus token-work scaling, measured rather than asserted |
| M20 — read-only API | ✅ | Hardened FastAPI, `snapshot_mode` and `assumptions` on every response, ARGUS-shaped evidence export |
| M21 — dashboard | ✅ | Diagnostic, workloads, findings and explainer views over a snapshot baked at image build |
| M22 — explainer | ✅ | A "how it works" view where the limits are as prominent as the results |
| M23 — deploy | ✅ | Single-container Render deploy, demo snapshot generated at image build, deployment-image CI lane |

## Cut, and why

| Milestone | Decision | Reason |
|---|---|---|
| M9–M13 — specialist agents | **Cut** | The pre-registered cut order chose the GPU lane over the agent layer when the two competed for the same days. The agent layer's own pre-registered expectation (Q5) was that it would *not* improve decision quality, only auditability — so cutting it removes a likely-null result rather than a likely finding. The decision ledger it would have written is not claimed anywhere in the product. |
| M17 — foundation forecasters | **Partly superseded** | The intent was to test whether a bigger model changes the answer. That question got answered with four classical families instead (seasonal-naive, drift, ETS, LightGBM): the most sophisticated of them never won a single cell, because the binding constraint is matching the model to the horizon, not model capacity. Chronos-Bolt/TimesFM remain untested. |
| M18 — Alibaba spot-GPU | **Cut** | No licence file at the repository root; the zero-cost audit will not redistribute or depend on unclearly licensed data. |
| M19 — carbon-aware deferral | **Cut** | Elegant, not load-bearing. First on the pre-registered cut list. |
| M24 — live Wikimedia lane | **Cut** | Replay carries the argument; a live arrival process is presentation, not evidence. |

## What the evaluation actually concluded

The project set out to show that a calibrated forecast beats conventional autoscaling. What
it measured is narrower and more useful:

1. **In the autoscaling regime, forecasting does not beat the trailing-percentile
   recommender Kubernetes already ships.** Minute-scale autocorrelation is ~0.98 on every
   workload measured, so recent load already carries nearly all the signal and an explicit
   model mostly adds its own error.
2. **In the commitment regime it does win** — where capacity is fixed for hours and reaction
   is structurally unavailable — but only on demand with genuine daily structure.
3. **Which regime you are in is cheap to measure — on aggregated demand.** Daily
   autocorrelation calls 27 of 28 cells correctly at a six-hour commitment across seven
   fleet-aggregate workloads.
4. **Q13, answered: it does not generalise.** On 43 individual Azure Functions workloads the
   same cutoff scores 51% at six hours and 47% at twelve — a coin flip, and worse than
   ignoring the diagnostic entirely — with a non-monotone relationship that no version of
   the rule predicted. Only the bottom of the range transfers: below ~0.20, forecasting
   failed to pay on both populations.

The shipped artifact is therefore a **diagnostic that tells you whether to build a forecaster
at all**, scoped to aggregated demand and stating that scope on its own results page. That is
a considerably smaller claim than the one the project started with, and it is the one the
measurements support.

Bitbrains is not on the critical path because its canonical host could not be verified; the
Materna fleet traces and the synthetic multi-resource regime carry the joint-provisioning
test without redistributing unclearly licensed data.
