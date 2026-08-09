# Evaluation

DELPHI evaluates forecasts with chronological rolling origins. Split boundaries travel with
every result; normalization and MASE scaling use training history only. The controller-facing
selection metric will be newsvendor loss once M4–M6 exist, so forecast accuracy alone never
decides the shipped controller.

## M2 baseline protocol

- Deterministic hourly synthetic traces: 14 days train, 3 days validation, 4 days test.
- Four non-overlapping 24-step rolling origins in the test interval.
- Quantiles: p50, p80, p90, p95, p99.
- Regimes: clean daily seasonality, known bursts, level shift, and near-noise.
- MASE denominator: train-only daily seasonal naive errors.
- WQL and empirical coverage are measured on unseen test windows.
- ARIMA uses one recorded cohort-level order, `(2,0,2)(1,1,1)[24]`, rather than spending a
  different automatic-search budget on each workload.

Run `make evaluate` to reproduce the table below. These are raw, uncalibrated M2 outputs;
coverage is expected to miss nominal levels until M3 adds and evaluates conformal calibration.

### Labelled synthetic regimes

| Regime | Model | MASE | WQL | Mean pinball | p95 coverage |
|---|---|---:|---:|---:|---:|
| clean_daily | `seasonal_naive_d:period=24:v1` | 1.002 | 0.040 | 1.034 | 0.958 |
| clean_daily | `seasonal_naive_w:period=168:v1` | 0.593 | 0.025 | 0.660 | 0.938 |
| clean_daily | `drift_naive:residual_window=168:v1` | 5.335 | 0.252 | 6.568 | 0.562 |
| clean_daily | `rolling_percentile:window=168:v1` | 3.461 | 0.121 | 3.156 | 1.000 |
| clean_daily | `statsforecast:auto_ets:season=24:v1` | 1.023 | 0.039 | 1.011 | 0.969 |
| clean_daily | `statsforecast:arima:season=24:v1` | 0.820 | 0.033 | 0.856 | 0.896 |
| clean_daily | `lightgbm_quantile:trees=80:leaves=15:seed=20260802:v1` | 0.656 | 0.038 | 1.000 | 0.948 |
| burst_known | `seasonal_naive_d:period=24:v1` | 0.687 | 0.044 | 1.143 | 1.000 |
| burst_known | `seasonal_naive_w:period=168:v1` | 0.417 | 0.027 | 0.702 | 0.958 |
| burst_known | `drift_naive:residual_window=168:v1` | 3.770 | 0.253 | 6.589 | 0.562 |
| burst_known | `rolling_percentile:window=168:v1` | 2.496 | 0.126 | 3.294 | 1.000 |
| burst_known | `statsforecast:auto_ets:season=24:v1` | 3.689 | 0.196 | 5.109 | 0.750 |
| burst_known | `statsforecast:arima:season=24:v1` | 0.621 | 0.052 | 1.354 | 1.000 |
| burst_known | `lightgbm_quantile:trees=80:leaves=15:seed=20260802:v1` | 0.466 | 0.048 | 1.255 | 0.958 |
| level_shift | `seasonal_naive_d:period=24:v1` | 0.628 | 0.032 | 1.373 | 1.000 |
| level_shift | `seasonal_naive_w:period=168:v1` | 3.966 | 0.093 | 4.040 | 0.906 |
| level_shift | `drift_naive:residual_window=168:v1` | 3.137 | 0.137 | 5.978 | 0.594 |
| level_shift | `rolling_percentile:window=168:v1` | 2.242 | 0.074 | 3.243 | 1.000 |
| level_shift | `statsforecast:auto_ets:season=24:v1` | 3.209 | 0.129 | 5.635 | 0.667 |
| level_shift | `statsforecast:arima:season=24:v1` | 0.484 | 0.023 | 1.000 | 1.000 |
| level_shift | `lightgbm_quantile:trees=80:leaves=15:seed=20260802:v1` | 0.881 | 0.035 | 1.510 | 0.979 |
| near_noise | `seasonal_naive_d:period=24:v1` | 0.887 | 0.073 | 2.048 | 0.958 |
| near_noise | `seasonal_naive_w:period=168:v1` | 0.915 | 0.081 | 2.261 | 0.906 |
| near_noise | `drift_naive:residual_window=168:v1` | 1.096 | 0.147 | 4.103 | 1.000 |
| near_noise | `rolling_percentile:window=168:v1` | 0.708 | 0.057 | 1.592 | 0.927 |
| near_noise | `statsforecast:auto_ets:season=24:v1` | 0.702 | 0.057 | 1.589 | 0.927 |
| near_noise | `statsforecast:arima:season=24:v1` | 0.723 | 0.059 | 1.642 | 0.938 |
| near_noise | `lightgbm_quantile:trees=80:leaves=15:seed=20260802:v1` | 0.724 | 0.063 | 1.769 | 0.833 |

### Azure Functions 2019 real-trace check

The real check uses the seeded top-1 workload, hourly means, ten days train, two days validation,
and two days test. It is a smoke comparison, not a claim that one busiest function represents
the full cohort. Reproduce it with `python scripts/evaluate_azure_baselines.py` after fetching
the archive.

| Regime | Model | MASE | WQL | Mean pinball | p95 coverage |
|---|---|---:|---:|---:|---:|
| azure_top1_hourly | `seasonal_naive_d:period=24:v1` | 0.926 | 0.043 | 1783.057 | 1.000 |
| azure_top1_hourly | `seasonal_naive_w:period=168:v1` | 0.330 | 0.020 | 820.859 | 0.979 |
| azure_top1_hourly | `drift_naive:residual_window=168:v1` | 0.711 | 0.039 | 1611.224 | 0.979 |
| azure_top1_hourly | `rolling_percentile:window=168:v1` | 0.453 | 0.025 | 1045.728 | 1.000 |
| azure_top1_hourly | `statsforecast:auto_ets:season=24:v1` | 0.617 | 0.044 | 1813.969 | 1.000 |
| azure_top1_hourly | `statsforecast:arima:season=24:v1` | 0.511 | 0.031 | 1271.995 | 1.000 |
| azure_top1_hourly | `lightgbm_quantile:trees=80:leaves=15:seed=20260802:v1` | 0.461 | 0.022 | 916.419 | 1.000 |

The weekly seasonal baseline is strongest on real-trace MASE (`0.330`), while LightGBM has the
best native-quantile WQL (`0.022`). Near-universal p95 over-coverage on the real slice and severe
under-coverage in some synthetic regimes show why raw intervals cannot be treated as calibrated.

## M3 calibration under drift

This experiment starts the test interval at a labelled synthetic level shift. Gamma is selected
once on the preceding validation stream from `{0.001, 0.005, 0.01, 0.02, 0.05}`; it is not tuned
on the test trace. Recovery is the first point at which trailing 24-hour coverage returns within
five percentage points of nominal p95. Run `make evaluate-calibration` to reproduce it.

**This answers Q2** — *does conformal calibration beat a fixed safety margin?* Registered
expectation: yes under drift, roughly neutral on stationary traces. **Verdict: confirmed.**
The fixed margin is the honest comparator and for some time this experiment did not include
one, measuring conformal only against an uncalibrated forecast — which is a strawman. Two are
now reported: the margin tuned on validation to hit nominal, and the conventional +15% that
Kubernetes VPA ships and that this repo's own `PercentileRecommender` defaults to.

| Method | p95 coverage | Mean p95 | Recovery steps (24 h window) |
|---|---:|---:|---:|
| `raw` | 0.906 | 98.751 | 46 |
| `fixed_margin_tuned_+0%` | 0.906 | 98.751 | 46 |
| `fixed_margin_conventional_+15%` | 0.917 | 113.563 | 46 |
| `split_conformal` | 0.903 | 98.505 | 46 |
| `aci_gamma_0.05` | **0.924** | **101.976** | **28** |

**ACI wins on every axis that matters.** Against the conventional +15% margin it holds better
coverage (0.924 versus 0.917) while provisioning **11% less capacity** (101.98 versus 113.56),
and it recovers from the shift in 28 steps where the margin never recovers inside the window
at all — a fixed margin cannot recover, because it does not respond to anything.

**The tuned margin is the more interesting result.** Raw p95 already covers 96.9% on the
stationary validation stream, so the smallest margin reaching the 95% target is **+0%**: an
operator tuning headroom honestly on pre-shift data would add none, and would then be
completely unprotected when the level moved. Buying safety by tuning a constant on quiet data
is not conservative, it only looks conservative.

Static split conformal does not survive the distribution shift in this slice either: its
validation correction slightly lowers test coverage. ACI remains below nominal over the full
transient but cuts recovery time by 18 steps. Coverage is retained observation-by-observation,
not only as the scalar summaries above.

---

## M4 simulator gate

Nothing downstream is trustworthy unless this passes, so it is a gate, not a task. Run
`make validate-simulator`; the script prints its own verdict.

| Check | Result |
|---|---|
| **Analytic** — event-driven queue vs the Erlang-C closed form | **PASS**, ≤2.2% relative error across c = 1–10, ρ = 0.5–0.85 |
| **Degenerate** — 5 cases | **PASS** |
| **Determinism** — byte-identical repeat runs | **PASS** |
| **Sensitivity** — ranking under 6 perturbed constants | **PASS**, 6/6 |

The sensitivity sweep immediately caught a real defect: the actuation delay was being
counted twice, once by controllers shifting their own forecast origin and again by the
simulator delaying the emitted plan. The tell was a flat, erratic margin in the lead-time
table. After the fix, proactive violations fell at every lead (0.127 → 0.028 at one step).

Every simplification in the simulator — the utilisation model, a modelled service-time
distribution, a modelled cold start — flatters the proactive methods. The sweep bounds
this; it cannot remove it.

## The assumption that governs every controller number

> A trace records what happened under the original operator's policy. Replaying it against
> a different policy is a counterfactual, valid only under an **open-loop assumption**:
> that the arrival process does not depend on the capacity decision.

For Azure Functions this holds cleanly — external users trigger functions, and our
provisioning cannot change how many events arrived. It would not hold for a batch trace
with queueing feedback, load shedding or retry amplification, and no such trace is used.

**Rankings transfer even when absolute numbers do not.** Frontier shapes and orderings
below are stated with confidence; dollar figures are illustrative.

### Controller-lane data and prices

Cohort: busiest 8 functions of day 1 by total invocations, seed `20260802`, then any
workload averaging under 1 invocation/minute dropped, then **5-minute bins** (declared, not
silent — per-minute replica decisions here are dominated by integer rounding).

**Most of Azure Functions is not a capacity-planning problem.** A decile-stratified sample
of 16 workloads returned 2 busy functions and 14 near-dead ones, several with 7 of 14 days
entirely absent (flagged `is_imputed`, never forward-filled). For those the whole answer is
scale-to-zero plus a cold-start budget. They are excluded by the rule above, and the
exclusion is itself the finding.

Price: **$0.0416/replica-hour** — Azure B2s, `southeastasia`, retail list, retrieved
2026-08-03 from the keyless Retail Prices API. List price is not what an enterprise pays,
so the *shape* of the frontier is the result, not the dollars.

**Tuning parity:** every controller swept over the same 7 risk settings (0.50, 0.70, 0.80,
0.90, 0.95, 0.98, 0.99). Our re-implementations of HPA and VPA/Autopilot are faithful to
the published control laws, not to the production systems — the same caveat BACC records.

## M8 pre-registered questions

### Q1 — Does forecasting beat reaction, and does the margin grow with lead time?

> ⚠️ **SUPERSEDED — the table below was produced by a simulator with an actuation-delay bug
> and every number in it is wrong.** The corrected measurement, and the reason the original
> conclusion inverted, are in
> [Correction: the actuation-delay bug (2026-08-08)](#correction-the-actuation-delay-bug-and-what-it-changed-2026-08-08).
> Retained so the correction can be checked against what it replaced.

**Partially confirmed, and the qualifier matters.**

| startup (steps) | reactive viol. | proactive viol. | margin | relative reduction |
|---:|---:|---:|---:|---:|
| 0 | 0.195 | 0.005 | +0.190 | 97.4% |
| 1 | 0.345 | 0.028 | +0.317 | 91.8% |
| 2 | 0.452 | 0.195 | +0.257 | 56.8% |
| 4 | 0.737 | 0.535 | +0.202 | 27.4% |
| 8 | 0.933 | 0.692 | +0.242 | 25.9% |

The absolute margin is larger at 8 steps than at 0, so the registered check passes — but it
is **not monotone**, and past ~2 steps of lead the reactive baseline exceeds 50% violations,
a **saturated regime where both controllers mostly fail**. The relative reduction is the
more informative reading and falls cleanly from 97.4% to 25.9%: forecasting helps most when
the lead is short enough that the forecast is still good. Read the ranking, not the levels.

### Q4 — Does a price-derived target beat a conventional fixed one?

**A qualified yes, with a real boundary — and a correction to the experiment's design.**

Sweeping C6's `q*` over the same grid as C5's compliance target makes the two arms
**identical by construction**: `q*` *is* the setting, so C6 reduces exactly to C5 and the
frontier tables show them producing byte-identical numbers. That is not a defect. **The
newsvendor contribution is not a different control law — it is that the setting is derived
from prices rather than chosen by convention.** The meaningful experiment therefore holds
the industry-default P95 fixed and varies the true cost ratio.

Synthetic `clean_daily`, total economic cost (capacity bill plus breaches priced at the
operator's own `C_u`), relative to the P95 default:

| C_u / C_o | derived q* | violations | capacity cost | total vs P95 default |
|---:|---:|---:|---:|---:|
| 1 | 0.500 | 0.2346 | 56.8 | **−5.00** |
| 3 | 0.750 | 0.1076 | 59.1 | **−2.82** |
| 9 | 0.900 | 0.0551 | 61.0 | **−1.20** |
| 19 | 0.950 | 0.0397 | 62.4 | +0.00 |
| 49 | 0.980 | 0.0293 | 64.5 | +2.72 |
| 99 | 0.990 | 0.0238 | 65.3 | +3.58 |

The **+0.00 at `C_u/C_o = 19`** is a consistency check, not a result: 19 is exactly the
ratio P95 implicitly asserts (`0.95/0.05`), so the two targets coincide there and must
agree. They do.

**Below 19 the default over-buys and the derived target wins** — up to $5.00 over the
scored window when failure and idle capacity cost the same. That region is real:
opportunistic, batch and spot-tier work all sit in it.

**Above 19 the derived target does not win in realised terms.** This is a negative against
our own hypothesis and is recorded as such. The likely mechanism is calibration, not the
identity: the newsvendor optimum is optimal *given the true demand distribution*, and deep
in the upper tail a seasonal-naive forecast's p98/p99 are its least trustworthy part.
Sizing off a miscalibrated tail buys capacity that does not buy down enough violations to
pay for itself. This is precisely why calibration rather than accuracy is the forecast
metric, and it makes foundation-model calibration the highest-value follow-up.

### Q10 — Is there a trace where no controller beats static provisioning?

**Yes, and it is a real one.** On Azure Functions workload 1, static at 6 replicas achieves
a 0.3% violation rate for $41.9, and the cheapest point on the whole frontier is static at
$34.9. The VPA/Autopilot-style percentile recommender dominates most of the rest.

**Neither calibrated controller reaches ≤1% violations on this workload at all.**

| family | cost @ ≤10% viol. | cost @ ≤5% viol. | cost @ ≤1% viol. |
|---|---:|---:|---:|
| C0 static | 41.9 | 41.9 | 41.9 |
| C1 reactive | 36.4 | 41.9 | 41.9 |
| **C2 percentile** | 37.2 | **39.7** | **40.5** |
| C3 proactive | 37.1 | 40.4 | 42.4 |
| C5 budget-paced | 39.2 | 39.2 | never reached |
| C6 newsvendor | 39.2 | 39.2 | never reached |

The reason is in the data, not mysterious: this function's demand is extremely stable
(mean ≈ 15,244 invocations/min, p95 ≈ 18,092), the frontier's dynamic range is only 23.8%
of its cheapest point, and there is very little for a forecaster to add. **A stable workload
does not need prediction, and reporting otherwise would be dishonest.**

## What didn't work

- **The newsvendor target does not beat the P95 default at high cost ratios** (Q4) — a
  negative against the core claim, with a probable cause in upper-tail calibration.
- **Neither calibrated controller reaches ≤1% violations on real Azure workload 1** (Q10),
  where a simple percentile recommender does.
- **Static provisioning is the cheapest point on that real frontier.**
- **Most of the Azure Functions population is unevaluable** for capacity planning.
- **`LightGBMQuantileForecaster` bands do not widen with horizon** (p99−p50 = 31.95 at step
  1, 30.12 at step 48): recursive one-step quantile models under-state long-horizon
  uncertainty. Per-horizon conformal calibration mitigates it; a direct multi-horizon model
  is the real fix and is not built.
- **The Azure trace's calendar anchor is a convention, not a fact.** Measured daily totals
  show no weekly dip and the two highest days are the ones the anchor calls Saturday and
  Sunday. Relative day/week lags are unaffected; `is_weekend` is unreliable and marked so.

## What would change these conclusions

- **A real `C_u`.** Every Q4 number is a sweep because the price of a breach is a business
  judgement nobody can measure. An operator with a real figure gets a point, not a curve.
- **A volatile workload.** Q10's negative comes from an unusually stable function. The claim
  that forecasting pays needs a bursty trace — the GPU/inference lane, where cold starts run
  to minutes, is the intended venue.
- **A calibrated upper tail**, which would settle Q4's high-ratio negative.
- **Queue-fidelity results.** Everything above uses the utilisation model. The event-driven
  queue is built and validated but has not yet been used for a controller comparison.

---

# GPU / LLM-inference lane (M14–M16)

Data: **Azure LLM inference traces 2024** — `code` (16.8M requests, 7 days from 2024-05-10)
and `conv` (27.3M requests, 7 days from 2024-05-12). **CC-BY 4.0**; cite Stojkovic et al.,
*DynamoLLM*, HPCA 2025. SHA-256 of both archives recorded in `docs/DATA.md`. Timestamps are
absolute UTC, so unlike the Azure Functions trace there is no calendar anchor to assume.

**Serving rates are modelled parameters, not measurements.** These traces publish tokens,
not hardware timings. A replica is modelled at 12,000 prefill tok/s, 2,500 decode tok/s,
240 s cold start, 85% utilisation ceiling. Every conclusion below is read from *rankings*
and from the cold-start sweep, never from absolute GPU-hour figures.

The capacity currency is **GPU-seconds of work per bin** — `prefill_tokens/prefill_rate +
decode_tokens/decode_rate`, the two phases contending for one device. A replica supplies
exactly `bin_seconds` of them, which lets the already-validated replay simulator score this
lane unchanged rather than introducing a second, unvalidated simulator.

## What the traces actually demand

| | `code` | `conv` |
|---|---:|---:|
| requests/min, mean → p95 | 1,667 → 4,625 (**2.8×**) | 2,709 → 4,024 (1.5×) |
| GPU-work/min, mean → p95 | 364 → 1,044 GPU-s (**2.9×**) | 483 → 699 GPU-s (1.4×) |
| phase split | **prefill 95.8% / decode 4.2%** | prefill 76.3% / decode 23.7% |
| context tokens/request, CV | 0.125 | 0.063 |
| generated tokens/request, CV | 0.177 | 0.081 |

`code` has **fewer** requests than `conv` but similar prefill load and 7.5× less decode
load — the raw prefill:decode token ratio is 110.7:1 versus 15.5:1. Two workloads on the
same hardware with completely different capacity shapes.

## M16 — is request-rate autoscaling structurally wrong here? (**this answers Q6**)

> **Q6, pre-registered:** *Is CPU-threshold scaling structurally wrong for GPU inference?*
> Registered expectation: yes, large, and widening with `startup_seconds`. **Verdict:
> confirmed.** The measurement is below; it was carried out under its milestone number and
> went for some time without being tied back to the question it answers, which is a
> bookkeeping failure in a project whose central discipline is exactly that ledger.

**Yes, and it is measurable.** Every proxy below is first rescaled to the true demand's
mean, so what remains is error in *shape*: a proxy that merely needed a different constant
would be a tuning problem, not a structural one.

| tracked signal | trace | correlation | mean abs rel. err | p95 abs rel. err | worst under-provision |
|---|---|---:|---:|---:|---:|
| `requests` | code | 0.9961 | 9.4% | 26.9% | **−39.2%** |
| `requests` | conv | 0.9749 | 5.4% | 11.0% | −15.1% |
| `prefill` | code | 1.0000 | 0.7% | 2.1% | −9.6% |
| `decode` | code | 0.9858 | 16.6% | 48.8% | **−57.3%** |
| `total_tokens` | code | 1.0000 | 0.6% | 1.7% | −7.5% |

**Correlation is a trap here.** Request count correlates with true GPU work at 0.9961 on
the `code` trace and still under-provisions by up to 39% in individual bins. A dashboard
showing r ≈ 0.996 would look like a solved problem; the SLO breaches happen in the tail.

Tracking the right signal, at essentially identical cost:

| controller | trace | violations on request count | violations on GPU work | GPU-hours |
|---|---|---:|---:|---:|
| reactive HPA | `code` | 0.1885 | **0.1521** (−19%) | 982 vs 997 |
| reactive HPA | `conv` | 0.1819 | **0.1299** (−29%) | 985 vs 980 |

**Caveat, stated rather than buried:** the effect is clear for the reactive controller but
*reverses slightly* for the proactive family on `code` (0.1260 on request count vs 0.1323
on GPU work). On a bursty trace the forecast error dominates the signal error, so this is
not a universal win and should not be reported as one.

## Q1 revisited — cold start is the mechanism

> ⚠️ **SUPERSEDED — both tables in this section predate the actuation-delay fix.** The `code`
> result in particular *inverted*: forecasting does help there, and the margin grows
> monotonically with cold start. See
> [Correction: the actuation-delay bug (2026-08-08)](#correction-the-actuation-delay-bug-and-what-it-changed-2026-08-08).

The synthetic result for Q1 was non-monotone and saturated. On real inference demand it
separates cleanly, and the two traces disagree in an informative way.

**`conv` — monotone confirmation, the cleanest evidence in the project:**

| startup | reactive viol. | proactive viol. | margin | relative reduction |
|---:|---:|---:|---:|---:|
| 0 s | 0.0922 | 0.0000 | +0.0922 | 100.0% |
| 60 s | 0.1017 | 0.0000 | +0.1017 | 100.0% |
| 240 s | 0.1299 | 0.0000 | +0.1299 | 100.0% |
| 600 s | 0.2033 | 0.0363 | +0.1670 | 82.2% |
| 1200 s | 0.3003 | 0.0694 | **+0.2309** | 76.9% |

The absolute margin grows **monotonically** with cold-start time, exactly as registered.
This is the venue Q10 said the claim needed: minutes-long cold starts on demand with
exploitable daily structure.

**`code` — forecasting does not help, at any lead time:**

| startup | reactive viol. | proactive viol. | margin |
|---:|---:|---:|---:|
| 0 s | 0.0887 | 0.1193 | −0.0306 |
| 60 s | 0.1073 | 0.1266 | −0.0193 |
| 240 s | 0.1521 | 0.1323 | +0.0198 |
| 600 s | 0.2870 | 0.3498 | −0.0628 |
| 1200 s | 0.4122 | 0.4335 | −0.0214 |

**A negative, and an informative one.** `code` is nearly twice as bursty as `conv`
(peak-to-mean 2.9× versus 1.4×), and its spikes are not recoverable from yesterday's
same-time value. Day-seasonal forecasting has nothing to exploit, so no amount of lead time
rescues it. **Burstiness defeats forecast-driven capacity control regardless of actuation
delay** — which is precisely the open challenge the workload-forecasting survey names, and
it is visible here rather than argued.

Taken together: forecasting pays when demand has exploitable structure, and *how much* it
pays scales with actuation delay. Both conditions are needed. Neither alone is enough.

## Methodology bug found and fixed

The first run of this lane showed proactive control losing to reactive at **every** lead
time on both traces. The cause was our own configuration, not the method: `refit_stride`
was set to one full season, so the controller was acting on **day-old forecasts**. Refitting
every 4 hours instead moved `code` from 0.2391 to 0.1323 violations.

**Refit cadence is a first-class tuning parameter and was being silently mis-set.** It is
now a named constant with the measurement recorded beside it. The general lesson: a
"forecasting doesn't help" result should always be checked against forecast *freshness*
before it is believed.

## Added to the negatives ledger

- **Forecasting does not help on the bursty `code` trace at any cold-start time.**
- **The right-signal advantage reverses for proactive controllers on `code`** — forecast
  error can dominate signal error.
- **Correlation of 0.9961 coexists with 39% under-provisioning** in the tail. Correlation is
  the wrong diagnostic for a capacity signal.

---

# Does forecasting help at all? (2026-08-04)

The honest answer, and the reason the project's claim changed.

## The wrong benchmark

Every earlier cold-start sweep compared proactive control against **reactive HPA**. That is
a strawman: everyone already knows threshold-reactive scaling is bad. The real incumbent is
the sliding-window percentile recommender that Borg Autopilot and Kubernetes VPA ship.

Putting every family on one frontier, with the incumbent given its own tuning dimension
(4 window lengths × 7 percentiles = 28 configurations, versus 7 for each other arm):

| trace | cold start | forecast points in usable region (≤10% viol.) | beats percentile @ ≤5%? |
|---|---:|---:|:--:|
| `code` | 60 s | 0 | **no** |
| `code` | 240 s | 0 | **no** |
| `code` | 600 s | 0 | **no** |
| `code` | 1200 s | 0 | **no** |
| `conv` | 60 s | 2 | **no** |
| `conv` | 240 s | 1 | **no** |
| `conv` | 600 s | 1 | **no** |
| `conv` | 1200 s | 0 | **no** |

**0 of 8.** And the gap *widens* with cold start — on `conv`, 12.0% → 17.5% → 18.8% → 23.7%
more expensive at a matched ≤5% violation rate — the opposite of the registered prediction.
On `code` the gap is far larger still: 68.0% at 60 s and 62.5% at 240 s.

> **Re-verified after the actuation-delay fix (2026-08-08).** Every controller in this
> experiment re-plans at every step, so all of them were affected by the bug and this table
> had to be re-measured. The verdict is unchanged at **0 of 8** — the counts of forecast
> points in the usable region moved only from 0 to 1 in two cells — and the widening gap is
> now cleanly *monotone*, where the original numbers turned over at the last step
> (27.2% → 24.1%). The project's central negative result is robust to the bug.

A methodological note on the verdict metric: counting raw Pareto points initially reported
"YES" in every setting, because a controller that is very cheap and very unsafe is
non-dominated simply by being more reckless than anything else. `newsvendor` appeared on
the frontier at a **0.5686** violation rate. The metric was changed to count only the
operationally usable region and to headline cost at a matched violation rate.

## Why — and it is not subtle

| trace | horizon | seasonal-naive (day-ago) MASE | trailing-mean MASE | winner |
|---|---:|---:|---:|---|
| `code` | 4 bins | 5.807 | **2.129** | trailing |
| `code` | 20 bins | 5.812 | **2.737** | trailing |
| `conv` | 4 bins | 3.637 | **1.303** | trailing |
| `conv` | 20 bins | 3.632 | **1.566** | trailing |

Autocorrelation of GPU work:

| trace | lag 1 min | lag 10 min | lag 1 h | lag 1 day |
|---|---:|---:|---:|---:|
| `code` | 0.989 | 0.981 | 0.933 | 0.730 |
| `conv` | 0.961 | 0.940 | 0.866 | **0.349** |

**Recent load carries almost all the signal; same-time-yesterday carries much less.** Every
GPU experiment had used `SeasonalNaiveForecaster(1440)`, which throws the strong signal
away. The percentile recommender wins because **it is itself a short-range forecaster**.

Retested with three predictors that do use recent load — persistence, drift, and the
seasonal model — the incumbent still won at every setting. **The conclusion is robust
across four forecasters, so it is not an artefact of one bad model choice.**

## What survives: the decision layer, not the forecasting layer

A percentile recommender is a forecaster. What it has never had is a principled way to
choose *which* percentile — VPA ships p95 by convention. That is exactly the gap the
newsvendor identity fills, and it is orthogonal to which predictor sits underneath.

**C7** keeps the winning predictor and replaces only the arbitrary part: `q*` sets the
percentile, ACI keeps it honest. Against the p95 convention, with margins matched and
total economic cost priced at the operator's own `C_u`:

| trace / cold start | C7 wins | ties | losses |
|---|---:|---:|---:|
| `code` @ 240 s | 6 | 0 | 0 |
| `code` @ 1200 s | 5 | 1 | 0 |
| `conv` @ 240 s | 6 | 0 | 0 |
| `conv` @ 1200 s | 2 | 0 | **4** |

**19 wins, 1 tie, 4 losses of 24.** Largest gain −$2,152 (`code` @ 240 s, `C_u/C_o` = 99);
largest loss +$593 (`conv` @ 1200 s, same ratio). The losses cluster where ACI's correction
hurts rather than helps.

**A second instrument artefact, caught before publication.** The first version of this
comparison had C7 at `margin=0.0` against a convention baseline at `PercentileRecommender`'s
default `margin=0.15`. C7 was simply buying 15% less capacity, and the "wins" were largely
that. A regression test now asserts that C7 with calibration disabled is **byte-identical**
to the plain recommender, so the ablation cannot silently drift again.

Decomposing at `C_u/C_o = 19`, where `q*` equals the 0.95 convention exactly and any
difference must be ACI alone: −$438, +$1, −$23, +$176 across the four settings. **ACI is
roughly neutral; the value is in the derived target, not the calibration.**

## The answer

**Forecasting, in the sense of bolting an explicit predictive model onto capacity control,
does not help on these traces.** A trailing percentile is already a near-optimal predictor
for load with 0.98 minute-scale autocorrelation, and an explicit model mostly adds its own
error.

**Deriving the target from prices does help** — 19 of 24 settings, and the mechanism is
understood rather than assumed.

So the project's claim narrows and sharpens: DELPHI is not "forecasting for capacity". It
is **a decision layer that tells you which quantile to buy, on top of whatever predictor
your data actually rewards** — which on these traces is the boring one already shipping in
Kubernetes.

## Correction: the claim was scoped too broadly (2026-08-04)

The section above concluded that forecasting does not help. That is true **only at
autoscaling lead times**, and stating it without that qualifier was an overclaim. Measured
across lead times on 1-minute bins:

| trace | 5 min | 30 min | 2 h | 6 h | 12 h | 24 h |
|---|---:|---:|---:|---:|---:|---:|
| `code` trailing | **2.419** | **3.514** | 7.267 | 15.167 | 18.878 | 5.012 |
| `code` day-ago | 4.756 | 4.756 | **4.756** | **4.756** | **4.756** | **4.756** |
| `conv` trailing | **1.304** | **1.702** | 2.895 | 3.247 | 4.395 | 2.758 |
| `conv` day-ago | 2.734 | 2.734 | **2.734** | **2.734** | **2.734** | **2.734** |

At a 24-hour planning horizon on hourly bins, seasonality wins outright:

| trace | trailing-24 h | day-ago | winner |
|---|---:|---:|---|
| `code` | 8.909 | **2.877** | day-ago, ~3× better |
| `conv` | 3.090 | **1.698** | day-ago, ~2× better |

**The crossover is at roughly 1–2 hours of lead time**, which is where minute-scale
persistence (autocorrelation 0.989) has decayed far enough that daily structure
(0.730 / 0.349) becomes the better signal.

Every GPU controller experiment used a lead of 4–20 minutes — **entirely inside the
trailing-window regime**. So the correct statement is:

> At autoscaling horizons (minutes), explicit forecasting adds nothing a trailing
> percentile does not already capture. At capacity-planning horizons (hours to a day),
> a trailing window is structurally useless and seasonal forecasting is 2–3× better.

Two caveats on the table above. The dip in trailing MASE at 24 h is not an anomaly: the
lagged window there ends 24 hours back, so it converges to same-time-yesterday by
construction. And **weekly seasonality could not be tested at all** — the traces are 7 days
long, so there is never a full week of history to look back on.

**Consequence for the project:** the newsvendor decision layer is horizon-independent and
still stands. But the forecasting layer has an untested regime where it should win, and the
GPU lane was built at the one lead time where it could not. Testing capacity planning —
hourly bins, day-ahead commitment, reserved-capacity pricing — is the outstanding work.

---

# Week-ahead: the horizon the earlier traces could not reach (2026-08-04)

The Azure traces are 7 days long, so week-ahead forecasting was not merely untested — it was
**untestable**: no held-out week exists and weekly seasonality cannot be learned from one
instance of it. That gap is now closed with a longer trace.

**Data: Bitbrains GWA-T-12 `rnd`** — 500 enterprise VMs (banks, insurers, credit-card
operators), **three consecutive months, 2013-06-30 to 2013-09-29, 91 days = 13.0 weeks**,
5-minute resolution, 284 MB. Aggregated to one fleet-level CPU demand signal, because the
capacity-planning question is "how much does the estate need", not "what will VM 417 do".

Provenance caveat carried in the source record: the canonical Grid Workloads Archive host has
been unreachable since 2026-08-02, so this comes from the @Large mirror and **the terms-of-use
page could not be read**. SHA-256 recorded; data never redistributed.

Estate size varies (median 501 VMs, range 0–549). Bins where the VM count collapses are
flagged `is_imputed` with zero quality — **an estate that shrank is not an estate that
idled** — leaving 25,216 of 26,208 bins usable.

## The workload has almost no exploitable long-range structure

| lag | 5 min | 1 h | 6 h | 1 day | 3 days | 1 week | 2 weeks |
|---|---:|---:|---:|---:|---:|---:|---:|
| autocorrelation | 0.985 | 0.868 | 0.546 | **0.248** | 0.204 | **0.271** | 0.169 |

Week-lag autocorrelation (0.271) is barely above day-lag (0.248), and both are weak.

## Week-ahead: nothing beats a flat average

Mean absolute error as a percentage of mean demand, one week ahead:

| predictor | error |
|---|---:|
| week-ago | 59.8% |
| trailing window | 58.5% |
| day-ago | 56.1% |
| **flat mean of the prior 4 weeks** | **53.1%** |

**The structureless predictor wins.** Rolling-origin over 8 held-out weeks confirms it:
week-ago wins 2, trailing 3, day-ago 3 — a coin flip, with MASE swinging from 6.4 to 18.6
between weeks.

## The generalisable finding

Contrast the two workloads measured in this project:

| workload | autocorrelation @ 1 day |
|---|---:|
| Azure LLM `code` (GPU work) | **0.737** |
| Azure LLM `conv` (GPU work) | 0.363 |
| Bitbrains `rnd` (fleet CPU) | **0.248** |

**Predictability is a property of the workload, not of the method.** The Azure inference
traces have a strong daily rhythm — developer and business hours are visible in the signal.
An aggregate of 500 heterogeneous enterprise VMs does not: individual rhythms wash out in
the sum, and what remains is close to a level plus noise.

So the complete answer across all three horizons tested:

| horizon | what wins | why |
|---|---|---|
| minutes | trailing percentile | persistence r ≈ 0.98; a heuristic already extracts it |
| hours to a day | seasonal, **where daily structure exists** | persistence has decayed; Azure LLM has r = 0.74 at a day |
| **one week** | **a flat average** | no predictor beat 53.1% error on the one fleet long enough to test |

**Consequence for the project.** The right first question is not "which forecaster" but
**"is this workload predictable at my lead time at all"** — and that is answerable in
seconds from an autocorrelation profile, before any model is built. That diagnostic is worth
more to an operator than another controller, and DELPHI should ship it.

Two limits on this section, stated rather than buried. It is **one fleet**, and a
13-week aggregate of business-critical VMs is not representative of all clusters. And MASE
here is measured against a 5-minute-ahead naive benchmark, which is a demanding denominator
for a week-ahead question — the honest comparison is *between* the predictors, where they
are within a few percent of each other and of a flat line.

---

# The commitment regime: where forecasting finally wins (2026-08-05)

Three earlier rounds found forecasting losing — at minute autoscaling horizons, and at a
week-ahead horizon. Both were tested in regimes where a controller can *react*, or where no
structure exists. This closes the remaining cell.

**A commitment cannot react.** Reserved instances, cluster sizing and procurement fix a
capacity level for hours, so the decision must cover the *entire* coming window including
its peak. A backward window can only assume the next window resembles the last; a forecast
can know a daily ramp falls inside it.

Both controllers perform the **same arithmetic** — the `q`-quantile of a window of demand
values — differing only in whether that window is observed or predicted. Keeping the
operation identical is what makes the comparison about information rather than about two
different sizing rules. A regression test asserts they agree exactly on a strictly periodic
series, where yesterday's window *is* the coming one.

Run through the validated simulator with prices, churn and a 240 s actuation delay.

## Result: two conditions, both necessary

Times forward dominated backward (cheaper **and** fewer violations) at the same quantile,
out of 7 settings:

| workload | daily autocorrelation | 1 h | 2 h | 6 h | 12 h |
|---|---:|---:|---:|---:|---:|
| Azure LLM `code` | **0.730** | 0/7 | 0/7 | **5/7** | **6/7** |
| Azure LLM `conv` | 0.349 | 0/7 | 0/7 | 0/7 | 0/7 |
| Bitbrains `rnd` fleet | 0.248 | 0/7 | 0/7 | 0/7 | 0/7 |

**Forecasting pays only where daily structure is strong *and* the commitment is long enough
that reaction is impossible.** Either condition alone is insufficient: `code` at 1–2 h has
the structure but can still react; `conv` and Bitbrains have long windows but nothing to
forecast.

Magnitudes on `code`, paired at identical settings:

| window | q | backward viol / cost | forward viol / cost | change |
|---|---:|---|---|---|
| 6 h | 0.95 | 0.3307 / $4,171 | 0.2849 / $3,684 | −14% violations, −12% cost |
| 6 h | 0.99 | 0.2809 / $4,478 | 0.2380 / $4,010 | −15% violations, −10% cost |
| 12 h | 0.90 | 0.4747 / $3,991 | 0.2564 / $3,786 | **−46% violations, −5% cost** |
| 12 h | 0.95 | 0.4694 / $4,154 | 0.2377 / $3,949 | **−49% violations, −5% cost** |
| 12 h | 0.99 | 0.4401 / $4,440 | 0.2161 / $4,235 | **−51% violations, −5% cost** |

At a 12-hour commitment the forecast **halves the violation rate while costing less**. This
is the first unambiguous, simulator-confirmed win for forecasting in the project.

Note the one consistent exception: at `q = 0.50` forward loses on violations at both
windows. Sizing a held commitment to a median is under-provisioning by construction, and a
better prediction of the median does not rescue that.

## Caveat: the absolute violation rates are high

At 6–12 hour commitments both families violate 20–50% of bins on `code`, because a single
held level cannot cover a signal with a ~2.9× peak-to-mean ratio without massive
over-buying. **Read the paired comparison, not the levels.** A real deployment would pair a
commitment with a small reactive tier for the peaks — which is exactly how reserved-plus-
on-demand purchasing works, and is not modelled here.

## The complete picture across every horizon tested

| horizon | regime | winner | why |
|---|---|---|---|
| minutes | autoscaling | trailing percentile | persistence r ≈ 0.98; a heuristic already extracts it |
| 1–2 h | commitment | trailing percentile | short enough that reaction still covers the error |
| **6–12 h** | **commitment** | **forecasting, dominant** | must cover a window you cannot react within |
| 1 week | commitment | flat average | no structure left to exploit |

**And the gate on all of it is the workload, not the method.** Daily autocorrelation
predicted every outcome here: 0.730 wins, 0.349 and 0.248 do not. That number costs seconds
to compute and tells an operator whether to build any of this — which remains the most
useful thing the project has produced.

---

# Does the answer survive a better forecaster? (2026-08-07)

Every controller experiment in this project used `SeasonalNaiveForecaster`. ARIMA, ETS,
LightGBM and drift were built and tested but never ran inside a control loop, so every
headline finding was strictly a finding about *one crude forecaster*. This closes that gap:
seven workloads × four forecasters × four quantiles, at two commitment windows.

Cells show how many times **forward dominated backward** — cheaper *and* fewer violations at
the same setting — out of 4 quantiles.

## 6-hour commitment

| workload | daily autocorr | seasonal-naive | drift | ets | lightgbm |
|---|---:|---:|---:|---:|---:|
| `azure-llm-code` | **0.730** | **4** | **2** | **4** | 0 |
| `materna-1` | 0.494 | 0 | 0 | 0 | 0 |
| `materna-3` | 0.469 | 0 | 0 | 0 | 0 |
| `materna-2` | 0.450 | 0 | 0 | 0 | 0 |
| `azure-llm-conv` | 0.349 | 0 | 0 | 0 | 0 |
| `bitbrains-rnd` | 0.248 | 0 | 0 | 0 | 0 |
| `bitbrains-fastStorage` | 0.197 | 0 | 0 | 0 | 0 |

## 12-hour commitment

| workload | daily autocorr | seasonal-naive | drift | ets | lightgbm |
|---|---:|---:|---:|---:|---:|
| `azure-llm-code` | **0.730** | **4** | **1** | **3** | 0 |
| `materna-1` | 0.494 | 0 | 0 | 0 | 0 |
| `materna-3` | 0.469 | 0 | 0 | 0 | 0 |
| `materna-2` | 0.450 | **2** | 0 | 0 | 0 |
| `azure-llm-conv` | 0.349 | 0 | 0 | 0 | 0 |
| `bitbrains-rnd` | 0.248 | 0 | 0 | 0 | 0 |
| `bitbrains-fastStorage` | 0.197 | 0 | 0 | 0 | 0 |

## Three things this establishes

**1. The positive result is not an artefact of one forecaster.** On `azure-llm-code`, ETS
reaches 4/4 at six hours and 3/4 at twelve, and drift reaches 2/4 and 1/4. Three independent
model families agree that this workload rewards forecasting at long commitments.

**2. The negative result is not an artefact either.** On the five low-autocorrelation
workloads, *no* forecaster won a single cell at six hours. Fifty-six cells, zero wins. The
earlier conclusion that "forecasting does not pay here" survives replacing the crude model
with better ones.

**3. LightGBM — the most sophisticated model in this repository — never won a single cell**,
at either window, on any workload. This is consistent with its known limitation, recorded
earlier in this document: it is a recursive one-step quantile model whose bands do not widen
with horizon (p99−p50 measured at 31.95 at step 1 versus 30.12 at step 48). Asked for a
median path across a 6–12 hour window, it has nothing useful to say. **Matching the model to
the horizon mattered more than model capacity did here.**

**Stated carefully, because the literature disagrees with the loose version.** Pre-training
does improve cloud-workload forecast *accuracy* — the CloudOps benchmark reports a 27% error
reduction over classical and deep baselines — so "sophistication does not help" would be
wrong. The claim supported here is narrower and is about the *decision*: a more accurate
forecaster did not produce better capacity decisions at these commitment horizons, which is
what the decision-focused learning literature predicts. Two scope limits apply. LightGBM is
the strongest model *available in this repo*, not in the field; and M17's foundation models
were cut, so Chronos-Bolt and TimesFM remain untested. The question of whether a genuinely
strong forecaster changes the commitment answer is **open, not settled**.

## The threshold is a rule of thumb, and it has an exception

The 0.50 cutoff used in the shipped diagnostic gets 27 of 28 cells right at six hours and 26
of 28 at twelve. But the exception is real and it is not in the direction that flatters the
rule: **`materna-2` at r = 0.450 — *below* the threshold — won 2 of 4 at twelve hours, while
`materna-1` at r = 0.494 — *above* it — won none.** The ordering by autocorrelation is
violated between two traces from the same provider.

Two candidate explanations, neither verified: `materna-2` is the trace spanning Christmas
and New Year, so it carries a genuine level shift a seasonal model may exploit; or 2 of 4
with a single forecaster is simply noise. **Reported as unresolved rather than explained
away.**

So the honest statement is: daily autocorrelation **orders** these workloads well and
predicts the extremes reliably, but it is not a calibrated boundary and a value near 0.45–0.50
does not settle the question. The dashboard says as much, and Q13 remains **open**.

---

# Correction: the actuation-delay bug, and what it changed (2026-08-08)

An audit of the M4–M23 code — the same treatment the M0–M3 audit got on 2026-08-03 — found a
bug in `apply_actuation_delay` that invalidated **every autoscaling-regime number in this
document**. It is written up here in full because it changed two headline answers, and
because the way it hid is more instructive than the bug itself.

## The bug

A controller's request is turned into serving capacity by `apply_actuation_delay`. When a
new request arrived while a change was still in flight, the code retargeted the change *and
restarted its clock*. So a request that kept moving never landed at all:

```
requested   1  2  3  4  5  6  7  8  9 10 11 12      (startup = 3 steps)
provisioned 1  1  1  1  1  1  1  1  1  1  1  1      ← before the fix
provisioned 1  1  1  1  5  5  5  5  9  9  9  9      ← after
```

A monotone ramp provisioned nothing, forever. Real reconciliation loops do not work this
way: pods already starting do not begin again because the desired count moved. The fix keeps
the supersede semantics the docstring describes — act on current desired state, not a
backlog — while preserving the in-flight change's original landing step.

## What it did *not* touch — the headline result is unaffected

The bug only bites when a request keeps moving while a change is in flight. Commitment
controllers hold capacity constant for a whole 6- or 12-hour window, so their plans change
once per window and then sit still — which both implementations handle identically.

This was verified rather than assumed: 400 commitment-shaped plans (both window lengths,
random levels, the deploy profile's 240 s startup) produce **byte-identical provisioned
capacity** under the buggy and the fixed model. Zero differences.

So the following are untouched, and every number in them stands:

- the commitment-regime result — forecasting wins on `azure-llm-code`;
- the 7 workloads x 4 forecasters x 4 quantiles study, including 27-of-28 and 26-of-28;
- the `materna-2` exception and the open Q13.

What *was* invalidated is the autoscaling regime, where controllers re-plan every step: the
Q1 lead-time sweep, the M8 frontier tables, and the GPU cold-start sweeps. Those were re-run
and are reported above.

## Why the gate could not see it

Every one of the four M4 validation checks passed before and after.

- **Erlang-C, degenerate, determinism** never exercise a *changing* request. They use
  constant or single-step plans, which the bug handled correctly.
- **The sensitivity sweep** checks that the *ranking* of controllers survives perturbation.
  The bug moved every controller the same way, so the ranking was stable and the check
  passed — while every absolute number underneath it was wrong.
- **The unit suite** had a test named `test_a_later_request_supersedes_one_still_in_flight`
  that asserted the superseding behaviour on a plan that stops changing after two steps. It
  passed under both implementations. All 178 tests passed before the fix, and all 178 still
  pass after it.

This is the 2026-08-03 lesson again, sharper: **a green gate is evidence that the checks
ran, not that the code is right.** The check that would have caught this — "a rising request
must eventually provision" — is a two-line property nobody had written, and it is now
`test_a_continuously_changing_request_still_lands`.

## What it changed — Q1 is refuted, and replaced by something better

The pre-registered Q1 prediction was that the proactive margin *grows* with actuation lead.
Corrected, on synthetic `clean_daily`:

| startup (steps) | % of season | reactive viol. | proactive viol. | margin |
|---:|---:|---:|---:|---:|
| 0 | 0.0% | 0.197 | 0.008 | +0.188 |
| 1 | 4.2% | 0.313 | 0.027 | +0.287 |
| 2 | 8.3% | 0.362 | 0.068 | **+0.293** |
| 3 | 12.5% | 0.392 | 0.140 | +0.252 |
| 4 | 16.7% | 0.407 | 0.235 | +0.172 |
| 6 | 25.0% | 0.448 | 0.365 | +0.083 |
| 8 | 33.3% | 0.433 | 0.397 | +0.037 |
| 10 | 41.7% | 0.457 | 0.447 | +0.010 |
| 12 | 50.0% | 0.537 | 0.505 | +0.032 |
| 16 | 66.7% | 0.692 | 0.585 | +0.107 |
| 20 | 83.3% | 0.790 | 0.633 | +0.157 |
| 24 | 100.0% | 0.408 | 0.340 | +0.068 |

**Q1 as registered is refuted.** The margin does not grow with lead time. But it is not
noise either — it has a clear shape, and the shape is the real answer:

> **The proactive margin is governed by where the actuation lead falls in the seasonal
> cycle, not by lead time as such.** It peaks at +0.293 around 8% of a cycle, collapses to
> +0.010 near half a cycle — the worst possible phase, where you are predicting the opposite
> part of the day — and recovers toward a full cycle, where a seasonal-naive forecaster
> realigns with the season it was built on.

Forecasting beat reaction at *every* lead tested; only the size of the advantage moved.

**This resolves a disagreement rather than creating one.** The GPU lane's cold starts (up to
1200 s) are at most 1.4% of a daily cycle — entirely on the rising limb — which is why that
lane sees the margin grow monotonically while the synthetic sweep sees it fall. Both are the
same curve sampled in different places.

Robustness: the reversal is not an artefact of one modelling choice. An independent pure
pipeline model (each request lands `lag` steps later, never cancelled) gives +0.190 → +0.053
against the fix's +0.190 → +0.038, agreeing closely at every lead; only the buggy model
produced a growing margin. Refit cadence was also swept (every 1, 6 and 24 steps) and makes
no material difference here, so the stale-forecast mechanism recorded in the GPU lane is not
what drives this.

## What it changed — the GPU `code` negative inverted

The recorded finding was *"burstiness defeats forecast-driven capacity control regardless of
actuation delay"*. Corrected:

| startup | reactive viol. | proactive viol. | margin | relative |
|---:|---:|---:|---:|---:|
| 0 s | 0.0899 | 0.1198 | −0.0299 | −33.2% |
| 60 s | 0.1082 | 0.1260 | −0.0179 | −16.5% |
| 240 s | 0.1477 | 0.1283 | **+0.0194** | 13.2% |
| 600 s | 0.1946 | 0.1299 | **+0.0648** | 33.3% |
| 1200 s | 0.2391 | 0.1356 | **+0.1035** | 43.3% |

Forecasting *does* help on `code` once the cold start exceeds roughly four minutes, and the
margin grows monotonically from there. **The claim that burstiness defeats forecasting
regardless of delay was an artefact of the bug and is withdrawn.** What survives is the
weaker and still useful statement: burstiness raises the cold-start threshold at which
forecasting starts paying — on `code` that threshold is ~240 s, while `conv` benefits at
every lead tested.

## What it changed — a caveat that turned out to be an artefact

The M4 gate carried a standing *saturation warning*: past ~2 steps of lead the reactive
baseline exceeded 50% violations, so "both controllers mostly fail" and only rankings were
trustworthy. Corrected, reactive plateaus at ~0.43 and the warning no longer fires. **The
saturated regime did not exist** — it was the bug preventing reactive from ever scaling up.
The corresponding L5 Trap 3 note is withdrawn for this sweep.

## A gate that asserted its own hypothesis

Check 4a required the margin to grow with lead, with the stated rationale that a margin
which fails to grow means "the simulator's actuation delay is not doing its job". That
inference is invalid, and it made the gate unfalsifiable in the wrong direction: a correct
simulator plus a wrong hypothesis was indistinguishable from a broken simulator. Indeed the
first run after fixing the bug reported **M4 gate: NOT GREEN** — the fix looked like a
regression.

Check 4a now gates on a genuine property of the simulator — *longer lag must never reduce
violations for either controller*, which holds — and reports the Q1 margin as a measurement.
Hypotheses are answered in this document, not enforced by the test suite.

## Also corrected: the diagnostic overclaimed

The shipped verdict string asserted that below the 0.50 threshold no forecaster beat a
trailing percentile *"at any commitment length"*. The project's own table on 2026-08-07
refutes this: `materna-2`, at r = 0.450, won 2 of 4 settings at a twelve-hour commitment.
The `HowItWorks` view carried the same overclaim, written when the study covered three
workloads and never updated when it grew to seven.

`classify()` now returns an explicitly **borderline** verdict for readings in 0.40–0.55,
naming the two traces that invert the ordering and telling the reader to measure rather than
trust the threshold. `test_borderline_verdicts_disclose_the_measured_exception` pins it.

## Added to the negatives ledger

- **Q1 as pre-registered is refuted.** The proactive margin does not grow with actuation
  lead; it peaks near 8% of a seasonal cycle and troughs near 50%.
- **A documented negative — "burstiness defeats forecasting on `code`" — was a bug
  artefact** and is withdrawn.
- **A documented caveat — the "saturated regime" — was a bug artefact** and is withdrawn.
- **A validation gate encoded a hypothesis as a pass condition**, so for five days a correct
  simulator would have been reported as broken.
- **The shipped diagnostic contradicted this document** for one day, in the direction that
  flattered the rule.

---

# Q13 answered: the threshold does not generalise (2026-08-08)

Q13 asked whether 0.50 is a real boundary or a round number that seven workloads flattered.
It is closer to the second, and the honest answer narrows the diagnostic's scope
considerably.

## Method

Azure Functions 2019 — CC-BY, already fetched and checksum-verified, so no new licence
question — aggregated to 5-minute bins. 400 functions loaded, 186 usable after excluding
sparse and degenerate series, 43 simulated. **Sampling was deliberately hostile:** stratified
by the diagnostic and oversampled in the 0.40–0.55 borderline band, so most evidence comes
from where the threshold is least defensible.

Two methodology corrections were needed first, and both changed the answer:

1. **Pareto dominance is the wrong scorer.** `evaluate_diagnostic.dominance` counts a cell
   only when forward is no worse on *both* cost and violations. On `762d22c5a3d7`
   (r = 0.892) forecasting cut violations from 0.193 to 0.072 — a factor of 2.7 — while
   costing 6% more, and that is recorded as *not a win*. Scored that way the question being
   answered is "must forecasting be free?", not "does forecasting pay?". Cells are now
   scored on total economic cost at the ratio the quantile implies, `C_u/C_o = q/(1-q)`,
   which is the newsvendor objective this project already uses for Q4.
2. **The replica-sizing heuristic degenerates on serverless traces.** `_fleet_workload`
   scales capacity by `median/8`; `2b373145c4fa` is 65% zeros, so its median is 0, the floor
   applies, and the run reported a mean of 17,413 replicas. Sizing off the p95 instead keeps
   the replica count sane at any sparsity and is applied identically to both controllers.

`dominance` also counts a *tie* — both controllers emitting an identical plan — as a forward
win. That never fired on the fleet traces (verified: every published win is strict, no plan
pair identical, including `materna-2`'s 2 of 4, so **the published 27-of-28 stands**) but it
fires often on low-volume functions, where it would manufacture agreement out of nothing.
Ties are now counted and excluded.

## Result

| daily autocorrelation | n | workloads where forecasting paid (6 h) | (12 h) |
|---|---:|---:|---:|
| < 0.20 | 8 | 25% | 12% |
| 0.20 – 0.35 | 6 | 67% | 83% |
| 0.35 – 0.50 | 13 | 69% | 69% |
| 0.50 – 0.70 | 7 | 57% | 71% |
| > 0.70 | 9 | 67% | 33% |

**The relationship is not monotone.** It rises out of the noise floor and then *falls* again
at the top: on serverless workloads, high daily autocorrelation does not imply that
forecasting pays.

Scored against the shipped 0.50 cutoff, accuracy is **22/43 (51%) at six hours and 20/43
(47%) at twelve** — a coin flip, and *below* the 58%/53% you would get by ignoring the
diagnostic and always predicting that forecasting pays. Fifteen workloads below the cutoff
paid anyway. The best cutoff achievable anywhere on this cohort is r >= 0.15 at 67%, which
is barely above that same do-nothing baseline.

## What survives

**The diagnostic is specific to fleet-aggregate demand.** It was derived on Bitbrains,
Materna and Azure LLM traces, every one of which sums thousands of VMs or requests. Those
aggregates are smooth, and daily structure is the dominant exploitable signal in them. An
individual serverless function is spiky and low-volume: it can carry high daily
autocorrelation while the variance that actually drives a commitment decision lives inside
the window, where a day-lagged correlation cannot see it.

What transfers is only the bottom of the range: **very low daily autocorrelation (below
~0.20) does predict that forecasting will not pay**, on both populations. Above that, on
serverless traces, the number carries little information.

Two candidate mechanisms, neither tested: within-window variance is the quantity that
matters and daily autocorrelation is a poor proxy for it on spiky demand; or integer replica
granularity dominates on low-volume functions, so both controllers round to the same
capacity regardless of what either predicts. Distinguishing them is the obvious next
experiment and is not claimed here.

## Added to the negatives ledger

- **The 0.50 threshold does not generalise beyond the fleet-aggregate traces it was derived
  from.** On individual Azure Functions workloads it is a coin flip, and worse than ignoring
  it entirely.
- **The relationship is non-monotone on serverless demand** — the top of the range behaves
  like the bottom, which no version of the rule predicted.
- **The win metric was measuring the wrong thing.** Pareto dominance asks whether forecasting
  is free; the newsvendor objective asks whether it pays. Only the second is the project's
  actual claim.

---

# Do these conclusions agree with anyone else? (2026-08-08)

A finding nobody else has ever seen is usually an instrument artefact. This section checks
each headline claim against published work, and records the places where the literature
disagrees or where DELPHI turns out to be re-deriving something with an established name.

## Where the literature agrees

**The newsvendor identity is textbook, and cloud provisioning is a named application of
it.** `CR = C_u/(C_u + C_o)` is the standard critical ratio; DELPHI's contribution is not
the identity but insisting that an autoscaler pinned at P95 is *asserting* `C_u/C_o = 19`
whether or not anyone decided that. Prior art on the non-stationary case with predictions
exists (arXiv 2305.07993) and is closer to this project than anything cited in `RESEARCH.md`.

**The percentile recommender is a genuinely strong incumbent, not a strawman.** Google's
Autopilot reports autopiloted jobs running at 23% slack versus 46% for hand-managed ones,
and a 10x reduction in jobs severely affected by OOMs. Choosing it rather than threshold HPA
as the baseline to beat is the right call, and it is why DELPHI's negative result is
interesting: most published proactive-autoscaling wins are measured against threshold HPA,
where 2–4x violation reductions are routine and where DELPHI also wins.

**Token work, not request rate, is the right signal for LLM serving.** Current practitioner
guidance says explicitly that CPU and memory are poor proxies for LLM load, that queue depth
and token counts are the signals to scale on, and that prefill and decode have different
costs. M16 measures what that guidance asserts.

**Cold start is the mechanism.** The standard account — an HPA polling every 15–30 s plus
30–90 s of provisioning means a two-minute spike is missed entirely — is the same mechanism
DELPHI's actuation-delay parameter models, and the same reason the margin depends on it.

## Where DELPHI is re-deriving something with a name

**The core thesis is decision-focused learning, and the project did not know it.** The claim
that forecast accuracy is the wrong objective because a capacity controller consumes a
*decision*, not a point estimate, is the founding premise of the predict-then-optimize /
decision-focused learning literature, whose position is that improved predictive accuracy
does not in general translate into improved decision quality — established both empirically
and theoretically. `RESEARCH.md` reached this independently via the calibration route and
cites neither term.

This is a credibility issue rather than a correctness one: the finding is sound and the
framing is standard, but presenting it as novel would misread the field. It should be cited
as what it is — a decision-focused evaluation of capacity control, applied to a domain where
the operations-research framing is not yet common.

**"Is this forecastable at all" is an existing research question with a better answer than
ours.** Spectral entropy is the established forecastability measure, and recent work
proposes spectral predictability specifically as a fast, training-free indicator of whether
forecasting will beat simple baselines — the same job as our diagnostic, computed better.
The literature's reason for preferring it is precisely the failure Q13 measured: a single
lagged correlation looks at one lag, while spectral measures capture structure across all
frequencies at once. **Our Q13 negative is what this literature predicts should happen.**
That is corroboration of the method and a clear signposted improvement: the diagnostic
should be spectral entropy, or autocorrelation plus spectral entropy, not a single lag.

Notably, BACC — the closest prior art — already orders its five Azure Functions traces by
autocorrelation *and spectral entropy* together, which is a hint this project should have
taken earlier.

## Where the literature disagrees, and what survives

**Pre-trained models do beat classical baselines on cloud-workload forecast accuracy.** The
CloudOps pre-training benchmark reports a 27% error reduction over classical and deep
baselines on its largest dataset. DELPHI's observation that LightGBM never won a commitment
cell must therefore not be read as "sophistication does not help forecasting" — it does.

The defensible claim is narrower and is the one this document should make: *a more accurate
forecaster did not produce better capacity decisions at these commitment horizons*, which is
exactly what decision-focused learning predicts and is a statement about the decision, not
about the model. It is also weakened by scope: LightGBM is the most sophisticated model *in
this repository*, not in the field. M17's foundation models were cut, so the strongest
available forecasters remain untested here, and the honest position is that the question is
open rather than answered.

## Net assessment

Six of the project's claims are corroborated by independent work, one is a re-derivation of
an established framework that should be cited rather than presented as new, and one — the
LightGBM result — needs narrowing to a statement about decisions rather than about models.
No claim was contradicted outright. The Q13 negative result is independently predicted by
the forecastability literature, which is the strongest single piece of evidence that the
measurement apparatus here is working correctly.

---

# Racing the predictability measures (2026-08-08)

Q13 showed the shipped diagnostic does not generalise, and the forecastability literature
offered a specific reason and a specific fix: a single lagged correlation reads one
frequency, spectral entropy reads all of them. **The fix was tried and it did not work.**

## What was raced

Three training-free measures, scored against the same target — did forecasting pay, by total
economic cost, at a 6 h and 12 h commitment — on the same 43 serverless workloads from Q13
plus the 5 non-GPU fleet workloads:

1. **Daily autocorrelation** — what shipped.
2. **Spectral predictability**, `1 − normalised spectral entropy` over a Welch-averaged
   periodogram — the field's standard forecastability feature.
3. **Low-frequency power fraction** — the share of spectral power at periods *longer than
   the commitment window*. Not from the literature; motivated by this project's own Q1
   result, that a decision fixed for N hours can only exploit structure slower than N hours.
   Alone among the three it is a function of the decision horizon, not of the series only.

Comparison is by **AUC**, which needs no cutoff. Reporting each measure at its own best
threshold would let a measure win by fitting a cutoff to 43 points, so best-threshold
accuracy is shown only as a clearly-labelled optimistic bound.

## Result

Serverless workloads, n = 43, with a 20,000-draw permutation test on each AUC:

| measure | AUC @ 6 h | p | AUC @ 12 h | p |
|---|---:|---:|---:|---:|
| daily autocorrelation | 0.539 | 0.67 | 0.438 | 0.50 |
| spectral predictability | 0.396 | 0.26 | 0.456 | 0.63 |
| low-frequency power | 0.595 | 0.31 | **0.700** | **0.027** |

**Spectral entropy failed.** At 0.396 and 0.456 it is at or below a coin flip, and on the
fleet workloads it is actively anti-predictive (AUC 0.167 at 6 h, though n = 5 there makes
that number nearly meaningless). The literature's recommended instrument did not rescue the
diagnostic on this population, and predicting that it would was wrong.

**The horizon-relative measure is the only one showing signal**, and it is not from the
literature — it comes from this project's own Q1 finding. At a twelve-hour commitment it
reaches AUC 0.700 with a nominal p = 0.027.

## Why that 0.700 is not being shipped

Six tests were run (3 measures x 2 windows). A Bonferroni-corrected threshold is p < 0.0083,
and 0.027 does not clear it. Picking the largest of six AUCs and quoting its uncorrected
p-value is precisely how a result gets manufactured, and this project's own doctrine says an
implausibly clean number is an artefact until proven otherwise.

So the honest status is: **a promising lead on one cohort, at one horizon, that does not
survive correction for the number of things tried.** It is not a finding, and the shipped
diagnostic is unchanged.

## What this means for the diagnostic

The product keeps daily autocorrelation and keeps the scope caveat Q13 forced onto it. That
is not because autocorrelation is good — on serverless it is a coin flip — but because
nothing tested beat it well enough to justify swapping the claim a visitor reads.

The measures are implemented, unit-tested against signals with known answers, and available
in `delphi.forecast.predictability` for the follow-up, which is now well defined: test
low-frequency power on a much larger cohort with the horizon fixed in advance, as a single
pre-registered hypothesis rather than one of six. On a fresh cohort with one test, p < 0.05
would mean something.

## Added to the negatives ledger

- **Spectral entropy did not improve on daily autocorrelation** for predicting whether
  forecasting pays, on either population — contradicting the expectation drawn from the
  forecastability literature.
- **The one measure that showed signal does not survive multiple-comparison correction**, and
  is reported as a lead rather than promoted into the product.

---

# Full-codebase audit (2026-08-09)

The 2026-08-03 audit covered M0–M3 and the 2026-08-08 one covered the control and product
layers. This pass covered what neither did: the queueing core, the adaptive controllers, the
Pareto machinery, the price client, and the deployed container. Four defects, one of them a
live security hole.

## 1. The budget pacer was steering on a frozen capacity trace

`adaptive.py` carried its **own copy** of the actuation-delay rule, so C5/C6 could reconstruct
what their past requests would have served and pace against it. When the clock-restart bug
was fixed in `apply_actuation_delay` on 2026-08-08, that copy was left behind — still
restarting the clock, still freezing.

Measured on a plan that moves every step, which is exactly what C5/C6 emit: the mirror
reported a flat 13 replicas for 40 steps while the simulator provisioned 13 → 24. **36 of 40
steps disagreed**, so the PI loop spent every run correcting against violations that never
happened.

Fixed by extracting `ActuationTracker`, with `apply_actuation_delay` implemented on top of it
so the batch and incremental forms *cannot* diverge. A test asserts they agree over 200
random plans, and another fails if the lag rule is re-implemented anywhere outside
`simulator.py` — this being the second time a copy of it drifted.

**Effect on published results: small and no conclusion moves.** Re-running the frontier
changes only C5/C6 rows: cost 66.6 → 66.5, churn 180 → 176, violation rates shifting by
around 0.001–0.008 in both directions. Q4 is bit-for-bit unchanged at 9 better / 3 tie / 6
worse of 18, and the "wins below 19:1, loses above" pattern holds. The GPU frontier verdict
is unchanged at 0 of 8.

## 2. Path traversal in the deployed dashboard

The SPA catch-all joined the request path onto the dist directory and served whatever it
found. Percent-encoded traversal survives URL normalisation, reaches the handler intact, and
read arbitrary files off the container:

```
/..%2f..%2frequirements-serve.txt   -> served the file
/%2e%2e/%2e%2e/%2e%2e/etc/hostname  -> served the file
```

Verified against the real deploy image under uvicorn, not merely in a test client. **The
container also ran as root**, so this was an arbitrary read of everything in it. No
credentials exist to steal — the project is keyless by design — but the source, the
dependency manifest and every system file were readable.

Render's edge happened to reject the encoded forms with a 400, so the live site was not
exploitable. That is an accident of the CDN rather than a control, and anyone following the
README's `docker run` had no such cover.

Fixed by resolving the candidate and requiring it to stay inside dist — which also closes
symlink escapes and absolute-path injection — and by dropping to an unprivileged user in the
image. Both are now asserted in the deploy-image CI lane against the running container, so a
regression fails the build rather than the internet.

## 3. Hypervolume was measuring the wrong staircase

`dominated_hypervolume` swept from cost zero and credited each strip with the *dearer*
point's violation rate, when over that interval only the cheaper point is affordable. A
single point at cost 5 with a 0.2 violation rate against a reference corner at cost 10
returned 8.0 where the true dominated rectangle is 5 x 0.8 = **4.0**.

This is not merely an inflation: it can reverse a ranking. A frontier reaching a 0.05
violation rate at cost 9 scored *above* one sitting at 0.45 for cost 5, where the corrected
areas are 1.75 and 3.25 respectively.

The existing test compared two frontiers where the error cancelled, so it passed throughout.
It now checks areas computed by hand. **No published result used this function** — it is
reported nowhere in this document — so nothing downstream moves.

## 4. The two fidelities were documented as the same system

`_synthesize_arrivals` claimed its service-time construction kept the utilisation and queue
models "describing the same physical system". They do not: `utilisation_target` derates
capacity in the utilisation model but is not applied in the queue, so at a target of 0.8 the
queue runs at rho = 0.8 where the utilisation model reports a step exactly at its limit.

Defensible for two models answering different questions, and harmless in practice — every
published result uses the utilisation model, and the Erlang-C validation sets the target to
1.0 where the discrepancy vanishes — but the docstring asserted something false about the
code. Corrected to state the difference and why it does not contaminate anything.

## Verified sound

The ACI update rule reproduces the published BACC form exactly, including the sign of the
correction under miscoverage. `pareto_front` is a correct two-objective sweep.
`cost_at_matched_violation` reports unreachable targets as unreachable rather than
substituting the nearest point. `parse_price_items` skips rows with a missing or non-positive
price rather than defaulting them, which would otherwise drive the newsvendor ratio to buy
unbounded capacity. `cheapest_hourly` refuses to convert monthly reservations into hourly
prices. The OData filter is now escaped — not a live injection path, since the values come
from config, but an apostrophe would have silently returned the wrong SKU's price.

## Added to the negatives ledger

- **A duplicated implementation of the actuation lag drifted from its original**, for the
  second time, and fed a control loop a capacity trace that never moved.
- **The deployed container was vulnerable to path traversal and ran as root.** It was
  unexploitable in production only because of a CDN behaviour nobody had designed for.
- **A Pareto summary statistic could rank two frontiers backwards**, and its test passed
  because the error cancelled between the two curves being compared.

---

# Q13b — the lead did not replicate (2026-08-09)

Pre-registered in `docs/PREREGISTRATION-Q13B.md` and committed, with the complete analysis
script, **before the confirmatory data existed**. The commit order is checkable in the git
history and is the only thing that makes the rest of this section worth reading.

## Result

One measure, one horizon, one test, on 80 workloads drawn fresh and explicitly disjoint from
the 43 the lead was found on.

| | exploratory (2026-08-08) | confirmatory (2026-08-09) |
|---|---:|---:|
| low-frequency power, AUC @ 12 h | 0.700 | **0.526** |
| p | 0.027 (one of six tests) | **0.359** (one-sided, pre-registered) |
| n measurable | 34 | 69 |

**Not confirmed.** The effect all but vanished. The design had roughly 0.9 power against a
true AUC of 0.700, so this is not a near miss on an underpowered test — 0.526 is what a
measure with essentially no signal looks like.

The honest reading is the boring one: **0.700 was the winner's curse.** It was the largest of
six AUCs computed on 34 workloads, and the largest of six noisy estimates is biased upward by
construction. Regression to the mean did the rest.

## Secondary, and deliberately not acted on

On the same fresh cohort, daily autocorrelation scored **AUC 0.601** and spectral
predictability **0.499**.

Autocorrelation scoring highest is exactly the sort of result that invites a second bite:
declare it vindicated, rebuild the diagnostic around it, quote 0.601. That is forbidden here
and the pre-registration says so in advance. These are descriptive numbers carrying no alpha,
computed on n = 69, where the permutation null from the exploratory run put the 95% ceiling
near 0.68 — so 0.601 is inside the noise band and establishes nothing. **No claim is made
from it, and the shipped diagnostic does not change.**

## What this closes

- **The low-frequency power measure is refuted**, not merely unconfirmed. It had one
  well-powered pre-registered chance and did not take it.
- **Spectral entropy is refuted twice over.** It lost the exploratory race (AUC 0.396 and
  0.456) and scored 0.499 — chance, to three decimals — on fresh data.
- **No training-free measure tested predicts whether forecasting pays on individual
  serverless workloads.** Daily autocorrelation, spectral entropy, and horizon-relative
  spectral power have all now been tried and all have failed on that population.

The diagnostic therefore stays exactly where Q13 left it: **valid on fleet-aggregate demand,
where it calls 27 of 28 cells correctly, and scoped out of everything else on the page a
visitor reads.** That scope is now supported by two independent failures to extend it rather
than one.

## A prediction of ours that was wrong, recorded as such

After the literature review on 2026-08-08 this project recommended replacing daily
autocorrelation with spectral entropy, calling it "the clearest improvement available". The
forecastability literature does support spectral measures in general, and the reasoning was
sound in the abstract. **On this problem it was wrong**, and measured twice to be wrong. The
recommendation is withdrawn from the README.

## Added to the negatives ledger

- **The Q13 replacement measure failed to replicate**, 0.700 → 0.526, against a
  pre-registered hypothesis with ~0.9 power.
- **A promising exploratory AUC was the winner's curse**, and pre-registration is what caught
  it rather than hindsight.
- **This project publicly recommended an improvement that its own next experiment refuted.**

---

# Q9 — independent per-resource sizing, measured at last (2026-08-09)

Registered as *"do independent per-resource forecasts over-provision the joint plan?"* with
the expectation "yes; the interesting part is by how much". **It had never been run.**
`generate_multi_resource` existed and was unit-tested, but no experiment used it — while
`docs/ROADMAP.md` claimed the synthetic regime "carries the joint-provisioning test". It did
not, and that sentence is now corrected.

## Result: the registered direction is refuted

A replica supplies a fixed amount of each resource, so the replicas needed at time *t* are set
by whichever resource is tightest: `r_t = max(cpu_t/cpu_per, memory_t/memory_per)`. Sizing on
the first half of the trace, scoring on the second, with each replica scaled so **either
resource alone** needs 8 replicas at its own target quantile:

| offset | correlation | joint repl. | indep. repl. | joint coverage | indep. coverage |
|---:|---:|---:|---:|---:|---:|
| 0.000 | +0.778 | 9 | 8 | 0.9688 | 0.8442 |
| 0.125 | +0.529 | 9 | 8 | 0.9688 | 0.8284 |
| 0.250 | +0.010 | 9 | 8 | 0.9688 | 0.8229 |
| 0.375 | −0.518 | 9 | 8 | 0.9688 | 0.8075 |
| 0.500 | −0.770 | 9 | 8 | 0.9688 | **0.8016** |

*(target q = 0.90; `offset` lags memory's daily cycle by a fraction of a day, so 0 means the
two resources rise together and 0.5 means one peaks while the other troughs.)*

**Independent sizing under-provisions. It does not over-provision.** The arithmetic makes this
inevitable and the registered expectation had the sign backwards: since `max(a, b) >= a`, the
quantile of the joint requirement is never below either marginal quantile, so combining
independently-sized resources can only buy less than the joint plan — never more.

What is genuinely measured is the **coverage** that costs, and how it scales. Asked for 90%,
the independent method delivers 84.4% when the two resources move together and **80.2%** when
they are anti-phase, against the joint plan's steady 96.9%. The shortfall widens monotonically
as correlation falls from +0.78 to −0.77, which is the mechanism showing itself: the more the
peaks avoid each other, the more often the resource you are not looking at is the binding one.

The practical statement: **you cannot reach a joint SLO by watching one dashboard per
resource.** The gap is not a tuning error to be closed with headroom on each chart; it is
structural, and it grows precisely as the resources become more independent.

## Three nulls before a signal, and why they are in the record

The first three designs returned exact zeros, and each was a broken instrument rather than an
absent effect. They are recorded because a null from an instrument that cannot detect the
thing is not evidence of absence, and because the sequence is the honest account of how this
number was arrived at.

1. **Peaks narrower than the tail.** The fixture injects one peak per resource spanning 1.56%
   of the series. A q = 0.95 quantile discards the top 5%, so the peaks sat entirely inside
   the discarded region and could not move any sizing decision. Diagnosed by checking where
   the peaks fell relative to the quantile, *before* interpreting the null.
2. **Peaks in only one half of the split.** Widening them did not help, because the fixture
   places the CPU peak at 33% of the series and the memory peak at 66% — so the training half
   contained only one of them and neither method could size for a peak it had never seen. The
   fix was decorrelation that *recurs*: anti-phase daily cycles rather than one-shot events,
   which is also what real multi-resource workloads look like.
3. **One resource dominating the maximum.** Scaling each replica off the resource's median
   equalises the two medians but not their ranges, so the wider-swinging resource was the
   binding one at every single step and decorrelation had nothing to act on — which is why
   the numbers were identical to four decimal places across every phase offset. Sizing so
   each resource alone needs the same replicas at its own target quantile fixed it.

Each change was made to an instrument that provably could not see the effect, not to a result
that was inconvenient. The tell in every case was a null that was *too clean* — identical
values across conditions that should have differed.

## Added to the negatives ledger

- **Q9's registered direction was wrong.** Independent per-resource sizing under-provisions
  rather than over-provisions, and the sign follows from `max(a, b) >= a` without needing an
  experiment at all.
- **A pre-registered question went unmeasured for the life of the project** while a shipped
  document implied it had been answered.
