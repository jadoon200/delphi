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

| Method | p95 coverage | Mean p95 | Recovery steps (24 h window) |
|---|---:|---:|---:|
| `raw` | 0.906 | 98.751 | 46 |
| `split_conformal` | 0.903 | 98.505 | 46 |
| `aci_gamma_0.05` | 0.927 | 102.122 | 28 |

Static split conformal does not survive the distribution shift in this slice: its validation
correction slightly lowers test coverage. ACI remains below nominal over the full transient but
cuts recovery time by 18 steps. Coverage is retained observation-by-observation, not only as the
three scalar summaries above.

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

## M16 — is request-rate autoscaling structurally wrong here?

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

**0 of 8.** And the gap *widens* with cold start — 11.5% → 15.8% → 27.2% → 24.1% more
expensive at a matched ≤5% violation rate — the opposite of the registered prediction.

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
