# Pre-registration — Q13b: does low-frequency power predict whether forecasting pays?

**Written and committed before the confirmatory data was generated.** The commit that adds
this file also adds `scripts/evaluate_q13b.py`, which is the complete analysis. Nothing below
was chosen after seeing a result; the git history is the evidence for that claim, and it is
the only evidence worth anything.

## Why a second test is needed

The exploratory run on 2026-08-08 raced three training-free measures against whether
forecasting paid, over 43 serverless workloads at two commitment windows. The share of
spectral power at periods longer than the commitment window reached **AUC 0.700 at twelve
hours, permutation p = 0.027**.

That is six tests (three measures x two windows). Bonferroni wants p < 0.0083 and 0.027 does
not clear it, so the result was recorded as a lead and deliberately not shipped. Reporting
the largest of six AUCs at its uncorrected p-value is how findings get manufactured.

A lead becomes a finding by surviving **one** pre-specified test on data that did not
generate it. That is what this is.

## The hypothesis, singular

> **H1.** On individual Azure Functions workloads, at a **twelve-hour** commitment,
> `low_frequency_power_fraction` is positively associated with whether forecasting beats a
> trailing-percentile commitment on total economic cost.
>
> **H0.** No association: AUC = 0.5.

One measure. One horizon. One test. Everything else was fixed before running.

## Everything fixed in advance

| Decision | Value | Why fixed now |
|---|---|---|
| Measure | `low_frequency_power_fraction(values, window_steps)` | The lead; no other measure is under test |
| Horizon | **12 hours** only | Where the lead appeared; testing both windows would restore the multiplicity problem |
| Outcome | Forecasting pays = ≥1 strict economic win across the 4 quantiles | Identical definition to the exploratory run |
| Scoring | Total economic cost at `C_u/C_o = q/(1-q)` | The newsvendor objective, as in Q4 |
| Forecaster | `SeasonalNaiveForecaster` | As in the exploratory run |
| Quantiles | 0.80, 0.90, 0.95, 0.99 | As in the exploratory run |
| Statistic | ROC AUC | Threshold-free; no cutoff to fit |
| Test | One-sided permutation, 20,000 draws, **α = 0.05** | One-sided because H1 predicts a direction |
| Sample size | **n = 80** | Power calculation below |
| Cohort | Fresh draw, seed `20260809`, **disjoint from the 43 exploratory workloads** | A confirmatory test on the generating data is not a test |
| Sampling | Uniform at random among usable workloads — **no stratification** | The exploratory run oversampled the borderline band; stratifying on the predictor under test would bias the AUC |
| Exclusions | Same `usable()` filters; workloads where all 4 quantiles tie are dropped as non-measurements | Fixed before running, as in the exploratory script |

### Power

Under H0 with a roughly balanced split at n = 80, the standard error of the AUC is about
`sqrt((n1+n2+1) / (12*n1*n2))` ≈ 0.065, so rejection needs an observed AUC above roughly
0.607. If the true AUC is the 0.700 the exploratory run suggested, power is approximately
**0.9**. If the true effect is appreciably smaller than that, this design will not detect it,
and a null result should be read as "not the effect size we thought", not "no effect".

## What counts as what

- **AUC > 0.5 with one-sided p < 0.05 → confirmed.** The measure predicts, and the diagnostic
  can be rebuilt on it.
- **p ≥ 0.05 → not confirmed.** The lead does not replicate, it is recorded as refuted, and
  the shipped diagnostic keeps daily autocorrelation with its existing scope caveat.
- **AUC < 0.5 → refuted**, whatever the p-value.

There is no third option, no re-slicing by band, and no switching to the six-hour window if
twelve is disappointing. Any further analysis is exploratory and will be labelled as such.

## Secondary, explicitly not the test

Daily autocorrelation and spectral predictability are computed on the same fresh cohort and
reported alongside, so the three can be compared on data none of them has seen. **These are
descriptive.** They carry no α, they cannot confirm anything, and the outcome of H1 does not
depend on them.
