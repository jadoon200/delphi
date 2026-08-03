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
