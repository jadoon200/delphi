# DELPHI

Calibrated time-series forecasting and auditable capacity control for cloud and AI-inference
workloads.

DELPHI treats capacity as a newsvendor decision: the relative cost of idle capacity and unmet
demand determines the demand quantile to provision. Forecasts therefore expose calibrated
quantiles—not a single point—and every later capacity decision records the evidence and
assumptions that produced it.

The tested M0 system spine, M1 demand layer, M2 forecast baselines, and M3 calibration layer are
complete. A validated, immutable `DemandSeries` unifies deterministic labelled regimes and the real Azure Functions 2019 wide
trace, with explicit missing-point quality, reproducible cohort selection, checksum-verified
fetching, and idempotent persistence. Forecasts are quantile-only and evaluated with chronological
rolling origins, MASE, WQL, empirical coverage, and a deliberate leakage test. Split conformal,
CQR, and ACI calibration retain realized coverage as a time series. Replay control and specialist
arbitration land as separately tested milestones; status is in
[docs/ROADMAP.md](docs/ROADMAP.md).

## Principles

- Free and licence-clean by default; no paid key or hosted model is required.
- Quantile calibration is measured on held-out time before a forecast may drive capacity.
- Controller comparisons state the open-loop replay assumption and report Pareto frontiers.
- DELPHI recommends plans for human review; it never applies changes to a real cluster.

## Development

```bash
make env
conda activate delphi
make install
make check
make fetch-azure    # 136 MB CC-BY trace, checksum-verified into ignored data/
make ingest-azure   # seeded top-volume + decile-stratified cohort into Postgres
```

Postgres uses host port `5436`, keeping it separate from the sibling portfolio services:

```bash
make up
make down
```

## Current architecture

```text
public trace / synthetic demand
              │
       canonical DemandSeries
              │
      calibrated quantiles       (next milestones)
              │
 newsvendor capacity controller  (planned)
              │
 specialist proposals + ledger   (planned)
              │
        human-reviewed plan
```

## Responsible scope

DELPHI is a replay and decision-support system. It does not control Kubernetes, cloud accounts,
or production infrastructure. Simulated savings are directional under an explicit open-loop
assumption; they are never presented as measured production savings.
