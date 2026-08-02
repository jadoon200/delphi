# DELPHI

Calibrated time-series forecasting and auditable capacity control for cloud and AI-inference
workloads.

DELPHI treats capacity as a newsvendor decision: the relative cost of idle capacity and unmet
demand determines the demand quantile to provision. Forecasts therefore expose calibrated
quantiles—not a single point—and every later capacity decision records the evidence and
assumptions that produced it.

The project is at **M0: the tested system spine**. It currently provides zero-cost configuration,
UTC-safe time handling, the canonical provenance/workload/demand schema, Alembic parity, and a
strict CI gate. Forecasting, calibration, replay control, and specialist arbitration land as
separately tested milestones; the public status is tracked in [docs/ROADMAP.md](docs/ROADMAP.md).

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

