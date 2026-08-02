# Roadmap

Milestones are marked complete only after their tests and documentation are green.

| Milestone | Status | Deliverable |
|---|---|---|
| M0 — system spine | ✅ | Config, UTC utilities, canonical DB schema, Alembic parity, zero-cost test, CI |
| M1 — demand data | ✅ | Validated `DemandSeries`, deterministic labelled synthetic regimes, and checksum-verified Azure Functions 2019 cohort ingest |
| M2 — forecast baselines | ⬜ | Rolling-origin seasonal-naive, percentile, classical and quantile baselines |
| M3 — calibration | ⬜ | Split conformal, CQR and adaptive conformal inference with measured coverage |
| M4–M8 — capacity control | ⬜ | Validated replay simulator, baseline controllers, newsvendor sizing, honest evaluation |
| M9–M13 — specialists | ⬜ | Typed proposals, deterministic supervisor, verifier and ablation |
| M14–M19 — inference lane | ⬜ | Token-aware serving model and CPU-vs-queue scaling experiment |
| M20–M24 — product | ⬜ | Read-only API, dashboard, explainer and free single-container deployment |

Bitbrains is not on the critical path because its canonical host and terms could not be verified;
the synthetic multi-resource regime preserves the joint-provisioning test without redistributing
unclearly licensed data.

The irreducible result is a cost-versus-violation Pareto frontier on real demand traces with
equal tuning budgets and a validated simulator. Product and agent claims remain subordinate to
that measurement.
