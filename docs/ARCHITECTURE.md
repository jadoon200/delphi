# Architecture

DELPHI is layered so forecasting quality and decision quality can fail independently and be
measured independently.

1. **Data** normalises every source into provenance-stamped, UTC-explicit demand points plus a
   workload profile that declares serving capacity and actuation delay.
2. **Forecast** emits quantiles over a horizon with an empirical coverage record.
3. **Control** selects a quantile from the under/over-provision cost ratio and evaluates the
   resulting plan in a validated replay simulator.
4. **Specialists** own forecast, sizing, cost, SLO, drift, carbon, and verification concerns.
   They communicate through typed proposals; deterministic code performs arbitration.
5. **Product** exposes the capacity picture, decision ledger, assumptions, and regret for human
   review. No real infrastructure is actuated.

The database spine currently implements layer 1. Later schemas are introduced only with the
milestone that proves their behaviour.

