# Architecture

DELPHI is layered so that **forecasting quality and decision quality can fail independently
and be measured independently**. That separation is the point: the project's central result
is that a better forecast does not produce a better decision, which is only observable if the
two are not entangled in one component.

```text
   public trace / synthetic demand
              │
      ┌───────▼────────┐
      │  1. Data       │  provenance-stamped, UTC-explicit demand points + a workload
      └───────┬────────┘  profile declaring serving capacity and actuation delay
              │
      ┌───────▼────────┐
      │  2. Diagnostic │  is this workload forecastable at all? training-free, seconds
      └───────┬────────┘  to compute — answered before anything is built
              │
      ┌───────▼────────┐
      │  3. Forecast   │  quantiles over a horizon, with an empirical coverage record;
      └───────┬────────┘  conformal / ACI calibration measured on held-out time
              │
      ┌───────▼────────┐
      │  4. Control    │  q* = C_u/(C_u+C_o) selects the quantile from the price of
      └───────┬────────┘  failure; the plan is scored in a validated replay simulator
              │
      ┌───────▼────────┐
      │  5. Product    │  read-only API + dashboard: capacity picture, findings,
      └────────────────┘  assumptions and limits. No real infrastructure is actuated.
```

## Layer notes

**1. Data.** Every source is normalised into the same `DemandSeries` — explicit UTC
timestamps, per-point imputation flags and quality, and a declared licence. Missing points
are represented, never forward-filled silently. Sources and their licence status are recorded
in [`DATA.md`](DATA.md), including the ones whose terms could not be verified.

**2. Diagnostic.** Daily autocorrelation, computed in seconds without training anything. It
is deliberately the first thing the product does, because the most useful answer this project
found is often *"do not build a forecaster for this workload"*. Its scope is narrow and
stated: it holds on fleet-aggregate demand and does not transfer to individual serverless
workloads — see Q13 and Q13b in [`EVAL.md`](EVAL.md).

**3. Forecast.** Forecasters emit quantiles, never a bare point estimate, because the layer
above consumes a quantile. Coverage is measured as a time series rather than assumed, and no
model's native quantile head is trusted without an empirical calibration layer over it. Five
model families are implemented; none of them changes the capacity decision.

**4. Control.** The compliance target is *derived* rather than chosen: a controller pinned at
p95 is asserting that unmet demand costs 19x idle capacity, and this layer states that ratio
instead of inheriting it from convention. Plans are scored in a replay simulator validated
against the Erlang-C closed form, with an explicit actuation delay — the parameter that
decides whether forecasting can pay at all.

**5. Product.** Read-only. Every response that shows a comparison carries its assumptions as
structured data, so a client cannot render the numbers without the caveats. Nothing here
actuates a real cluster, and the deployed snapshot declares itself a demo rather than
inferring liveness from freshness.

## What is deliberately absent

An earlier revision of this document described a sixth layer of **specialist agents**
exchanging typed proposals under a supervisor, with a decision ledger. **That layer was cut
and does not exist.** Its own pre-registered expectation (Q5) was that it would not improve
decision quality, only auditability, so cutting it removed a likely-null result rather than a
likely finding. Nothing in the product claims a ledger, and this document should not have
gone on describing one. The cut and its reasoning are recorded in [`ROADMAP.md`](ROADMAP.md).

The database spine implements layer 1. Later layers hold their state in baked snapshots
rather than in schemas, because every experiment here takes minutes to hours and none of it
can run inside a web request.
