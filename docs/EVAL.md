# Evaluation

> A trace records demand under the original operator's policy. Replaying it against a different
> capacity policy is a counterfactual and is valid only when arrivals are independent of the
> capacity decision. DELPHI reports replay rankings and Pareto frontiers; it does not claim
> measured production savings.

No forecast or controller results exist yet. This is deliberate: M0 establishes the instruments
that future results depend on.

Every result will state chronological split boundaries, sample size, trace granularity, simulator
fidelity, tuning trial count, and seed. Forecasts report MASE and weighted quantile loss alongside
empirical quantile coverage and sharpness. Controllers report the cost-versus-violation frontier,
regret, capacity-hours, churn, escalations, and decision latency.

The simulator is not trusted until it reproduces Erlang-C, passes degenerate cases, is deterministic,
and survives sensitivity sweeps over every assumed constant. Negative controls—shuffled demand,
reversed time, and near-noise traces—must perform poorly or the evaluation is measuring leakage.

