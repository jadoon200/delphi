"""The read-only snapshot the API serves.

Every experiment in this project takes minutes to hours — parsing 44M requests, refitting
LightGBM at every commitment boundary. None of that can happen inside a web request, so the
API serves a **precomputed snapshot** built by ``scripts/build_snapshot.py`` and baked into
the deploy image, exactly as the sibling projects do.

The snapshot records how and when it was generated. ``snapshot_mode`` is set explicitly
rather than inferred from freshness: a baked demo whose ``generated_at`` happens to be
recent is still a demo, and inferring otherwise is how a sibling project ended up showing a
"live" badge over synthetic data.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SnapshotMode = Literal["live", "demo", "replay"]


class WorkloadSummary(BaseModel):
    """One workload, with the diagnostic that decides whether forecasting can help it."""

    workload_id: str
    label: str
    source_id: str
    licence: str
    resource_kind: str
    unit: str
    step_seconds: int
    bins: int
    days: float
    #: The headline diagnostic. Measured, not assumed.
    daily_autocorrelation: float
    minute_autocorrelation: float
    weekly_autocorrelation: float | None = None
    peak_to_mean: float
    mean_demand: float
    #: Whether the measured structure predicts that forecasting will pay at a long
    #: commitment. Derived from the threshold the evaluation established, not hand-set.
    forecastable: bool
    verdict: str


class DemandPoint(BaseModel):
    ts: datetime
    value: float
    is_imputed: bool = False


class DemandSeriesPayload(BaseModel):
    workload_id: str
    step_seconds: int
    downsample_factor: int = Field(
        1, description="Points are averaged in blocks of this size to bound response size."
    )
    points: list[DemandPoint]


class HorizonRow(BaseModel):
    """Which predictor wins at a given lead time, and by how much."""

    horizon_label: str
    horizon_steps: int
    trailing_mase: float
    seasonal_mase: float
    winner: str


class FrontierPointPayload(BaseModel):
    controller_id: str
    family: str
    setting: float
    violation_rate: float
    cost: float
    mean_replicas: float
    on_pareto_front: bool


class Finding(BaseModel):
    """A pre-registered question and what the measurement actually said."""

    question_id: str
    question: str
    prior: str
    answer: str
    verdict: Literal["confirmed", "refuted", "mixed", "open"]
    evidence: str


class Snapshot(BaseModel):
    """Everything the dashboard needs, in one immutable document."""

    generated_at: datetime
    snapshot_mode: SnapshotMode
    delphi_version: str
    #: Stated on every response that shows a controller comparison. Not a footnote.
    assumptions: list[str]
    workloads: list[WorkloadSummary]
    horizons: dict[str, list[HorizonRow]]
    frontiers: dict[str, list[FrontierPointPayload]]
    findings: list[Finding]

    def workload(self, workload_id: str) -> WorkloadSummary | None:
        return next((w for w in self.workloads if w.workload_id == workload_id), None)


#: The threshold the evaluation established: below this, no forecaster beat a trailing
#: window on any workload tested, at any commitment length.
FORECASTABLE_AUTOCORRELATION = 0.50

STANDING_ASSUMPTIONS: list[str] = [
    "Open-loop assumption: replaying a trace against a different policy is a counterfactual, "
    "valid only because every arrival process here is externally driven and cannot respond "
    "to provisioning. Rankings transfer; absolute costs carry a sim-to-real disclaimer.",
    "Simulated numbers are a ceiling by construction. Every simplification in the "
    "simulator — the utilisation model, modelled service times, modelled cold starts — "
    "flatters the proactive methods.",
    "DELPHI recommends capacity; it never applies a change to anything real.",
    "Retail list prices are not what an enterprise pays. The shape of a frontier is the "
    "result, not the absolute dollars.",
]


def classify(daily_autocorrelation: float) -> tuple[bool, str]:
    """Turn the measured diagnostic into a plain-language verdict."""
    if daily_autocorrelation >= FORECASTABLE_AUTOCORRELATION:
        return True, (
            "Strong daily structure. Forecasting is expected to pay at commitment windows "
            "of roughly six hours or longer, where reaction is impossible."
        )
    if daily_autocorrelation >= 0.30:
        return False, (
            "Weak daily structure. No forecaster tested beat a trailing percentile at any "
            "commitment length on workloads in this range."
        )
    return False, (
        "Effectively no daily structure. A trailing percentile — or at long horizons a flat "
        "average — is the right tool; an explicit forecast only adds its own error."
    )
