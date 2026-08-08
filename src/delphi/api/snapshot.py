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

#: How much weight the diagnostic itself can carry at a given reading. ``borderline`` is not
#: a hedge — it is the range where the measured ordering demonstrably broke down, and
#: collapsing it into the yes/no answer is how the shipped verdict came to overclaim.
DiagnosticBand = Literal["strong", "borderline", "weak", "none"]


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
    #: How far this reading can be trusted. Read this before ``forecastable``.
    band: DiagnosticBand = "weak"
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


#: The threshold the evaluation established. It is a rule of thumb, not a calibrated
#: boundary: across 7 workloads x 4 forecasters x 4 quantiles it calls 27 of 28 cells
#: correctly at a six-hour commitment and 26 of 28 at twelve hours. The known exception runs
#: *against* the rule — `materna-2`, at r = 0.450, won 2 of 4 cells at twelve hours while
#: `materna-1` at r = 0.494 won none — so a value near 0.45-0.50 does not settle the
#: question. See `docs/EVAL.md`; Q13 is open.
FORECASTABLE_AUTOCORRELATION = 0.50

#: Below this, the evaluation found no exceptions at all: no forecaster won a single cell.
NO_STRUCTURE_AUTOCORRELATION = 0.30

#: Width of the band around the threshold where the ordering was observed to break down.
INDETERMINATE_BAND = (0.40, 0.55)

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


def classify_band(daily_autocorrelation: float) -> DiagnosticBand:
    """Which evidential band a reading falls in. The single source of this threshold."""
    low, high = INDETERMINATE_BAND
    if low <= daily_autocorrelation <= high:
        return "borderline"
    if daily_autocorrelation > high:
        return "strong"
    if daily_autocorrelation >= NO_STRUCTURE_AUTOCORRELATION:
        return "weak"
    return "none"


def classify(daily_autocorrelation: float) -> tuple[bool, str]:
    """Turn the measured diagnostic into a plain-language verdict.

    The verdict states its own reliability. Near the threshold the measured ordering
    genuinely broke down, and saying so is the point of shipping a diagnostic rather than
    a number.
    """
    band = classify_band(daily_autocorrelation)
    if band == "borderline":
        return daily_autocorrelation >= FORECASTABLE_AUTOCORRELATION, (
            "Borderline daily structure, and this is the range where the diagnostic is "
            "least trustworthy. Two traces from the same provider inverted here: one at "
            "0.450 beat a trailing percentile at a twelve-hour commitment while one at "
            "0.494 did not. Treat this as 'measure it yourself', not as an answer."
        )
    if band == "strong":
        return True, (
            "Strong daily structure. Forecasting is expected to pay at commitment windows "
            "of roughly six hours or longer, where reaction is impossible."
        )
    if band == "weak":
        return False, (
            "Weak daily structure. No forecaster tested beat a trailing percentile here at "
            "a six-hour commitment; the one exception measured, at twelve hours, sat higher "
            "in the borderline band."
        )
    return False, (
        "Effectively no daily structure. A trailing percentile — or at long horizons a flat "
        "average — is the right tool; an explicit forecast only adds its own error."
    )
