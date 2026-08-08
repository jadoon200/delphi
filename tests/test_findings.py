"""The shipped findings must not contradict `docs/EVAL.md`.

This has now gone wrong three times, in the same way each time: a measurement changed, the
evaluation was updated, and a string baked into the served surface was not. The verdict text
claimed nothing below 0.50 ever won; the explainer claimed a scope the data did not support;
and the Q13 finding went on asserting that daily autocorrelation "predicted every outcome
measured so far" for hours after the experiment that refuted it.

Prose cannot be fully pinned by a test. What *can* be pinned is that a finding does not
assert the specific things the evaluation has since disproved, and that the numbers it
quotes are the ones the current scripts actually produce.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_snapshot import findings  # type: ignore[import-not-found]

from delphi.api.snapshot import Finding


@pytest.fixture(scope="module")
def by_id() -> dict[str, Finding]:
    return {finding.question_id: finding for finding in findings()}


def test_every_finding_carries_a_verdict_and_its_evidence(by_id: dict[str, Finding]) -> None:
    assert by_id, "the snapshot must ship findings"
    for finding in by_id.values():
        assert finding.verdict in {"confirmed", "refuted", "open"}
        assert len(finding.evidence) > 80, f"{finding.question_id} evidence is too thin"
        assert finding.prior, f"{finding.question_id} has no pre-registered prior"


def test_q13_does_not_claim_the_threshold_generalises(by_id: dict[str, Finding]) -> None:
    """Q13 was refuted on 43 serverless workloads. The shipped text must say so."""
    q13 = by_id["Q13"]
    text = f"{q13.answer} {q13.evidence}".lower()

    assert q13.verdict == "refuted"
    assert "predicted every outcome" not in text, "this is the claim Q13 disproved"
    assert "coin flip" in text or "51%" in text, "the negative result must be stated"
    assert "serverless" in text or "azure functions" in text, "the scope must be named"


def test_q1_records_the_percentile_recommender_as_the_incumbent(by_id: dict[str, Finding]) -> None:
    """Beating threshold HPA proves little; the finding must name what it measured against."""
    q1 = by_id["Q1"]
    text = f"{q1.answer} {q1.evidence}".lower()
    assert q1.verdict == "refuted"
    assert "percentile" in text


def test_q4_quotes_counts_the_frontier_script_can_produce(by_id: dict[str, Finding]) -> None:
    """The previous text claimed 19 of 24; the script emits three traces x six ratios."""
    q4 = by_id["Q4"]
    text = f"{q4.answer} {q4.evidence}"
    assert "19 of 24" not in text, "that count is not reproducible from evaluate_frontier.py"
    assert "18" in text or "9 of 18" in text


def test_no_finding_claims_an_unqualified_win_for_forecasting(by_id: dict[str, Finding]) -> None:
    """The project's headline is that forecasting loses to the incumbent in the autoscaling
    regime and wins only under commitment. No finding may state the unscoped version."""
    for finding in by_id.values():
        text = f"{finding.answer} {finding.evidence}".lower()
        if "forecasting" in text and finding.verdict == "confirmed":
            assert any(
                word in text
                for word in ("commitment", "aggregat", "below 19", "price", "static", "week")
            ), f"{finding.question_id} claims a win without naming the regime it holds in"
