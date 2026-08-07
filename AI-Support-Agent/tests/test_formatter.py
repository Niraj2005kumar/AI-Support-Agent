"""
tests/test_formatter.py
=======================

Unit tests for the formatter node's output-shaping logic.
"""

from __future__ import annotations

from nodes.formatter import compute_confidence, format_output
from state import (
    ConfidenceLevel,
    FinalStatus,
    TriageLabel,
    VerificationStatus,
)


class TestComputeConfidence:
    """Confidence level mapping."""

    def test_high_confidence(self) -> None:
        assert compute_confidence(0.9) == ConfidenceLevel.HIGH.value

    def test_medium_confidence(self) -> None:
        assert compute_confidence(0.6) == ConfidenceLevel.MEDIUM.value

    def test_low_confidence(self) -> None:
        assert compute_confidence(0.2) == ConfidenceLevel.LOW.value


class TestFormatOutput:
    """Final output shape and status resolution."""

    def test_success_output_shape(self, answerable_state) -> None:
        out = format_output(answerable_state)
        assert out["question"] == answerable_state["question"]
        assert out["sources"] == answerable_state["sources"]
        assert out["verification"] == VerificationStatus.PASSED.value
        assert out["status"] == FinalStatus.SUCCESS.value
        assert out["confidence"] in {
            ConfidenceLevel.HIGH.value,
            ConfidenceLevel.MEDIUM.value,
            ConfidenceLevel.LOW.value,
        }

    def test_out_of_scope_safe_response(self, out_of_scope_state) -> None:
        out = format_output(out_of_scope_state)
        assert out["status"] == FinalStatus.SAFE_RESPONSE.value
        assert out["verification"] == VerificationStatus.NOT_APPLICABLE.value

    def test_verification_failed_returns_safe_failure(self) -> None:
        state = {
            "question": "q",
            "answer": "some unverified answer",
            "sources": [],
            "verification": VerificationStatus.FAILED.value,
            "retry_count": 1,  # >= MAX_RETRIES
            "documents": [],
        }
        out = format_output(state)
        assert out["status"] == FinalStatus.FAILED.value
        assert "couldn't confidently verify" in out["answer"].lower()

    def test_escalated_status(self) -> None:
        state = {
            "question": "q",
            "answer": "",
            "sources": [],
            "triage_label": TriageLabel.ESCALATION_REQUIRED.value,
            "verification": VerificationStatus.NOT_APPLICABLE.value,
            "retry_count": 0,
            "documents": [],
        }
        out = format_output(state)
        assert out["status"] == FinalStatus.ESCALATED.value
