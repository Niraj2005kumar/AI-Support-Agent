"""
tests/test_graph.py
===================

Unit tests for the graph routing functions.

These tests exercise the conditional route logic without compiling the full
graph, keeping them fast and model-free.
"""

from __future__ import annotations

import graph
from state import TriageLabel, VerificationStatus
from config import Verification

RETRIEVAL = graph.NODE_RETRIEVAL
FORMATTER = graph.NODE_FORMATTER
GENERATOR = graph.NODE_GENERATOR


class TestRouteAfterTriage:
    """Routing after the triage node."""

    def test_answerable_goes_to_retrieval(self) -> None:
        state = {"triage_label": TriageLabel.ANSWERABLE.value}
        assert graph._route_after_triage(state) == RETRIEVAL

    def test_out_of_scope_goes_to_formatter(self) -> None:
        state = {"triage_label": TriageLabel.OUT_OF_SCOPE.value}
        assert graph._route_after_triage(state) == FORMATTER

    def test_clarification_required_goes_to_formatter(self) -> None:
        state = {"triage_label": TriageLabel.CLARIFICATION_REQUIRED.value}
        assert graph._route_after_triage(state) == FORMATTER

    def test_escalation_goes_to_formatter(self) -> None:
        state = {"triage_label": TriageLabel.ESCALATION_REQUIRED.value}
        assert graph._route_after_triage(state) == FORMATTER

    def test_direct_answer_question_stays_answerable(self) -> None:
        state = {"triage_label": TriageLabel.ANSWERABLE.value}
        assert graph._route_after_triage(state) == RETRIEVAL

    def test_clarification_required_question_routes_to_formatter(self) -> None:
        state = {"triage_label": TriageLabel.CLARIFICATION_REQUIRED.value}
        assert graph._route_after_triage(state) == FORMATTER

    def test_out_of_scope_question_routes_to_formatter(self) -> None:
        state = {"triage_label": TriageLabel.OUT_OF_SCOPE.value}
        assert graph._route_after_triage(state) == FORMATTER


class TestRouteAfterVerifier:
    """Routing after the verifier node."""

    def test_passed_goes_to_formatter(self) -> None:
        state = {"verification": VerificationStatus.PASSED.value, "retry_count": 0}
        assert graph._route_after_verifier(state) == FORMATTER

    def test_failed_with_retries_goes_to_generator(self) -> None:
        state = {
            "verification": VerificationStatus.FAILED.value,
            "retry_count": 0,
        }
        assert graph._route_after_verifier(state) == GENERATOR

    def test_failed_without_retries_goes_to_formatter(self) -> None:
        state = {
            "verification": VerificationStatus.FAILED.value,
            "retry_count": Verification.MAX_RETRIES,
        }
        assert graph._route_after_verifier(state) == FORMATTER

    def test_verification_failure_retries_generator(self) -> None:
        state = {
            "verification": VerificationStatus.FAILED.value,
            "retry_count": Verification.MAX_RETRIES - 1,
        }
        assert graph._route_after_verifier(state) == GENERATOR
