"""
tests/test_triage.py
====================

Unit tests for the triage node's classification logic.

These tests exercise the deterministic keyword rules. Semantic-similarity
paths are covered separately / optionally since they require the embedding
model.
"""

from __future__ import annotations

import pytest

from nodes.triage import triage_question
from state import TriageLabel


class TestTriageAnswerable:
    """Questions that should be classified as answerable."""

    @pytest.mark.parametrize(
        "question",
        [
            "Can a read-only user create API credentials?",
            "What roles and permissions does OrbitDesk have?",
            "How do I regenerate my API key?",
            "Who can invite team members to the workspace?",
        ],
    )
    def test_domain_keyword_questions_are_answerable(self, question: str) -> None:
        decision = triage_question(question)
        assert decision.label == TriageLabel.ANSWERABLE


class TestTriageOutOfScope:
    """Questions that are clearly unrelated to OrbitDesk."""

    @pytest.mark.parametrize(
        "question",
        [
            "What is the weather in Paris today?",
            "Tell me a recipe for chocolate cake.",
            "Who won the football match last night?",
        ],
    )
    def test_unrelated_questions_are_out_of_scope(self, question: str) -> None:
        decision = triage_question(question)
        assert decision.label == TriageLabel.OUT_OF_SCOPE


class TestTriageClarification:
    """Ambiguous or too-short questions."""

    @pytest.mark.parametrize(
        "question",
        ["hi", "?", "help", ""],
    )
    def test_vague_questions_need_clarification(self, question: str) -> None:
        decision = triage_question(question)
        assert decision.label == TriageLabel.CLARIFICATION_REQUIRED


class TestTriageEscalation:
    """Questions that trigger escalation to a human."""

    @pytest.mark.parametrize(
        "question",
        [
            "I need a refund for my subscription.",
            "There was a data breach in my account.",
            "I want to file a lawsuit against OrbitDesk.",
        ],
    )
    def test_sensitive_questions_are_escalated(self, question: str) -> None:
        decision = triage_question(question)
        assert decision.label == TriageLabel.ESCALATION_REQUIRED


class TestTriageEdgeCases:
    """Boundary and edge behaviour."""

    def test_empty_question_returns_clarification(self) -> None:
        decision = triage_question("   ")
        assert decision.label == TriageLabel.CLARIFICATION_REQUIRED
