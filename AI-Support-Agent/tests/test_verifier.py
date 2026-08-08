"""
tests/test_verifier.py
======================

Unit tests for the verifier node's grounding checks.

The lexical-support path is deterministic and does not require an LLM, so it
can be tested directly.
"""

from __future__ import annotations

from nodes.generator import generate_answer_for_question
from nodes.verifier import _keyword_support, verify_answer


class TestKeywordSupport:
    """The lexical-overlap heuristic."""

    def test_high_overlap_returns_high_support(self, sample_documents) -> None:
        answer = (
            "Only Owners and Admins can create workspace API credentials. "
            "Read-only users cannot create credentials."
        )
        support = _keyword_support(answer, sample_documents)
        assert support >= 0.35

    def test_low_overlap_returns_low_support(self, sample_documents) -> None:
        answer = "The sky is blue and bananas are yellow."
        support = _keyword_support(answer, sample_documents)
        assert support < 0.35


class TestVerifyAnswer:
    """End-to-end verify logic (lexical path)."""

    def test_grounded_answer_passes(self, sample_documents) -> None:
        answer = (
            "No. Only Owners and Admins can create workspace API credentials."
        )
        passed, note = verify_answer("Can a read-only user create API credentials?", answer, sample_documents)
        assert passed is True
        assert note

    def test_hallucinated_answer_fails(self, sample_documents) -> None:
        # Contains mostly unsupported content.
        answer = "SpaceX will launch a rocket to Mars next Tuesday."
        passed, _note = verify_answer("What is the capital of France?", answer, sample_documents)
        assert passed is False

    def test_short_valid_answer_is_not_rejected(self, sample_documents, monkeypatch) -> None:
        documents = [
            {
                "filename": "03_scheduled_exports.md",
                "content": (
                    "Changing the workspace timezone does not immediately rewrite existing recurring "
                    "export schedules. Existing schedules retain the timezone stored when they were "
                    "last saved and display a Timezone update pending notice. To apply the new "
                    "workspace timezone to an existing recurring schedule: open the schedule, review the "
                    "displayed next-run time, select Save schedule, and confirm the notice disappears."
                ),
                "score": 0.91,
                "metadata": {"path": "03_scheduled_exports.md"},
            }
        ]

        monkeypatch.setattr("nodes.generator.generate_answer", lambda _prompt: "Resave the schedule.")

        answer = generate_answer_for_question(
            "My scheduled exports stopped after changing workspace timezone.",
            documents,
        )

        assert answer == "Resave the schedule."

    def test_empty_answer_fails(self, sample_documents) -> None:
        passed, note = verify_answer("question", "", sample_documents)
        assert passed is False
        assert "empty" in note.lower()
