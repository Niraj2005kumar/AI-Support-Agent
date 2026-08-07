"""
state.py
========

Typed state definitions for the LangGraph workflow.

In LangGraph, a *state* is a shared, mutable dictionary that is passed from
node to node. Each node reads the fields it needs and returns a partial update
that is merged back into the graph state.

This module defines:

    * ``AnswerState``   — the primary state flowing through the graph.
    * ``TriageDecision``— a small dataclass describing the triage outcome.
    * ``RetrievedDoc``  — a dataclass describing a single retrieved document.

Using ``TypedDict`` keeps the state lightweight (no Pydantic overhead per
node), while dataclasses give us clear, type-checked structures for the
intermediate objects produced by the triage and retrieval stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict


class TriageLabel(str, Enum):
    """
    The four possible outcomes of the triage node.

    These values are stored in the state as strings and drive the conditional
    routing in the LangGraph edges.
    """

    ANSWERABLE = "Answerable"
    CLARIFICATION_REQUIRED = "Clarification Required"
    OUT_OF_SCOPE = "Out of Scope"
    ESCALATION_REQUIRED = "Escalation Required"


class VerificationStatus(str, Enum):
    """Possible verification outcomes for a generated answer."""

    PASSED = "Passed"
    FAILED = "Failed"
    NOT_APPLICABLE = "Not Applicable"


class ConfidenceLevel(str, Enum):
    """Human-readable confidence levels mapped from a numeric score."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class FinalStatus(str, Enum):
    """Overall status of the pipeline run."""

    SUCCESS = "Success"
    SAFE_RESPONSE = "Safe Response"
    ESCALATED = "Escalated"
    FAILED = "Failed"


# ---------------------------------------------------------------------------
# Intermediate data structures
# ---------------------------------------------------------------------------
@dataclass
class RetrievedDoc:
    """
    A single document retrieved from the FAISS index.

    Attributes
    ----------
    filename : str
        Name of the source Markdown file (or case id) in the knowledge base.
    content : str
        The full text content of the document.
    score : float
        Relevance score returned by FAISS (lower is more relevant for L2).
    metadata : dict[str, Any]
        Extra metadata (source path, case id, tags, etc.).
    """

    filename: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriageDecision:
    """
    The result of analysing the user's question in the triage node.

    Attributes
    ----------
    label : TriageLabel
        The classification of the question.
    follow_up_question : str | None
        A clarifying question to ask the user, if applicable.
    reason : str
        A short, human-readable reason for the classification.
    """

    label: TriageLabel
    follow_up_question: str | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class AnswerState(TypedDict, total=False):
    """
    The shared state object flowing through the LangGraph.

    Because ``total=False`` all fields are optional; nodes write only the
    subset they own. This keeps the contract flexible and forward-compatible.

    Fields
    ------
    question : str
        The user's original question.
    triage_label : str
        The classification produced by the triage node.
    follow_up_question : str | None
        A clarifying question to ask the user, if needed.
    documents : list[RetrievedDoc]
        Documents retrieved by the retrieval node.
    answer : str
        The final, generated answer text.
    sources : list[str]
        Filenames of the documents that grounded the answer.
    verification : str
        Verification status (see VerificationStatus).
    verification_notes : str
        Human-readable notes from the verifier.
    confidence : str
        Confidence level (see ConfidenceLevel).
    confidence_score : float
        Numeric confidence in the range [0, 1].
    status : str
        Overall pipeline status (see FinalStatus).
    retry_count : int
        Number of generation retries performed so far.
    raw_output : dict[str, Any]
        The final, schema-conforming output dictionary.
    """

    question: str
    triage_label: str
    follow_up_question: str | None
    documents: list[RetrievedDoc]
    answer: str
    sources: list[str]
    verification: str
    verification_notes: str
    confidence: str
    confidence_score: float
    status: str
    retry_count: int
    raw_output: dict[str, Any]


def initial_state(question: str) -> AnswerState:
    """
    Build an initial graph state from a user question.

    Parameters
    ----------
    question : str
        The user's question to process.

    Returns
    -------
    AnswerState
        A fully initialised state dictionary with safe defaults.
    """
    return {
        "question": question,
        "triage_label": "",
        "follow_up_question": None,
        "documents": [],
        "answer": "",
        "sources": [],
        "verification": VerificationStatus.NOT_APPLICABLE.value,
        "verification_notes": "",
        "confidence": ConfidenceLevel.LOW.value,
        "confidence_score": 0.0,
        "status": FinalStatus.FAILED.value,
        "retry_count": 0,
        "raw_output": {},
    }
