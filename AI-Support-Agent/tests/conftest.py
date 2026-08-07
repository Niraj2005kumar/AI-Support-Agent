"""
tests/conftest.py
=================

Shared pytest fixtures for the test suite.

These fixtures provide lightweight, deterministic sample documents and states
so unit tests can run quickly without loading the heavy local models.
"""

from __future__ import annotations

import pytest

from state import (
    ConfidenceLevel,
    FinalStatus,
    TriageLabel,
    VerificationStatus,
)

# ---------------------------------------------------------------------------
# Sample documents (mirroring real knowledge base content)
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    {
        "filename": "02_roles_and_permissions.md",
        "content": (
            "OrbitDesk has four roles: Owner, Admin, Editor, and Read-only. "
            "Only Owners and Admins can create workspace API credentials. "
            "Read-only users cannot modify settings or create credentials."
        ),
        "score": 0.82,
        "metadata": {"path": "02_roles_and_permissions.md"},
    },
    {
        "filename": "05_api_credentials.md",
        "content": (
            "Workspace API credentials can only be created by users with "
            "Owner or Admin roles. Each credential includes a key and secret. "
            "Credentials are used to authenticate API requests."
        ),
        "score": 0.78,
        "metadata": {"path": "05_api_credentials.md"},
    },
]


@pytest.fixture
def sample_documents() -> list[dict]:
    """Return a list of sample retrieved documents."""
    return [dict(d) for d in SAMPLE_DOCS]


@pytest.fixture
def answerable_state(sample_documents) -> dict:
    """Return a state that has passed triage and retrieval."""
    return {
        "question": "Can a read-only user create API credentials?",
        "triage_label": TriageLabel.ANSWERABLE.value,
        "follow_up_question": None,
        "documents": sample_documents,
        "sources": [d["filename"] for d in sample_documents],
        "answer": (
            "No. Only Owners and Admins can create workspace API credentials."
        ),
        "verification": VerificationStatus.PASSED.value,
        "verification_notes": "Lexical support ratio 0.80 >= 0.35.",
        "confidence": ConfidenceLevel.HIGH.value,
        "confidence_score": 0.85,
        "status": FinalStatus.SUCCESS.value,
        "retry_count": 0,
        "raw_output": {},
    }


@pytest.fixture
def out_of_scope_state() -> dict:
    """Return a state classified as out of scope."""
    return {
        "question": "What's the weather today?",
        "triage_label": TriageLabel.OUT_OF_SCOPE.value,
        "follow_up_question": "I can only assist with questions about OrbitDesk.",
        "documents": [],
        "sources": [],
        "answer": "I can only assist with questions about OrbitDesk.",
        "verification": VerificationStatus.NOT_APPLICABLE.value,
        "verification_notes": "",
        "confidence": ConfidenceLevel.LOW.value,
        "confidence_score": 0.5,
        "status": FinalStatus.SAFE_RESPONSE.value,
        "retry_count": 0,
        "raw_output": {},
    }
