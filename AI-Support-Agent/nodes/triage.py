"""
nodes/triage.py
===============

Triage node: classifies the user's question before any retrieval happens.

The triage node decides whether the question can be answered by the knowledge
base at all. It returns a :class:`TriageDecision` with one of four labels:

    * ``Answerable``            — proceed to retrieval.
    * ``Clarification Required``— ask the user a follow-up question.
    * ``Out of Scope``          — refuse safely (not about OrbitDesk).
    * ``Escalation Required``   — hand off to a human agent.

The classification is deterministic and does NOT call the LLM, keeping it fast
and cheap. It uses:

    1. Keyword rules for escalation triggers and out-of-scope signals.
    2. A set of OrbitDesk domain keywords to confirm the topic is in scope.
    3. A semantic similarity check against canonical OrbitDesk topic phrases
       using the local embedding model, for questions that are not obviously
       keyword-matched.
"""

from __future__ import annotations

import re

import numpy as np

from config import SafeResponses
from state import TriageDecision, TriageLabel
from utils.embeddings import encode_query, get_embedding_model
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical topics present in the knowledge base (for semantic matching).
# ---------------------------------------------------------------------------
_TOPIC_PHRASES: list[str] = [
    "OrbitDesk product overview and features",
    "roles and permissions in OrbitDesk",
    "users and team member roles",
    "API credentials and authentication",
    "API keys creation and management",
    "billing and subscription plans",
    "security and safe responses",
    "data privacy and encryption",
    "resolved support cases and troubleshooting",
    "The Great 2025 Data Indexing Outage",
]

# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------
# Signals that strongly indicate the question is about a human-handled issue.
_ESCALATION_KEYWORDS: list[str] = [
    "refund",
    "legal",
    "lawsuit",
    "compliance",
    "security breach",
    "data breach",
    "account hacked",
    "compromised",
    "sue",
    "attorney",
    "police",
    "fraud",
    "emergency",
    "urgent legal",
]

# Broad signals that the question is not about OrbitDesk at all.
_OUT_OF_SCOPE_KEYWORDS: list[str] = [
    "weather",
    "football",
    "cooking",
    "recipe",
    "movie",
    "politics",
    "stock market",
    "your competitor",
    "not orbitdesk",
    "unrelated",
]

# Core OrbitDesk domain vocabulary. If a question contains any of these, it is
# very likely in scope.
_DOMAIN_KEYWORDS: list[str] = [
    "orbitdesk",
    "api",
    "credential",
    "role",
    "roles",
    "permission",
    "permissions",
    "admin",
    "owner",
    "analyst",
    "viewer",
    "read-only",
    "editor",
    "workspace",
    "dashboard",
    "connection",
    "schedule",
    "export",
    "billing",
    "subscription",
    "plan",
    "security",
    "privacy",
    "encryption",
    "data",
    "ticket",
    "support",
    "case",
    "indexing",
    "outage",
    "invite",
    "inviting",
    "member",
    "members",
    "team",
]


def _lower(text: str) -> str:
    """Return a lower-cased, whitespace-normalised copy of ``text``."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Return True if ``text`` contains any of ``keywords`` as substrings."""
    return any(kw in text for kw in keywords)


def _semantic_similarity(question: str) -> float:
    """
    Compute the maximum cosine similarity between the question and the known
    OrbitDesk topic phrases.

    Parameters
    ----------
    question : str
        The user's question.

    Returns
    -------
    float
        The highest similarity score in [0, 1].
    """
    try:
        model = get_embedding_model()
        topic_vectors = model.encode(
            _TOPIC_PHRASES, normalize_embeddings=True, convert_to_numpy=True
        )
        query_vec = model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        )
        # Inner product of unit vectors == cosine similarity.
        sims = np.dot(topic_vectors, query_vec.T).flatten()
        return float(sims.max())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic similarity check failed: %s", exc)
        return 0.0


def _classify(question: str) -> TriageDecision:
    """
    Apply the deterministic rules to classify ``question``.

    Parameters
    ----------
    question : str
        The user's question.

    Returns
    -------
    TriageDecision
        The classification result.
    """
    q = _lower(question)

    # 1. Escalation: serious / human-handled issues take priority.
    if _contains_any(q, _ESCALATION_KEYWORDS):
        logger.info("Triage -> Escalation Required (keyword match).")
        return TriageDecision(
            label=TriageLabel.ESCALATION_REQUIRED,
            reason="Question contains escalation-triggering keyword(s).",
            follow_up_question=SafeResponses.ESCALATED,
        )

    # 2. Out of scope: clearly unrelated to OrbitDesk.
    if _contains_any(q, _OUT_OF_SCOPE_KEYWORDS):
        logger.info("Triage -> Out of Scope (keyword match).")
        return TriageDecision(
            label=TriageLabel.OUT_OF_SCOPE,
            reason="Question appears unrelated to OrbitDesk.",
            follow_up_question=SafeResponses.OUT_OF_SCOPE,
        )

    # 3. Clarification required: too vague / too short to answer well.
    if len(question.strip()) < 8:
        logger.info("Triage -> Clarification Required (too vague).")
        return TriageDecision(
            label=TriageLabel.CLARIFICATION_REQUIRED,
            reason="Question is too short or vague.",
            follow_up_question=(
                "Which aspect of OrbitDesk would you like to know about "
                "(e.g. roles, permissions, API credentials, billing, security)?"
            ),
        )

    if "sync" in q and not any(kw in q for kw in ["slack", "salesforce", "zendesk"]):
        logger.info("Triage -> Clarification Required (vague sync query).")
        return TriageDecision(
            label=TriageLabel.CLARIFICATION_REQUIRED,
            reason="Question references sync but is too vague.",
            follow_up_question=(
                "Could you please specify which sync integration you are referring to "
                "(e.g., Slack sync, Salesforce sync, or Zendesk sync)?"
            ),
        )

    # 4. Domain check: does the question reference an OrbitDesk topic?
    if _contains_any(q, _DOMAIN_KEYWORDS):
        logger.info("Triage -> Answerable (domain keyword match).")
        return TriageDecision(
            label=TriageLabel.ANSWERABLE,
            reason="Question references an OrbitDesk topic.",
        )

    # 5. Semantic fallback: is the question semantically close to a known
    #    OrbitDesk topic even without exact keywords?
    sim = _semantic_similarity(question)
    if sim >= 0.45:
        logger.info("Triage -> Answerable (semantic match, sim=%.2f).", sim)
        return TriageDecision(
            label=TriageLabel.ANSWERABLE,
            reason=f"Semantically related to OrbitDesk (sim={sim:.2f}).",
        )

    # 6. Otherwise treat as out of scope / unclear.
    logger.info("Triage -> Out of Scope (no match, sim=%.2f).", sim)
    return TriageDecision(
        label=TriageLabel.OUT_OF_SCOPE,
        reason="No OrbitDesk topic matched.",
        follow_up_question=SafeResponses.OUT_OF_SCOPE,
    )


def triage_question(question: str) -> TriageDecision:
    """
    Public entry point: classify a user question.

    Parameters
    ----------
    question : str
        The user's question.

    Returns
    -------
    TriageDecision
        The classification result.
    """
    if not question or not question.strip():
        return TriageDecision(
            label=TriageLabel.CLARIFICATION_REQUIRED,
            reason="Empty question.",
            follow_up_question=(
                "Please describe what you'd like help with, e.g. "
                "'Can a read-only user create API credentials?'"
            ),
        )
    return _classify(question)


# LangGraph node signature: takes full state, returns a partial update.
def run_triage(state: dict) -> dict:
    logger.info("Running Triage...")
    question = str(state.get("question", ""))
    decision = triage_question(question)

    logger.info("Triage decision: %s", decision.label.value)

    result = {
        "triage_label": decision.label.value,
        "follow_up_question": decision.follow_up_question,
    }

    # If the question is not answerable, we can pre-populate the final status
    # so the formatter can produce a safe response immediately.
    if decision.label == TriageLabel.OUT_OF_SCOPE:
        result["status"] = "Safe Response"
        result["answer"] = SafeResponses.OUT_OF_SCOPE
    elif decision.label == TriageLabel.CLARIFICATION_REQUIRED:
        result["status"] = "Safe Response"
        result["answer"] = SafeResponses.NEEDS_CLARIFICATION
    elif decision.label == TriageLabel.ESCALATION_REQUIRED:
        result["status"] = "Escalated"
        result["answer"] = SafeResponses.ESCALATED

    return result
