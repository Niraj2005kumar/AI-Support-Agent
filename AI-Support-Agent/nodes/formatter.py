"""
nodes/formatter.py
==================

Formatter node: shapes the final pipeline state into the schema-conforming
output dictionary returned to the user.

The formatter is the single place that decides the user-visible fields:

    * ``question``  — the user's question.
    * ``answer``    — the final answer (or safe response).
    * ``sources``   — the retrieved documents that grounded the answer.
    * ``verification`` — Passed / Failed / Not Applicable.
    * ``confidence``   — High / Medium / Low.
    * ``status``       — Success / Safe Response / Escalated / Failed.

It also computes the human-readable confidence level from a numeric score and
finalises the overall status based on the triage label, verification result,
and retry state.
"""

from __future__ import annotations

from config import SafeResponses, Verification
from state import ConfidenceLevel, FinalStatus, TriageLabel, VerificationStatus
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_confidence(score: float) -> str:
    """
    Map a numeric confidence score to a human-readable level.

    Parameters
    ----------
    score : float
        Confidence in [0, 1].

    Returns
    -------
    str
        ``High``, ``Medium``, or ``Low``.
    """
    if score >= Verification.HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.HIGH.value
    if score >= Verification.MEDIUM_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.MEDIUM.value
    return ConfidenceLevel.LOW.value


def _resolve_final_status(state: dict) -> str:
    """
    Determine the final overall status from the state.

    Priority order:
        1. Escalation.
        2. Safe responses (out of scope / clarification / not-in-KB).
        3. Verification success -> Success.
        4. Verification failure after retries -> Failed.
        5. Default -> Failed.

    Parameters
    ----------
    state : dict
        The current graph state.

    Returns
    -------
    str
        The final status value.
    """
    triage_label = state.get("triage_label", "")
    answer = state.get("answer", "")
    verification = state.get("verification", VerificationStatus.NOT_APPLICABLE.value)
    retry_count = int(state.get("retry_count", 0))

    if triage_label == TriageLabel.ESCALATION_REQUIRED.value:
        return FinalStatus.ESCALATED.value

    # Safe responses that bypass the generator.
    if answer in {
        SafeResponses.OUT_OF_SCOPE,
        SafeResponses.NEEDS_CLARIFICATION,
        SafeResponses.NOT_IN_KNOWLEDGE_BASE,
    }:
        return FinalStatus.SAFE_RESPONSE.value

    if verification == VerificationStatus.PASSED.value:
        return FinalStatus.SUCCESS.value

    # Verification failed and we exhausted retries.
    if verification == VerificationStatus.FAILED.value and retry_count >= Verification.MAX_RETRIES:
        return FinalStatus.FAILED.value

    return FinalStatus.FAILED.value


def _compute_confidence_score(state: dict) -> float:
    """
    Compute a numeric confidence score for the answer.

    Uses the verification outcome and the mean retrieval score as a proxy.

    Parameters
    ----------
    state : dict
        The current graph state.

    Returns
    -------
    float
        Confidence score in [0, 1].
    """
    verification = state.get("verification", VerificationStatus.NOT_APPLICABLE.value)

    # Base score from verification.
    if verification == VerificationStatus.PASSED.value:
        base = 0.85
    elif verification == VerificationStatus.NOT_APPLICABLE.value:
        base = 0.5
    else:
        base = 0.2

    # Boost using average retrieval similarity.
    docs = state.get("documents", []) or []
    if docs:
        avg_score = sum(float(d.get("score", 0.0)) for d in docs) / len(docs)
        base = 0.5 * base + 0.5 * min(avg_score, 1.0)

    return round(min(max(base, 0.0), 1.0), 3)


def format_output(state: dict) -> dict:
    """
    Build the final schema-conforming output dictionary.

    Parameters
    ----------
    state : dict
        The current graph state.

    Returns
    -------
    dict
        The final output dictionary with keys matching ``output_schema.json``.
    """
    question = str(state.get("question", ""))
    answer = str(state.get("answer", ""))
    sources = list(state.get("sources", []) or [])
    verification = str(state.get("verification", VerificationStatus.NOT_APPLICABLE.value))
    confidence_score = _compute_confidence_score(state)
    confidence = compute_confidence(confidence_score)
    status = _resolve_final_status(state)

    # If verification ultimately failed, replace the answer with a safe
    # response rather than risk exposing an unverified answer.
    if (
        verification == VerificationStatus.FAILED.value
        and int(state.get("retry_count", 0)) >= Verification.MAX_RETRIES
    ):
        answer = SafeResponses.VERIFICATION_FAILED
        status = FinalStatus.FAILED.value
        confidence = ConfidenceLevel.LOW.value
        confidence_score = 0.1

    output = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "verification": verification,
        "confidence": confidence,
        "status": status,
    }

    logger.info("Formatted output: status=%s, verification=%s", status, verification)
    return output


# LangGraph node signature: takes full state, returns a partial update.
def run_formatter(state: dict) -> dict:
    """
    LangGraph formatter node.

    Produces the final output dictionary and stores it in
    ``state["raw_output"]``.

    Parameters
    ----------
    state : dict
        The current graph state.

    Returns
    -------
    dict
        A partial state update containing ``raw_output``.
    """
    output = format_output(state)
    return {"raw_output": output}
