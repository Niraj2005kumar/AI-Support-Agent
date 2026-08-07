
from __future__ import annotations

from config import SafeResponses, Verification
from state import ConfidenceLevel, FinalStatus, TriageLabel, VerificationStatus
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_VERIFICATION_VALUES = {"Passed", "Failed", "Not Applicable"}
VALID_CONFIDENCE_VALUES = {"High", "Medium", "Low"}
VALID_STATUS_VALUES = {"Success", "Safe Response", "Escalated", "Failed"}
REQUIRED_OUTPUT_FIELDS = [
    "question",
    "answer",
    "sources",
    "verification",
    "confidence",
    "status",
]


def compute_confidence(score: float) -> str:

    if score >= Verification.HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.HIGH.value
    if score >= Verification.MEDIUM_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.MEDIUM.value
    return ConfidenceLevel.LOW.value


def _resolve_final_status(state: dict) -> str:

    triage_label = state.get("triage_label", "")
    answer = state.get("answer", "")
    verification = state.get("verification", VerificationStatus.NOT_APPLICABLE.value)
    retry_count = int(state.get("retry_count", 0))

    if triage_label == TriageLabel.ESCALATION_REQUIRED.value:
        return FinalStatus.ESCALATED.value

    # Safe responses that bypass the generator or match corresponding triage labels
    if triage_label in {TriageLabel.OUT_OF_SCOPE.value, TriageLabel.CLARIFICATION_REQUIRED.value}:
        return FinalStatus.SAFE_RESPONSE.value

    if answer in {
        SafeResponses.OUT_OF_SCOPE,
        SafeResponses.NEEDS_CLARIFICATION,
        SafeResponses.NOT_IN_KNOWLEDGE_BASE,
    }:
        return FinalStatus.SAFE_RESPONSE.value

    if verification == VerificationStatus.PASSED.value:
        return FinalStatus.SUCCESS.value

    if verification == VerificationStatus.FAILED.value and retry_count >= Verification.MAX_RETRIES:
        return FinalStatus.FAILED.value

    return FinalStatus.FAILED.value


def _compute_confidence_score(state: dict) -> float:

    verification = state.get("verification", VerificationStatus.NOT_APPLICABLE.value)


    if verification == VerificationStatus.PASSED.value:
        base = 0.85
    elif verification == VerificationStatus.NOT_APPLICABLE.value:
        base = 0.5
    else:
        base = 0.2

    docs = state.get("documents", []) or []
    if docs:
        avg_score = sum(float(d.get("score", 0.0)) for d in docs) / len(docs)
        base = 0.5 * base + 0.5 * min(avg_score, 1.0)

    return round(min(max(base, 0.0), 1.0), 3)


def _validate_runtime_output(output: dict) -> dict:

    if not isinstance(output, dict):
        raise ValueError("Runtime output must be a dictionary.")

    missing = [field for field in REQUIRED_OUTPUT_FIELDS if field not in output]
    if missing:
        raise ValueError(f"Missing required output fields: {missing}")

    question = output.get("question", "")
    answer = output.get("answer", "")
    sources = output.get("sources", [])
    verification = output.get("verification", "")
    confidence = output.get("confidence", "")
    status = output.get("status", "")

    if not isinstance(question, str):
        raise ValueError("Question must be a string.")
    if not isinstance(answer, str):
        raise ValueError("Answer must be a string.")
    if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
        raise ValueError("Sources must be a list of strings.")
    if verification not in VALID_VERIFICATION_VALUES:
        raise ValueError(f"Verification value is invalid: {verification!r}")
    if confidence not in VALID_CONFIDENCE_VALUES:
        raise ValueError(f"Confidence value is invalid: {confidence!r}")
    if status not in VALID_STATUS_VALUES:
        raise ValueError(f"Status value is invalid: {status!r}")
    if status == "Success" and not answer.strip():
        raise ValueError("Answer cannot be empty.")

    return output


def format_output(state: dict) -> dict:

    question = str(state.get("question", ""))
    answer = str(state.get("answer", ""))
    sources = list(state.get("sources", []) or [])
    verification = str(state.get("verification", VerificationStatus.NOT_APPLICABLE.value))
    confidence_score = _compute_confidence_score(state)
    confidence = compute_confidence(confidence_score)
    status = _resolve_final_status(state)

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

    try:
        validated = _validate_runtime_output(output)
    except ValueError as exc:
        logger.error("Runtime output validation failed: %s", exc)
        safe_output = {
            "question": question,
            "answer": SafeResponses.VERIFICATION_FAILED,
            "sources": [],
            "verification": VerificationStatus.FAILED.value,
            "confidence": ConfidenceLevel.LOW.value,
            "status": FinalStatus.FAILED.value,
        }
        logger.warning("Returning safe fallback output.")
        return _validate_runtime_output(safe_output)

    logger.info("Formatted output: status=%s, verification=%s", status, verification)
    return validated


# LangGraph node signature: takes full state, returns a partial update.
def run_formatter(state: dict) -> dict:
    logger.info("Formatting Output...")
    output = format_output(state)
    return {"raw_output": output}
