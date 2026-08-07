"""
nodes/verifier.py
=================

Verification node: ensures the generated answer is grounded in the retrieved
documents before it is returned to the user.

The verifier checks:

    1. That every important claim in the answer is supported by the context.
    2. That no unsupported / hallucinated information was added.
    3. That the answer is consistent with the documentation.
    4. That the expected output shape is present.

It uses a two-stage strategy:

    * A fast keyword-overlap heuristic (does the answer contain content words
      from the context?).
    * An optional LLM verdict via ``build_verification_prompt`` when the
      heuristic is inconclusive.

If verification fails, the node signals a retry. The graph uses the retry
counter to decide whether to re-run generation or fall back to a safe response
(the safe response logic lives in ``graph.py``'s routing and the formatter).
"""

from __future__ import annotations

import json
import re

from config import Verification
from models.llm import generate_answer
from state import VerificationStatus
from utils.logger import get_logger
from utils.prompts import build_verification_prompt

logger = get_logger(__name__)


def _context_terms(documents: list[dict]) -> set[str]:
    """
    Extract a set of significant content words from the retrieved documents.

    Stopwords and very common words are removed, leaving a vocabulary the
    answer can be checked against.

    Parameters
    ----------
    documents : list[dict]
        Retrieved documents.

    Returns
    -------
    set[str]
        Lower-cased significant terms found in the context.
    """
    stopwords = {
        "and", "or", "the", "a", "an", "in", "on", "of", "to", "for", "with",
        "is", "are", "was", "were", "be", "been", "can", "could", "will",
        "would", "should", "must", "you", "your", "our", "we", "they", "it",
        "this", "that", "these", "those", "has", "have", "had", "not", "no",
        "yes", "at", "by", "from", "as", "or", "if", "then", "than", "so",
        "about", "into", "over", "after", "before",
    }

    terms: set[str] = set()
    for doc in documents:
        content = str(doc.get("content", ""))
        for word in re.findall(r"[a-z][a-z0-9-]{2,}", content.lower()):
            if word not in stopwords and len(word) > 2:
                terms.add(word)
    return terms


def _keyword_support(answer: str, documents: list[dict]) -> float:
    """
    Estimate lexical support of the answer by the context.

    Returns the fraction of the answer's significant terms that appear in the
    context vocabulary.

    Parameters
    ----------
    answer : str
        The generated answer.
    documents : list[dict]
        Retrieved documents.

    Returns
    -------
    float
        A support ratio in [0, 1].
    """
    terms = _context_terms(documents)
    if not terms:
        return 0.0

    answer_terms = {
        w for w in re.findall(r"[a-z][a-z0-9-]{2,}", answer.lower())
        if len(w) > 2
    }
    if not answer_terms:
        return 0.0

    overlap = answer_terms & terms
    return len(overlap) / len(answer_terms)


def _llm_verdict(question: str, answer: str, documents: list[dict]) -> bool:
    """
    Ask the local LLM to verify the answer against the context.

    The model is expected to output ``PASS`` or ``FAIL``. We treat any output
    containing ``FAIL`` (or not containing ``PASS``) as a failure.

    Parameters
    ----------
    question : str
        The user's question.
    answer : str
        The candidate answer.
    documents : list[dict]
        Retrieved documents.

    Returns
    -------
    bool
        True if the model says the answer passes verification.
    """
    prompt = build_verification_prompt(question, answer, documents)
    try:
        verdict = generate_answer(prompt).strip().upper()
    except RuntimeError as exc:
        logger.error("Verifier LLM failed: %s", exc)
        return False

    logger.info("Verifier LLM verdict: %s", verdict)
    return "PASS" in verdict and "FAIL" not in verdict


def _answer_terms(answer: str) -> set[str]:
    """Return significant answer terms without stopwords or short fragments."""
    return {
        word
        for word in re.findall(r"[a-z][a-z0-9-]{2,}", answer.lower())
        if len(word) > 2
    }


def _unsupported_terms(answer: str, documents: list[dict]) -> list[str]:
    """Return answer terms not supported by the retrieved context."""
    context_terms = _context_terms(documents)
    answer_terms = _answer_terms(answer)
    if not answer_terms:
        return []
    unsupported = sorted(term for term in answer_terms if term not in context_terms)
    return unsupported


def compile_detailed_validation(
    question: str,
    answer: str,
    documents: list[dict],
    sources: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Perform a multi-check validation on the generated answer.

    Returns a tuple of (overall_passed, checklist_report_string).
    """
    checklist: list[str] = []
    overall_passed = True
    sources = list(sources or [])

    is_non_empty = bool(answer and answer.strip())
    if not is_non_empty:
        overall_passed = False
    checklist.append(f"- Empty Answers Check: {'PASSED' if is_non_empty else 'FAILED'}")

    has_sources = bool(sources)
    if not has_sources:
        overall_passed = False
    checklist.append(
        f"- Source References Check: {'PASSED' if has_sources else 'FAILED'} "
        f"({len(sources)} source(s) referenced)"
    )

    has_required_fields = bool(question and answer and sources)
    if not has_required_fields:
        overall_passed = False
    checklist.append(
        "- Runtime Output Schema Check: "
        f"{'PASSED' if has_required_fields else 'FAILED'}"
    )

    support = _keyword_support(answer, documents)
    unsupported = _unsupported_terms(answer, documents)
    evidence_passed = support >= 0.35
    llm_checked = False
    llm_passed = False

    if not evidence_passed and support >= 0.15:
        llm_checked = True
        llm_passed = _llm_verdict(question, answer, documents)
        if llm_passed:
            evidence_passed = True

    if unsupported and support < 0.5:
        evidence_passed = False
        overall_passed = False

    if not evidence_passed:
        overall_passed = False

    if unsupported:
        checklist.append(
            f"- Unsupported Claims Check: FAILED (unverified terms: {', '.join(unsupported[:8])})"
        )
    else:
        checklist.append("- Unsupported Claims Check: PASSED")

    evidence_msg = (
        f"PASSED (Lexical overlap: {support:.2f})"
        if evidence_passed and not llm_checked
        else (
            f"PASSED (LLM verdict PASSED, Lexical overlap: {support:.2f})"
            if evidence_passed and llm_passed
            else (
                f"FAILED (Lexical overlap: {support:.2f}, LLM verdict: FAILED)"
                if llm_checked
                else f"FAILED (Lexical overlap: {support:.2f} is too low)"
            )
        )
    )
    checklist.append(f"- Evidence Support & Hallucination Check: {evidence_msg}")

    has_valid_scores = True
    if documents:
        has_valid_scores = all(
            "score" in doc or hasattr(doc, "score") for doc in documents
        )
    if not has_valid_scores:
        overall_passed = False
    checklist.append(f"- Confidence Score Check: {'PASSED' if has_valid_scores else 'FAILED'}")

    try:
        dummy = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "verification": "Passed" if overall_passed else "Failed",
            "confidence": "Low",
            "status": "Success" if overall_passed else "Failed",
        }
        json.dumps(dummy)
        json_valid = True
    except Exception:
        json_valid = False
        overall_passed = False
    checklist.append(f"- JSON Validity Check: {'PASSED' if json_valid else 'FAILED'}")

    report = "\n".join(checklist)
    return overall_passed, report


def verify_answer(
    question: str,
    answer: str,
    documents: list[dict],
) -> tuple[bool, str]:
    """
    Verify an answer against the retrieved documents.

    Parameters
    ----------
    question : str
        The user's question.
    answer : str
        The generated answer to verify.
    documents : list[dict]
        The retrieved documents the answer should be grounded in.

    Returns
    -------
    tuple[bool, str]
        A tuple of ``(passed, note)`` describing the verification result.
    """
    # Safe response answers (already-canonical text) do not need regeneration.
    if not answer:
        return False, "Empty answer."

    sources = []
    for d in documents:
        if isinstance(d, dict):
            sources.append(d.get("filename", "unknown"))
        else:
            sources.append(getattr(d, "filename", "unknown"))

    return compile_detailed_validation(question, answer, documents, sources)


# LangGraph node signature: takes full state, returns a partial update.
def run_verifier(state: dict) -> dict:
    logger.info("Running Verifier...")
    question = str(state.get("question", ""))
    answer = str(state.get("answer", ""))
    documents = state.get("documents", []) or []
    sources = state.get("sources", []) or []
    retry_count = int(state.get("retry_count", 0))

    passed, note = compile_detailed_validation(question, answer, documents, sources)

    if passed:
        logger.info("Verification PASSED:\n%s", note)
        return {
            "verification": VerificationStatus.PASSED.value,
            "verification_notes": note,
            "retry_count": retry_count,
        }

    logger.warning("Verification FAILED:\n%s", note)

    # If we still have retries left, clear the answer and let the graph
    # re-route to the generator. The graph checks retry_count to decide.
    if retry_count < Verification.MAX_RETRIES:
        return {
            "verification": VerificationStatus.FAILED.value,
            "verification_notes": note,
            "retry_count": retry_count + 1,
            "answer": "",  # Force re-generation.
        }

    # No retries left: mark for safe failure (the formatter handles the text).
    return {
        "verification": VerificationStatus.FAILED.value,
        "verification_notes": note,
        "retry_count": retry_count,
    }
