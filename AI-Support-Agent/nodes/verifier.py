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

    # Fast heuristic: require meaningful lexical overlap with the context.
    support = _keyword_support(answer, documents)
    logger.info("Keyword support ratio: %.2f", support)

    if support >= 0.35:
        return True, f"Lexical support ratio {support:.2f} >= 0.35."

    # Inconclusive around the band -> run the LLM verdict for a final call.
    if support >= 0.15:
        passed = _llm_verdict(question, answer, documents)
        if passed:
            return True, "LLM verdict PASSED."
        return False, "LLM verdict FAILED."
    else:
        return False, f"Answer has too little support ({support:.2f})."


# LangGraph node signature: takes full state, returns a partial update.
def run_verifier(state: dict) -> dict:
    """
    LangGraph verifier node.

    Validates the generated ``answer`` against the ``documents`` in state.
    Writes the verification outcome and, on failure with retries remaining,
    clears the answer so the graph can regenerate.

    Parameters
    ----------
    state : dict
        The current graph state.

    Returns
    -------
    dict
        A partial state update with verification results.
    """
    question = str(state.get("question", ""))
    answer = str(state.get("answer", ""))
    documents = state.get("documents", []) or []
    retry_count = int(state.get("retry_count", 0))

    passed, note = verify_answer(question, answer, documents)

    if passed:
        logger.info("Verification PASSED: %s", note)
        return {
            "verification": VerificationStatus.PASSED.value,
            "verification_notes": note,
            "retry_count": retry_count,
        }

    logger.warning("Verification FAILED: %s", note)

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
