"""
nodes/generator.py
==================

Generator node: produces a grounded answer using the local LLM.

This node runs after retrieval and:

    1. Builds a prompt from the user's question and the retrieved documents
       (see ``utils.prompts.build_generation_prompt``).
    2. Passes the prompt to the local Hugging Face model.
    3. Returns the generated answer to the state.

Grounding is enforced by the prompt: the model is instructed to answer ONLY
from the provided context and to explicitly say when the answer is missing.
The generator never reaches outside the retrieved documents.
"""

from __future__ import annotations

from config import SafeResponses
from models.llm import generate_answer
from utils.logger import get_logger
from utils.prompts import build_generation_prompt

_INCOMPLETE_TAILS = {
    "if",
    "and",
    "or",
    "but",
    "because",
    "when",
    "while",
    "since",
    "then",
    "therefore",
    "however",
}


def _looks_incomplete(answer: str) -> bool:
    """Return True when the generated answer appears truncated or unfinished."""
    text = answer.strip()
    if not text:
        return True

    lowered = text.lower()
    if lowered.endswith(tuple(sorted(_INCOMPLETE_TAILS, key=len, reverse=True))):
        return True

    # A concise grounded answer may legitimately be very short (for example,
    # "Resave the schedule." or "Open the schedule."). Reject only obvious
    # mid-sentence fragments; do not reject valid answers merely because they are
    # brief or omit a domain keyword.
    if text.endswith((".", "!", "?")):
        return False

    return False

logger = get_logger(__name__)


def generate_answer_for_question(
    question: str,
    documents: list[dict],
) -> str:
    """
    Generate a grounded answer for a question given retrieved documents.

    Parameters
    ----------
    question : str
        The user's question.
    documents : list[dict]
        Retrieved documents, each with ``filename`` and ``content``.

    Returns
    -------
    str
        The generated answer. If the model fails or produces an empty answer,
        the safe "not in knowledge base" response is returned.
    """
    if not documents:
        logger.warning("No documents provided to generator; returning safe response.")
        return SafeResponses.NOT_IN_KNOWLEDGE_BASE

    prompt = build_generation_prompt(question, documents)

    try:
        answer = generate_answer(prompt)
    except RuntimeError as exc:
        logger.error("Generator failed: %s", exc)
        return SafeResponses.NOT_IN_KNOWLEDGE_BASE

    answer = answer.strip()

    # Guard against empty / whitespace-only generations.
    if not answer:
        logger.warning("LLM returned an empty answer.")
        return SafeResponses.NOT_IN_KNOWLEDGE_BASE

    if _looks_incomplete(answer):
        logger.warning("LLM returned an incomplete answer; rejecting it.")
        return SafeResponses.NOT_IN_KNOWLEDGE_BASE

    return answer


# LangGraph node signature: takes full state, returns a partial update.
def run_generator(state: dict) -> dict:
    logger.info("Running Generator...")
    question = str(state.get("question", ""))
    documents = state.get("documents", []) or []

    answer = generate_answer_for_question(question, documents)
    logger.info("Generated answer (%d chars).", len(answer))

    return {"answer": answer}
