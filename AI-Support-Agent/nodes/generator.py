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

    # Guard against empty / whitespace-only generations.
    if not answer or not answer.strip():
        logger.warning("LLM returned an empty answer.")
        return SafeResponses.NOT_IN_KNOWLEDGE_BASE

    return answer.strip()


# LangGraph node signature: takes full state, returns a partial update.
def run_generator(state: dict) -> dict:
    """
    LangGraph generator node.

    Reads ``question`` and ``documents`` from state, generates an answer, and
    writes it back.

    Parameters
    ----------
    state : dict
        The current graph state.

    Returns
    -------
    dict
        A partial state update containing the generated ``answer``.
    """
    question = str(state.get("question", ""))
    documents = state.get("documents", []) or []

    answer = generate_answer_for_question(question, documents)
    logger.info("Generated answer (%d chars).", len(answer))

    return {"answer": answer}
