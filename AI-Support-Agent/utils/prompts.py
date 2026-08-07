"""
utils/prompts.py
================

Prompt templates for the OrbitDesk AI Support Agent.

All LLM-facing text in the project is centralised here so that:

    * Prompt engineering changes live in one obvious place.
    * The generator, verifier, and triage (when LLM-assisted) share consistent
      instructions.
    * The "grounding" behaviour — answer ONLY from provided context — is
      encoded once and reused everywhere.

Every template is deterministic: it accepts structured inputs and returns a
formatted string ready to pass to the local model.
"""

from __future__ import annotations

from typing import Any

from config import Retrieval, SafeResponses


def _build_context_block(
    documents: list[dict[str, Any]] | list[Any],
    *,
    max_chars: int | None = None,
    max_docs: int | None = None,
) -> str:
    """
    Build a grounded context block from retrieved documents.

    Parameters
    ----------
    documents : list[dict[str, Any]] | list[Any]
        Retrieved documents. Each item must support attribute access (or have
        ``filename`` / ``content`` keys).
    max_chars : int | None, optional
        Maximum characters of each document to include.
    max_docs : int | None, optional
        Maximum number of documents to include.

    Returns
    -------
    str
        A formatted context block labelled with source filenames.
    """
    max_chars = max_chars or Retrieval.MAX_DOC_CHARS
    max_docs = max_docs or Retrieval.MAX_CONTEXT_DOCS

    blocks: list[str] = []
    for doc in documents[:max_docs]:
        # Support both dict-like and attribute-like documents.
        if isinstance(doc, dict):
            filename = doc.get("filename", "unknown")
            content = doc.get("content", "")
        else:
            filename = getattr(doc, "filename", "unknown")
            content = getattr(doc, "content", "")

        snippet = content[:max_chars]
        blocks.append(f"Source: {filename}\n{snippet}")

    context = "\n\n".join(blocks)
    if not context.strip():
        return "No relevant documentation was found."
    return context


def build_generation_prompt(
    question: str,
    documents: list[dict[str, Any]] | list[Any],
) -> str:
    """
    Build the prompt used to generate an answer from grounded context.

    The prompt explicitly forbids using outside knowledge and tells the model
    to say when the answer is not in the knowledge base, preventing
    hallucination.

    Parameters
    ----------
    question : str
        The user's question.
    documents : list[dict[str, Any]] | list[Any]
        Retrieved, relevant documents.

    Returns
    -------
    str
        A fully formatted generation prompt.
    """
    context = _build_context_block(documents)

    return (
        "You are OrbitDesk Assistant, a customer support agent for the SaaS "
        "product OrbitDesk.\n"
        "Answer ONLY using the documentation context provided below.\n"
        "You must NEVER use any external, general, or background knowledge.\n"
        "If the answer is contained in the context, give a concise and accurate "
        "answer in 2-3 complete sentences that ends with a period.\n"
        "Do not stop mid-sentence, mid-clause, or with a trailing connector like 'if' or 'and'.\n"
        "If the answer is NOT contained in the context, reply exactly: "
        f"{SafeResponses.NOT_IN_KNOWLEDGE_BASE}\n"
        "\n"
        "=== CONTEXT ===\n"
        f"{context}\n"
        "\n"
        "=== QUESTION ===\n"
        f"{question}\n"
        "\n"
        "=== ANSWER ===\n"
    )


def build_verification_prompt(
    question: str,
    answer: str,
    documents: list[dict[str, Any]] | list[Any],
) -> str:
    """
    Build a prompt that asks the model to verify an answer against context.

    The model is asked to confirm every important claim is supported, that the
    answer does not introduce unsupported information, and whether it is
    consistent with the documentation. The model must output a single
    structured verdict line.

    Parameters
    ----------
    question : str
        The user's question.
    answer : str
        The candidate answer to verify.
    documents : list[dict[str, Any]] | list[Any]
        The retrieved documents the answer should be grounded in.

    Returns
    -------
    str
        A fully formatted verification prompt.
    """
    context = _build_context_block(documents)

    return (
        "You are a strict quality inspector for a customer support agent.\n"
        "Verify whether the provided ANSWER is fully supported by the "
        "provided CONTEXT.\n"
        "Consider:\n"
        "1. Is every important claim in the ANSWER supported by the CONTEXT?\n"
        "2. Does the ANSWER introduce any information NOT present in the "
        "CONTEXT?\n"
        "3. Is the ANSWER consistent with the documentation?\n"
        "\n"
        "Respond with exactly one line:\n"
        "PASS if fully supported, CONSISTENT, and with no unsupported info.\n"
        "FAIL otherwise.\n"
        "\n"
        "=== CONTEXT ===\n"
        f"{context}\n"
        "\n"
        "=== QUESTION ===\n"
        f"{question}\n"
        "\n"
        "=== ANSWER ===\n"
        f"{answer}\n"
        "\n"
        "=== VERDICT (PASS or FAIL) ===\n"
    )


def build_clarification_message(follow_up: str) -> str:
    """
    Compose a user-facing clarification message.

    Parameters
    ----------
    follow_up : str
        A specific follow-up question for the user.

    Returns
    -------
    str
        A friendly clarification message combining the safe prefix and the
        specific question.
    """
    return f"{SafeResponses.NEEDS_CLARIFICATION}\n\n{follow_up}"
