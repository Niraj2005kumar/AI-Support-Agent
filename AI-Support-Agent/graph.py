"""
graph.py
========

LangGraph orchestration for the OrbitDesk AI Support Agent.

This module assembles the individual pipeline nodes into a compiled
``StateGraph`` with conditional routing that implements the full workflow:

    START
      → triage
          ├─ Answerable            → retrieval
          ├─ Clarification Required → formatter
          ├─ Out of Scope          → formatter
          └─ Escalation Required   → formatter
      → retrieval
      → generator
      → verifier
          ├─ Passed                → formatter
          ├─ Failed (retries left) → generator  (retry loop)
          └─ Failed (no retries)   → formatter
      → formatter
      → END

The vector store is injected into the state under the private key
``__vector_store`` so the retrieval node can access it without polluting the
public schema.
"""

from __future__ import annotations

from typing import Any

from config import Verification
from langgraph.graph import END, START, StateGraph

from nodes.formatter import run_formatter
from nodes.generator import run_generator
from nodes.retrieval import run_retrieval
from nodes.triage import run_triage
from nodes.verifier import run_verifier
from state import AnswerState, TriageLabel, VerificationStatus
from utils.logger import get_logger
from utils.vector_store import VectorStore

logger = get_logger(__name__)

# Node names (string constants for clarity).
NODE_TRIAGE = "triage"
NODE_RETRIEVAL = "retrieval"
NODE_GENERATOR = "generator"
NODE_VERIFIER = "verifier"
NODE_FORMATTER = "formatter"


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------
def _route_after_triage(state: dict) -> str:
    """
    Route to ``retrieval`` if the question is answerable, otherwise go
    directly to the ``formatter`` to produce a safe response.

    Parameters
    ----------
    state : dict
        The current graph state.

    Returns
    -------
    str
        The name of the next node.
    """
    label = state.get("triage_label", "")
    if label == TriageLabel.ANSWERABLE.value:
        return NODE_RETRIEVAL
    logger.info("Triage not answerable (%s); routing to formatter.", label)
    return NODE_FORMATTER


def _route_after_verifier(state: dict) -> str:
    """
    Route after the verifier:

        * Passed / Not Applicable -> formatter.
        * Failed with retries left -> generator (retry loop).
        * Failed with no retries left -> formatter (safe failure).

    Parameters
    ----------
    state : dict
        The current graph state.

    Returns
    -------
    str
        The name of the next node.
    """
    verification = state.get("verification", VerificationStatus.NOT_APPLICABLE.value)
    retry_count = int(state.get("retry_count", 0))

    if verification == VerificationStatus.PASSED.value:
        return NODE_FORMATTER

    if verification == VerificationStatus.FAILED.value:
        if retry_count < Verification.MAX_RETRIES:
            logger.info("Verification failed; retry generator (attempt %d).", retry_count)
            return NODE_GENERATOR
        logger.warning("Verification failed; no retries left -> formatter.")
    return NODE_FORMATTER


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(vector_store: VectorStore) -> Any:
    """
    Build and compile the LangGraph for the support agent.

    Parameters
    ----------
    vector_store : VectorStore
        The FAISS vector store built from the knowledge base. Injected into
        the initial state so the retrieval node can access it.

    Returns
    -------
    Any
        A compiled LangGraph application ready to be invoked.
    """
    # Bind the vector store to the retrieval node via closure so the node's
    # signature stays protocol-compatible with LangGraph.
    def _retrieval_with_store(state: dict) -> dict:
        state = dict(state)
        state["__vector_store"] = vector_store
        return run_retrieval(state)

    builder: StateGraph = StateGraph(AnswerState)

    # Add nodes.
    builder.add_node(NODE_TRIAGE, run_triage)
    builder.add_node(NODE_RETRIEVAL, _retrieval_with_store)
    builder.add_node(NODE_GENERATOR, run_generator)
    builder.add_node(NODE_VERIFIER, run_verifier)
    builder.add_node(NODE_FORMATTER, run_formatter)

    # Edges.
    builder.add_edge(START, NODE_TRIAGE)
    builder.add_conditional_edges(
        NODE_TRIAGE,
        _route_after_triage,
        {
            NODE_RETRIEVAL: NODE_RETRIEVAL,
            NODE_FORMATTER: NODE_FORMATTER,
        },
    )
    builder.add_edge(NODE_RETRIEVAL, NODE_GENERATOR)
    builder.add_edge(NODE_GENERATOR, NODE_VERIFIER)
    builder.add_conditional_edges(
        NODE_VERIFIER,
        _route_after_verifier,
        {
            NODE_GENERATOR: NODE_GENERATOR,
            NODE_FORMATTER: NODE_FORMATTER,
        },
    )
    builder.add_edge(NODE_FORMATTER, END)

    logger.info("LangGraph compiled successfully.")
    return builder.compile()


def run_agent(graph: Any, question: str) -> dict:
    """
    Invoke the compiled graph with a user question and return the final output.

    Parameters
    ----------
    graph : Any
        The compiled LangGraph application.
    question : str
        The user's question.

    Returns
    -------
    dict
        The final schema-conforming output dictionary (``raw_output``).
    """
    from state import initial_state

    state = initial_state(question)
    result = graph.invoke(state)
    return result.get("raw_output", {})
