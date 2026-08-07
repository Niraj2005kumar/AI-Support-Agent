"""
nodes/retrieval.py
==================

Retrieval node: searches the local knowledge base via FAISS.

This node runs only for questions classified ``Answerable`` by the triage
node. It:

    1. Encodes the user's question into an embedding.
    2. Searches the FAISS vector store for the Top-K most similar documents.
    3. Applies the similarity threshold to discard irrelevant results.
    4. Returns the retrieved documents (content + metadata) to the state.

Because the vector store is built exclusively from the local knowledge base,
this node structurally cannot return unrelated or external content.
"""

from __future__ import annotations

from config import SafeResponses
from state import RetrievedDoc
from utils.logger import get_logger
from utils.vector_store import VectorStore

logger = get_logger(__name__)


def retrieve_documents(
    vector_store: VectorStore,
    question: str,
    k: int | None = None,
) -> list[RetrievedDoc]:
    """
    Retrieve the most relevant documents for ``question`` from the store.

    Parameters
    ----------
    vector_store : VectorStore
        The FAISS vector store built from the knowledge base.
    question : str
        The user's question.
    k : int | None, optional
        Number of results to request. Defaults to the configured Top-K.

    Returns
    -------
    list[RetrievedDoc]
        The retrieved documents, most relevant first. Empty if none met the
        similarity threshold or the store was empty.
    """
    results = vector_store.search(question, k=k)

    docs: list[RetrievedDoc] = []
    for r in results:
        docs.append(
            RetrievedDoc(
                filename=str(r["filename"]),
                content=str(r["content"]),
                score=float(r["score"]),
                metadata=dict(r["metadata"]),
            )
        )

    logger.info("Retrieved %d document(s).", len(docs))
    return docs


# LangGraph node signature: takes full state, returns a partial update.
def run_retrieval(state: dict) -> dict:
    """
    LangGraph retrieval node.

    Reads the question from state, retrieves documents from the injected
    vector store, and writes them back to state (along with source filenames).

    The ``vector_store`` is expected to be attached to the state under the
    private key ``"__vector_store"`` by the graph builder, so it is not part
    of the public schema.

    Parameters
    ----------
    state : dict
        The current graph state.

    Returns
    -------
    dict
        A partial state update with retrieved documents and sources.
    """
    question = str(state.get("question", ""))
    vector_store: VectorStore | None = state.get("__vector_store")

    if vector_store is None:
        logger.error("No vector store available for retrieval.")
        return {
            "documents": [],
            "sources": [],
            "answer": SafeResponses.NOT_IN_KNOWLEDGE_BASE,
            "status": "Safe Response",
            "verification": "Not Applicable",
        }

    docs = retrieve_documents(vector_store, question)

    if not docs:
        logger.info("No relevant documents found for question.")
        return {
            "documents": [],
            "sources": [],
            "answer": SafeResponses.NOT_IN_KNOWLEDGE_BASE,
            "status": "Safe Response",
            "verification": "Not Applicable",
        }

    return {
        "documents": [d.__dict__ for d in docs],
        "sources": [d.filename for d in docs],
    }
