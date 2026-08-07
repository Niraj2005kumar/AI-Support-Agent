"""
utils/vector_store.py
=====================

FAISS vector database construction and search.

This module provides the retrieval backbone of the agent:

    * ``VectorStore`` builds a FAISS ``IndexFlatIP`` (inner product) over the
      embeddings of all knowledge base documents. Because both document and
      query embeddings are normalised to unit length, inner product equals
      cosine similarity — perfect for semantic search.
    * ``search`` returns the Top-K most similar documents together with their
      similarity scores and metadata.
    * A metadata row lookup maps an index position back to the original
      ``Document`` object so results can be consumed downstream.

Only content that exists in the local knowledge base is ever indexed, so the
retriever structurally cannot pull in unrelated or external documents.
"""

from __future__ import annotations

import numpy as np
import faiss

from config import Paths, Retrieval
from utils.embeddings import encode_texts, encode_query, get_embedding_dimension
from utils.loader import Document
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    A small wrapper around a FAISS index for document retrieval.

    Parameters
    ----------
    documents : list[Document]
        The knowledge base documents to index.

    Attributes
    ----------
    index : faiss.Index
        The underlying FAISS index (inner-product).
    documents : list[Document]
        Original documents, aligned by position with index rows.
    """

    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("Cannot build a vector store from an empty document list.")

        self.documents: list[Document] = documents
        self._dimension: int = get_embedding_dimension()

        logger.info(
            "Building FAISS index over %d document(s) (dim=%d)...",
            len(documents),
            self._dimension,
        )

        # Embed all document contents into a single matrix.
        texts = [doc.content for doc in documents]
        emb_matrix: np.ndarray = encode_texts(texts)

        # Inner-product index; normalised vectors -> cosine similarity.
        self.index: faiss.Index = faiss.IndexFlatIP(self._dimension)
        self.index.add(emb_matrix)
        logger.info("FAISS index built successfully with %d vectors.", self.index.ntotal)

    # -----------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        min_similarity: float | None = None,
    ) -> list[dict[str, object]]:
        """
        Retrieve the Top-K most relevant documents for a query.

        Parameters
        ----------
        query : str
            The user's question or search query.
        k : int | None, optional
            Number of results to return. Defaults to ``config.Retrieval.TOP_K``.
        min_similarity : float | None, optional
            Minimum cosine similarity threshold. Results below this are
            dropped. Defaults to ``config.Retrieval.MIN_SIMILARITY``.

        Returns
        -------
        list[dict[str, object]]
            A list of result dictionaries. Each contains:
                * ``filename`` — source file name.
                * ``content``  — full document content.
                * ``score``    — cosine similarity in [0, 1].
                * ``metadata`` — the document's metadata.
            Results are sorted by descending score (most relevant first).
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty; returning no results.")
            return []

        k = k or Retrieval.TOP_K
        min_sim = min_similarity if min_similarity is not None else Retrieval.MIN_SIMILARITY
        k = min(k, self.index.ntotal)

        query_vec = encode_query(query)

        # FAISS returns similarities (inner product) in descending order for
        # IndexFlatIP.
        scores, indices = self.index.search(query_vec, k)

        results: list[dict[str, object]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS uses -1 for missing neighbours.
            similarity = float(score)
            if similarity < min_sim:
                logger.debug(
                    "Dropping doc '%s' below similarity threshold %.3f",
                    self.documents[idx].filename,
                    min_sim,
                )
                continue

            doc = self.documents[idx]
            results.append(
                {
                    "filename": doc.filename,
                    "content": doc.content,
                    "score": similarity,
                    "metadata": doc.metadata,
                }
            )

        # Sort descending by similarity so the most relevant is first.
        results.sort(key=lambda r: float(r["score"]), reverse=True)

        logger.info(
            "Retrieved %d result(s) (of %d requested) for query.",
            len(results),
            k,
        )
        return results

    # -----------------------------------------------------------------------
    def save_local(self, path) -> None:
        """Persist the FAISS index to ``path`` for future fast loads."""
        if path is None:
            path = Paths.VECTOR_CACHE_DIR / "index.bin"
        import os
        os.makedirs(path.parent, exist_ok=True) if hasattr(path, "parent") else None
        faiss.write_index(self.index, str(path))
        logger.info("FAISS index written to %s", path)

    @classmethod
    def load_local(cls, path, documents: list[Document]) -> "VectorStore":
        """Load a persisted FAISS index and reconnect it to ``documents``."""
        index = faiss.read_index(str(path))
        self = cls.__new__(cls)
        self.documents = documents
        self.index = index
        self._dimension = index.d
        logger.info("FAISS index loaded from %s (%d vectors).", path, index.ntotal)
        return self
