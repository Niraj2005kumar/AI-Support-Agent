"""
utils/embeddings.py
===================

Local, offline sentence-embedding utilities.

This module wraps the Sentence Transformers model used to convert text into
dense vector representations. These vectors form the basis of the FAISS
vector store and are also used to measure semantic similarity during triage.

Key design decisions:

    * The model is loaded lazily and cached as a module-level singleton so it
      is only loaded once per process.
    * Encoding is performed on the configured device (CPU by default), keeping
      the entire pipeline fully offline.
    * Both whole documents and individual queries can be encoded; queries are
      normalised to unit length to make cosine similarity work with FAISS L2.

This module deliberately has no network calls — the model files come from the
local Hugging Face cache after the initial download.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Models
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton (lazily initialised)
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Return the cached Sentence Transformer model, loading it on first call.

    Uses a module-level singleton to avoid repeatedly loading the model into
    memory, which is expensive both in time and RAM.

    Returns
    -------
    SentenceTransformer
        The configured, ready-to-use embedding model.
    """
    global _model  # noqa: PLW0603
    if _model is None:
        logger.info("Loading Embeddings...")
        _model = SentenceTransformer(Models.EMBEDDING_MODEL, device=Models.DEVICE)
    return _model


def encode_texts(texts: list[str], *, batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of texts into a 2D numpy array of embeddings.

    Parameters
    ----------
    texts : list[str]
        The texts to embed (documents or queries).
    batch_size : int, optional
        Batch size passed to the model for efficient, chunked encoding.

    Returns
    -------
    np.ndarray
        A float32 array of shape ``(len(texts), dimension)`` with the
        embeddings normalised to unit length.
    """
    if not texts:
        return np.zeros((0, get_embedding_model().get_sentence_embedding_dimension()))

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # Unit vectors -> cosine = dot product.
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single query string into a normalised embedding vector.

    Parameters
    ----------
    query : str
        The user's question or search query.

    Returns
    -------
    np.ndarray
        A 2D array of shape ``(1, dimension)`` containing the query vector.
    """
    return encode_texts([query])


def get_embedding_dimension() -> int:
    """
    Return the dimensionality of the embedding vectors.

    Returns
    -------
    int
        The number of dimensions produced by the configured model.
    """
    return get_embedding_model().get_sentence_embedding_dimension()


def release_embedding_model() -> None:
    """
    Free the embedding model from memory.

    Useful in long-running integration tests or when the model is no longer
    needed. The model will be lazily re-loaded on the next ``encode_*`` call.
    """
    global _model  # noqa: PLW0603
    if _model is not None:
        del _model
        _model = None
        logger.info("Embedding model released from memory.")
