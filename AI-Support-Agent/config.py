"""
config.py
=========

Central configuration for the OrbitDesk AI Support Agent.

This module is the single source of truth for every tunable setting in the
project:

    * Filesystem paths (knowledge base, JSON data files, vector index cache).
    * Local model identifiers (Hugging Face, fully offline after first load).
    * Retrieval hyper-parameters (Top-K, similarity threshold).
    * Verification / retry behaviour (max retries, confidence thresholds).
    * Language of *safe* responses used when the agent cannot or must not
      answer (out-of-scope, missing-in-KB, verification failure).

All paths are resolved relative to the project root so the application can be
launched from any working directory without breaking imports or file lookups.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# ``__file__`` lives in the project root (config.py). We anchor all relative
# paths to this location so the app works regardless of how it is launched.
PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
class Paths:
    """Centralised, root-relative paths used across the project."""

    # Knowledge base (Markdown documentation).
    KNOWLEDGE_BASE_DIR: Path = PROJECT_ROOT / "knowledge_base"

    # Resolved support cases (JSON) used as additional retrieval context.
    RESOLVED_CASES_FILE: Path = PROJECT_ROOT / "resolved_cases.json"

    # Sample questions used for smoke testing / demos.
    SAMPLE_QUESTIONS_FILE: Path = PROJECT_ROOT / "sample_questions.json"

    # Output schema (JSON) that the final response must conform to.
    OUTPUT_SCHEMA_FILE: Path = PROJECT_ROOT / "output_schema.json"

    # Optional cache directory for pre-built FAISS indexes.
    VECTOR_CACHE_DIR: Path = PROJECT_ROOT / ".vector_cache"

    # Logging directory.
    LOG_DIR: Path = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Local models (Hugging Face, offline after initial download/cache)
# ---------------------------------------------------------------------------
class Models:
    """Identifiers for the local, offline models."""

    # Small, fast Seq2Seq model used for answer generation.
    LLM_NAME: str = os.getenv("ORBITDESK_LLM", "google/flan-t5-small")

    # Sentence embedding model used to build and query the FAISS index.
    EMBEDDING_MODEL: str = os.getenv(
        "ORBITDESK_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # Optional device override ("cpu", "cuda", "mps"). Defaults to CPU.
    DEVICE: str = os.getenv("ORBITDESK_DEVICE", "cpu")

    # Generation hyper-parameters for the local LLM.
    MAX_NEW_TOKENS: int = int(os.getenv("ORBITDESK_MAX_TOKENS", "200"))
    TEMPERATURE: float = float(os.getenv("ORBITDESK_TEMPERATURE", "0.2"))
    TOP_P: float = float(os.getenv("ORBITDESK_TOP_P", "0.9"))


# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
class Retrieval:
    """Hyper-parameters that control the FAISS search behaviour."""

    # Number of most-relevant documents to return to the generator.
    TOP_K: int = int(os.getenv("ORBITDESK_TOP_K", "3"))

    # Minimum cosine similarity (0..1) for a document to be considered
    # relevant. Documents below this threshold are discarded.
    MIN_SIMILARITY: float = float(os.getenv("ORBITDESK_MIN_SIM", "0.35"))

    # Maximum number of characters of a single document passed to the LLM.
    MAX_DOC_CHARS: int = int(os.getenv("ORBITDESK_MAX_DOC_CHARS", "1000"))

    # Maximum number of documents to include in a single prompt context.
    MAX_CONTEXT_DOCS: int = int(os.getenv("ORBITDESK_MAX_CTX_DOCS", "3"))


# ---------------------------------------------------------------------------
# Verification / retry behaviour
# ---------------------------------------------------------------------------
class Verification:
    """Controls how the verifier validates answers and retries generation."""

    # Maximum number of generation retries after a failed verification.
    MAX_RETRIES: int = int(os.getenv("ORBITDESK_MAX_RETRIES", "1"))

    # Minimum confidence score (0..1) for an answer to be "High".
    HIGH_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("ORBITDESK_HIGH_CONF", "0.75")
    )

    # Minimum confidence score (0..1) for an answer to be "Medium".
    MEDIUM_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("ORBITDESK_MED_CONF", "0.45")
    )


# ---------------------------------------------------------------------------
# Safe / fallback response templates
# ---------------------------------------------------------------------------
class SafeResponses:
    """
    Canonical, non-hallucinating responses.

    These messages are returned verbatim whenever the agent decides it must
    not generate an open-ended answer. They keep the system safe and honest.
    """

    # Question is unrelated to OrbitDesk.
    OUT_OF_SCOPE: str = (
        "I can only assist with questions about OrbitDesk. "
        "This topic is outside my knowledge base, so I'm unable to help with it."
    )

    # The question is too vague to answer confidently.
    NEEDS_CLARIFICATION: str = (
        "I need a bit more detail to help you accurately. "
        "Could you clarify one or more of the following?"
    )

    # The retrieval stage could not find any relevant documentation.
    NOT_IN_KNOWLEDGE_BASE: str = (
        "I couldn't find this information in the OrbitDesk knowledge base. "
        "Please contact support for further assistance."
    )

    # The answer could not be verified even after retries.
    VERIFICATION_FAILED: str = (
        "I couldn't confidently verify an answer for your question. "
        "To avoid giving you incorrect information, I've passed this to our "
        "support team for review."
    )

    # The question has been escalated to a human agent.
    ESCALATED: str = (
        "This looks like a complex issue that requires a human agent. "
        "I've flagged it for escalation — a support specialist will follow up."
    )


# ---------------------------------------------------------------------------
# Logging settings
# ---------------------------------------------------------------------------
class Logging:
    """Logging configuration for the application."""

    # Default log level: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    LEVEL: str = os.getenv("ORBITDESK_LOG_LEVEL", "INFO")

    # Log file name (stored under Paths.LOG_DIR).
    FILE_NAME: str = "orbitdesk_agent.log"

    # Whether to also stream logs to the console.
    CONSOLE_ENABLED: bool = True


# ---------------------------------------------------------------------------
# Convenience initialisation
# ---------------------------------------------------------------------------
def ensure_directories() -> None:
    """
    Create any directories the application needs at runtime.

    Safe to call multiple times; missing directories are created, existing
    ones are left untouched.
    """
    Paths.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Paths.VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    Paths.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)


# Execute directory setup on import so downstream imports can rely on the
# folders existing.
ensure_directories()


# ---------------------------------------------------------------------------
# Read-only convenience properties
# ---------------------------------------------------------------------------
def get_project_root() -> str:
    """Return the absolute project root as a string."""
    return str(PROJECT_ROOT)


def get_embedding_dimension() -> int:
    """
    Return the embedding dimension for the configured Sentence Transformer.

    NOTE: This is hard-coded to 384 because ``all-MiniLM-L6-v2`` produces
    fixed 384-dimensional vectors. If the embedding model is changed, this
    value must be updated to match the new model's output dimension.
    """
    return 384
