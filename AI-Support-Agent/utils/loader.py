"""
utils/loader.py
===============

Knowledge base loading and normalisation.

This module is the single entry point for turning the raw knowledge base
(``knowledge_base/*.md`` and ``resolved_cases.json``) into a normalised list of
``Document`` objects that the rest of the pipeline can consume.

Responsibilities:

    * Load all Markdown files from the knowledge base directory.
    * Load resolved support cases from the JSON file.
    * Normalise both into a unified ``Document`` structure.
    * Attach useful metadata (source path, tags, case id, etc.).
    * Strip Markdown formatting noise so the embedding model sees clean text.

The pipeline NEVER reaches outside these sources — everything the agent knows
comes from here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from config import Paths
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """
    A single, normalised knowledge base entry.

    Attributes
    ----------
    id : str
        A unique identifier for the document (e.g. ``md:01_product_overview``).
    filename : str
        The source file name (``01_product_overview.md``).
    content : str
        Cleaned, human-readable text content.
    source_type : str
        ``"md"`` for Markdown docs or ``"case"`` for resolved cases.
    metadata : dict[str, Any]
        Extra metadata such as source path, tags, or case id.
    """

    id: str
    filename: str
    content: str
    source_type: str = "md"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Markdown cleaning helpers
# ---------------------------------------------------------------------------
def _parse_front_matter(text: str) -> tuple[str, dict[str, object]]:
    """Strip a simple YAML front-matter block and keep its metadata."""
    if not text.startswith("---"):
        return text, {}

    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?", text, flags=re.DOTALL)
    if not match:
        return text, {}

    front_matter = {}
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            items = [part.strip().strip("\"'") for part in inner.split(",") if part.strip()]
            parsed_value = items
        elif value.lower() in {"true", "false"}:
            parsed_value = value.lower() == "true"
        elif value.lower() in {"null", "none"}:
            parsed_value = None
        elif value.startswith(("\"", "'")) and value.endswith(("\"", "'")):
            parsed_value = value[1:-1]
        else:
            parsed_value = value

        front_matter[key] = parsed_value

    remainder = text[match.end() :]
    return remainder, front_matter


def _clean_markdown(text: str) -> str:
    """
    Remove common Markdown syntax noise to produce cleaner embedding text.

    Parameters
    ----------
    text : str
        Raw Markdown content.

    Returns
    -------
    str
        Cleaned plain text with Markdown markers removed.
    """
    text, _ = _parse_front_matter(text)
    # Remove code fences.
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    # Remove inline code backticks.
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove heading markers.
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers.
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    # Remove link syntax but keep the link text.
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Remove horizontal rules.
    text = re.sub(r"^\s*---+\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines / spaces.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_markdown_documents(directory: Path | None = None) -> list[Document]:
    """
    Load and normalise all Markdown documents from the knowledge base.

    Parameters
    ----------
    directory : Path | None
        The knowledge base directory. Defaults to ``config.Paths.KNOWLEDGE_BASE_DIR``.

    Returns
    -------
    list[Document]
        A list of normalised Markdown documents.
    """
    kb_dir = directory or Paths.KNOWLEDGE_BASE_DIR
    documents: list[Document] = []

    if not kb_dir.exists():
        logger.warning("Knowledge base directory not found: %s", kb_dir)
        return documents

    for file_path in sorted(kb_dir.glob("*.md")):
        try:
            raw = file_path.read_text(encoding="utf-8")
            content_without_front_matter, metadata = _parse_front_matter(raw)
            cleaned = _clean_markdown(content_without_front_matter)
            doc_id = f"md:{file_path.stem}"
            metadata = {
                **metadata,
                "path": str(file_path),
                "tags": metadata.get("tags", [file_path.stem]),
            }
            documents.append(
                Document(
                    id=doc_id,
                    filename=file_path.name,
                    content=cleaned,
                    source_type="md",
                    metadata=metadata,
                )
            )
            logger.debug("Loaded Markdown document: %s", file_path.name)
        except OSError as exc:
            logger.error("Failed to read %s: %s", file_path, exc)

    logger.info("Loaded %d Markdown document(s).", len(documents))
    return documents


def load_resolved_cases(path: Path | None = None) -> list[Document]:
    """
    Load and normalise resolved support cases from the JSON file.

    The JSON file is expected to contain a list of case objects. Each case
    should have at least ``id``, ``question`` (or ``title``) and ``answer``
    (or ``resolution``) fields. Unknown fields are preserved in metadata.

    Parameters
    ----------
    path : Path | None
        The JSON file path. Defaults to ``config.Paths.RESOLVED_CASES_FILE``.

    Returns
    -------
    list[Document]
        A list of normalised case documents.
    """
    cases_file = path or Paths.RESOLVED_CASES_FILE
    documents: list[Document] = []

    if not cases_file.exists():
        logger.warning("Resolved cases file not found: %s", cases_file)
        return documents

    try:
        raw = cases_file.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load resolved cases: %s", exc)
        return documents

    for idx, case in enumerate(data):
        if not isinstance(case, dict):
            logger.warning("Skipping non-object case at index %d", idx)
            continue

        case_id = str(case.get("id", f"case_{idx}"))
        question = str(case.get("question", case.get("title", "")))
        resolution = str(case.get("answer", case.get("resolution", "")))

        if not question or not resolution:
            logger.warning("Skipping incomplete case: %s", case_id)
            continue

        content = f"Q: {question}\nA: {resolution}"
        documents.append(
            Document(
                id=f"case:{case_id}",
                filename=f"resolved_case_{case_id}.txt",
                content=content,
                source_type="case",
                metadata={"case_id": case_id, **case},
            )
        )

    logger.info("Loaded %d resolved case(s).", len(documents))
    return documents


def load_all_documents() -> list[Document]:
    """
    Load the complete knowledge base as a single normalised list.

    Combines Markdown documentation and resolved support cases. Callers can
    rely on this as the definitive source of all retrievable content.

    Returns
    -------
    list[Document]
        All documents (Markdown + cases) in a unified format.
    """
    docs = load_markdown_documents()
    docs.extend(load_resolved_cases())
    logger.info("Total documents loaded: %d", len(docs))
    return docs


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def documents_to_records(documents: Iterable[Document]) -> list[dict[str, Any]]:
    """
    Convert ``Document`` objects to plain dictionaries.

    Useful for interop with libraries that expect JSON-serialisable records
    (e.g. building the FAISS index or writing debug output).

    Parameters
    ----------
    documents : Iterable[Document]
        The documents to convert.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionary records.
    """
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "content": doc.content,
            "source_type": doc.source_type,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]
