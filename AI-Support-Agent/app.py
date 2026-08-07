"""
app.py
======

Command-line entry point for the OrbitDesk AI Support Agent.

This module:

    1. Loads the local knowledge base (Markdown docs + resolved cases).
    2. Builds the FAISS vector store from the documents.
    3. Compiles the LangGraph pipeline.
    4. Processes a user question supplied via argument, interactive prompt,
       or a sample question.
    5. Prints the final schema-conforming JSON output.

Usage examples:
    python app.py --question "Can a read-only user create API credentials?"
    python app.py --sample
    python app.py          # interactive prompt
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from config import Paths
from graph import build_graph, run_agent
from utils.loader import load_all_documents
from utils.logger import get_logger
from utils.vector_store import VectorStore

logger = get_logger(__name__)


def build_pipeline() -> tuple[Any, list[str]]:
    """
    Build the complete pipeline: load docs, build vector store, compile graph.

    Returns
    -------
    tuple[Any, list[str]]
        A tuple of (compiled graph, list of loaded document filenames).
    """
    logger.info("Loading knowledge base...")
    documents = load_all_documents()

    if not documents:
        logger.error("Knowledge base is empty. Nothing to answer from.")
        sys.exit("Knowledge base is empty. Please add documents to knowledge_base/.")

    logger.info("Building vector store...")
    vector_store = VectorStore(documents)

    logger.info("Compiling pipeline graph...")
    graph = build_graph(vector_store)

    filenames = sorted({d.filename for d in documents})
    return graph, filenames


def load_sample_questions() -> list[str]:
    """
    Load sample questions from ``sample_questions.json``.

    Returns
    -------
    list[str]
        A list of sample question strings. Empty if the file is missing or
        invalid.
    """
    sample_file = Paths.SAMPLE_QUESTIONS_FILE
    if not sample_file.exists():
        logger.warning("Sample questions file not found: %s", sample_file)
        return []

    try:
        data = json.loads(sample_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(q) for q in data if str(q).strip()]
        return []
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load sample questions: %s", exc)
        return []


def print_output(output: dict) -> None:
    """Pretty-print the final output dictionary as JSON."""
    print("\n" + "=" * 60)
    print("OrbitDesk AI Support Agent — Response")
    print("=" * 60)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """
    Application entry point.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 on success, non-zero on error).
    """
    parser = argparse.ArgumentParser(
        description="OrbitDesk AI Support Agent (offline LangGraph pipeline)."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="The user's question to answer.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run all questions from sample_questions.json and print outputs.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (repeated prompts).",
    )
    args = parser.parse_args(argv)

    try:
        graph, _filenames = build_pipeline()
    except SystemExit:
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to build pipeline: %s", exc)
        return 1

    # --- Sample mode --------------------------------------------------------
    if args.sample:
        samples = load_sample_questions()
        if not samples:
            logger.warning("No sample questions available.")
        for q in samples:
            logger.info("Processing sample question: %s", q)
            output = run_agent(graph, q)
            print_output(output)
        return 0

    # --- Interactive mode -----------------------------------------------------
    if args.interactive:
        print("Enter questions (type 'exit' or 'quit' to stop):")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit", "q"}:
                break
            if not q:
                continue
            output = run_agent(graph, q)
            print_output(output)
        return 0

    # --- Single question mode ------------------------------------------------
    question = args.question
    if not question:
        question = input("Enter your question: ").strip()

    if not question:
        logger.error("No question provided.")
        parser.print_usage()
        return 1

    output = run_agent(graph, question)
    print_output(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
