"""
tests/test_loader.py
====================

Unit tests for the knowledge base loader, using a temporary directory with
sample Markdown and JSON files.
"""

from __future__ import annotations

import json

from config import Paths
from utils.loader import _clean_markdown, load_markdown_documents, load_resolved_cases
from utils.vector_store import VectorStore


class TestCleanMarkdown:
    """Markdown cleaning behaviour."""

    def test_strips_headings_and_bold(self) -> None:
        raw = "# Title\n\nSome **bold** text with `inline`."
        cleaned = _clean_markdown(raw)
        assert "Title" in cleaned
        assert "**bold**" not in cleaned
        assert "`inline`" not in cleaned

    def test_collapses_blank_lines(self) -> None:
        raw = "Line one.\n\n\n\nLine two."
        cleaned = _clean_markdown(raw)
        assert "\n\n\n" not in cleaned


class TestLoadMarkdown:


    def test_loads_md_files(self, tmp_path_factory) -> None:
        kb = tmp_path_factory.mktemp("kb")
        (kb / "01_test.md").write_text("# Doc\n\nHello world.", encoding="utf-8")
        (kb / "02_test.md").write_text("# Doc2\n\nSecond doc.", encoding="utf-8")

        docs = load_markdown_documents(kb)
        assert len(docs) == 2
        filenames = {d.filename for d in docs}
        assert filenames == {"01_test.md", "02_test.md"}
        assert all(d.source_type == "md" for d in docs)

    def test_extracts_yaml_front_matter_metadata(self, tmp_path_factory) -> None:
        kb = tmp_path_factory.mktemp("kb")
        (kb / "01_frontmatter.md").write_text(
            "---\ndocument_id: KB-001\ntitle: OrbitDesk Overview\nupdated: 2026-07-01\nstatus: current\ntags: [overview, workspace]\n---\n\n# Doc\n\nHello world.",
            encoding="utf-8",
        )

        docs = load_markdown_documents(kb)
        assert len(docs) == 1
        assert docs[0].metadata["document_id"] == "KB-001"
        assert docs[0].metadata["title"] == "OrbitDesk Overview"
        assert docs[0].metadata["tags"] == ["overview", "workspace"]

    def test_real_kb_returns_multi_document_results(self) -> None:
        docs = load_markdown_documents(Paths.KNOWLEDGE_BASE_DIR)
        assert len(docs) >= 2

        store = VectorStore(docs)
        results = store.search(
            "Who can create API credentials and who can invite team members?",
            k=3,
        )
        assert len(results) >= 2
        filenames = {result["filename"] for result in results}
        assert any("02_roles_and_permissions.md" in name for name in filenames)
        assert any("05_api_credentials.md" in name for name in filenames)


class TestLoadResolvedCases:
    """Resolved cases JSON loading."""

    def test_loads_cases(self, tmp_path_factory) -> None:
        cases_file = tmp_path_factory.mktemp("cases") / "cases.json"
        cases = [
            {
                "id": "case-1",
                "question": "How do I reset my password?",
                "answer": "Go to Settings > Security and click Reset.",
            }
        ]
        cases_file.write_text(json.dumps(cases), encoding="utf-8")

        docs = load_resolved_cases(cases_file)
        assert len(docs) == 1
        assert docs[0].source_type == "case"
        assert docs[0].metadata["case_id"] == "case-1"

    def test_skips_incomplete_cases(self, tmp_path_factory) -> None:
        cases_file = tmp_path_factory.mktemp("cases2") / "cases.json"
        cases = [
            {"id": "ok", "question": "q", "answer": "a"},
            {"id": "bad", "question": "", "answer": ""},
        ]
        cases_file.write_text(json.dumps(cases), encoding="utf-8")

        docs = load_resolved_cases(cases_file)
        assert len(docs) == 1
