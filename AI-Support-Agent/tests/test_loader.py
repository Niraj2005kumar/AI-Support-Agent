"""
tests/test_loader.py
====================

Unit tests for the knowledge base loader, using a temporary directory with
sample Markdown and JSON files.
"""

from __future__ import annotations

import json

from utils.loader import _clean_markdown, load_markdown_documents, load_resolved_cases


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


class TestLoadMarkdown(tmp_path_factory):
    """Markdown directory loading."""

    def test_loads_md_files(self, tmp_path_factory) -> None:
        kb = tmp_path_factory.mktemp("kb")
        (kb / "01_test.md").write_text("# Doc\n\nHello world.", encoding="utf-8")
        (kb / "02_test.md").write_text("# Doc2\n\nSecond doc.", encoding="utf-8")

        docs = load_markdown_documents(kb)
        assert len(docs) == 2
        filenames = {d.filename for d in docs}
        assert filenames == {"01_test.md", "02_test.md"}
        assert all(d.source_type == "md" for d in docs)


class TestLoadResolvedCases(tmp_path_factory):
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
