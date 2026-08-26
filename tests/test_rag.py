"""
tests/test_rag.py — Unit tests for RAG parsing, chunking, and ranking logic.

Offline tests that do not make external API calls.
"""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.rag import (
    _chunk_text,
    _extract_heading,
    _parse_document,
    _rank_boost,
    RetrievedChunk,
)
from agent.config import KNOWLEDGE_BASE_PATH


class TestChunking:
    def test_chunk_text_small(self):
        text = "This is a short test sentence for chunking."
        chunks = _chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_split_with_overlap(self):
        text = "word " * 500  # 500 words
        chunks = _chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        # Overlap check
        assert len(chunks[0]) > 0


class TestHeadingExtraction:
    def test_single_heading(self):
        md = "# Main Title\nSome content below"
        assert _extract_heading(md) == "Main Title"

    def test_subheadings_returns_last(self):
        md = "# Main Title\nIntro text\n## Section 1\nSection text\n### Subsection A\nSub text"
        assert _extract_heading(md) == "Subsection A"

    def test_no_heading(self):
        md = "Just plain text without markdown headings."
        assert _extract_heading(md) == ""


class TestDocumentParsing:
    def test_parse_valid_markdown_file(self):
        doc_path = Path(KNOWLEDGE_BASE_PATH) / "01-returns-policy-current.md"
        assert doc_path.exists()
        meta, body = _parse_document(doc_path)
        assert isinstance(meta, dict)
        assert meta.get("status") == "active"
        assert meta.get("policy_authority") == "official"
        assert "30 days" in body or "return" in body.lower()

    def test_parse_internal_document(self):
        doc_path = Path(KNOWLEDGE_BASE_PATH) / "14-internal-content-migration-notes.md"
        assert doc_path.exists()
        meta, body = _parse_document(doc_path)
        assert meta.get("audience") == "internal"


class TestRankBoost:
    def test_official_active_boost(self):
        chunk = RetrievedChunk(
            text="sample",
            source_file="01-returns-policy-current.md",
            heading="Policy",
            status="active",
            policy_authority="official",
            audience="customer",
            document_id="01-returns",
            score=0.8,
        )
        boost = _rank_boost(chunk)
        assert boost == pytest.approx(0.25)

    def test_superseded_penalty(self):
        chunk = RetrievedChunk(
            text="sample",
            source_file="02-returns-policy-legacy.md",
            heading="Policy",
            status="superseded",
            policy_authority="official",
            audience="customer",
            document_id="02-returns",
            score=0.8,
        )
        boost = _rank_boost(chunk)
        # official (+0.15) and superseded (-0.30) -> -0.15
        assert boost == pytest.approx(-0.15)


class TestRetrievedChunk:
    def test_citation(self):
        chunk = RetrievedChunk(
            text="sample text",
            source_file="05-domestic-shipping.md",
            heading="Standard Shipping",
            status="active",
            policy_authority="official",
            audience="customer",
            document_id="05-shipping",
            score=0.9,
        )
        assert chunk.citation() == "05-domestic-shipping.md § Standard Shipping"
        d = chunk.to_dict()
        assert d["source_file"] == "05-domestic-shipping.md"
        assert d["score"] == 0.9
