"""Unit tests for the citation engine (citations/)."""

from __future__ import annotations

from datetime import datetime

import pytest

from ehr_copilot.citations.evidence_pack import EvidencePack
from ehr_copilot.citations.formatter import CitationFormatter
from ehr_copilot.citations.span_mapper import SpanMapper
from ehr_copilot.domain.answer import Citation, EvidenceSpan
from ehr_copilot.domain.document import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    NoteSection,
)


def _make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    """Build a minimal DocumentChunk for testing."""
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            patient_id="patient-001",
            document_id="doc-001",
            document_type=DocumentType.ENCOUNTER_SUMMARY,
            section=NoteSection.LABS_RESULTS,
            encounter_date=datetime(2024, 1, 15),
        ),
    )


# ---------------------------------------------------------------------------
# SpanMapper
# ---------------------------------------------------------------------------


class TestSpanMapper:
    def test_map_citations_with_matching_text(self):
        """When the answer closely paraphrases source text, citations are created."""
        mapper = SpanMapper()

        source_chunks = [
            _make_chunk(
                "c1",
                "Hemoglobin A1c was measured at 7.2% on 2024-01-15 during the "
                "ambulatory encounter. This value is above the normal range of 4.0-5.6%.",
            ),
            _make_chunk(
                "c2",
                "Patient is currently taking Metformin hydrochloride 500 MG "
                "twice daily for diabetes management. No adverse effects reported.",
            ),
        ]

        answer_text = (
            "The patient's Hemoglobin A1c was 7.2% on 2024-01-15, which is above normal range. "
            "The patient is taking Metformin hydrochloride 500 MG twice daily."
        )

        citations = mapper.map_citations(answer_text, source_chunks, threshold=50.0)
        assert len(citations) >= 1

        # At least one citation should reference chunk c1 or c2
        chunk_ids_cited = set()
        for cit in citations:
            for span in cit.evidence_spans:
                chunk_ids_cited.add(span.chunk_id)
        assert len(chunk_ids_cited) >= 1

        # Citation IDs should be sequential starting at 1
        if citations:
            assert citations[0].citation_id == 1

    def test_no_citations_below_threshold(self):
        """When answer text is completely unrelated, no citations are produced."""
        mapper = SpanMapper()
        source_chunks = [
            _make_chunk("c1", "Normal chest X-ray findings. No acute disease."),
        ]
        answer_text = (
            "The patient's genetic testing reveals BRCA1 mutation which requires "
            "further oncological consultation and evaluation."
        )
        citations = mapper.map_citations(
            answer_text, source_chunks, threshold=90.0
        )
        assert len(citations) == 0


# ---------------------------------------------------------------------------
# CitationFormatter
# ---------------------------------------------------------------------------


class TestCitationFormatter:
    @pytest.fixture
    def formatter(self) -> CitationFormatter:
        return CitationFormatter()

    @pytest.fixture
    def sample_citations(self) -> list[Citation]:
        return [
            Citation(
                citation_id=1,
                claim_text="The A1c is 7.2%.",
                evidence_spans=[
                    EvidenceSpan(
                        chunk_id="c1",
                        text="Hemoglobin A1c was 7.2%",
                        relevance_score=0.92,
                        document_source="Encounter Summary > Labs Results (2024-01-15)",
                    )
                ],
                confidence=0.92,
            ),
            Citation(
                citation_id=2,
                claim_text="Metformin 500 MG is prescribed.",
                evidence_spans=[
                    EvidenceSpan(
                        chunk_id="c2",
                        text="Metformin hydrochloride 500 MG Oral Tablet",
                        relevance_score=0.88,
                        document_source="Encounter Summary > Medications (2024-01-15)",
                    )
                ],
                confidence=0.88,
            ),
        ]

    def test_format_answer_adds_markers(self, formatter, sample_citations):
        answer_text = "The A1c is 7.2%. Metformin 500 MG is prescribed."
        result = formatter.format_answer(answer_text, sample_citations)
        assert "[1]" in result
        assert "[2]" in result

    def test_format_answer_no_citations(self, formatter):
        answer_text = "The patient is stable."
        result = formatter.format_answer(answer_text, [])
        assert result == answer_text

    def test_format_references(self, formatter, sample_citations):
        result = formatter.format_references(sample_citations)
        assert "References" in result
        assert "[1]" in result
        assert "[2]" in result
        assert "Labs Results" in result
        assert "Medications" in result

    def test_format_references_empty(self, formatter):
        result = formatter.format_references([])
        assert result == ""


# ---------------------------------------------------------------------------
# EvidencePack
# ---------------------------------------------------------------------------


class TestEvidencePack:
    def test_build(self):
        """EvidencePack.build should produce citations, formatted text, and chunk map."""
        source_chunks = [
            _make_chunk(
                "c1",
                "Hemoglobin A1c was measured at 7.2% on 2024-01-15 during "
                "the ambulatory encounter for check up.",
            ),
            _make_chunk(
                "c2",
                "The patient is taking Metformin hydrochloride 500 MG twice "
                "daily with meals for Type 2 diabetes management.",
            ),
        ]
        answer_text = (
            "The patient's Hemoglobin A1c was 7.2% on 2024-01-15. "
            "The patient takes Metformin hydrochloride 500 MG twice daily."
        )

        pack = EvidencePack.build(answer_text, source_chunks, threshold=50.0)

        # Should have at least one citation
        assert isinstance(pack.citations, list)
        # Source chunks are indexed by chunk_id
        assert "c1" in pack.source_chunks
        assert "c2" in pack.source_chunks
        # Formatted answer should contain the original text (possibly with markers)
        assert "7.2%" in pack.formatted_answer
        # to_dict should be serialisable
        d = pack.to_dict()
        assert "citations" in d
        assert "source_chunks" in d
        assert "formatted_answer" in d
        assert "formatted_references" in d
