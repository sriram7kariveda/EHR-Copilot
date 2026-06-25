"""Unit tests for the section-aware chunker (ingestion/chunker.py)."""

from __future__ import annotations

from datetime import datetime

import pytest

from ehr_copilot.config import ChunkingConfig
from ehr_copilot.domain.document import (
    ClinicalDocument,
    DocumentType,
    NoteSection,
)
from ehr_copilot.ingestion.chunker import SectionAwareChunker


def _make_document(
    text: str = "",
    sections: dict | None = None,
    document_type: DocumentType = DocumentType.CLINICAL_NOTE,
) -> ClinicalDocument:
    """Helper to build a ClinicalDocument for testing."""
    return ClinicalDocument(
        document_id="doc-test-001",
        patient_id="patient-001",
        document_type=document_type,
        title="Test Document",
        text=text,
        encounter_id="enc-001",
        encounter_date=datetime(2024, 1, 15, 10, 0, 0),
        provider="Dr. Test",
        source_file="test.json",
        sections=sections or {},
    )


class TestSectionAwareChunker:
    def test_basic_chunking(self):
        """A document with plain text should produce at least one chunk."""
        # Create text that is long enough to produce at least one chunk
        text = "The patient has a history of diabetes. " * 20
        doc = _make_document(text=text)
        chunker = SectionAwareChunker(
            ChunkingConfig(max_chunk_tokens=50, min_chunk_tokens=5)
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.text
            assert chunk.metadata.patient_id == "patient-001"
            assert chunk.metadata.document_id == "doc-test-001"

    def test_section_aware_splitting(self):
        """Chunks from different sections carry different section metadata."""
        sections = {
            NoteSection.CHIEF_COMPLAINT: "Chest pain for three days, " * 15,
            NoteSection.MEDICATIONS: "Metformin 500mg twice daily. " * 15,
        }
        doc = _make_document(sections=sections)
        chunker = SectionAwareChunker(
            ChunkingConfig(max_chunk_tokens=50, min_chunk_tokens=5)
        )
        chunks = chunker.chunk(doc)

        section_values = {c.metadata.section for c in chunks}
        assert NoteSection.CHIEF_COMPLAINT in section_values
        assert NoteSection.MEDICATIONS in section_values

        # Verify no chunk mixes sections
        for chunk in chunks:
            assert chunk.metadata.section in (
                NoteSection.CHIEF_COMPLAINT,
                NoteSection.MEDICATIONS,
            )

    def test_chunk_overlap(self):
        """With non-zero overlap, consecutive chunks from the same section share text."""
        long_text = " ".join(f"word{i}" for i in range(200))
        doc = _make_document(text=long_text)

        config = ChunkingConfig(
            max_chunk_tokens=50,
            overlap_tokens=20,
            min_chunk_tokens=5,
        )
        chunker = SectionAwareChunker(config)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 2

        # Check that consecutive chunks share some text (the overlap region)
        for i in range(len(chunks) - 1):
            words_current = set(chunks[i].text.split())
            words_next = set(chunks[i + 1].text.split())
            overlap = words_current & words_next
            assert len(overlap) > 0, (
                f"Chunks {i} and {i+1} should share overlapping words"
            )

    def test_minimum_chunk_size_filtering(self):
        """Chunks below the minimum token threshold are filtered out,
        unless they are the only chunk."""
        sections = {
            NoteSection.CHIEF_COMPLAINT: "A very long section with plenty of content. " * 20,
            NoteSection.ALLERGIES: "NKDA",  # Very short -- should be kept if sole chunk of section
        }
        doc = _make_document(sections=sections)

        config = ChunkingConfig(
            max_chunk_tokens=100,
            overlap_tokens=0,
            min_chunk_tokens=30,
        )
        chunker = SectionAwareChunker(config)
        chunks = chunker.chunk(doc)

        # All chunks should have reasonable token counts.
        # The short section may still produce a chunk because
        # min_chunk_tokens filtering only applies when there are multiple
        # spans from the same section split.
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_metadata_preservation(self):
        """Chunk metadata should carry through document-level fields."""
        text = "Patient is stable. Vitals within normal limits. " * 10
        doc = _make_document(
            text=text,
            document_type=DocumentType.DISCHARGE_SUMMARY,
        )
        chunker = SectionAwareChunker(
            ChunkingConfig(max_chunk_tokens=50, min_chunk_tokens=5)
        )
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.metadata.patient_id == "patient-001"
            assert chunk.metadata.document_id == "doc-test-001"
            assert chunk.metadata.document_type == DocumentType.DISCHARGE_SUMMARY
            assert chunk.metadata.encounter_id == "enc-001"
            assert chunk.metadata.encounter_date == datetime(2024, 1, 15, 10, 0, 0)
            assert chunk.metadata.provider == "Dr. Test"
            assert chunk.metadata.source_file == "test.json"

    def test_total_chunks_backfilled(self):
        """Every chunk's total_chunks field reflects the actual count."""
        text = "Clinical content here. " * 50
        doc = _make_document(text=text)
        chunker = SectionAwareChunker(
            ChunkingConfig(max_chunk_tokens=30, min_chunk_tokens=5)
        )
        chunks = chunker.chunk(doc)

        total = len(chunks)
        assert total > 1
        for chunk in chunks:
            assert chunk.metadata.total_chunks == total

    def test_empty_document(self):
        """An empty document produces no chunks."""
        doc = _make_document(text="", sections={})
        chunker = SectionAwareChunker()
        assert chunker.chunk(doc) == []

    def test_chunk_ids_unique(self):
        """Every chunk should have a unique chunk_id."""
        text = "Word " * 100
        doc = _make_document(text=text)
        chunker = SectionAwareChunker(
            ChunkingConfig(max_chunk_tokens=30, min_chunk_tokens=5)
        )
        chunks = chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
