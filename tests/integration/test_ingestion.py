"""Integration tests for the FHIR ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from ehr_copilot.config import ChunkingConfig
from ehr_copilot.domain.document import DocumentType, NoteSection
from ehr_copilot.ingestion.chunker import SectionAwareChunker
from ehr_copilot.ingestion.fhir_parser import FHIRBundleParser
from ehr_copilot.ingestion.note_segmenter import NoteSegmenter


class TestFHIRIngestion:
    def test_load_synthea_bundle(self, sample_fhir_bundle_path: Path):
        """Loading the small Synthea bundle should produce a PatientContext,
        documents, and structured resources."""
        parser = FHIRBundleParser()
        patient_ctx, documents, resources = parser.parse(sample_fhir_bundle_path)

        # Patient context
        assert patient_ctx.patient_id.value == "patient-001"
        assert patient_ctx.demographics.full_name == "John Michael Smith"
        assert patient_ctx.demographics.gender.value == "male"
        assert patient_ctx.source == "synthea"

        # Structured resources
        assert len(resources["encounters"]) == 2
        assert len(resources["conditions"]) == 2
        assert len(resources["observations"]) >= 3
        assert len(resources["medications"]) == 2

        # Documents (one encounter summary per encounter)
        assert len(documents) >= 2
        for doc in documents:
            assert doc.patient_id == "patient-001"
            assert doc.text  # non-empty

    def test_full_ingestion_pipeline(self, sample_fhir_bundle_path: Path):
        """The full pipeline: FHIR parse -> segment -> chunk."""
        parser = FHIRBundleParser()
        _patient_ctx, documents, _resources = parser.parse(sample_fhir_bundle_path)

        segmenter = NoteSegmenter()
        chunker = SectionAwareChunker(
            ChunkingConfig(max_chunk_tokens=200, overlap_tokens=20, min_chunk_tokens=10)
        )

        all_chunks = []
        for doc in documents:
            # If the document has no sections, try segmenting
            if not doc.sections and doc.text:
                sections = segmenter.segment(doc.text)
                doc.sections = sections

            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)

        # We should get at least as many chunks as documents
        assert len(all_chunks) >= len(documents)

        # Every chunk should have the correct patient_id
        for chunk in all_chunks:
            assert chunk.metadata.patient_id == "patient-001"
            assert chunk.chunk_id
            assert chunk.text

    def test_chunk_count_and_metadata(self, sample_fhir_bundle_path: Path):
        """Chunks should preserve encounter metadata from the FHIR source."""
        parser = FHIRBundleParser()
        _patient_ctx, documents, _resources = parser.parse(sample_fhir_bundle_path)

        chunker = SectionAwareChunker(
            ChunkingConfig(max_chunk_tokens=500, overlap_tokens=0, min_chunk_tokens=5)
        )

        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunker.chunk(doc))

        assert len(all_chunks) >= 1

        # Check that encounter IDs flow through to chunk metadata
        encounter_ids = {c.metadata.encounter_id for c in all_chunks if c.metadata.encounter_id}
        assert len(encounter_ids) >= 1

        # Check document types
        doc_types = {c.metadata.document_type for c in all_chunks}
        assert DocumentType.ENCOUNTER_SUMMARY in doc_types

        # Encounter dates should be present on chunks from encounter summaries
        dated_chunks = [c for c in all_chunks if c.metadata.encounter_date is not None]
        assert len(dated_chunks) >= 1
