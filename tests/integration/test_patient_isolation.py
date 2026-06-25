"""Integration tests for patient data isolation.

Verifies that per-patient indices maintain strict data separation, and
that destroying a patient index properly releases resources.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ehr_copilot.config import BM25Config, FAISSConfig, RetrievalConfig
from ehr_copilot.domain.document import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    NoteSection,
)
from ehr_copilot.indexing.patient_index import PatientIndex


def _make_chunks(patient_id: str, prefix: str, count: int = 3) -> list[DocumentChunk]:
    """Generate test chunks for a specific patient."""
    chunks = []
    for i in range(count):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{prefix}-chunk-{i}",
                text=f"Patient {patient_id} clinical note chunk {i}. "
                     f"This contains data specific to patient {patient_id}.",
                metadata=ChunkMetadata(
                    patient_id=patient_id,
                    document_id=f"{prefix}-doc-{i}",
                    document_type=DocumentType.ENCOUNTER_SUMMARY,
                    section=NoteSection.LABS_RESULTS,
                    encounter_date=datetime(2024, 1, 15 + i),
                ),
                token_count=20,
            )
        )
    return chunks


class TestPatientIsolation:
    """Test that patient indices are isolated from one another."""

    def test_query_one_patient_does_not_return_other(self):
        """Loading two patients and querying one should not return
        chunks from the other patient."""
        dimension = 16  # Small dimension for testing

        # Create mock embedding model
        mock_embedding = MagicMock()

        # Encode returns random but deterministic embeddings
        def mock_encode(texts):
            np.random.seed(42)
            return np.random.randn(len(texts), dimension).astype(np.float32)

        def mock_encode_query(text):
            np.random.seed(99)
            return np.random.randn(dimension).astype(np.float32)

        mock_embedding.encode = mock_encode
        mock_embedding.encode_query = mock_encode_query

        faiss_cfg = FAISSConfig(index_type="FlatIP")
        bm25_cfg = BM25Config()
        retrieval_cfg = RetrievalConfig(
            top_k_dense=5, top_k_sparse=5, final_top_k=3
        )

        # Build index for patient A
        chunks_a = _make_chunks("patient-A", "a")
        index_a = PatientIndex(faiss_cfg, bm25_cfg, dimension)
        index_a.build(chunks_a, mock_embedding)
        retriever_a = index_a.get_retriever(retrieval_cfg)

        # Build index for patient B
        chunks_b = _make_chunks("patient-B", "b")
        index_b = PatientIndex(faiss_cfg, bm25_cfg, dimension)
        index_b.build(chunks_b, mock_embedding)
        retriever_b = index_b.get_retriever(retrieval_cfg)

        # Query patient A's retriever
        results_a = retriever_a.retrieve("clinical note for patient A")
        for chunk, _score in results_a:
            assert chunk.metadata.patient_id == "patient-A", (
                f"Expected patient-A data, got {chunk.metadata.patient_id}"
            )

        # Query patient B's retriever
        results_b = retriever_b.retrieve("clinical note for patient B")
        for chunk, _score in results_b:
            assert chunk.metadata.patient_id == "patient-B", (
                f"Expected patient-B data, got {chunk.metadata.patient_id}"
            )

        # Clean up
        index_a.destroy()
        index_b.destroy()

    def test_destroy_frees_resources(self):
        """Destroying a patient index should clear the stores and reset state."""
        dimension = 16
        mock_embedding = MagicMock()
        mock_embedding.encode = lambda texts: np.random.randn(
            len(texts), dimension
        ).astype(np.float32)

        faiss_cfg = FAISSConfig(index_type="FlatIP")
        bm25_cfg = BM25Config()

        chunks = _make_chunks("patient-X", "x", count=5)
        index = PatientIndex(faiss_cfg, bm25_cfg, dimension)

        # Before build
        assert index.is_built is False
        assert index.chunk_count == 0

        # After build
        index.build(chunks, mock_embedding)
        assert index.is_built is True
        assert index.chunk_count == 5

        # After destroy
        index.destroy()
        assert index.is_built is False
        assert index.chunk_count == 0

        # Should not be able to get a retriever after destroy
        with pytest.raises(RuntimeError, match="not been built"):
            index.get_retriever(RetrievalConfig())
