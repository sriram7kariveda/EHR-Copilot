"""Registry that tracks active per-patient indices."""

from __future__ import annotations

import logging
import threading

from ehr_copilot.config import IndexingConfig
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.indexing.embedding import EmbeddingModel
from ehr_copilot.indexing.patient_index import PatientIndex

logger = logging.getLogger(__name__)


class IndexRegistry:
    """Singleton-like registry mapping ``patient_id`` to :class:`PatientIndex`.

    Thread-safe thanks to an internal lock.  The class can be instantiated
    multiple times (e.g. in tests), but a module-level convenience instance
    is also provided as :data:`default_registry`.
    """

    def __init__(self) -> None:
        self._indices: dict[str, PatientIndex] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_index(
        self,
        patient_id: str,
        chunks: list[DocumentChunk],
        embedding_model: EmbeddingModel,
        config: IndexingConfig,
    ) -> PatientIndex:
        """Create, build, and register an index for *patient_id*.

        If an index already exists for the patient it is destroyed first.

        Parameters
        ----------
        patient_id:
            Unique patient identifier.
        chunks:
            Pre-chunked clinical document chunks.
        embedding_model:
            Shared embedding model used for dense vectors.
        config:
            Indexing configuration (FAISS, BM25 hyper-parameters).

        Returns
        -------
        The newly built :class:`PatientIndex`.
        """
        with self._lock:
            # Tear down any pre-existing index for this patient.
            existing = self._indices.pop(patient_id, None)
            if existing is not None:
                existing.destroy()
                logger.info(
                    "Replaced existing index for patient %s.", patient_id
                )

            patient_index = PatientIndex(
                faiss_config=config.faiss,
                bm25_config=config.bm25,
                dimension=embedding_model.dimension,
            )
            patient_index.build(chunks, embedding_model)
            self._indices[patient_id] = patient_index

            logger.info(
                "Index registered for patient %s (%d chunks).",
                patient_id,
                patient_index.chunk_count,
            )
            return patient_index

    def get_index(self, patient_id: str) -> PatientIndex | None:
        """Return the index for *patient_id*, or ``None`` if not found."""
        with self._lock:
            return self._indices.get(patient_id)

    def destroy_index(self, patient_id: str) -> bool:
        """Destroy and de-register the index for *patient_id*.

        Returns
        -------
        ``True`` if an index was found and destroyed, ``False`` otherwise.
        """
        with self._lock:
            patient_index = self._indices.pop(patient_id, None)
            if patient_index is None:
                return False
            patient_index.destroy()
            logger.info("Index destroyed for patient %s.", patient_id)
            return True

    def list_patients(self) -> list[str]:
        """Return a list of patient IDs that currently have active indices."""
        with self._lock:
            return list(self._indices.keys())

    def destroy_all(self) -> None:
        """Destroy every registered index and clear the registry."""
        with self._lock:
            for pid, patient_index in self._indices.items():
                patient_index.destroy()
                logger.debug("Destroyed index for patient %s.", pid)
            count = len(self._indices)
            self._indices.clear()
            logger.info("All %d patient indices destroyed.", count)


# Module-level convenience instance.
default_registry = IndexRegistry()
