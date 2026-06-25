"""Abstract base classes for the ingestion pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ehr_copilot.domain.document import ClinicalDocument, DocumentChunk


class IngestorBase(ABC):
    """Base class for all data ingestors.

    An ingestor reads from a source (file, directory, database) and produces
    a list of ``ClinicalDocument`` objects ready for chunking and indexing.
    """

    @abstractmethod
    def ingest(self, source: Path) -> list[ClinicalDocument]:
        """Ingest data from *source* and return clinical documents.

        Parameters
        ----------
        source:
            Path to the source file or directory.

        Returns
        -------
        list[ClinicalDocument]
            Parsed clinical documents.
        """
        ...


class ChunkerBase(ABC):
    """Base class for document chunkers.

    A chunker splits a single ``ClinicalDocument`` into smaller
    ``DocumentChunk`` objects suitable for embedding and retrieval.
    """

    @abstractmethod
    def chunk(self, document: ClinicalDocument) -> list[DocumentChunk]:
        """Split *document* into retrieval-ready chunks.

        Parameters
        ----------
        document:
            The clinical document to chunk.

        Returns
        -------
        list[DocumentChunk]
            Ordered list of document chunks.
        """
        ...
