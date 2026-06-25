"""Abstract base classes for indexing and retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ehr_copilot.domain.document import DocumentChunk


class IndexStore(ABC):
    """Abstract interface for a document-chunk index (dense or sparse)."""

    @abstractmethod
    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray | None = None,
    ) -> None:
        """Insert chunks (and optional pre-computed embeddings) into the store."""
        ...

    @abstractmethod
    def search(
        self,
        query: str | np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return the *top_k* most relevant chunks with similarity scores."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all data from the store and free resources."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        ...


class RetrieverBase(ABC):
    """Abstract interface for a retriever that accepts a text query."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve the *top_k* most relevant chunks for a natural-language query."""
        ...
