"""FAISS-backed dense vector index (IndexFlatIP for cosine similarity)."""

from __future__ import annotations

import logging

import faiss
import numpy as np

from ehr_copilot.config import FAISSConfig
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.indexing.base import IndexStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(IndexStore):
    """Dense vector store backed by a FAISS ``IndexFlatIP`` index.

    Because all embeddings are L2-normalised before insertion, inner-product
    search is equivalent to cosine similarity.
    """

    def __init__(self, config: FAISSConfig, dimension: int) -> None:
        self._config = config
        self._dimension = dimension
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self._chunks: list[DocumentChunk] = []

    # ------------------------------------------------------------------
    # IndexStore interface
    # ------------------------------------------------------------------

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray | None = None,
    ) -> None:
        """Add chunks with their pre-computed embeddings to the FAISS index.

        Parameters
        ----------
        chunks:
            Document chunks to store.
        embeddings:
            A numpy array of shape ``(len(chunks), dimension)`` with dtype
            float32.  Must be provided; dense indexing requires vectors.

        Raises
        ------
        ValueError
            If *embeddings* is ``None`` or has a shape mismatch.
        """
        if embeddings is None:
            raise ValueError(
                "FAISSVectorStore.add() requires pre-computed embeddings."
            )
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) does not match "
                f"number of chunks ({len(chunks)})."
            )
        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match "
                f"configured dimension ({self._dimension})."
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)
        self._chunks.extend(chunks)

        logger.debug(
            "Added %d vectors to FAISS index (total: %d)",
            len(chunks),
            self._index.ntotal,
        )

    def search(
        self,
        query: str | np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search the index with a query embedding.

        Parameters
        ----------
        query:
            A numpy array of shape ``(dimension,)`` or ``(1, dimension)``.
            String queries are **not** supported; embed the query first.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        List of ``(DocumentChunk, score)`` tuples sorted by descending score.
        """
        if isinstance(query, str):
            raise TypeError(
                "FAISSVectorStore.search() requires a numpy embedding, not a "
                "string.  Encode the query first via EmbeddingModel."
            )

        if self._index.ntotal == 0:
            return []

        query_vec = np.asarray(query, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        actual_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, actual_k)

        results: list[tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[int(idx)], float(score)))
        return results

    def clear(self) -> None:
        """Reset the FAISS index and free associated memory."""
        self._index.reset()
        self._chunks.clear()
        logger.debug("FAISS index cleared.")

    def __len__(self) -> int:
        return self._index.ntotal
