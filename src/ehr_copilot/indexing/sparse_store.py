"""BM25S-backed sparse retrieval index."""

from __future__ import annotations

import logging
from typing import Any

import bm25s
import numpy as np

from ehr_copilot.config import BM25Config
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.indexing.base import IndexStore

logger = logging.getLogger(__name__)


class BM25SparseStore(IndexStore):
    """Sparse keyword index powered by the ``bm25s`` library."""

    def __init__(self, config: BM25Config) -> None:
        self._config = config
        self._retriever: bm25s.BM25 | None = None
        self._chunks: list[DocumentChunk] = []

    # ------------------------------------------------------------------
    # IndexStore interface
    # ------------------------------------------------------------------

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray | None = None,
    ) -> None:
        """Tokenize chunk texts and build a BM25 index.

        Parameters
        ----------
        chunks:
            Document chunks to index.
        embeddings:
            Ignored for sparse indexing; accepted for interface compatibility.
        """
        if not chunks:
            return

        self._chunks.extend(chunks)

        corpus = [c.text for c in self._chunks]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        self._retriever = bm25s.BM25(k1=self._config.k1, b=self._config.b)
        self._retriever.index(corpus_tokens)

        logger.debug(
            "BM25 index built with %d documents (k1=%.2f, b=%.2f)",
            len(self._chunks),
            self._config.k1,
            self._config.b,
        )

    def search(
        self,
        query: str | np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search the BM25 index with a text query.

        Parameters
        ----------
        query:
            A plain-text query string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        List of ``(DocumentChunk, score)`` tuples sorted by descending BM25
        score.
        """
        if self._retriever is None or len(self._chunks) == 0:
            return []

        if not isinstance(query, str):
            raise TypeError(
                "BM25SparseStore.search() requires a string query, not an "
                "embedding vector."
            )

        query_tokens = bm25s.tokenize([query], stopwords="en")
        actual_k = min(top_k, len(self._chunks))
        results, scores = self._retriever.retrieve(query_tokens, k=actual_k)

        output: list[tuple[DocumentChunk, float]] = []
        for idx, score in zip(results[0], scores[0]):
            idx_int = int(idx)
            if 0 <= idx_int < len(self._chunks):
                output.append((self._chunks[idx_int], float(score)))
        return output

    def clear(self) -> None:
        """Reset the BM25 index and free associated memory."""
        self._retriever = None
        self._chunks.clear()
        logger.debug("BM25 index cleared.")

    def __len__(self) -> int:
        return len(self._chunks)
