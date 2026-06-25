"""Hybrid retriever combining dense and sparse results via RRF + cross-encoder reranking."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from ehr_copilot.config import RetrievalConfig
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.indexing.base import RetrieverBase
from ehr_copilot.indexing.embedding import EmbeddingModel
from ehr_copilot.indexing.sparse_store import BM25SparseStore
from ehr_copilot.indexing.vector_store import FAISSVectorStore

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class HybridRetriever(RetrieverBase):
    """Fuses dense (FAISS) and sparse (BM25) retrieval with RRF scoring,
    then optionally reranks the top candidates with a cross-encoder.

    Pipeline::

        Dense top-20 ─┐
                       ├─ RRF fusion → top-30 → Cross-Encoder rerank → top-8
        Sparse top-20 ┘
    """

    _cross_encoder: CrossEncoder | None = None
    _cross_encoder_model: str | None = None

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        sparse_store: BM25SparseStore,
        embedding_model: EmbeddingModel,
        config: RetrievalConfig,
    ) -> None:
        self._vector_store = vector_store
        self._sparse_store = sparse_store
        self._embedding_model = embedding_model
        self._config = config

    # ------------------------------------------------------------------
    # Cross-encoder lazy loading (shared across instances)
    # ------------------------------------------------------------------

    @classmethod
    def _get_cross_encoder(cls, model_name: str) -> CrossEncoder:
        """Lazily load the cross-encoder model (once, shared across all instances)."""
        if cls._cross_encoder is None or cls._cross_encoder_model != model_name:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker: %s", model_name)
            cls._cross_encoder = CrossEncoder(model_name)
            cls._cross_encoder_model = model_name
        return cls._cross_encoder

    # ------------------------------------------------------------------
    # RetrieverBase interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve chunks by fusing dense and sparse rankings, then reranking.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Override for the number of final results.

        Returns
        -------
        Fused and reranked list of ``(DocumentChunk, score)`` tuples.
        """
        # 1. Encode query for dense retrieval.
        query_embedding = self._embedding_model.encode_query(query)

        # 2. Dense retrieval.
        dense_results = self._vector_store.search(
            query_embedding,
            top_k=self._config.top_k_dense,
        )

        # 3. Sparse retrieval.
        sparse_results = self._sparse_store.search(
            query,
            top_k=self._config.top_k_sparse,
        )

        # 4. Apply Reciprocal Rank Fusion.
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # 5. Cross-encoder reranking (if enabled).
        reranker_cfg = self._config.reranker
        if reranker_cfg.enabled and len(fused) > 0:
            # Take top-K candidates for reranking.
            rerank_k = min(reranker_cfg.top_k_rerank, len(fused))
            candidates = fused[:rerank_k]
            reranked = self._cross_encoder_rerank(query, candidates, reranker_cfg.model)
            # Return final_top_k from the reranked results.
            final_k = min(self._config.final_top_k, len(reranked))
            return reranked[:final_k]

        # Fallback: no reranking, just return RRF results.
        final_k = min(self._config.final_top_k, len(fused))
        return fused[:final_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: list[tuple[DocumentChunk, float]],
        model_name: str,
    ) -> list[tuple[DocumentChunk, float]]:
        """Rerank candidates using a cross-encoder model.

        The cross-encoder scores each (query, chunk_text) pair jointly,
        providing much higher precision than bi-encoder retrieval alone.

        Parameters
        ----------
        query:
            The original query text.
        candidates:
            RRF-fused candidates to rerank.
        model_name:
            Cross-encoder model identifier.

        Returns
        -------
        Reranked list sorted by descending cross-encoder score.
        """
        cross_encoder = self._get_cross_encoder(model_name)

        # Build (query, document) pairs for the cross-encoder.
        pairs = [(query, chunk.text) for chunk, _score in candidates]

        # Score all pairs in a single batch.
        scores = cross_encoder.predict(pairs, show_progress_bar=False)

        # Pair scores back with chunks and sort by descending score.
        scored = [
            (chunk, float(ce_score))
            for (chunk, _rrf_score), ce_score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "Cross-encoder reranked %d candidates (top score: %.4f)",
            len(scored),
            scored[0][1] if scored else 0.0,
        )

        return scored

    def _reciprocal_rank_fusion(
        self,
        *rankings: list[tuple[DocumentChunk, float]],
    ) -> list[tuple[DocumentChunk, float]]:
        """Merge multiple ranked lists using RRF.

        Parameters
        ----------
        *rankings:
            Each ranking is a list of ``(DocumentChunk, original_score)``
            tuples already sorted by descending relevance.

        Returns
        -------
        Merged list sorted by descending RRF score.
        """
        k = self._config.rrf_k

        # Accumulate RRF scores keyed by chunk_id.
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        chunk_map: dict[str, DocumentChunk] = {}

        for ranking in rankings:
            for rank, (chunk, _original_score) in enumerate(ranking, start=1):
                rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)
                chunk_map[chunk.chunk_id] = chunk

        # Sort by fused score descending.
        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids]
