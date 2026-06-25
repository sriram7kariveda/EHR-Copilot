"""Retrieval agent -- hybrid search over patient documents."""

from __future__ import annotations

import logging
import time

from ehr_copilot.agents.base import AgentBase, AgentContext, AgentResult
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.domain.query import ClinicalQuery
from ehr_copilot.indexing.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RetrievalAgent(AgentBase[ClinicalQuery, list[DocumentChunk]]):
    """Retrieves relevant document chunks for a clinical query.

    The agent delegates to a :class:`HybridRetriever` (dense + sparse fusion)
    and returns the top-k chunks with their RRF scores.
    """

    name: str = "retrieval"

    def __init__(
        self,
        retriever: HybridRetriever,
        top_k: int = 8,
    ) -> None:
        self._retriever = retriever
        self._top_k = top_k

    async def run(
        self,
        input_data: ClinicalQuery,
        context: AgentContext,
    ) -> AgentResult[list[DocumentChunk]]:
        start = time.perf_counter()

        # HybridRetriever.retrieve is synchronous -- call it directly.
        scored_chunks = self._retriever.retrieve(
            query=input_data.text,
            top_k=self._top_k,
        )

        # Separate chunks from scores for the output.
        chunks = [chunk for chunk, _score in scored_chunks]
        scores = [score for _chunk, score in scored_chunks]

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Retrieval returned %d chunks for query %s (top score=%.4f)",
            len(chunks),
            input_data.query_id,
            scores[0] if scores else 0.0,
        )

        return AgentResult(
            agent_name=self.name,
            output=chunks,
            latency_ms=round(elapsed_ms, 2),
            metadata={
                "num_chunks": len(chunks),
                "scores": [round(s, 4) for s in scores],
                "chunk_ids": [c.chunk_id for c in chunks],
            },
        )
