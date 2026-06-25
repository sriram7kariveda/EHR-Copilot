"""Query Decomposition Agent -- breaks complex queries into sub-queries.

For complex query types (COMPARISON, TEMPORAL_NUMERIC, SUMMARY), a single
retrieval pass may miss relevant evidence.  This agent decomposes the query
into focused sub-queries so each can retrieve independently, then the
pipeline merges the chunk sets for broader evidence coverage.

Justification: MA-RAG (arXiv 2025), CoT-RAG (EMNLP 2025 Findings).
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from ehr_copilot.agents.base import AgentBase, AgentContext, AgentResult
from ehr_copilot.domain.query import QueryIntent, QueryType
from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)

# Query types that benefit from decomposition.
_DECOMPOSE_TYPES = {
    QueryType.COMPARISON,
    QueryType.TEMPORAL_NUMERIC,
    QueryType.SUMMARY,
}


class DecompositionInput(BaseModel):
    """Input bundle for the query decomposer."""

    query_text: str
    intent: QueryIntent


class DecompositionResult(BaseModel):
    """Output of the query decomposer."""

    original_query: str
    sub_queries: list[str] = Field(default_factory=list)
    was_decomposed: bool = False


class QueryDecomposerAgent(AgentBase[DecompositionInput, DecompositionResult]):
    """Decomposes complex clinical queries into focused sub-queries.

    Simple queries (FACTUAL, MEDICATION) pass through unchanged.
    Complex queries (COMPARISON, TEMPORAL_NUMERIC, SUMMARY) are broken
    into 2-4 sub-queries for targeted retrieval.

    This is a key component of the tree architecture: the Triage Agent
    determines *whether* to decompose, and this agent determines *how*.
    """

    name: str = "query_decomposer"

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def run(
        self,
        input_data: DecompositionInput,
        context: AgentContext,
    ) -> AgentResult[DecompositionResult]:
        start = time.perf_counter()

        query_text = input_data.query_text
        intent = input_data.intent

        # Only decompose complex query types.
        if intent.query_type not in _DECOMPOSE_TYPES:
            result = DecompositionResult(
                original_query=query_text,
                sub_queries=[query_text],
                was_decomposed=False,
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "Query decomposition skipped (type=%s)", intent.query_type.value,
            )
            return AgentResult(
                agent_name=self.name,
                output=result,
                latency_ms=round(elapsed, 2),
                metadata={"skipped": True, "query_type": intent.query_type.value},
            )

        # Ask LLM to decompose.
        prompt = f"""You are a clinical query decomposer. Break the following complex clinical question into 2-4 simpler, focused sub-questions that together cover the full scope of the original question.

Original question: {query_text}

Rules:
1. Each sub-question should be answerable from a single type of clinical document
2. Keep sub-questions specific and clinically precise
3. Preserve patient context in each sub-question
4. Return ONLY a JSON array of strings

Example:
Original: "Compare the patient's blood pressure trend with their medication changes over the past year"
Output: ["What were the patient's blood pressure readings over the past year?", "What antihypertensive medication changes were made over the past year?", "Were there any documented correlations between medication changes and blood pressure values?"]

Return a JSON array:"""

        llm_response = await self._llm.generate(
            LLMRequest(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1024,
            )
        )

        # Parse sub-queries.
        try:
            parsed = ResponseParser.parse_json_block(llm_response.text)
            if isinstance(parsed, list):
                sub_queries = [str(q) for q in parsed if isinstance(q, str)]
            else:
                sub_queries = [query_text]
        except (ValueError, TypeError):
            logger.warning("Failed to parse decomposition; using original query.")
            sub_queries = [query_text]

        if not sub_queries:
            sub_queries = [query_text]

        result = DecompositionResult(
            original_query=query_text,
            sub_queries=sub_queries,
            was_decomposed=len(sub_queries) > 1,
        )

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Decomposed query into %d sub-queries (type=%s)",
            len(sub_queries),
            intent.query_type.value,
        )

        return AgentResult(
            agent_name=self.name,
            output=result,
            latency_ms=round(elapsed, 2),
            metadata={
                "llm_latency_ms": llm_response.latency_ms,
                "model": llm_response.model,
                "num_sub_queries": len(sub_queries),
                "was_decomposed": result.was_decomposed,
                "query_type": intent.query_type.value,
            },
        )
