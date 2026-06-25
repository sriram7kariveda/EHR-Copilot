"""Retrieval Sufficiency Evaluator -- judges if retrieved evidence is adequate.

After retrieval, this agent evaluates whether the chunks actually contain
enough information to answer the query.  Based on the evaluation, the
pipeline takes one of three actions:

    SUFFICIENT   -> proceed to reasoning (evidence is good)
    INSUFFICIENT -> reformulate query and re-retrieve (evidence gaps)
    AMBIGUOUS    -> expand retrieval with broader search (unclear coverage)

This implements the Corrective RAG (CRAG) pattern from Yan et al. (2024).
"""

from __future__ import annotations

import logging
import time
from enum import Enum

from pydantic import BaseModel, Field

from ehr_copilot.agents.base import AgentBase, AgentContext, AgentResult
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class RetrievalVerdict(str, Enum):
    """Evaluation of retrieval quality."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    AMBIGUOUS = "ambiguous"


class RetrievalEvalInput(BaseModel):
    """Input bundle for the retrieval evaluator."""

    query_text: str
    chunks: list[DocumentChunk]

    model_config = {"arbitrary_types_allowed": True}


class RetrievalEvaluation(BaseModel):
    """Output of the retrieval sufficiency evaluator."""

    verdict: RetrievalVerdict
    coverage_score: float  # 0.0-1.0, how well chunks cover the query
    missing_aspects: list[str] = Field(default_factory=list)
    reformulated_query: str | None = None


class RetrievalEvaluatorAgent(AgentBase[RetrievalEvalInput, RetrievalEvaluation]):
    """Evaluates whether retrieved chunks are sufficient to answer the query.

    This is the CRAG (Corrective RAG) component of the tree architecture.
    It prevents the reasoning agent from generating answers based on
    inadequate evidence, reducing hallucination at the source.

    Configuration:
        Model: Same LLM as other agents (Qwen 3 8B via OpenRouter)
        Temperature: 0.0 (deterministic evaluation)
        Max tokens: 1024 (structured JSON output)
    """

    name: str = "retrieval_evaluator"

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def run(
        self,
        input_data: RetrievalEvalInput,
        context: AgentContext,
    ) -> AgentResult[RetrievalEvaluation]:
        start = time.perf_counter()

        query = input_data.query_text
        chunks = input_data.chunks

        # Build evidence summary for LLM evaluation (top 10 for efficiency).
        evidence_summary = "\n".join(
            f"[{i+1}] {c.display_source}: {c.text[:300]}..."
            for i, c in enumerate(chunks[:10])
        )

        prompt = f"""You are a clinical evidence evaluator. Assess whether the retrieved evidence chunks are SUFFICIENT to answer the clinical query.

Query: {query}

Retrieved evidence (top {min(len(chunks), 10)} chunks):
{evidence_summary}

Evaluate:
1. Does the evidence contain the specific information needed to answer the query?
2. Are there any critical aspects of the query that NO chunk addresses?
3. Is the evidence relevant or mostly noise?

Respond in JSON:
{{
    "verdict": "sufficient" | "insufficient" | "ambiguous",
    "coverage_score": 0.0-1.0,
    "missing_aspects": ["list of what information is missing"],
    "reformulated_query": "a better query to find missing info (only if insufficient)"
}}"""

        llm_response = await self._llm.generate(
            LLMRequest(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1024,
            )
        )

        # Parse response.
        try:
            parsed = ResponseParser.parse_json_block(llm_response.text)
            verdict_str = parsed.get("verdict", "sufficient").lower()
            try:
                verdict = RetrievalVerdict(verdict_str)
            except ValueError:
                verdict = RetrievalVerdict.SUFFICIENT

            evaluation = RetrievalEvaluation(
                verdict=verdict,
                coverage_score=float(parsed.get("coverage_score", 0.5)),
                missing_aspects=list(parsed.get("missing_aspects", [])),
                reformulated_query=parsed.get("reformulated_query"),
            )
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse retrieval evaluation: %s", exc)
            evaluation = RetrievalEvaluation(
                verdict=RetrievalVerdict.SUFFICIENT,
                coverage_score=0.5,
                missing_aspects=[],
            )

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "Retrieval evaluation: %s (coverage=%.2f, missing=%d aspects)",
            evaluation.verdict.value,
            evaluation.coverage_score,
            len(evaluation.missing_aspects),
        )

        return AgentResult(
            agent_name=self.name,
            output=evaluation,
            latency_ms=round(elapsed, 2),
            metadata={
                "llm_latency_ms": llm_response.latency_ms,
                "model": llm_response.model,
                "verdict": evaluation.verdict.value,
                "coverage_score": evaluation.coverage_score,
            },
        )
