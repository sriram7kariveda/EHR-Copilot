"""Pipeline orchestrator -- wires all agents into a tree-structured flow."""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from typing import TYPE_CHECKING

from ehr_copilot.agents.base import AgentContext
from ehr_copilot.agents.chunk_filter import filter_and_rerank_chunks, get_branch_top_k
from ehr_copilot.agents.critic import CriticAgent, CriticInput, CriticOutput
from ehr_copilot.agents.entity_verifier import verify_entities
from ehr_copilot.agents.numeric_validator import (
    NumericValidationInput,
    NumericValidatorAgent,
)
from ehr_copilot.agents.query_decomposer import (
    QueryDecomposerAgent,
    DecompositionInput,
    DecompositionResult,
)
from ehr_copilot.agents.reasoning import ReasoningAgent, ReasoningInput
from ehr_copilot.agents.retrieval import RetrievalAgent
from ehr_copilot.agents.retrieval_evaluator import (
    RetrievalEvaluatorAgent,
    RetrievalEvalInput,
    RetrievalVerdict,
)
from ehr_copilot.agents.router import RouterAgent
from ehr_copilot.agents.temporal_validator import (
    TemporalValidationInput,
    TemporalValidatorAgent,
)
from ehr_copilot.config import AgentsConfig
from ehr_copilot.domain.answer import (
    CopilotAnswer,
    CriticVerdict,
    DraftAnswer,
    ValidationResult,
)
from ehr_copilot.domain.audit import AuditEventType
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.domain.query import ClinicalQuery, QueryIntent
from ehr_copilot.domain.query import QueryType
from ehr_copilot.indexing.hybrid_retriever import HybridRetriever

if TYPE_CHECKING:
    from ehr_copilot.audit.logger import AuditLogger

logger = logging.getLogger(__name__)

# Tree routing: each query type maps to a specialised reasoning prompt.
_REASONING_PROMPT_MAP: dict[QueryType, str] = {
    QueryType.FACTUAL: "reasoning_cot.txt",
    QueryType.TEMPORAL: "reasoning_temporal.txt",
    QueryType.NUMERIC: "reasoning_numeric.txt",
    QueryType.TEMPORAL_NUMERIC: "reasoning_temporal.txt",
    QueryType.MEDICATION: "reasoning_medication.txt",
    QueryType.SUMMARY: "reasoning_summary.txt",
    QueryType.COMPARISON: "reasoning_comparison.txt",
    QueryType.UNKNOWN: "reasoning_cot.txt",
}

# Maximum retrieval re-attempts when evaluator says INSUFFICIENT.
_MAX_RETRIEVAL_RETRIES = 1


class CopilotPipeline:
    """Orchestrates the multi-agent tree pipeline for clinical queries.

    Tree architecture (9 agents, 7 query-type branches)::

        1. Triage Agent          -- classify query intent (tree root)
        2. Query Decomposer      -- break complex queries into sub-queries
                                    (COMPARISON, TEMPORAL_NUMERIC, SUMMARY)
        3. Retrieval             -- branch-specific top_k + chunk filtering
        4. Retrieval Evaluator   -- CRAG: sufficient / insufficient / ambiguous
                                    If INSUFFICIENT -> reformulate + re-retrieve
        5. Reasoning             -- branch-specific CoT prompt
        6. Temporal Validator    -- (TEMPORAL / TEMPORAL_NUMERIC branches only)
        7. Numeric Validator     -- (NUMERIC / TEMPORAL_NUMERIC branches only)
        8. Critic (DPO-trained)  -- approve / revise / abstain (all branches)
                                    If REVISED -> loop back to step 5
        9. Entity Verifier       -- deterministic hallucination removal

    The Triage Agent is justified because it controls 4 dimensions:
        (a) Whether to decompose the query (step 2)
        (b) Retrieval depth and chunk filtering strategy (step 3)
        (c) Which specialised reasoning prompt to use (step 5)
        (d) Which validators to activate (steps 6-7)
    """

    def __init__(
        self,
        router: RouterAgent,
        query_decomposer: QueryDecomposerAgent,
        retrieval: RetrievalAgent,
        retrieval_evaluator: RetrievalEvaluatorAgent,
        reasoning: ReasoningAgent,
        temporal_validator: TemporalValidatorAgent,
        numeric_validator: NumericValidatorAgent,
        critic: CriticAgent,
        config: AgentsConfig,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._router = router
        self._query_decomposer = query_decomposer
        self._retrieval = retrieval
        self._retrieval_evaluator = retrieval_evaluator
        self._reasoning = reasoning
        self._temporal_validator = temporal_validator
        self._numeric_validator = numeric_validator
        self._critic = critic
        self._config = config
        self._audit_logger = audit_logger

    async def _audit(
        self,
        context: AgentContext,
        event_type: AuditEventType,
        data: dict,
    ) -> None:
        """Log an audit event, silently ignoring failures."""
        if self._audit_logger is None:
            return
        try:
            await self._audit_logger.log(
                session_id=context.session_id,
                patient_id=context.patient_id,
                event_type=event_type,
                data=data,
            )
        except Exception:
            logger.warning("Failed to write audit event %s", event_type.value, exc_info=True)

    async def _retrieve_chunks(
        self,
        query: ClinicalQuery,
        retriever: HybridRetriever,
        intent: QueryIntent,
        context: AgentContext,
    ) -> tuple[list[DocumentChunk], float]:
        """Run branch-specific retrieval + chunk filtering."""
        branch_top_k = get_branch_top_k(intent.query_type)
        original_retriever = self._retrieval._retriever
        original_top_k = self._retrieval._top_k
        self._retrieval._retriever = retriever
        self._retrieval._top_k = branch_top_k

        try:
            retrieval_result = await self._retrieval.run(query, context)
        finally:
            self._retrieval._retriever = original_retriever
            self._retrieval._top_k = original_top_k

        chunks = retrieval_result.output
        chunks = filter_and_rerank_chunks(chunks, intent.query_type)

        logger.info(
            "Tree branch %s: top_k=%d, %d chunks after filtering",
            intent.query_type.value, branch_top_k, len(chunks),
        )
        return chunks, retrieval_result.latency_ms

    async def run(
        self,
        query: ClinicalQuery,
        retriever: HybridRetriever,
    ) -> CopilotAnswer:
        pipeline_start = time.perf_counter()

        context = AgentContext(
            session_id=query.session_id,
            patient_id=query.patient_id,
            query_id=query.query_id,
        )

        step_latencies: dict[str, float] = {}

        # ------------------------------------------------------------------
        # Step 1: Triage -- classify the query (tree root)
        # ------------------------------------------------------------------
        router_result = await self._router.run(query.text, context)
        intent: QueryIntent = router_result.output
        step_latencies["triage"] = router_result.latency_ms
        query.intent = intent

        await self._audit(context, AuditEventType.ROUTE_CLASSIFIED, {
            "query_id": query.query_id,
            "query_type": intent.query_type.value,
            "requires_temporal": intent.requires_temporal,
            "requires_numeric": intent.requires_numeric,
            "latency_ms": router_result.latency_ms,
        })

        # ------------------------------------------------------------------
        # Step 2: Query Decomposition (complex queries only)
        # ------------------------------------------------------------------
        decomp_input = DecompositionInput(
            query_text=query.text,
            intent=intent,
        )
        decomp_result = await self._query_decomposer.run(decomp_input, context)
        decomposition: DecompositionResult = decomp_result.output
        step_latencies["decomposition"] = decomp_result.latency_ms

        logger.info(
            "Query decomposition: %d sub-queries (decomposed=%s)",
            len(decomposition.sub_queries),
            decomposition.was_decomposed,
        )

        # ------------------------------------------------------------------
        # Step 3: Retrieve -- for each sub-query, merge results
        # ------------------------------------------------------------------
        all_chunks: list[DocumentChunk] = []
        seen_chunk_ids: set[str] = set()
        total_retrieval_ms = 0.0

        for sub_query_text in decomposition.sub_queries:
            sub_query = ClinicalQuery(
                query_id=query.query_id,
                patient_id=query.patient_id,
                session_id=query.session_id,
                text=sub_query_text,
                intent=intent,
            )
            sub_chunks, latency = await self._retrieve_chunks(
                sub_query, retriever, intent, context,
            )
            total_retrieval_ms += latency

            for chunk in sub_chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)

        chunks = all_chunks
        step_latencies["retrieval"] = round(total_retrieval_ms, 2)

        # ------------------------------------------------------------------
        # Step 4: Retrieval Sufficiency Evaluation (CRAG pattern)
        # ------------------------------------------------------------------
        ret_eval_input = RetrievalEvalInput(
            query_text=query.text,
            chunks=chunks,
        )
        eval_result = await self._retrieval_evaluator.run(ret_eval_input, context)
        retrieval_eval = eval_result.output
        step_latencies["retrieval_eval"] = eval_result.latency_ms

        if retrieval_eval.verdict == RetrievalVerdict.INSUFFICIENT:
            # Re-retrieve with reformulated query.
            reformulated = retrieval_eval.reformulated_query or query.text
            logger.info(
                "Retrieval INSUFFICIENT (coverage=%.2f). Re-retrieving with: %s",
                retrieval_eval.coverage_score,
                reformulated[:100],
            )
            retry_query = ClinicalQuery(
                query_id=query.query_id,
                patient_id=query.patient_id,
                session_id=query.session_id,
                text=reformulated,
                intent=intent,
            )
            retry_chunks, retry_ms = await self._retrieve_chunks(
                retry_query, retriever, intent, context,
            )
            step_latencies["retrieval_retry"] = round(retry_ms, 2)

            # Merge new chunks with existing (deduplicated).
            for chunk in retry_chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)

            logger.info(
                "After re-retrieval: %d total chunks", len(chunks),
            )

        elif retrieval_eval.verdict == RetrievalVerdict.AMBIGUOUS:
            logger.info(
                "Retrieval AMBIGUOUS (coverage=%.2f). Proceeding with caution.",
                retrieval_eval.coverage_score,
            )

        await self._audit(context, AuditEventType.RETRIEVAL_COMPLETED, {
            "query_id": query.query_id,
            "num_chunks": len(chunks),
            "chunk_ids": [c.chunk_id for c in chunks],
            "retrieval_verdict": retrieval_eval.verdict.value,
            "coverage_score": retrieval_eval.coverage_score,
        })

        # ------------------------------------------------------------------
        # Steps 5-8: Reasoning -> Validation -> Critic loop
        # ------------------------------------------------------------------
        max_retries = self._config.max_retry_loops
        draft: DraftAnswer | None = None
        critic_output: CriticOutput | None = None
        temporal_validation: ValidationResult | None = None
        numeric_validation: ValidationResult | None = None

        branch_prompt = _REASONING_PROMPT_MAP.get(
            intent.query_type, "reasoning_cot.txt"
        )
        logger.info("Tree branch prompt: %s", branch_prompt)

        for attempt in range(1 + max_retries):
            # Step 5: Reasoning -- branch-specific CoT prompt.
            reasoning_input = ReasoningInput(
                query=query,
                chunks=chunks,
                intent=intent,
                prompt_template=branch_prompt,
            )
            reasoning_result = await self._reasoning.run(reasoning_input, context)
            draft = reasoning_result.output
            step_latencies[f"reasoning_{attempt}"] = reasoning_result.latency_ms

            await self._audit(context, AuditEventType.REASONING_COMPLETED, {
                "query_id": query.query_id,
                "attempt": attempt,
                "confidence": draft.confidence,
                "source_chunk_ids": draft.source_chunk_ids,
                "latency_ms": reasoning_result.latency_ms,
            })

            # Step 6: Temporal validation (TEMPORAL branches only).
            temporal_validation = None
            if intent.requires_temporal:
                tv_input = TemporalValidationInput(
                    draft_answer=draft,
                    chunks=chunks,
                    intent=intent,
                )
                tv_result = await self._temporal_validator.run(tv_input, context)
                temporal_validation = tv_result.output
                step_latencies[f"temporal_{attempt}"] = tv_result.latency_ms

            # Step 7: Numeric validation (NUMERIC branches only).
            numeric_validation = None
            if intent.requires_numeric:
                nv_input = NumericValidationInput(
                    draft_answer=draft,
                    chunks=chunks,
                )
                nv_result = await self._numeric_validator.run(nv_input, context)
                numeric_validation = nv_result.output
                step_latencies[f"numeric_{attempt}"] = nv_result.latency_ms

            if temporal_validation is not None or numeric_validation is not None:
                await self._audit(context, AuditEventType.VALIDATION_COMPLETED, {
                    "query_id": query.query_id,
                    "attempt": attempt,
                    "temporal_valid": temporal_validation.valid if temporal_validation else None,
                    "numeric_valid": numeric_validation.valid if numeric_validation else None,
                    "temporal_issues": len(temporal_validation.issues) if temporal_validation else 0,
                    "numeric_issues": len(numeric_validation.issues) if numeric_validation else 0,
                })

            # Step 8: Critic -- approve / revise / abstain.
            critic_input = CriticInput(
                query_text=query.text,
                draft_answer=draft,
                chunks=chunks,
                temporal_validation=temporal_validation,
                numeric_validation=numeric_validation,
            )
            critic_result = await self._critic.run(critic_input, context)
            critic_output = critic_result.output
            step_latencies[f"critic_{attempt}"] = critic_result.latency_ms

            await self._audit(context, AuditEventType.CRITIC_VERDICT, {
                "query_id": query.query_id,
                "attempt": attempt,
                "verdict": critic_output.verdict.value,
                "abstention_reason": critic_output.abstention_reason,
                "latency_ms": critic_result.latency_ms,
            })

            if critic_output.verdict != CriticVerdict.REVISED:
                break

            if attempt < max_retries:
                logger.info(
                    "Critic requested revision (attempt %d/%d), re-running reasoning.",
                    attempt + 1, max_retries,
                )
                if critic_output.revised_text:
                    draft = DraftAnswer(
                        text=critic_output.revised_text,
                        reasoning_trace=draft.reasoning_trace,
                        source_chunk_ids=draft.source_chunk_ids,
                        confidence=draft.confidence,
                    )
            else:
                logger.info(
                    "Critic requested revision but max retries (%d) exhausted.",
                    max_retries,
                )

        # ------------------------------------------------------------------
        # Step 9: Entity Verification + Build CopilotAnswer
        # ------------------------------------------------------------------
        assert draft is not None
        assert critic_output is not None

        if critic_output.verdict == CriticVerdict.REVISED and critic_output.revised_text:
            final_text = critic_output.revised_text
        elif critic_output.verdict == CriticVerdict.ABSTAINED:
            final_text = (
                critic_output.abstention_reason
                or "I cannot provide a reliable answer based on the available evidence."
            )
        else:
            final_text = draft.text

        if critic_output.verdict != CriticVerdict.ABSTAINED and chunks:
            chunk_texts = [c.text for c in chunks]
            verified_text, grounded, removed = verify_entities(final_text, chunk_texts)
            if removed:
                logger.info(
                    "Entity verifier removed %d hallucinated entities: %s",
                    len(removed), removed,
                )
            final_text = verified_text
            step_latencies["entity_verification"] = 0.0

        pipeline_elapsed_ms = (time.perf_counter() - pipeline_start) * 1000

        answer = CopilotAnswer(
            answer_id=str(uuid4()),
            query_id=query.query_id,
            patient_id=query.patient_id,
            text=final_text,
            verdict=critic_output.verdict,
            confidence=draft.confidence,
            reasoning_trace=draft.reasoning_trace,
            temporal_validation=temporal_validation,
            numeric_validation=numeric_validation,
            abstention_reason=critic_output.abstention_reason,
            latency_ms=round(pipeline_elapsed_ms, 2),
        )

        logger.info(
            "Pipeline completed: answer_id=%s verdict=%s latency=%.1fms steps=%s",
            answer.answer_id,
            answer.verdict.value,
            answer.latency_ms,
            step_latencies,
        )

        return answer
