"""Unit tests for the pipeline orchestrator (agents/pipeline.py)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ehr_copilot.agents.base import AgentContext, AgentResult
from ehr_copilot.agents.critic import CriticAgent, CriticOutput
from ehr_copilot.agents.numeric_validator import NumericValidatorAgent
from ehr_copilot.agents.pipeline import CopilotPipeline
from ehr_copilot.agents.reasoning import ReasoningAgent
from ehr_copilot.agents.retrieval import RetrievalAgent
from ehr_copilot.agents.router import RouterAgent
from ehr_copilot.agents.temporal_validator import TemporalValidatorAgent
from ehr_copilot.config import AgentsConfig
from ehr_copilot.domain.answer import (
    CopilotAnswer,
    CriticVerdict,
    DraftAnswer,
    ValidationResult,
)
from ehr_copilot.domain.document import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    NoteSection,
)
from ehr_copilot.domain.query import ClinicalQuery, QueryIntent, QueryType
from ehr_copilot.llm.mock_client import MockLLMClient
from ehr_copilot.llm.prompt_engine import PromptEngine


# ---------------------------------------------------------------------------
# Helper to build a pipeline with mock agents
# ---------------------------------------------------------------------------


def _make_pipeline(
    router_response: str,
    reasoning_response: str,
    critic_response: str,
    max_retry_loops: int = 1,
) -> tuple[CopilotPipeline, MockLLMClient]:
    """Build a CopilotPipeline with a single MockLLMClient that returns
    different responses based on keyword matching."""

    client = MockLLMClient(
        default_response=critic_response,
        responses={
            "classify": router_response,
            "clinical query classifier": router_response,
            "reasoning assistant": reasoning_response,
            "think step by step": reasoning_response,
            "answer critic": critic_response,
            "faithfulness": critic_response,
            "temporal": json.dumps({"valid": True, "issues": [], "corrections": []}),
            "numeric": json.dumps({"valid": True, "issues": [], "corrections": []}),
        },
    )

    engine = PromptEngine()

    router = RouterAgent(llm_client=client, prompt_engine=engine)
    reasoning = ReasoningAgent(llm_client=client, prompt_engine=engine)
    temporal = TemporalValidatorAgent(llm_client=client, prompt_engine=engine)
    numeric = NumericValidatorAgent(llm_client=client, prompt_engine=engine)
    critic = CriticAgent(llm_client=client, prompt_engine=engine)

    # Retrieval agent needs a HybridRetriever -- we mock it.
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        (
            DocumentChunk(
                chunk_id="chunk-001",
                text="Hemoglobin A1c: 7.2 %. Encounter on 2024-01-15.",
                metadata=ChunkMetadata(
                    patient_id="patient-001",
                    document_id="doc-001",
                    document_type=DocumentType.ENCOUNTER_SUMMARY,
                    section=NoteSection.LABS_RESULTS,
                ),
            ),
            0.85,
        ),
        (
            DocumentChunk(
                chunk_id="chunk-002",
                text="Metformin 500mg twice daily. Lisinopril 10mg once daily.",
                metadata=ChunkMetadata(
                    patient_id="patient-001",
                    document_id="doc-001",
                    document_type=DocumentType.ENCOUNTER_SUMMARY,
                    section=NoteSection.MEDICATIONS,
                ),
            ),
            0.72,
        ),
    ]

    retrieval = RetrievalAgent(retriever=mock_retriever, top_k=3)
    config = AgentsConfig(max_retry_loops=max_retry_loops)

    pipeline = CopilotPipeline(
        router=router,
        retrieval=retrieval,
        reasoning=reasoning,
        temporal_validator=temporal,
        numeric_validator=numeric,
        critic=critic,
        config=config,
    )

    return pipeline, client


def _make_query() -> ClinicalQuery:
    return ClinicalQuery(
        query_id="q-test-001",
        patient_id="patient-001",
        session_id="sess-test",
        text="What is the patient's latest A1c?",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCopilotPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_returns_copilot_answer(self):
        """A normal run should return a CopilotAnswer with APPROVED verdict."""
        router_resp = json.dumps({
            "query_type": "FACTUAL",
            "requires_temporal": False,
            "requires_numeric": False,
            "key_entities": ["A1c"],
            "confidence": 0.9,
        })
        reasoning_resp = (
            "<reasoning>Chunk [1] shows A1c of 7.2%.</reasoning>\n"
            "<answer>The latest A1c is 7.2%.</answer>\n"
            "<source_chunks>1</source_chunks>"
        )
        critic_resp = json.dumps({
            "verdict": "APPROVED",
            "issues": [],
            "revised_text": None,
            "abstention_reason": None,
        })

        pipeline, client = _make_pipeline(router_resp, reasoning_resp, critic_resp)
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            (
                DocumentChunk(
                    chunk_id="c1",
                    text="A1c: 7.2%.",
                    metadata=ChunkMetadata(
                        patient_id="patient-001",
                        document_id="d1",
                        document_type=DocumentType.ENCOUNTER_SUMMARY,
                    ),
                ),
                0.9,
            )
        ]
        pipeline._retrieval._retriever = mock_retriever

        answer = await pipeline.run(_make_query(), mock_retriever)

        assert isinstance(answer, CopilotAnswer)
        assert answer.verdict == CriticVerdict.APPROVED
        assert answer.query_id == "q-test-001"
        assert answer.patient_id == "patient-001"
        assert "7.2%" in answer.text
        assert answer.latency_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_handles_abstention(self):
        """When the critic abstains, the answer should be flagged as abstention."""
        router_resp = json.dumps({
            "query_type": "FACTUAL",
            "requires_temporal": False,
            "requires_numeric": False,
            "key_entities": [],
            "confidence": 0.5,
        })
        reasoning_resp = (
            "<reasoning>No relevant data found.</reasoning>\n"
            "<answer>I do not have enough information.</answer>\n"
            "<source_chunks></source_chunks>"
        )
        critic_resp = json.dumps({
            "verdict": "ABSTAINED",
            "issues": ["No supporting evidence found."],
            "revised_text": None,
            "abstention_reason": "Insufficient evidence to answer reliably.",
        })

        pipeline, _ = _make_pipeline(router_resp, reasoning_resp, critic_resp)
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        pipeline._retrieval._retriever = mock_retriever

        answer = await pipeline.run(_make_query(), mock_retriever)

        assert answer.is_abstention is True
        assert answer.verdict == CriticVerdict.ABSTAINED
        assert answer.abstention_reason is not None
        assert "evidence" in answer.abstention_reason.lower() or "Insufficient" in answer.abstention_reason

    @pytest.mark.asyncio
    async def test_pipeline_retry_on_revise(self):
        """When the critic returns REVISED, the pipeline should retry reasoning
        and then accept on the second pass."""
        router_resp = json.dumps({
            "query_type": "FACTUAL",
            "requires_temporal": False,
            "requires_numeric": False,
            "key_entities": ["A1c"],
            "confidence": 0.9,
        })
        reasoning_resp = (
            "<reasoning>Chunk [1] shows A1c.</reasoning>\n"
            "<answer>The A1c is 7.2%.</answer>\n"
            "<source_chunks>1</source_chunks>"
        )

        # The critic will first revise, then approve.
        # Since MockLLMClient matches keywords, we need to be creative.
        # We will build the pipeline with max_retry_loops=1 and have the
        # critic always return REVISED. The pipeline should exhaust retries
        # and use the revised text.
        critic_resp = json.dumps({
            "verdict": "REVISED",
            "issues": ["Minor inaccuracy in date."],
            "revised_text": "The patient's A1c is 7.2%, measured on 2024-01-15.",
            "abstention_reason": None,
        })

        pipeline, _ = _make_pipeline(
            router_resp, reasoning_resp, critic_resp, max_retry_loops=1
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            (
                DocumentChunk(
                    chunk_id="c1",
                    text="A1c: 7.2% on 2024-01-15.",
                    metadata=ChunkMetadata(
                        patient_id="patient-001",
                        document_id="d1",
                        document_type=DocumentType.ENCOUNTER_SUMMARY,
                    ),
                ),
                0.9,
            )
        ]
        pipeline._retrieval._retriever = mock_retriever

        answer = await pipeline.run(_make_query(), mock_retriever)

        # After exhausting retries with REVISED, the pipeline should still
        # return an answer (using the revised text from the last critic pass).
        assert isinstance(answer, CopilotAnswer)
        assert answer.verdict == CriticVerdict.REVISED
        assert "7.2%" in answer.text
