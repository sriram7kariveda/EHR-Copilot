"""Unit tests for the router agent (agents/router.py)."""

from __future__ import annotations

import json

import pytest

from ehr_copilot.agents.base import AgentContext
from ehr_copilot.agents.router import RouterAgent
from ehr_copilot.domain.query import QueryType
from ehr_copilot.llm.mock_client import MockLLMClient
from ehr_copilot.llm.prompt_engine import PromptEngine


@pytest.fixture
def agent_context() -> AgentContext:
    return AgentContext(
        session_id="test-session",
        patient_id="patient-001",
        query_id="query-001",
    )


def _make_router(response_text: str) -> RouterAgent:
    """Build a RouterAgent backed by a MockLLMClient with a fixed response."""
    client = MockLLMClient(default_response=response_text)
    engine = PromptEngine()
    return RouterAgent(llm_client=client, prompt_engine=engine)


class TestRouterAgent:
    @pytest.mark.asyncio
    async def test_classifies_factual_query(self, agent_context):
        response = json.dumps({
            "query_type": "FACTUAL",
            "requires_temporal": False,
            "requires_numeric": False,
            "key_entities": ["blood type"],
            "confidence": 0.93,
        })
        router = _make_router(response)
        result = await router.run("What is the patient's blood type?", agent_context)

        intent = result.output
        assert intent.query_type == QueryType.FACTUAL
        assert intent.requires_temporal is False
        assert intent.requires_numeric is False
        assert intent.confidence == pytest.approx(0.93)
        assert "blood type" in intent.key_entities

    @pytest.mark.asyncio
    async def test_classifies_temporal_query(self, agent_context):
        response = json.dumps({
            "query_type": "TEMPORAL",
            "requires_temporal": True,
            "requires_numeric": False,
            "key_entities": ["encounter", "visit"],
            "confidence": 0.87,
        })
        router = _make_router(response)
        result = await router.run("When was the last visit?", agent_context)

        intent = result.output
        assert intent.query_type == QueryType.TEMPORAL
        assert intent.requires_temporal is True

    @pytest.mark.asyncio
    async def test_fallback_to_unknown_on_parse_failure(self, agent_context):
        """If the LLM returns garbage, the router should fall back to UNKNOWN."""
        router = _make_router("This is not JSON at all, sorry!")
        result = await router.run("Tell me something", agent_context)

        intent = result.output
        assert intent.query_type == QueryType.UNKNOWN
        assert intent.confidence == 0.0
