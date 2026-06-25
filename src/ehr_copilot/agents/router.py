"""Router agent -- classifies a clinical query into a structured intent."""

from __future__ import annotations

import logging
import time

from ehr_copilot.agents.base import AgentBase, AgentContext, AgentResult
from ehr_copilot.domain.query import QueryIntent, QueryType
from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.prompt_engine import PromptEngine
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class RouterAgent(AgentBase[str, QueryIntent]):
    """Classifies a raw query string into a :class:`QueryIntent`.

    The agent renders the ``router.txt`` prompt template, sends the query to
    the LLM, and parses the JSON response into a ``QueryIntent`` model.  If
    parsing fails the agent falls back to ``QueryType.UNKNOWN`` with zero
    confidence.
    """

    name: str = "router"

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_engine: PromptEngine,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_engine

    async def run(
        self,
        input_data: str,
        context: AgentContext,
    ) -> AgentResult[QueryIntent]:
        start = time.perf_counter()

        # Render the classification prompt.
        prompt_text = self._prompts.render("router.txt", query=input_data)

        # Ask the LLM to classify.
        llm_response = await self._llm.generate(
            LLMRequest(
                prompt=prompt_text,
                temperature=0.0,
                max_tokens=2048,
            )
        )

        # Parse the JSON response.
        try:
            parsed = ResponseParser.parse_json_block(llm_response.text)

            # Resolve the query type enum value.
            raw_type = parsed.get("query_type", "unknown")
            try:
                query_type = ResponseParser.parse_enum_value(raw_type, QueryType)
            except ValueError:
                query_type = QueryType.UNKNOWN

            intent = QueryIntent(
                query_type=query_type,
                requires_temporal=bool(parsed.get("requires_temporal", False)),
                requires_numeric=bool(parsed.get("requires_numeric", False)),
                key_entities=list(parsed.get("key_entities", [])),
                confidence=float(parsed.get("confidence", 0.0)),
            )
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Router failed to parse LLM response, falling back to UNKNOWN: %s",
                exc,
            )
            intent = QueryIntent(
                query_type=QueryType.UNKNOWN,
                requires_temporal=False,
                requires_numeric=False,
                key_entities=[],
                confidence=0.0,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Router classified query as %s (confidence=%.2f, temporal=%s, numeric=%s)",
            intent.query_type.value,
            intent.confidence,
            intent.requires_temporal,
            intent.requires_numeric,
        )

        return AgentResult(
            agent_name=self.name,
            output=intent,
            latency_ms=round(elapsed_ms, 2),
            metadata={
                "llm_latency_ms": llm_response.latency_ms,
                "model": llm_response.model,
            },
        )
