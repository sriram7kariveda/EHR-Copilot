"""Reasoning agent -- chain-of-thought synthesis over retrieved evidence."""

from __future__ import annotations

import logging
import re
import time

from pydantic import BaseModel, Field

from ehr_copilot.agents.base import AgentBase, AgentContext, AgentResult
from ehr_copilot.domain.answer import DraftAnswer
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.domain.query import ClinicalQuery, QueryIntent
from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.prompt_engine import PromptEngine
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class ReasoningInput(BaseModel):
    """Input bundle for the reasoning agent."""

    query: ClinicalQuery
    chunks: list[DocumentChunk]
    intent: QueryIntent
    prompt_template: str = "reasoning_cot.txt"  # branch-specific prompt


class ReasoningAgent(AgentBase[ReasoningInput, DraftAnswer]):
    """Generates a draft answer via chain-of-thought reasoning over evidence.

    The agent constructs a prompt that includes the patient query alongside the
    retrieved document chunks and instructs the LLM to reason step-by-step
    before producing a final answer.  The response is parsed to extract:

    * the reasoning trace
    * the final answer text with citations
    * the list of source chunk IDs used
    """

    name: str = "reasoning"

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_engine: PromptEngine,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_engine

    async def run(
        self,
        input_data: ReasoningInput,
        context: AgentContext,
    ) -> AgentResult[DraftAnswer]:
        start = time.perf_counter()

        # Build the branch-specific chain-of-thought prompt.
        prompt_text = self._prompts.render(
            input_data.prompt_template,
            query=input_data.query.text,
            chunks=input_data.chunks,
        )

        llm_response = await self._llm.generate(
            LLMRequest(
                prompt=prompt_text,
                temperature=0.1,
                max_tokens=8192,
            )
        )

        # Parse the structured response.
        reasoning_trace = self._extract_reasoning(llm_response.text)
        answer_text = self._extract_answer(llm_response.text)
        source_chunk_ids = self._extract_source_chunk_ids(
            llm_response.text, input_data.chunks
        )

        draft = DraftAnswer(
            text=answer_text,
            reasoning_trace=reasoning_trace,
            source_chunk_ids=source_chunk_ids,
            confidence=0.0,  # The critic will assign confidence later.
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Reasoning produced draft answer with %d source chunks",
            len(source_chunk_ids),
        )

        return AgentResult(
            agent_name=self.name,
            output=draft,
            latency_ms=round(elapsed_ms, 2),
            metadata={
                "llm_latency_ms": llm_response.latency_ms,
                "model": llm_response.model,
                "num_source_chunks": len(source_chunk_ids),
            },
        )

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_reasoning(text: str) -> str:
        """Extract the content between <reasoning> tags."""
        try:
            return ResponseParser.extract_between_tags(text, "reasoning")
        except ValueError:
            logger.debug("No <reasoning> tags found; using full text as trace.")
            return text.strip()

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract the content between <answer> tags."""
        try:
            return ResponseParser.extract_between_tags(text, "answer")
        except ValueError:
            logger.debug(
                "No <answer> tags found; falling back to text after reasoning."
            )
            # Fallback: strip out <reasoning> block and return the rest.
            cleaned = re.sub(
                r"<reasoning>.*?</reasoning>",
                "",
                text,
                flags=re.DOTALL,
            ).strip()
            return cleaned if cleaned else text.strip()

    @staticmethod
    def _extract_source_chunk_ids(
        text: str,
        chunks: list[DocumentChunk],
    ) -> list[str]:
        """Map cited chunk numbers back to chunk IDs.

        The LLM cites chunks as ``[1], [2]`` etc.  The ``<source_chunks>``
        block should contain a comma-separated list of the 1-based indices.
        """
        try:
            raw = ResponseParser.extract_between_tags(text, "source_chunks")
        except ValueError:
            raw = ""

        # Parse comma-separated integers from the source_chunks block.
        indices: list[int] = []
        for token in re.findall(r"\d+", raw):
            idx = int(token)
            if 1 <= idx <= len(chunks):
                indices.append(idx)

        # If the explicit block was empty, scan the answer text for [N] refs.
        if not indices:
            for match in re.finditer(r"\[(\d+)\]", text):
                idx = int(match.group(1))
                if 1 <= idx <= len(chunks):
                    indices.append(idx)

        # Deduplicate while preserving order.
        seen: set[int] = set()
        unique: list[int] = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)

        return [chunks[i - 1].chunk_id for i in unique]
