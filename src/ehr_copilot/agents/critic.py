"""Critic agent -- hallucination detection, evidence faithfulness, abstention."""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from ehr_copilot.agents.base import AgentBase, AgentContext, AgentResult
from ehr_copilot.domain.answer import CriticVerdict, DraftAnswer, ValidationResult
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.prompt_engine import PromptEngine
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class CriticInput(BaseModel):
    """Input bundle for the critic agent."""

    query_text: str
    draft_answer: DraftAnswer
    chunks: list[DocumentChunk]
    temporal_validation: ValidationResult | None = None
    numeric_validation: ValidationResult | None = None


class CriticOutput(BaseModel):
    """Output from the critic agent."""

    verdict: CriticVerdict
    revised_text: str | None = None
    abstention_reason: str | None = None
    issues: list[str] = Field(default_factory=list)


class CriticAgent(AgentBase[CriticInput, CriticOutput]):
    """Cross-references the draft answer against source evidence.

    The critic evaluates faithfulness by checking whether every claim in the
    draft answer is supported by at least one evidence chunk.  It also reviews
    any issues flagged by the temporal and numeric validators and decides:

    * **APPROVED** -- the answer is faithful and accurate.
    * **REVISED** -- the answer has fixable issues; the critic provides a
      corrected version of the text.
    * **ABSTAINED** -- there is insufficient evidence or critical errors; the
      system should decline to answer.
    """

    name: str = "critic"

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_engine: PromptEngine,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_engine

    async def run(
        self,
        input_data: CriticInput,
        context: AgentContext,
    ) -> AgentResult[CriticOutput]:
        start = time.perf_counter()

        # Collect validation issues for the prompt.
        temporal_issues: list[str] = []
        if input_data.temporal_validation and input_data.temporal_validation.issues:
            temporal_issues = input_data.temporal_validation.issues

        numeric_issues: list[str] = []
        if input_data.numeric_validation and input_data.numeric_validation.issues:
            numeric_issues = input_data.numeric_validation.issues

        prompt_text = self._prompts.render(
            "critic.txt",
            query=input_data.query_text,
            draft_answer=input_data.draft_answer.text,
            chunks=input_data.chunks,
            temporal_issues=temporal_issues,
            numeric_issues=numeric_issues,
        )

        llm_response = await self._llm.generate(
            LLMRequest(
                prompt=prompt_text,
                temperature=0.0,
                max_tokens=8192,
            )
        )

        # Parse the critic's structured response.
        try:
            parsed = ResponseParser.parse_json_block(llm_response.text)

            raw_verdict = parsed.get("verdict", "ABSTAINED")
            verdict = self._parse_verdict(raw_verdict)

            output = CriticOutput(
                verdict=verdict,
                revised_text=parsed.get("revised_text"),
                abstention_reason=parsed.get("abstention_reason"),
                issues=list(parsed.get("issues", [])),
            )
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Critic failed to parse LLM response, defaulting to ABSTAINED: %s",
                exc,
            )
            output = CriticOutput(
                verdict=CriticVerdict.ABSTAINED,
                abstention_reason=f"Critic parse error: {exc}",
                issues=[f"Critic parse error (auto-abstained): {exc}"],
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Critic verdict: %s (issues=%d)",
            output.verdict.value,
            len(output.issues),
        )

        return AgentResult(
            agent_name=self.name,
            output=output,
            latency_ms=round(elapsed_ms, 2),
            metadata={
                "llm_latency_ms": llm_response.latency_ms,
                "verdict": output.verdict.value,
                "num_issues": len(output.issues),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_verdict(raw: str) -> CriticVerdict:
        """Map the LLM's verdict string to the CriticVerdict enum.

        The LLM may respond with "APPROVED", "REVISED", or "ABSTAINED" in
        various casings.  We normalise and match.
        """
        cleaned = raw.strip().upper()
        mapping = {
            "APPROVED": CriticVerdict.APPROVED,
            "REVISED": CriticVerdict.REVISED,
            "ABSTAINED": CriticVerdict.ABSTAINED,
            # Common variants the LLM might produce.
            "APPROVE": CriticVerdict.APPROVED,
            "REVISE": CriticVerdict.REVISED,
            "ABSTAIN": CriticVerdict.ABSTAINED,
        }
        return mapping.get(cleaned, CriticVerdict.ABSTAINED)
