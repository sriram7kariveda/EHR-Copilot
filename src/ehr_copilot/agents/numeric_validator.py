"""Numeric validator agent -- UCUM unit checking and math verification."""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel

from ehr_copilot.agents.base import AgentBase, AgentContext, AgentResult
from ehr_copilot.domain.answer import DraftAnswer, ValidationResult
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.prompt_engine import PromptEngine
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class NumericValidationInput(BaseModel):
    """Input bundle for the numeric validator agent."""

    draft_answer: DraftAnswer
    chunks: list[DocumentChunk]


class NumericValidatorAgent(AgentBase[NumericValidationInput, ValidationResult]):
    """Validates numeric claims in a draft answer against source evidence.

    The agent extracts numeric values, units, and calculations from the draft
    answer and cross-references them against the retrieved chunks.  It checks:

    * Whether numeric values are accurately quoted from the evidence.
    * Whether units are correct and consistent (UCUM-aware).
    * Whether any calculations or comparisons are mathematically accurate.
    """

    name: str = "numeric_validator"

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_engine: PromptEngine,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_engine

    async def run(
        self,
        input_data: NumericValidationInput,
        context: AgentContext,
    ) -> AgentResult[ValidationResult]:
        start = time.perf_counter()

        prompt_text = self._prompts.render(
            "numeric_check.txt",
            draft_answer=input_data.draft_answer.text,
            chunks=input_data.chunks,
        )

        llm_response = await self._llm.generate(
            LLMRequest(
                prompt=prompt_text,
                temperature=0.0,
                max_tokens=4096,
            )
        )

        # Parse the validation result.
        try:
            parsed = ResponseParser.parse_json_block(llm_response.text)
            result = ValidationResult(
                valid=bool(parsed.get("valid", False)),
                issues=list(parsed.get("issues", [])),
                corrections=list(parsed.get("corrections", [])),
            )
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Numeric validator failed to parse LLM response: %s", exc
            )
            result = ValidationResult(
                valid=False,
                issues=["Numeric validation could not be completed (parse error)"],
                corrections=[],
                details={"parse_error": str(exc)},
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Numeric validation: valid=%s, issues=%d",
            result.valid,
            len(result.issues),
        )

        return AgentResult(
            agent_name=self.name,
            output=result,
            latency_ms=round(elapsed_ms, 2),
            metadata={
                "llm_latency_ms": llm_response.latency_ms,
                "num_issues": len(result.issues),
            },
        )
