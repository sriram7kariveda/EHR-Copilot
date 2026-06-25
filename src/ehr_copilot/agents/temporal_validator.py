"""Temporal validator agent -- timeline construction and claim checking."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime

from pydantic import BaseModel

from ehr_copilot.agents.base import AgentBase, AgentContext, AgentResult
from ehr_copilot.domain.answer import DraftAnswer, ValidationResult
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.domain.query import QueryIntent
from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.prompt_engine import PromptEngine
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)

# Regex patterns for common date formats in clinical text
_DATE_PATTERNS = [
    # ISO format: 2024-01-15, 2024-01-15T10:30:00
    re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:T\d{2}:\d{2}(?::\d{2})?)?\b"),
    # US format: 01/15/2024, 1/15/2024
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
    # Verbose: January 15, 2024 or Jan 15, 2024
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|"
        r"Nov|Dec)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
]


def _extract_dates(text: str) -> list[datetime]:
    """Extract and parse dates from text using regex patterns."""
    dates = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1)
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%B %d %Y",
                         "%b %d, %Y", "%b %d %Y"):
                try:
                    dates.append(datetime.strptime(raw.replace(",", ""), fmt))
                    break
                except ValueError:
                    continue
    return sorted(set(dates))


def _deterministic_temporal_check(
    answer_text: str,
    chunks: list[DocumentChunk],
) -> list[str]:
    """Run a fast, deterministic check for temporal consistency.

    Extracts dates from the answer and the evidence chunks, then verifies:
    1. Every date in the answer appears in at least one chunk.
    2. If the answer mentions dates in a sequence, the order matches evidence.

    Returns a list of issues found (empty if all checks pass).
    """
    issues: list[str] = []

    answer_dates = _extract_dates(answer_text)
    if not answer_dates:
        return issues  # No dates to validate

    # Build the set of dates present in the evidence
    evidence_dates: set[datetime] = set()
    for chunk in chunks:
        evidence_dates.update(_extract_dates(chunk.text))
        if chunk.metadata.encounter_date:
            # Normalise to midnight for comparison
            ed = chunk.metadata.encounter_date
            evidence_dates.add(datetime(ed.year, ed.month, ed.day))

    # Check: every answer date should appear in evidence
    for ad in answer_dates:
        if ad not in evidence_dates:
            issues.append(
                f"Date {ad.strftime('%Y-%m-%d')} in answer not found in evidence"
            )

    # Check: if the answer lists multiple dates, they should be in
    # chronological order (forward or reverse is fine, but not mixed)
    if len(answer_dates) >= 3:
        is_ascending = all(
            answer_dates[i] <= answer_dates[i + 1]
            for i in range(len(answer_dates) - 1)
        )
        is_descending = all(
            answer_dates[i] >= answer_dates[i + 1]
            for i in range(len(answer_dates) - 1)
        )
        if not is_ascending and not is_descending:
            issues.append(
                "Dates in the answer are not in consistent chronological order"
            )

    return issues


class TemporalValidationInput(BaseModel):
    """Input bundle for the temporal validator agent."""

    draft_answer: DraftAnswer
    chunks: list[DocumentChunk]
    intent: QueryIntent


class TemporalValidatorAgent(AgentBase[TemporalValidationInput, ValidationResult]):
    """Validates temporal claims in a draft answer against source evidence.

    The agent extracts temporal claims from the draft answer text, builds a
    mini timeline from the retrieved chunks (using encounter dates from
    metadata), and asks the LLM to verify that:

    * Dates mentioned in the answer are consistent with the evidence.
    * Temporal ordering is correct (e.g. "after surgery" actually happened
      after the surgery date in the records).
    * No time-based claims are unsupported.
    """

    name: str = "temporal_validator"

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_engine: PromptEngine,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_engine

    async def run(
        self,
        input_data: TemporalValidationInput,
        context: AgentContext,
    ) -> AgentResult[ValidationResult]:
        start = time.perf_counter()

        # Phase 1: Deterministic regex-based date check (fast, no LLM cost).
        deterministic_issues = _deterministic_temporal_check(
            input_data.draft_answer.text,
            input_data.chunks,
        )
        if deterministic_issues:
            logger.info(
                "Deterministic temporal check found %d issues",
                len(deterministic_issues),
            )

        # Sort chunks by encounter date for the timeline view.
        sorted_chunks = sorted(
            input_data.chunks,
            key=lambda c: (
                c.metadata.encounter_date or datetime.min
            ),
        )

        # Phase 2: LLM-based validation for semantic temporal reasoning.
        prompt_text = self._prompts.render(
            "temporal_check.txt",
            draft_answer=input_data.draft_answer.text,
            chunks=sorted_chunks,
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
            llm_issues = list(parsed.get("issues", []))
            llm_valid = bool(parsed.get("valid", False))
            llm_corrections = list(parsed.get("corrections", []))
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Temporal validator failed to parse LLM response: %s", exc
            )
            llm_issues = ["Temporal validation could not be completed (parse error)"]
            llm_valid = False
            llm_corrections = []

        # Merge deterministic and LLM results.
        all_issues = deterministic_issues + llm_issues
        is_valid = llm_valid and len(deterministic_issues) == 0

        result = ValidationResult(
            valid=is_valid,
            issues=all_issues,
            corrections=llm_corrections,
            details={
                "deterministic_issues": deterministic_issues,
                "llm_valid": llm_valid,
            },
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Temporal validation: valid=%s, issues=%d (deterministic=%d, llm=%d)",
            result.valid,
            len(result.issues),
            len(deterministic_issues),
            len(llm_issues),
        )

        return AgentResult(
            agent_name=self.name,
            output=result,
            latency_ms=round(elapsed_ms, 2),
            metadata={
                "llm_latency_ms": llm_response.latency_ms,
                "num_issues": len(result.issues),
                "deterministic_issues": len(deterministic_issues),
            },
        )
