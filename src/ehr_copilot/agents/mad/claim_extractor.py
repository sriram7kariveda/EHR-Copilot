"""Claim Extractor — decomposes answers into atomic clinical claims.

Each claim is a single verifiable clinical fact (one drug, one dose,
one diagnosis, one lab value). Claims are marked as material (dangerous
if wrong) or contextual (background info).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import Enum

from pydantic import BaseModel, Field

from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)

# Threshold (in characters) beyond which we chunk the answer before
# sending it to the LLM so that the prompt + response fit comfortably.
_CHUNK_CHAR_LIMIT = 800


class ClaimType(str, Enum):
    MATERIAL = "material"      # Drug/dose/diagnosis/lab — dangerous if wrong
    CONTEXTUAL = "contextual"  # Background info — less critical


class Claim(BaseModel):
    """A single atomic clinical claim extracted from an answer."""

    claim_id: int
    text: str
    claim_type: ClaimType = ClaimType.CONTEXTUAL
    prior_confidence: float = 0.5  # Initial confidence (0.5 = uncertain)


# ---------------------------------------------------------------------------
# Regex fallback helpers
# ---------------------------------------------------------------------------

_NUMBERED_ITEM_RE = re.compile(
    r"(?:^|\n)\s*\d+[\.\)]\s+(.+?)(?=\n\s*\d+[\.\)]|\n\s*$|$)",
    re.DOTALL,
)
_BULLET_ITEM_RE = re.compile(
    r"(?:^|\n)\s*[-*•]\s+(.+?)(?=\n\s*[-*•]|\n\s*$|$)",
    re.DOTALL,
)


def _regex_split_claims(answer_text: str) -> list[Claim]:
    """Last-resort fallback: use regex to split numbered / bulleted lists
    into individual claims so we never collapse a 12-item list into 1."""
    items: list[str] = []

    # Try numbered list first (most common for diagnosis lists)
    items = [m.group(1).strip() for m in _NUMBERED_ITEM_RE.finditer(answer_text)]

    # Fall back to bullet points
    if len(items) <= 1:
        items = [m.group(1).strip() for m in _BULLET_ITEM_RE.finditer(answer_text)]

    if len(items) <= 1:
        # Cannot split further — return the whole answer as one claim
        return [Claim(
            claim_id=1,
            text=answer_text[:500],
            claim_type=ClaimType.MATERIAL,
            prior_confidence=0.5,
        )]

    claims: list[Claim] = []
    for idx, item_text in enumerate(items, start=1):
        if not item_text:
            continue
        claims.append(Claim(
            claim_id=idx,
            text=item_text,
            claim_type=ClaimType.MATERIAL,
            prior_confidence=0.5,
        ))
    return claims


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_answer(answer_text: str, limit: int = _CHUNK_CHAR_LIMIT) -> list[str]:
    """Split a long answer into segments, preferring to break at sentence
    or list-item boundaries so that each chunk can be processed independently."""
    if len(answer_text) <= limit:
        return [answer_text]

    # Split on numbered/bulleted list items first — these are natural
    # boundaries in clinical answers.
    segments: list[str] = re.split(r"(?=\n\s*\d+[\.\)]|\n\s*[-*•])", answer_text)
    segments = [s.strip() for s in segments if s.strip()]

    # If splitting produced only one segment, fall back to sentence splitting
    if len(segments) <= 1:
        segments = re.split(r"(?<=[.;])\s+", answer_text)
        segments = [s.strip() for s in segments if s.strip()]

    # Merge small segments back together until they approach *limit*
    chunks: list[str] = []
    current = ""
    for seg in segments:
        if current and len(current) + len(seg) + 1 > limit:
            chunks.append(current)
            current = seg
        else:
            current = f"{current} {seg}".strip() if current else seg
    if current:
        chunks.append(current)

    return chunks if chunks else [answer_text]


class ClaimExtractor:
    """Extracts atomic clinical claims from a draft answer."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def extract(self, answer_text: str) -> list[Claim]:
        chunks = _chunk_answer(answer_text)

        if len(chunks) == 1:
            claims = await self._extract_chunk(chunks[0])
        else:
            logger.info(
                "Answer is %d chars — split into %d chunks for claim extraction.",
                len(answer_text),
                len(chunks),
            )
            # Process chunks concurrently
            chunk_results = await asyncio.gather(
                *(self._extract_chunk(c) for c in chunks)
            )
            claims = [claim for batch in chunk_results for claim in batch]

        # ----- Fallback: if the LLM collapsed a long answer into very
        # few claims, use regex to break numbered/bulleted lists apart.
        if len(claims) <= 1 and len(answer_text) > _CHUNK_CHAR_LIMIT:
            regex_claims = _regex_split_claims(answer_text)
            if len(regex_claims) > len(claims):
                logger.warning(
                    "LLM returned only %d claim(s) for a %d-char answer. "
                    "Falling back to regex splitting (%d claims).",
                    len(claims),
                    len(answer_text),
                    len(regex_claims),
                )
                claims = regex_claims

        # Re-number claim IDs sequentially
        for idx, claim in enumerate(claims, start=1):
            claim.claim_id = idx

        return claims

    # ------------------------------------------------------------------
    # Single-chunk extraction
    # ------------------------------------------------------------------

    async def _extract_chunk(self, text: str) -> list[Claim]:
        prompt = f"""You are a clinical claim extractor. Decompose the following medical answer into atomic claims. Each claim should be a single verifiable clinical fact.

IMPORTANT — granularity rules:
- Each diagnosis is a SEPARATE claim.
- Each medication is a SEPARATE claim.
- Each lab value / vital sign is a SEPARATE claim.
- Each procedure or recommendation is a SEPARATE claim.
- NEVER group multiple diagnoses, medications, or lab values into one claim.
  For example, if the text lists 12 diagnoses, you MUST return at least 12 claims.

Answer:
{text}

For each claim, determine if it is:
- MATERIAL: Drug names, dosages, diagnoses, lab values, procedures, clinical recommendations — dangerous if wrong
- CONTEXTUAL: Background info, general descriptions, temporal context — less critical if slightly off

Respond in JSON array format:
[
  {{"claim_id": 1, "text": "the specific claim text", "type": "material", "confidence": 0.5}},
  {{"claim_id": 2, "text": "another claim", "type": "contextual", "confidence": 0.5}}
]

Rules:
- One fact per claim (not "patient takes metformin and lisinopril" — split into two claims)
- If the answer contains a numbered list, EACH numbered item must become its own claim
- Include specific values: doses, dates, lab values, diagnosis codes
- Mark drug names, doses, diagnoses, lab values as MATERIAL
- Mark general descriptions, explanations as CONTEXTUAL
- Set initial confidence to 0.5 for all claims (will be updated by verification)

Return ONLY the JSON array:"""

        response = await self._llm.generate(
            LLMRequest(prompt=prompt, temperature=0.1, max_tokens=4096)
        )

        try:
            parsed = ResponseParser.parse_json_block(response.text)
            if not isinstance(parsed, list):
                parsed = [parsed]

            claims: list[Claim] = []
            for item in parsed:
                claims.append(Claim(
                    claim_id=int(item.get("claim_id", len(claims) + 1)),
                    text=str(item.get("text", "")),
                    claim_type=ClaimType.MATERIAL if item.get("type", "").lower() == "material" else ClaimType.CONTEXTUAL,
                    prior_confidence=float(item.get("confidence", 0.5)),
                ))
            return claims

        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse claims from chunk: %s. Using regex fallback.", exc)
            return _regex_split_claims(text)
