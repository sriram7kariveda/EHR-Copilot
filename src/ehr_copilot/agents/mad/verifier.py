"""Verifier (Agent A) — verifies clinical claims against evidence.

For each claim, checks if the evidence supports it and assigns a
confidence score. After receiving challenges from Agent B, revises
confidence scores (but NOT claim text).
"""

from __future__ import annotations

import json
import logging
from enum import Enum

from pydantic import BaseModel, Field

from ehr_copilot.agents.mad.claim_extractor import Claim, ClaimType
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class VerificationVerdict(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    PARTIAL = "partial"


class VerifiedClaim(BaseModel):
    """A claim with verification result from Agent A."""

    claim: Claim
    verdict: VerificationVerdict = VerificationVerdict.PARTIAL
    confidence: float = 0.5
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""


class Verifier:
    """Agent A — verifies claims against source evidence."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def verify_claims(
        self,
        claims: list[Claim],
        chunks: list[DocumentChunk],
    ) -> list[VerifiedClaim]:
        """Initial verification of all claims against evidence."""
        evidence_text = "\n".join(
            f"[{i+1}] {c.display_source}: {c.text[:500]}"
            for i, c in enumerate(chunks[:15])
        )

        claims_text = "\n".join(
            f"  Claim {c.claim_id}: \"{c.text}\" (type: {c.claim_type.value})"
            for c in claims
        )

        prompt = f"""You are a STRICT clinical evidence verifier. Your job is to catch hallucinated medical claims. Be skeptical — only mark a claim as "supported" if the evidence EXPLICITLY and DIRECTLY states the same fact.

Claims to verify:
{claims_text}

Evidence:
{evidence_text}

STRICT RULES:
- "supported" ONLY if the evidence contains the EXACT same fact (same drug name, same dose, same diagnosis, same value)
- "not_supported" if the claim says something the evidence does NOT mention, OR if the claim contradicts the evidence, OR if the claim adds details not in the evidence
- "partial" if the evidence mentions the topic but with different specifics (e.g., claim says "500mg" but evidence says "250mg")
- When in doubt, choose "not_supported" — false negatives are better than false positives in medical contexts
- A claim that PARAPHRASES the evidence in a misleading way is NOT supported

For each claim, respond with:
- verdict: "supported" | "not_supported" | "partial"
- confidence: 0.0-1.0
- evidence_chunks: list of chunk numbers [1], [2] that you checked
- reasoning: brief explanation of what matches or doesn't match

Respond as JSON array:
[
  {{"claim_id": 1, "verdict": "supported", "confidence": 0.85, "evidence_chunks": [1, 3], "reasoning": "Chunk 1 explicitly states..."}},
  ...
]

Return ONLY the JSON array:"""

        response = await self._llm.generate(
            LLMRequest(prompt=prompt, temperature=0.1, max_tokens=4096)
        )

        verified = []
        try:
            parsed = ResponseParser.parse_json_block(response.text)
            if not isinstance(parsed, list):
                parsed = [parsed]

            claim_map = {c.claim_id: c for c in claims}

            for item in parsed:
                cid = int(item.get("claim_id", 0))
                claim = claim_map.get(cid)
                if not claim:
                    continue

                verdict_str = item.get("verdict", "partial").lower()
                verdict = {
                    "supported": VerificationVerdict.SUPPORTED,
                    "not_supported": VerificationVerdict.NOT_SUPPORTED,
                    "partial": VerificationVerdict.PARTIAL,
                }.get(verdict_str, VerificationVerdict.PARTIAL)

                # Map chunk indices to chunk IDs
                chunk_indices = item.get("evidence_chunks", [])
                chunk_ids = []
                for idx in chunk_indices:
                    if isinstance(idx, int) and 1 <= idx <= len(chunks):
                        chunk_ids.append(chunks[idx - 1].chunk_id)

                verified.append(VerifiedClaim(
                    claim=claim,
                    verdict=verdict,
                    confidence=min(max(float(item.get("confidence", 0.5)), 0.0), 1.0),
                    evidence_chunk_ids=chunk_ids,
                    reasoning=str(item.get("reasoning", "")),
                ))

        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse verification: %s", exc)
            for claim in claims:
                verified.append(VerifiedClaim(
                    claim=claim,
                    verdict=VerificationVerdict.PARTIAL,
                    confidence=0.5,
                    reasoning=f"Parse error: {exc}",
                ))

        # Add any claims that weren't in the response
        verified_ids = {v.claim.claim_id for v in verified}
        for claim in claims:
            if claim.claim_id not in verified_ids:
                verified.append(VerifiedClaim(
                    claim=claim,
                    verdict=VerificationVerdict.PARTIAL,
                    confidence=0.5,
                ))

        logger.info(
            "Verified %d claims: %d supported, %d not_supported, %d partial",
            len(verified),
            sum(1 for v in verified if v.verdict == VerificationVerdict.SUPPORTED),
            sum(1 for v in verified if v.verdict == VerificationVerdict.NOT_SUPPORTED),
            sum(1 for v in verified if v.verdict == VerificationVerdict.PARTIAL),
        )

        return verified

    async def revise_verdicts(
        self,
        verified_claims: list[VerifiedClaim],
        challenges: list[dict],
        chunks: list[DocumentChunk],
    ) -> list[VerifiedClaim]:
        """Revise confidence after seeing Agent B's challenges.

        IMPORTANT: Only confidence and reasoning are revised, NOT the claim text.
        """
        if not challenges:
            return verified_claims

        challenges_text = "\n".join(
            f"  Challenge for Claim {ch['claim_id']}: [{ch['challenge_type']}] {ch['challenge_text']}"
            for ch in challenges
        )

        claims_text = "\n".join(
            f"  Claim {v.claim.claim_id}: \"{v.claim.text}\" — current verdict: {v.verdict.value}, confidence: {v.confidence:.2f}"
            for v in verified_claims
        )

        prompt = f"""You are a clinical evidence verifier. You previously verified claims. Agent B has raised challenges. Revise your CONFIDENCE (not the claims) based on these challenges.

Your previous verdicts:
{claims_text}

Challenges raised:
{challenges_text}

For each challenged claim, decide:
- Should confidence go UP (challenge was weak/irrelevant)?
- Should confidence go DOWN (challenge raised a valid concern)?
- Should verdict change (supported → partial, or partial → not_supported)?

Respond as JSON array (include ALL claims, not just challenged ones):
[
  {{"claim_id": 1, "verdict": "supported", "confidence": 0.75, "reasoning": "Challenge about X was valid, reducing confidence"}},
  ...
]

Return ONLY the JSON array:"""

        response = await self._llm.generate(
            LLMRequest(prompt=prompt, temperature=0.1, max_tokens=2048)
        )

        try:
            parsed = ResponseParser.parse_json_block(response.text)
            if not isinstance(parsed, list):
                parsed = [parsed]

            revision_map = {int(item.get("claim_id", 0)): item for item in parsed}

            for vc in verified_claims:
                rev = revision_map.get(vc.claim.claim_id)
                if rev:
                    verdict_str = rev.get("verdict", vc.verdict.value).lower()
                    vc.verdict = {
                        "supported": VerificationVerdict.SUPPORTED,
                        "not_supported": VerificationVerdict.NOT_SUPPORTED,
                        "partial": VerificationVerdict.PARTIAL,
                    }.get(verdict_str, vc.verdict)
                    vc.confidence = min(max(float(rev.get("confidence", vc.confidence)), 0.0), 1.0)
                    vc.reasoning = str(rev.get("reasoning", vc.reasoning))

        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse revision: %s", exc)

        return verified_claims
