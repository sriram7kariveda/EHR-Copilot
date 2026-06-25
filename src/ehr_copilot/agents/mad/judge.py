"""Judge (Blind) — independently scores claims without seeing confidence.

The Judge is deliberately "blind" to Agent A's confidence scores to
prevent anchoring bias. It sees only the claims, final verdicts, and
evidence — then scores each claim independently.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from ehr_copilot.agents.mad.claim_extractor import Claim, ClaimType
from ehr_copilot.agents.mad.verifier import VerifiedClaim
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class JudgeVerdict(BaseModel):
    """Judge's independent score for a single claim."""

    claim_id: int
    score: float  # 1.0 = fully supported, 0.5 = partial, 0.0 = unsupported
    reasoning: str = ""
    correction_signal: str = ""  # What needs to be fixed (for REVISED)


class JudgeResult(BaseModel):
    """Complete judge output for all claims."""

    verdicts: list[JudgeVerdict] = Field(default_factory=list)
    aggregate_score: float = 0.0
    material_min_score: float = 0.0
    contextual_mean_score: float = 0.0
    correction_signals: list[str] = Field(default_factory=list)


class Judge:
    """Blind Judge — scores claims without seeing confidence."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def score_claims(
        self,
        verified_claims: list[VerifiedClaim],
        chunks: list[DocumentChunk],
    ) -> JudgeResult:
        """Score each claim independently. Blind to confidence scores."""
        evidence_text = "\n".join(
            f"[{i+1}] {c.display_source}: {c.text[:400]}"
            for i, c in enumerate(chunks[:15])
        )

        # Deliberately exclude confidence scores (blind judge)
        claims_text = "\n".join(
            f"  Claim {v.claim.claim_id} ({v.claim.claim_type.value}): \"{v.claim.text}\" — verdict: {v.verdict.value}"
            for v in verified_claims
        )

        prompt = f"""You are a STRICT independent clinical judge. Your job is to catch hallucinated medical information. Score each claim based ONLY on the evidence provided. You do NOT know how confident the verifier was.

Claims and their verification status:
{claims_text}

Evidence:
{evidence_text}

SCORING RULES (be strict — patient safety depends on this):
- score 1.0: The evidence EXPLICITLY states this exact fact (same drug, dose, diagnosis, value)
- score 0.5: The evidence discusses the topic but the claim adds details or specifics NOT in the evidence
- score 0.0: The evidence does NOT mention this fact, OR the claim contradicts the evidence, OR the claim fabricates information

IMPORTANT:
- If a claim mentions a specific drug/dose/diagnosis that is NOT in the evidence, score 0.0 even if it sounds plausible
- If a claim generalizes or paraphrases in a way that changes the medical meaning, score 0.0
- When in doubt, score 0.0 — it is safer to flag a correct claim than to approve a hallucinated one

Respond as JSON array:
[
  {{"claim_id": 1, "score": 1.0, "reasoning": "Evidence chunk [2] explicitly confirms...", "correction_signal": ""}},
  {{"claim_id": 2, "score": 0.0, "reasoning": "No evidence supports this claim", "correction_signal": "Remove claim about X, evidence shows Y instead"}},
  ...
]

Return ONLY the JSON array:"""

        response = await self._llm.generate(
            LLMRequest(prompt=prompt, temperature=0.0, max_tokens=4096)
        )

        verdicts = []
        try:
            parsed = ResponseParser.parse_json_block(response.text)
            if not isinstance(parsed, list):
                parsed = [parsed]

            for item in parsed:
                verdicts.append(JudgeVerdict(
                    claim_id=int(item.get("claim_id", 0)),
                    score=min(max(float(item.get("score", 0.5)), 0.0), 1.0),
                    reasoning=str(item.get("reasoning", "")),
                    correction_signal=str(item.get("correction_signal", "")),
                ))

        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse judge verdicts: %s", exc)
            for vc in verified_claims:
                verdicts.append(JudgeVerdict(
                    claim_id=vc.claim.claim_id,
                    score=0.5,
                    reasoning=f"Parse error: {exc}",
                ))

        # Add missing claims
        judged_ids = {v.claim_id for v in verdicts}
        for vc in verified_claims:
            if vc.claim.claim_id not in judged_ids:
                verdicts.append(JudgeVerdict(
                    claim_id=vc.claim.claim_id,
                    score=0.5,
                    reasoning="Not judged (missing from response)",
                ))

        # Compute aggregates
        claim_map = {vc.claim.claim_id: vc.claim for vc in verified_claims}
        material_scores = []
        contextual_scores = []

        for v in verdicts:
            claim = claim_map.get(v.claim_id)
            if claim and claim.claim_type == ClaimType.MATERIAL:
                material_scores.append(v.score)
            else:
                contextual_scores.append(v.score)

        material_min = min(material_scores) if material_scores else 1.0
        # If no contextual claims, default to 1.0 (don't penalize)
        contextual_mean = sum(contextual_scores) / len(contextual_scores) if contextual_scores else 1.0

        # Weighted aggregate: material claims dominate (weakest link)
        aggregate = 0.7 * material_min + 0.3 * contextual_mean

        correction_signals = [
            v.correction_signal for v in verdicts
            if v.correction_signal and v.score < 1.0
        ]

        logger.info(
            "Judge scored %d claims: material_min=%.2f, contextual_mean=%.2f, aggregate=%.2f",
            len(verdicts), material_min, contextual_mean, aggregate,
        )

        return JudgeResult(
            verdicts=verdicts,
            aggregate_score=aggregate,
            material_min_score=material_min,
            contextual_mean_score=contextual_mean,
            correction_signals=correction_signals,
        )
