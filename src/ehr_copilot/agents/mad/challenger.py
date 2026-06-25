"""Challenger (Agent B) — adversarial medical challenge of verified claims.

For every SUPPORTED claim, Agent B must attempt relevant medical challenges.
This forces comprehensive review and catches edge cases the Verifier missed.

Challenge types are domain-specific to clinical EHR:
- CONTRAINDICATION: Patient-specific contraindications
- DOSAGE_CHECK: Dose appropriateness for patient demographics
- INTERACTION: Drug-drug or drug-condition interactions
- GUIDELINE_CURRENCY: Whether recommendation matches current guidelines
- GAP_FINDING: Important information the answer omitted
"""

from __future__ import annotations

import json
import logging
from enum import Enum

from ehr_copilot.agents.mad.verifier import VerifiedClaim, VerificationVerdict
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.llm.base import LLMClient, LLMRequest
from ehr_copilot.llm.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class ChallengeType(str, Enum):
    CONTRAINDICATION = "contraindication"
    DOSAGE_CHECK = "dosage_check"
    INTERACTION = "interaction"
    GUIDELINE_CURRENCY = "guideline_currency"
    GAP_FINDING = "gap_finding"


class Challenge:
    """A challenge raised by Agent B against a claim."""

    def __init__(
        self,
        claim_id: int,
        challenge_type: ChallengeType,
        challenge_text: str,
        evidence_found: str = "",
        severity: str = "medium",
    ):
        self.claim_id = claim_id
        self.challenge_type = challenge_type
        self.challenge_text = challenge_text
        self.evidence_found = evidence_found
        self.severity = severity  # low, medium, high

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "challenge_type": self.challenge_type.value,
            "challenge_text": self.challenge_text,
            "evidence_found": self.evidence_found,
            "severity": self.severity,
        }


class Challenger:
    """Agent B — challenges SUPPORTED claims with medical queries."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def challenge_claims(
        self,
        verified_claims: list[VerifiedClaim],
        chunks: list[DocumentChunk],
        query_text: str = "",
    ) -> list[Challenge]:
        """Challenge all SUPPORTED claims with medical-specific queries."""
        # Only challenge SUPPORTED claims (True→Skeptic rule)
        supported = [v for v in verified_claims if v.verdict == VerificationVerdict.SUPPORTED]

        if not supported:
            logger.info("No SUPPORTED claims to challenge")
            return []

        evidence_text = "\n".join(
            f"[{i+1}] {c.display_source}: {c.text[:400]}"
            for i, c in enumerate(chunks[:15])
        )

        claims_text = "\n".join(
            f"  Claim {v.claim.claim_id} ({v.claim.claim_type.value}): \"{v.claim.text}\" — confidence: {v.confidence:.2f}"
            for v in supported
        )

        prompt = f"""You are an adversarial clinical auditor. Agent A has verified these claims as SUPPORTED. Your job is to CHALLENGE them by finding potential issues.

Patient query: {query_text}

SUPPORTED claims to challenge:
{claims_text}

Evidence available:
{evidence_text}

For each claim, attempt these challenge types (skip if not applicable):

1. CONTRAINDICATION: "Given the patient's conditions/history, are there contraindications?"
2. DOSAGE_CHECK: "Is the dosage/frequency appropriate for this specific patient (age, weight, renal function)?"
3. INTERACTION: "Are there drug-drug or drug-condition interactions not mentioned?"
4. GUIDELINE_CURRENCY: "Does this align with current clinical guidelines, or could it be outdated?"
5. GAP_FINDING: "What important clinical information is MISSING from the answer that should be mentioned?"

Rules:
- Only raise challenges where you find actual evidence or clinical reasoning to support the challenge
- Do NOT challenge just to challenge — be specific and evidence-based
- For each challenge, cite evidence chunks if applicable
- Rate severity: "high" (patient safety risk), "medium" (accuracy concern), "low" (minor omission)

Respond as JSON array:
[
  {{"claim_id": 1, "challenge_type": "contraindication", "challenge_text": "Evidence shows patient has renal impairment [3], which is a contraindication for...", "evidence_found": "chunk 3 mentions...", "severity": "high"}},
  ...
]

Return empty array [] if no valid challenges found. Return ONLY the JSON array:"""

        response = await self._llm.generate(
            LLMRequest(prompt=prompt, temperature=0.2, max_tokens=4096)
        )

        challenges = []
        try:
            parsed = ResponseParser.parse_json_block(response.text)
            if not isinstance(parsed, list):
                parsed = [parsed] if parsed else []

            for item in parsed:
                ctype_str = item.get("challenge_type", "gap_finding").lower()
                try:
                    ctype = ChallengeType(ctype_str)
                except ValueError:
                    ctype = ChallengeType.GAP_FINDING

                challenges.append(Challenge(
                    claim_id=int(item.get("claim_id", 0)),
                    challenge_type=ctype,
                    challenge_text=str(item.get("challenge_text", "")),
                    evidence_found=str(item.get("evidence_found", "")),
                    severity=str(item.get("severity", "medium")),
                ))

        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse challenges: %s", exc)

        logger.info(
            "Agent B raised %d challenges against %d SUPPORTED claims",
            len(challenges), len(supported),
        )

        return challenges
