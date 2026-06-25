"""Debate Engine — orchestrates the Multi-Agent Debate (MAD) pipeline.

Runs 2 cycles of adversarial debate between Verifier and Challenger,
then the Blind Judge scores claims. Returns a CriticOutput compatible
with the existing pipeline interface.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from ehr_copilot.agents.base import AgentBase, AgentContext, AgentResult
from ehr_copilot.agents.critic import CriticInput, CriticOutput
from ehr_copilot.agents.mad.claim_extractor import ClaimExtractor, ClaimType
from ehr_copilot.agents.mad.challenger import Challenger
from ehr_copilot.agents.mad.judge import Judge, JudgeResult
from ehr_copilot.agents.mad.storage import DebateStorage
from ehr_copilot.agents.mad.verifier import Verifier, VerificationVerdict
from ehr_copilot.domain.answer import CriticVerdict

logger = logging.getLogger(__name__)

MAX_CYCLES = 2
EARLY_EXIT_THRESHOLD = 0.95  # Exit early ONLY if all material claims very high confidence


class DebateResult(BaseModel):
    """Full debate result with transcript and scores."""

    critic_output: CriticOutput
    judge_result: JudgeResult
    num_claims: int = 0
    num_challenges: int = 0
    num_cycles: int = 0
    debate_latency_ms: float = 0.0


class DebateEngine(AgentBase[CriticInput, CriticOutput]):
    """Orchestrates the Multi-Agent Debate for hallucination detection.

    Replaces the single Critic agent with a structured 3-agent debate.
    Returns the same CriticOutput interface so the pipeline doesn't change.
    """

    name: str = "debate_engine"

    def __init__(
        self,
        claim_extractor: ClaimExtractor,
        verifier: Verifier,
        challenger: Challenger,
        judge: Judge,
        storage: DebateStorage | None = None,
    ) -> None:
        self._claim_extractor = claim_extractor
        self._verifier = verifier
        self._challenger = challenger
        self._judge = judge
        self._storage = storage

    async def run(
        self,
        input_data: CriticInput,
        context: AgentContext,
    ) -> AgentResult[CriticOutput]:
        start = time.perf_counter()

        # Step 1: Extract claims from draft answer
        claims = await self._claim_extractor.extract(input_data.draft_answer.text)
        logger.info("Extracted %d claims (%d material)",
                     len(claims),
                     sum(1 for c in claims if c.claim_type == ClaimType.MATERIAL))

        # Step 2: Initial verification (Agent A)
        verified = await self._verifier.verify_claims(claims, input_data.chunks)

        total_challenges = 0
        actual_cycles = 0

        for cycle in range(MAX_CYCLES):
            actual_cycles += 1

            # Check early exit: all material claims high confidence + SUPPORTED
            material_claims = [
                v for v in verified
                if v.claim.claim_type == ClaimType.MATERIAL
            ]
            if material_claims and all(
                v.confidence >= EARLY_EXIT_THRESHOLD
                and v.verdict == VerificationVerdict.SUPPORTED
                for v in material_claims
            ):
                logger.info("Early exit at cycle %d: all material claims ≥ %.2f",
                            cycle + 1, EARLY_EXIT_THRESHOLD)
                break

            # Step 3: Challenge (Agent B)
            challenges = await self._challenger.challenge_claims(
                verified, input_data.chunks, query_text=input_data.query_text,
            )
            total_challenges += len(challenges)

            if not challenges:
                logger.info("No challenges raised in cycle %d", cycle + 1)
                break

            # Step 4: Revise verdicts (Agent A responds to challenges)
            challenge_dicts = [ch.to_dict() for ch in challenges]

            # Store confidence before revision for reward computation
            confidence_before = {v.claim.claim_id: v.confidence for v in verified}

            verified = await self._verifier.revise_verdicts(
                verified, challenge_dicts, input_data.chunks,
            )

            # Log challenges to storage
            if self._storage:
                for ch in challenges:
                    conf_after = next(
                        (v.confidence for v in verified if v.claim.claim_id == ch.claim_id),
                        confidence_before.get(ch.claim_id, 0.5),
                    )
                    self._storage.log_challenge(
                        query_id=context.query_id,
                        claim_id=ch.claim_id,
                        challenge_type=ch.challenge_type.value,
                        challenge_text=ch.challenge_text,
                        severity=ch.severity,
                        confidence_before=confidence_before.get(ch.claim_id, 0.5),
                        confidence_after=conf_after,
                    )

        # Step 5: Judge (Blind) scores all claims
        judge_result = await self._judge.score_claims(verified, input_data.chunks)

        # Step 6: Routing decision → CriticOutput
        critic_output = self._route_decision(judge_result, input_data, total_challenges)

        # Step 7: Log to storage
        if self._storage:
            self._log_to_storage(
                context, input_data, claims, verified, judge_result, critic_output,
            )

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "Debate complete: %d claims, %d challenges, %d cycles, verdict=%s, aggregate=%.2f",
            len(claims), total_challenges, actual_cycles,
            critic_output.verdict.value, judge_result.aggregate_score,
        )

        return AgentResult(
            agent_name=self.name,
            output=critic_output,
            latency_ms=round(elapsed, 2),
            metadata={
                "num_claims": len(claims),
                "num_challenges": total_challenges,
                "num_cycles": actual_cycles,
                "aggregate_score": judge_result.aggregate_score,
                "material_min": judge_result.material_min_score,
                "contextual_mean": judge_result.contextual_mean_score,
            },
        )

    def _route_decision(
        self,
        judge_result: JudgeResult,
        input_data: CriticInput,
        total_challenges: int = 0,
    ) -> CriticOutput:
        """Convert judge scores to APPROVED / REVISED / ABSTAINED."""
        # Hard block: any material claim scored 0.0
        if judge_result.material_min_score == 0.0:
            return CriticOutput(
                verdict=CriticVerdict.ABSTAINED,
                abstention_reason="Material clinical claim unsupported by evidence. "
                                  + "; ".join(judge_result.correction_signals[:3]),
                issues=judge_result.correction_signals,
            )

        # Approve if high confidence
        if judge_result.aggregate_score >= 0.9:
            return CriticOutput(
                verdict=CriticVerdict.APPROVED,
                issues=[],
            )

        # Otherwise revise
        correction_text = input_data.draft_answer.text
        if judge_result.correction_signals:
            correction_text += "\n\n[Corrections needed: " + "; ".join(
                judge_result.correction_signals[:5]
            ) + "]"
        return CriticOutput(
            verdict=CriticVerdict.REVISED,
            revised_text=correction_text,
                issues=judge_result.correction_signals,
            )

        # Zone 2, no challenges, material claims OK — approve
        return CriticOutput(
            verdict=CriticVerdict.APPROVED,
            issues=[],
        )

    def _log_to_storage(self, context, input_data, claims, verified, judge_result, critic_output):
        """Log full debate trajectory to SQLite for GRPO training."""
        # Build judge score map
        judge_scores = {v.claim_id: v.score for v in judge_result.verdicts}

        self._storage.log_query(
            query_id=context.query_id,
            query_text=input_data.query_text,
            answer_text=input_data.draft_answer.text,
            num_claims=len(claims),
            aggregate_score=judge_result.aggregate_score,
            routing_decision=critic_output.verdict.value,
        )

        for vc in verified:
            js = judge_scores.get(vc.claim.claim_id, 0.5)
            brier = DebateStorage.compute_brier_reward(vc.confidence, js)

            self._storage.log_claim(
                query_id=context.query_id,
                claim_id=vc.claim.claim_id,
                claim_text=vc.claim.text,
                claim_type=vc.claim.claim_type.value,
                stage="final",
                verdict=vc.verdict.value,
                confidence=vc.confidence,
                judge_score=js,
                brier_reward=brier,
            )

        for jv in judge_result.verdicts:
            self._storage.log_judge_verdict(
                query_id=context.query_id,
                claim_id=jv.claim_id,
                judge_score=jv.score,
                reasoning=jv.reasoning,
                correction_signal=jv.correction_signal,
            )
