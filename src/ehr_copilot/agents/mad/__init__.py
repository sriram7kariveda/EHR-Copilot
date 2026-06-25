"""Multi-Agent Debate (MAD) module for hallucination detection.

Replaces the single Critic agent with a 3-agent adversarial debate:
- Verifier (Agent A): verifies claims against evidence
- Challenger (Agent B): challenges verified claims with medical queries
- Judge (Blind): scores claims without seeing confidence scores

Reference: guardrails-enterprise MAD architecture, adapted for clinical EHR domain.
"""

from ehr_copilot.agents.mad.debate_engine import DebateEngine, DebateResult

__all__ = ["DebateEngine", "DebateResult"]
