"""SQLite trajectory storage for GRPO training data.

Logs every debate step (claims, verifications, challenges, judge verdicts)
with agent prompts and computed rewards. This data is used in Phase 3
for GRPO training of the Verifier and Challenger agents.

Reward formulas:
- Verifier (Agent A): Brier score = 2 * confidence * judge_score - confidence²
- Challenger (Agent B): +1 if challenged wrong claim, -1 if challenged correct claim
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS debate_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT,
    query_text TEXT,
    answer_text TEXT,
    num_claims INTEGER,
    aggregate_score REAL,
    routing_decision TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS debate_claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT,
    claim_id INTEGER,
    claim_text TEXT,
    claim_type TEXT,
    stage TEXT,  -- 'initial', 'post_verify', 'post_revise'
    verdict TEXT,
    confidence REAL,
    judge_score REAL,
    brier_reward REAL,
    verifier_prompt TEXT
);

CREATE TABLE IF NOT EXISTS debate_challenges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT,
    claim_id INTEGER,
    challenge_type TEXT,
    challenge_text TEXT,
    severity TEXT,
    confidence_before REAL,
    confidence_after REAL,
    challenger_reward REAL,
    challenger_prompt TEXT
);

CREATE TABLE IF NOT EXISTS debate_judge_verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT,
    claim_id INTEGER,
    judge_score REAL,
    reasoning TEXT,
    correction_signal TEXT,
    judge_prompt TEXT
);
"""


class DebateStorage:
    """SQLite storage for debate trajectories."""

    def __init__(self, db_path: str = "data/debate_trajectories.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(_SCHEMA)

    def log_query(
        self,
        query_id: str,
        query_text: str,
        answer_text: str,
        num_claims: int,
        aggregate_score: float,
        routing_decision: str,
    ):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO debate_queries (query_id, query_text, answer_text, num_claims, aggregate_score, routing_decision, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (query_id, query_text, answer_text, num_claims, aggregate_score, routing_decision, datetime.utcnow().isoformat()),
            )

    def log_claim(
        self,
        query_id: str,
        claim_id: int,
        claim_text: str,
        claim_type: str,
        stage: str,
        verdict: str,
        confidence: float,
        judge_score: float = -1.0,
        brier_reward: float = 0.0,
        verifier_prompt: str = "",
    ):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO debate_claims (query_id, claim_id, claim_text, claim_type, stage, verdict, confidence, judge_score, brier_reward, verifier_prompt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (query_id, claim_id, claim_text, claim_type, stage, verdict, confidence, judge_score, brier_reward, verifier_prompt),
            )

    def log_challenge(
        self,
        query_id: str,
        claim_id: int,
        challenge_type: str,
        challenge_text: str,
        severity: str,
        confidence_before: float,
        confidence_after: float,
        challenger_reward: float = 0.0,
        challenger_prompt: str = "",
    ):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO debate_challenges (query_id, claim_id, challenge_type, challenge_text, severity, confidence_before, confidence_after, challenger_reward, challenger_prompt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (query_id, claim_id, challenge_type, challenge_text, severity, confidence_before, confidence_after, challenger_reward, challenger_prompt),
            )

    def log_judge_verdict(
        self,
        query_id: str,
        claim_id: int,
        judge_score: float,
        reasoning: str = "",
        correction_signal: str = "",
        judge_prompt: str = "",
    ):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO debate_judge_verdicts (query_id, claim_id, judge_score, reasoning, correction_signal, judge_prompt) VALUES (?, ?, ?, ?, ?, ?)",
                (query_id, claim_id, judge_score, reasoning, correction_signal, judge_prompt),
            )

    @staticmethod
    def compute_brier_reward(confidence: float, judge_score: float) -> float:
        """Brier reward for Agent A. Rewards calibrated confidence."""
        return 2.0 * confidence * judge_score - confidence ** 2

    @staticmethod
    def compute_challenger_reward(
        judge_score: float,
        confidence_delta: float,
    ) -> float:
        """Reward for Agent B.
        +1 if challenged a wrong claim (judge_score < 1.0) and moved confidence down.
        -1 if challenged a correct claim (judge_score = 1.0) — gaslighting penalty.
        """
        if judge_score >= 1.0:
            return -1.0  # Gaslighting: challenged a correct claim
        if confidence_delta <= -0.1:
            return 1.0  # Good challenge: moved confidence down on wrong claim
        return 0.0  # Ineffective challenge
