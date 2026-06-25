"""Collect full pipeline trajectories for MARL training.

Runs the pipeline on MedHallu data and logs every agent's input/output.
This generates the training data for shared-reward MARL across 4 agents:
Triage, CRAG Evaluator, Verifier, Challenger.

The shared reward = whether the full pipeline correctly classified
the answer as hallucinated or not (binary detection accuracy).

Usage:
    python scripts/collect_pipeline_trajectories.py \
        --data_path data/medhallu_eval_2k.jsonl \
        --output_path data/marl_trajectories.jsonl \
        --count 1000
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


async def collect_trajectory(
    llm_pipeline, llm_debate, pair: dict, pair_idx: int, is_hallucinated: bool,
) -> dict:
    """Run one answer through the pipeline and collect all agent outputs."""
    from ehr_copilot.llm.base import LLMRequest
    import re

    answer = pair["hallucinated"] if is_hallucinated else pair["ground_truth"]
    evidence = "\n".join(f"[{i+1}] {k[:600]}" for i, k in enumerate(pair["knowledge"]))
    query = pair["question"]

    trajectory = {
        "pair_idx": pair_idx,
        "is_hallucinated": is_hallucinated,
        "query": query,
        "answer": answer,
        "evidence": evidence,
    }

    # Agent 1: Triage — classify query type
    triage_prompt = f"""Classify this clinical query into one of: FACTUAL, TEMPORAL, NUMERIC, TEMPORAL_NUMERIC, MEDICATION, SUMMARY, COMPARISON.

Query: {query}

Respond in JSON: {{"query_type": "...", "requires_temporal": true/false, "requires_numeric": true/false}}
Return ONLY JSON:"""

    triage_resp = await llm_pipeline.generate(LLMRequest(prompt=triage_prompt, temperature=0.0, max_tokens=256))
    trajectory["triage_prompt"] = triage_prompt
    trajectory["triage_output"] = triage_resp.text

    # Agent 4: CRAG Evaluator — is evidence sufficient?
    crag_prompt = f"""Assess if this evidence is sufficient to answer the query.

Query: {query}
Evidence: {evidence[:1500]}

Respond in JSON: {{"verdict": "sufficient"|"insufficient"|"ambiguous", "coverage_score": 0.0-1.0}}
Return ONLY JSON:"""

    crag_resp = await llm_pipeline.generate(LLMRequest(prompt=crag_prompt, temperature=0.0, max_tokens=256))
    trajectory["crag_prompt"] = crag_prompt
    trajectory["crag_output"] = crag_resp.text

    # Agent 8: Verifier — verify claims
    verifier_prompt = f"""You are a STRICT clinical evidence verifier. Verify if this answer is supported by evidence.

Answer: {answer[:500]}
Evidence: {evidence[:1500]}

Respond in JSON: {{"verdict": "supported"|"not_supported"|"partial", "confidence": 0.0-1.0}}
Return ONLY JSON:"""

    verifier_resp = await llm_debate.generate(LLMRequest(prompt=verifier_prompt, temperature=0.1, max_tokens=256))
    trajectory["verifier_prompt"] = verifier_prompt
    trajectory["verifier_output"] = verifier_resp.text

    # Agent 9: Challenger — challenge claims
    challenger_prompt = f"""You are an adversarial clinical auditor. Challenge this answer if you find issues.

Answer: {answer[:500]}
Evidence: {evidence[:1500]}

If issues found, respond as JSON array: [{{"challenge_type": "...", "challenge_text": "..."}}]
If no issues, respond: []
Return ONLY JSON:"""

    challenger_resp = await llm_debate.generate(LLMRequest(prompt=challenger_prompt, temperature=0.2, max_tokens=512))
    trajectory["challenger_prompt"] = challenger_prompt
    trajectory["challenger_output"] = challenger_resp.text

    # Compute per-agent rewards based on shared outcome
    # Shared reward: did the pipeline correctly classify this answer?
    # For hallucinated answers: pipeline should flag (verifier NOT_SUPPORTED, challenger finds issues)
    # For good answers: pipeline should approve (verifier SUPPORTED, challenger no issues)

    v_correct = False
    try:
        v_match = re.search(r'"verdict"\s*:\s*"(\w+)"', verifier_resp.text)
        if v_match:
            v_verdict = v_match.group(1).lower()
            if is_hallucinated and v_verdict in ("not_supported", "partial"):
                v_correct = True
            elif not is_hallucinated and v_verdict == "supported":
                v_correct = True
    except Exception:
        pass

    c_has_challenges = "challenge_type" in challenger_resp.text and "[]" not in challenger_resp.text
    c_correct = (is_hallucinated and c_has_challenges) or (not is_hallucinated and not c_has_challenges)

    # Shared reward: both agents correct = 1.0, one correct = 0.5, both wrong = -1.0
    if v_correct and c_correct:
        shared_reward = 1.0
    elif v_correct or c_correct:
        shared_reward = 0.0
    else:
        shared_reward = -1.0

    trajectory["shared_reward"] = shared_reward
    trajectory["verifier_correct"] = v_correct
    trajectory["challenger_correct"] = c_correct

    return trajectory


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/medhallu_eval_2k.jsonl")
    parser.add_argument("--output_path", default="data/marl_trajectories.jsonl")
    parser.add_argument("--pipeline_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--debate_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    pairs = []
    with open(args.data_path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    pairs = pairs[:args.count]
    logger.info("Loaded %d pairs", len(pairs))

    # Load models
    from ehr_copilot.llm.local_client import LocalLLMClient

    logger.info("Loading pipeline model: %s", args.pipeline_model)
    llm_pipeline = LocalLLMClient(model_name=args.pipeline_model)

    logger.info("Loading debate model: %s", args.debate_model)
    llm_debate = LocalLLMClient(model_name=args.debate_model)

    # Collect trajectories
    trajectories = []
    start = time.time()

    for i, pair in enumerate(pairs):
        # Hallucinated version
        t1 = await collect_trajectory(llm_pipeline, llm_debate, pair, i, is_hallucinated=True)
        trajectories.append(t1)

        # Ground truth version
        t2 = await collect_trajectory(llm_pipeline, llm_debate, pair, i, is_hallucinated=False)
        trajectories.append(t2)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            correct = sum(1 for t in trajectories if t["shared_reward"] > 0)
            logger.info(
                "  [%d/%d] %d trajectories, shared accuracy: %.1f%%, time: %.0fs",
                i + 1, len(pairs), len(trajectories),
                100 * correct / len(trajectories), elapsed,
            )

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")

    correct = sum(1 for t in trajectories if t["shared_reward"] > 0)
    logger.info("Saved %d trajectories to %s", len(trajectories), output_path)
    logger.info("Shared accuracy: %.1f%%", 100 * correct / len(trajectories))


if __name__ == "__main__":
    asyncio.run(main())
