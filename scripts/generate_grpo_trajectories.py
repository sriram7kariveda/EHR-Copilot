"""Generate GRPO training trajectories from MedHallu + MedNLI datasets.

MedHallu (10K): Run debate on hallucinated vs ground truth answers.
  - Hallucinated answer → debate should catch it (Verifier low confidence, Judge 0.0/0.5)
  - Ground truth answer → debate should approve it (Verifier high confidence, Judge 1.0)

MedNLI (14K): Direct Verifier training data.
  - Premise (evidence) + Hypothesis (claim) + Label (entailment/contradiction/neutral)
  - Maps to: evidence + claim → SUPPORTED/NOT_SUPPORTED/PARTIAL

Output: JSONL trajectories with rewards for GRPO training.

Usage:
    python scripts/generate_grpo_trajectories.py \
        --output_dir data/grpo_trajectories \
        --medhallu_count 1000 \
        --mednli_count 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def generate_medhallu_trajectories(count: int = 1000) -> list[dict]:
    """Generate debate trajectories from MedHallu dataset.

    Each MedHallu example produces TWO trajectories:
    1. Hallucinated answer → Verifier should be skeptical, Judge should score low
    2. Ground truth answer → Verifier should be confident, Judge should score high
    """
    from datasets import load_dataset

    logger.info("Loading MedHallu from HuggingFace...")
    labeled = load_dataset("UTAustin-AIHealth/MedHallu", name="pqa_labeled", split="train")
    artificial = load_dataset("UTAustin-AIHealth/MedHallu", name="pqa_artificial", split="train")
    logger.info("  pqa_labeled: %d, pqa_artificial: %d", len(labeled), len(artificial))

    trajectories = []
    random.seed(42)

    for ds in [labeled, artificial]:
        indices = list(range(len(ds)))
        random.shuffle(indices)

        for idx in indices:
            if len(trajectories) >= count * 2:
                break

            row = ds[idx]
            question = row["Question"]
            knowledge = row["Knowledge"]
            ground_truth = row["Ground Truth"]
            hallucinated = row["Hallucinated Answer"]
            category = row.get("Category of Hallucination", "")

            # Build evidence text from knowledge
            evidence = "\n".join(f"[{i+1}] {k[:600]}" for i, k in enumerate(knowledge))

            # Trajectory 1: Hallucinated answer (should be caught)
            trajectories.append({
                "type": "medhallu_hallucinated",
                "query": question,
                "answer": hallucinated,
                "evidence": evidence,
                "expected_verdict": "not_supported",
                "expected_judge_score": 0.0,
                "ground_truth": ground_truth,
                "hallucination_category": category,
                # Verifier reward: should have LOW confidence on hallucinated claims
                "verifier_target_confidence": 0.2,
                # Challenger reward: should find issues
                "challenger_should_challenge": True,
                # Judge reward: should score LOW
                "judge_target_score": 0.0,
            })

            # Trajectory 2: Ground truth answer (should be approved)
            trajectories.append({
                "type": "medhallu_ground_truth",
                "query": question,
                "answer": ground_truth,
                "evidence": evidence,
                "expected_verdict": "supported",
                "expected_judge_score": 1.0,
                "ground_truth": ground_truth,
                "hallucination_category": "",
                # Verifier reward: should have HIGH confidence
                "verifier_target_confidence": 0.9,
                # Challenger reward: should NOT find real issues
                "challenger_should_challenge": False,
                # Judge reward: should score HIGH
                "judge_target_score": 1.0,
            })

    random.shuffle(trajectories)
    logger.info("Generated %d MedHallu trajectories", len(trajectories))
    return trajectories


def generate_mednli_trajectories(count: int = 5000) -> list[dict]:
    """Generate Verifier training data from MedNLI.

    MedNLI has: premise (clinical text), hypothesis (claim), label.
    Labels map directly to Verifier verdicts:
    - entailment → SUPPORTED (confidence ~0.9)
    - contradiction → NOT_SUPPORTED (confidence ~0.1)
    - neutral → PARTIAL (confidence ~0.5)
    """
    from datasets import load_dataset

    logger.info("Loading MedNLI from HuggingFace...")
    try:
        ds = load_dataset("bigbio/mednli", name="mednli_bigbio_te", split="train")
    except Exception:
        # Try alternate loading
        try:
            ds = load_dataset("bigbio/mednli", split="train")
        except Exception as e:
            logger.warning("Could not load MedNLI: %s. Trying physionet/mednli...", e)
            try:
                ds = load_dataset("physionet/mednli", split="train")
            except Exception as e2:
                logger.warning("Could not load MedNLI from any source: %s", e2)
                return []

    logger.info("MedNLI loaded: %d examples", len(ds))

    # Map labels
    label_map = {
        "entailment": ("supported", 0.9, 1.0),
        "contradiction": ("not_supported", 0.1, 0.0),
        "neutral": ("partial", 0.5, 0.5),
        # Integer labels (some versions)
        0: ("entailment", 0.9, 1.0),
        1: ("neutral", 0.5, 0.5),
        2: ("contradiction", 0.1, 0.0),
    }

    trajectories = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices:
        if len(trajectories) >= count:
            break

        row = ds[idx]

        # Handle different column names across dataset versions
        premise = row.get("premise") or row.get("text_1") or row.get("sentence1", "")
        hypothesis = row.get("hypothesis") or row.get("text_2") or row.get("sentence2", "")
        label = row.get("label") or row.get("gold_label", "")

        if not premise or not hypothesis:
            continue

        # Convert label to string if integer
        if isinstance(label, int):
            label_info = label_map.get(label, ("partial", 0.5, 0.5))
        else:
            label_str = str(label).lower().strip()
            label_info = label_map.get(label_str, ("partial", 0.5, 0.5))

        verdict, target_conf, target_judge = label_info

        trajectories.append({
            "type": "mednli",
            "query": f"Is this claim supported by the evidence?",
            "answer": hypothesis,
            "evidence": f"[1] {premise}",
            "expected_verdict": verdict,
            "expected_judge_score": target_judge,
            "ground_truth": "",
            "hallucination_category": "",
            "verifier_target_confidence": target_conf,
            "challenger_should_challenge": verdict == "not_supported",
            "judge_target_score": target_judge,
        })

    random.shuffle(trajectories)
    logger.info("Generated %d MedNLI trajectories", len(trajectories))
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Generate GRPO training trajectories")
    parser.add_argument("--output_dir", default="data/grpo_trajectories")
    parser.add_argument("--medhallu_count", type=int, default=1000,
                        help="MedHallu examples to use (produces 2x trajectories)")
    parser.add_argument("--mednli_count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate trajectories
    medhallu = generate_medhallu_trajectories(count=args.medhallu_count)
    mednli = generate_mednli_trajectories(count=args.mednli_count)

    # Combine and split
    all_trajectories = medhallu + mednli
    random.shuffle(all_trajectories)

    split_idx = int(len(all_trajectories) * 0.9)
    train = all_trajectories[:split_idx]
    eval_set = all_trajectories[split_idx:]

    # Write
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    with open(train_path, "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")

    with open(eval_path, "w") as f:
        for t in eval_set:
            f.write(json.dumps(t) + "\n")

    # Stats
    medhallu_count = sum(1 for t in all_trajectories if t["type"].startswith("medhallu"))
    mednli_count = sum(1 for t in all_trajectories if t["type"] == "mednli")

    print(f"\n{'='*50}")
    print(f"GRPO Trajectory Generation Complete")
    print(f"{'='*50}")
    print(f"MedHallu trajectories: {medhallu_count}")
    print(f"  - Hallucinated: {sum(1 for t in all_trajectories if t['type'] == 'medhallu_hallucinated')}")
    print(f"  - Ground truth: {sum(1 for t in all_trajectories if t['type'] == 'medhallu_ground_truth')}")
    print(f"MedNLI trajectories:   {mednli_count}")
    print(f"Total:                 {len(all_trajectories)}")
    print(f"Train:                 {len(train)}")
    print(f"Eval:                  {len(eval_set)}")
    print(f"Train path:            {train_path}")
    print(f"Eval path:             {eval_path}")


if __name__ == "__main__":
    main()
