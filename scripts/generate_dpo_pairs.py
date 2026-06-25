#!/usr/bin/env python3
"""Generate DPO preference pairs for critic agent training.

Combines two data sources:
  1. Local EHR eval results (70 pairs from our 10-patient evaluation)
  2. MedHallu dataset (500 pairs from HuggingFace — real medical hallucinations)

Total: ~570 preference pairs for DPO training.

Output: JSONL file compatible with HuggingFace TRL DPOTrainer.

Usage:
    python scripts/generate_dpo_pairs.py [--medhallu_count 500] [--skip_local]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RESULTS_PATH = Path("results/eval_results_10patients_merged.json")
GT_EVAL_PATH = Path("results/ground_truth_eval.json")
OUTPUT_PATH = Path("data/dpo_pairs.jsonl")

# Critic prompt template (same as production critic.txt)
CRITIC_PROMPT_TEMPLATE = """You are a clinical answer critic. Your job is to ensure the answer is grounded in source evidence and clinically useful.

Query: {query}
Draft Answer: {draft_answer}

Evidence chunks:
{evidence_text}

Evaluate:
1. Are the KEY clinical claims in the answer supported by the evidence?
2. Are there any clearly hallucinated facts not found in the evidence?
3. Are any issues critical enough to make the answer dangerous or misleading?

Guidelines:
- APPROVE if the answer is mostly accurate and clinically useful.
- REVISE if the answer has fixable inaccuracies — provide a corrected version.
- ABSTAIN only if evidence is clearly insufficient or the answer contains dangerous misinformation.
- Prefer REVISE over ABSTAIN when possible.

Respond in JSON:
{{
  "verdict": "APPROVED" or "REVISED" or "ABSTAINED",
  "issues": ["list of issues found"],
  "revised_text": "corrected answer text if REVISED, null otherwise",
  "abstention_reason": "reason if ABSTAINED, null otherwise"
}}"""


def build_critic_prompt(query: str, draft_answer: str, evidence_text: str) -> str:
    return CRITIC_PROMPT_TEMPLATE.format(
        query=query,
        draft_answer=draft_answer,
        evidence_text=evidence_text,
    )


# ---------------------------------------------------------------------------
# Source 1: Local EHR evaluation results (same as before)
# ---------------------------------------------------------------------------

def build_evidence_text(result: dict) -> str:
    """Extract evidence text from citation spans."""
    evidence_parts = []
    citations = result.get("citations", [])
    if not citations and "evidence_pack" in result:
        citations = result["evidence_pack"].get("citations", [])

    for i, cit in enumerate(citations, 1):
        for span in cit.get("evidence_spans", []):
            text = span.get("text", "")
            if text:
                evidence_parts.append(f"[{i}] {text[:800]}")

    if not evidence_parts and "evidence_pack" in result:
        chunks = result["evidence_pack"].get("source_chunks", {})
        for chunk_id, chunk_data in chunks.items():
            text = chunk_data if isinstance(chunk_data, str) else chunk_data.get("text", "")
            if text:
                evidence_parts.append(f"[{len(evidence_parts)+1}] {text[:800]}")

    return "\n".join(evidence_parts) if evidence_parts else "No evidence available."


def generate_local_pairs() -> list[dict]:
    """Generate pairs from our local EHR evaluation results."""
    if not RESULTS_PATH.exists() or not GT_EVAL_PATH.exists():
        print("  [SKIP] Local eval results not found")
        return []

    with open(RESULTS_PATH) as f:
        eval_data = json.load(f)
    with open(GT_EVAL_PATH) as f:
        gt_data = json.load(f)

    gt_scores = {}
    for r in gt_data["per_result"]:
        if "RAG" in r["model"]:
            key = (r["patient_id"], r["query_index"])
            gt_scores[key] = r

    pairs = []

    for pr in eval_data["patient_results"]:
        pid = pr["patient_id"]
        for qi, result in enumerate(pr.get("rag_results", [])):
            if "error" in result:
                continue

            answer_text = result.get("answer_text", "")
            query = result.get("query", "")
            if not answer_text or not query:
                continue

            evidence_text = build_evidence_text(result)
            if evidence_text == "No evidence available.":
                continue

            prompt = build_critic_prompt(query, answer_text, evidence_text)
            gt = gt_scores.get((pid, qi), {})
            halluc_rate = gt.get("hallucination_rate", 0)
            ef1 = gt.get("entity_f1", {})
            f1 = ef1.get("f1", 0)

            # Good answer → APPROVE vs false ABSTAIN
            if halluc_rate == 0 and f1 >= 0.5:
                pairs.append({
                    "prompt": prompt,
                    "chosen": json.dumps({
                        "verdict": "APPROVED",
                        "issues": [],
                        "revised_text": None,
                        "abstention_reason": None,
                    }, indent=2),
                    "rejected": json.dumps({
                        "verdict": "ABSTAINED",
                        "issues": ["Insufficient evidence to verify all claims."],
                        "revised_text": None,
                        "abstention_reason": "Cannot verify all clinical claims against the provided evidence.",
                    }, indent=2),
                    "source": "ehr_local",
                })

            # Hallucinated answer → REVISE vs false APPROVE
            if halluc_rate > 0.1:
                pairs.append({
                    "prompt": prompt,
                    "chosen": json.dumps({
                        "verdict": "REVISED",
                        "issues": [f"Some claims lack direct evidence support (estimated {halluc_rate:.0%} ungrounded)."],
                        "revised_text": answer_text,
                        "abstention_reason": None,
                    }, indent=2),
                    "rejected": json.dumps({
                        "verdict": "APPROVED",
                        "issues": [],
                        "revised_text": None,
                        "abstention_reason": None,
                    }, indent=2),
                    "source": "ehr_local",
                })

    return pairs


# ---------------------------------------------------------------------------
# Source 2: MedHallu dataset from HuggingFace
# ---------------------------------------------------------------------------

def download_medhallu(count: int = 500) -> list[dict]:
    """Download MedHallu dataset and convert to critic DPO pairs.

    MedHallu fields:
      - Question: medical question
      - Knowledge: list of evidence passages
      - Ground_Truth: correct answer
      - Hallucinated_Answer: hallucinated answer
      - Difficulty_Level: easy/medium/hard
      - Category_of_Hallucination: type of hallucination
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] 'datasets' package not installed. Run: pip install datasets")
        print("  Trying direct parquet download instead...")
        return download_medhallu_parquet(count)

    print(f"  Downloading MedHallu from HuggingFace (target: {count} pairs)...")

    # Load both splits
    pairs = []

    # pqa_labeled (1000 expert-labeled, higher quality)
    try:
        labeled = load_dataset("UTAustin-AIHealth/MedHallu", name="pqa_labeled", split="train")
        print(f"  Loaded pqa_labeled: {len(labeled)} rows")
        pairs.extend(_convert_medhallu_rows(labeled, count))
    except Exception as e:
        print(f"  [WARN] Could not load pqa_labeled: {e}")

    # pqa_artificial (9000 synthetic, if we need more)
    if len(pairs) < count:
        remaining = count - len(pairs)
        try:
            artificial = load_dataset("UTAustin-AIHealth/MedHallu", name="pqa_artificial", split="train")
            print(f"  Loaded pqa_artificial: {len(artificial)} rows")
            pairs.extend(_convert_medhallu_rows(artificial, remaining))
        except Exception as e:
            print(f"  [WARN] Could not load pqa_artificial: {e}")

    return pairs[:count]


def download_medhallu_parquet(count: int = 500) -> list[dict]:
    """Fallback: download MedHallu via parquet URL if datasets lib unavailable."""
    try:
        import pandas as pd
        url = "hf://datasets/UTAustin-AIHealth/MedHallu@refs/convert/parquet/pqa_labeled/train/0000.parquet"
        df = pd.read_parquet(url)
        print(f"  Downloaded parquet: {len(df)} rows")
        rows = df.to_dict("records")
        return _convert_medhallu_rows(rows, count)
    except Exception as e:
        print(f"  [ERROR] Parquet download failed: {e}")
        return []


def _convert_medhallu_rows(rows, count: int) -> list[dict]:
    """Convert MedHallu rows to critic DPO pairs.

    For each row, we create TWO types of pairs:
      1. Grounded answer → chosen=APPROVE, rejected=false REVISE
      2. Hallucinated answer → chosen=REVISE(with ground truth), rejected=false APPROVE
    """
    pairs = []
    indices = list(range(len(rows)))
    random.seed(42)
    random.shuffle(indices)

    for idx in indices:
        if len(pairs) >= count:
            break

        row = rows[idx]
        question = row["Question"] if isinstance(row, dict) else row["Question"]
        knowledge = row["Knowledge"] if isinstance(row, dict) else row["Knowledge"]
        ground_truth = row["Ground_Truth"] if isinstance(row, dict) else row["Ground_Truth"]
        hallucinated = row["Hallucinated_Answer"] if isinstance(row, dict) else row["Hallucinated_Answer"]
        difficulty = row.get("Difficulty_Level", "medium") if isinstance(row, dict) else getattr(row, "Difficulty_Level", "medium")
        category = row.get("Category_of_Hallucination", "") if isinstance(row, dict) else getattr(row, "Category_of_Hallucination", "")

        # Build evidence text from knowledge passages
        if isinstance(knowledge, list):
            evidence_text = "\n".join(f"[{i+1}] {k[:800]}" for i, k in enumerate(knowledge))
        else:
            evidence_text = f"[1] {str(knowledge)[:800]}"

        # --- Pair: Hallucinated answer → chosen=REVISE, rejected=false APPROVE ---
        prompt_halluc = build_critic_prompt(question, hallucinated, evidence_text)
        pairs.append({
            "prompt": prompt_halluc,
            "chosen": json.dumps({
                "verdict": "REVISED",
                "issues": [f"Answer contains hallucinated information ({category}). "
                           f"Key claims are not supported by the provided evidence."],
                "revised_text": ground_truth,
                "abstention_reason": None,
            }, indent=2),
            "rejected": json.dumps({
                "verdict": "APPROVED",
                "issues": [],
                "revised_text": None,
                "abstention_reason": None,
            }, indent=2),
            "source": f"medhallu_{difficulty}",
        })

        if len(pairs) >= count:
            break

        # --- Pair: Ground truth answer → chosen=APPROVE, rejected=false ABSTAIN ---
        prompt_gt = build_critic_prompt(question, ground_truth, evidence_text)
        pairs.append({
            "prompt": prompt_gt,
            "chosen": json.dumps({
                "verdict": "APPROVED",
                "issues": [],
                "revised_text": None,
                "abstention_reason": None,
            }, indent=2),
            "rejected": json.dumps({
                "verdict": "ABSTAINED",
                "issues": ["Insufficient evidence to verify all claims."],
                "revised_text": None,
                "abstention_reason": "Cannot verify claims against provided evidence.",
            }, indent=2),
            "source": f"medhallu_{difficulty}",
        })

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medhallu_count", type=int, default=500,
                        help="Number of pairs to generate from MedHallu")
    parser.add_argument("--skip_local", action="store_true",
                        help="Skip local EHR evaluation pairs")
    parser.add_argument("--skip_medhallu", action="store_true",
                        help="Skip MedHallu download")
    args = parser.parse_args()

    print("=" * 60)
    print("DPO PREFERENCE PAIR GENERATOR v2")
    print("  Sources: Local EHR Eval + MedHallu (HuggingFace)")
    print("=" * 60)

    all_pairs = []

    # Source 1: Local EHR pairs
    if not args.skip_local:
        print("\n[1/2] Generating pairs from local EHR evaluation...")
        local_pairs = generate_local_pairs()
        print(f"  Generated {len(local_pairs)} local pairs")
        all_pairs.extend(local_pairs)
    else:
        print("\n[1/2] Skipping local EHR pairs")

    # Source 2: MedHallu
    if not args.skip_medhallu:
        print(f"\n[2/2] Downloading MedHallu dataset ({args.medhallu_count} pairs)...")
        medhallu_pairs = download_medhallu(args.medhallu_count)
        print(f"  Generated {len(medhallu_pairs)} MedHallu pairs")
        all_pairs.extend(medhallu_pairs)
    else:
        print("\n[2/2] Skipping MedHallu")

    # Shuffle
    random.seed(42)
    random.shuffle(all_pairs)

    # Write full output (with metadata)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # Write HF-compatible output (prompt/chosen/rejected only)
    hf_path = OUTPUT_PATH.parent / "dpo_pairs_hf.jsonl"
    with open(hf_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }) + "\n")

    # Stats
    from collections import Counter
    sources = Counter(p.get("source", "unknown") for p in all_pairs)

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(all_pairs)} preference pairs")
    print(f"{'=' * 60}")
    print(f"  By source:")
    for src, cnt in sources.most_common():
        print(f"    {src}: {cnt}")
    print(f"\n  Output files:")
    print(f"    Full (with metadata): {OUTPUT_PATH}")
    print(f"    HF-compatible:        {hf_path}")


if __name__ == "__main__":
    main()
