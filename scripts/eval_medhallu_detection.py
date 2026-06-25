"""Track 2: Hallucination Detection Eval on MedHallu dataset.

Runs the MAD debate on MedHallu pairs and measures:
- Detection F1: binary classification (hallucinated vs not)
- Precision: % of flagged answers that are actually hallucinated
- Recall: % of hallucinated answers caught
- Per-difficulty breakdown (Easy/Medium/Hard)
- Compares: single Critic vs MAD (base) vs MAD (GRPO-trained)

Usage:
    python scripts/eval_medhallu_detection.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --count 5000 \
        --configs base,grpo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def load_medhallu(count: int = 5000, seed: int = 42):
    """Load MedHallu pairs (separate from GRPO training split)."""
    from datasets import load_dataset

    logger.info("Loading MedHallu...")
    labeled = load_dataset("UTAustin-AIHealth/MedHallu", name="pqa_labeled", split="train")
    artificial = load_dataset("UTAustin-AIHealth/MedHallu", name="pqa_artificial", split="train")
    logger.info("  pqa_labeled: %d, pqa_artificial: %d", len(labeled), len(artificial))

    pairs = []
    random.seed(seed)

    # Use a different seed offset than GRPO training (seed=42) to avoid overlap
    for ds in [labeled, artificial]:
        indices = list(range(len(ds)))
        random.seed(seed + 100)  # Different from GRPO training seed
        random.shuffle(indices)

        for idx in indices:
            if len(pairs) >= count:
                break
            row = ds[idx]
            pairs.append({
                "question": row["Question"],
                "knowledge": row["Knowledge"],
                "ground_truth": row["Ground Truth"],
                "hallucinated": row["Hallucinated Answer"],
                "difficulty": row.get("Difficulty_Level", "unknown"),
                "category": row.get("Category of Hallucination", ""),
            })

    logger.info("Loaded %d MedHallu pairs", len(pairs))
    return pairs


def build_evidence_text(knowledge: list[str]) -> str:
    return "\n".join(f"[{i+1}] {k[:600]}" for i, k in enumerate(knowledge))


async def run_single_critic(llm, answer: str, evidence: str, query: str) -> dict:
    """Run old-style single Critic (baseline)."""
    from ehr_copilot.llm.base import LLMRequest

    prompt = f"""You are a clinical answer critic. Evaluate if this answer is grounded in the evidence.

Query: {query}
Answer: {answer[:800]}
Evidence: {evidence[:1500]}

Respond in JSON:
{{"verdict": "APPROVED" | "REVISED" | "ABSTAINED", "confidence": 0.0-1.0}}

Return ONLY the JSON:"""

    response = await llm.generate(LLMRequest(prompt=prompt, temperature=0.0, max_tokens=256))

    import re
    verdict = "REVISED"  # default: flag as problematic
    confidence = 0.5
    try:
        v_match = re.search(r'"verdict"\s*:\s*"(\w+)"', response.text)
        if v_match:
            verdict = v_match.group(1).upper()
        c_match = re.search(r'"confidence"\s*:\s*([\d.]+)', response.text)
        if c_match:
            confidence = float(c_match.group(1))
    except Exception:
        pass

    is_flagged = verdict in ("REVISED", "ABSTAINED")
    return {"verdict": verdict, "confidence": confidence, "is_flagged": is_flagged}


async def run_mad_debate(debate_engine, answer: str, evidence_chunks, query: str) -> dict:
    """Run MAD debate and return detection result."""
    from ehr_copilot.agents.critic import CriticInput
    from ehr_copilot.agents.base import AgentContext
    from ehr_copilot.domain.answer import DraftAnswer, CriticVerdict

    critic_input = CriticInput(
        query_text=query,
        draft_answer=DraftAnswer(
            text=answer,
            reasoning_trace="",
            source_chunk_ids=[],
            confidence=0.0,
        ),
        chunks=evidence_chunks,
    )
    context = AgentContext(session_id="eval", patient_id="eval", query_id="eval")

    result = await debate_engine.run(critic_input, context)

    is_flagged = result.output.verdict in (CriticVerdict.REVISED, CriticVerdict.ABSTAINED)
    return {
        "verdict": result.output.verdict.value,
        "aggregate_score": result.metadata.get("aggregate_score", 0.5),
        "num_claims": result.metadata.get("num_claims", 0),
        "num_challenges": result.metadata.get("num_challenges", 0),
        "is_flagged": is_flagged,
    }


def compute_detection_metrics(results: list[dict]) -> dict:
    """Compute binary classification metrics for hallucination detection."""
    tp = fp = tn = fn = 0

    for r in results:
        is_hallucinated = r["is_hallucinated"]  # ground truth
        is_flagged = r["is_flagged"]  # model prediction

        if is_hallucinated and is_flagged:
            tp += 1
        elif not is_hallucinated and is_flagged:
            fp += 1
        elif not is_hallucinated and not is_flagged:
            tn += 1
        else:
            fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": len(results),
    }


def bootstrap_ci(results: list[dict], n_bootstrap: int = 10000, ci: float = 0.95) -> dict:
    """Compute bootstrap confidence intervals for F1."""
    f1_scores = []
    n = len(results)
    for _ in range(n_bootstrap):
        sample = [results[random.randint(0, n-1)] for _ in range(n)]
        metrics = compute_detection_metrics(sample)
        f1_scores.append(metrics["f1"])

    f1_scores.sort()
    lower_idx = int((1 - ci) / 2 * n_bootstrap)
    upper_idx = int((1 + ci) / 2 * n_bootstrap)

    return {
        "f1_mean": np.mean(f1_scores),
        "f1_ci_lower": f1_scores[lower_idx],
        "f1_ci_upper": f1_scores[upper_idx],
    }


async def evaluate_config(
    config_name: str,
    pairs: list[dict],
    llm,
    debate_engine=None,
) -> list[dict]:
    """Run evaluation for one config on all pairs."""
    from ehr_copilot.domain.document import DocumentChunk, ChunkMetadata, DocumentType

    results = []

    for i, pair in enumerate(pairs):
        evidence_text = build_evidence_text(pair["knowledge"])

        # Build mock chunks for MAD debate
        chunks = []
        for j, k in enumerate(pair["knowledge"]):
            chunks.append(DocumentChunk(
                chunk_id=f"medhallu-{i}-{j}",
                text=k,
                metadata=ChunkMetadata(
                    patient_id="medhallu", document_id=f"doc-{i}",
                    document_type=DocumentType.CLINICAL_NOTE,
                ),
            ))

        # Test on hallucinated answer
        if config_name == "single_critic":
            halluc_result = await run_single_critic(llm, pair["hallucinated"], evidence_text, pair["question"])
            gt_result = await run_single_critic(llm, pair["ground_truth"], evidence_text, pair["question"])
        else:
            halluc_result = await run_mad_debate(debate_engine, pair["hallucinated"], chunks, pair["question"])
            gt_result = await run_mad_debate(debate_engine, pair["ground_truth"], chunks, pair["question"])

        results.append({
            "pair_idx": i,
            "is_hallucinated": True,
            "is_flagged": halluc_result["is_flagged"],
            "difficulty": pair.get("difficulty", "unknown"),
            **{f"detail_{k}": v for k, v in halluc_result.items()},
        })
        results.append({
            "pair_idx": i,
            "is_hallucinated": False,
            "is_flagged": gt_result["is_flagged"],
            "difficulty": pair.get("difficulty", "unknown"),
            **{f"detail_{k}": v for k, v in gt_result.items()},
        })

        if (i + 1) % 50 == 0:
            partial = compute_detection_metrics(results)
            logger.info(
                "  %s [%d/%d] F1=%.3f P=%.3f R=%.3f Acc=%.3f",
                config_name, i + 1, len(pairs),
                partial["f1"], partial["precision"], partial["recall"], partial["accuracy"],
            )

    return results


async def main():
    parser = argparse.ArgumentParser(description="MedHallu Hallucination Detection Eval")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--data_path", default="data/medhallu_eval_2k.jsonl",
                        help="Pre-generated JSONL (use instead of HuggingFace download)")
    parser.add_argument("--configs", default="single_critic,mad_base,mad_grpo",
                        help="Comma-separated: single_critic,mad_base,mad_grpo")
    parser.add_argument("--grpo_verifier", default="models/verifier-grpo-v2")
    parser.add_argument("--grpo_challenger", default="models/challenger-grpo-v2")
    parser.add_argument("--output", default="results/medhallu_detection_results.json")
    parser.add_argument("--seed", type=int, default=142)  # Different from training seed
    args = parser.parse_args()

    configs = args.configs.split(",")

    # Load data (prefer pre-generated JSONL to avoid HuggingFace download on compute nodes)
    if args.data_path and os.path.exists(args.data_path):
        logger.info("Loading from pre-generated %s", args.data_path)
        pairs = []
        with open(args.data_path) as f:
            for line in f:
                pairs.append(json.loads(line.strip()))
        pairs = pairs[:args.count]
        logger.info("Loaded %d pairs from file", len(pairs))
    else:
        pairs = load_medhallu(count=args.count, seed=args.seed)

    # Load model
    from ehr_copilot.llm.local_client import LocalLLMClient
    logger.info("Loading LLM: %s", args.model)
    llm = LocalLLMClient(model_name=args.model)

    # Build debate engines
    from ehr_copilot.agents.mad.claim_extractor import ClaimExtractor
    from ehr_copilot.agents.mad.verifier import Verifier
    from ehr_copilot.agents.mad.challenger import Challenger
    from ehr_copilot.agents.mad.judge import Judge
    from ehr_copilot.agents.mad.debate_engine import DebateEngine

    debate_base = None
    debate_grpo = None

    if "mad_base" in configs or "mad_grpo" in configs:
        extractor = ClaimExtractor(llm)
        verifier = Verifier(llm)
        challenger = Challenger(llm)
        judge = Judge(llm)
        debate_base = DebateEngine(
            claim_extractor=extractor, verifier=verifier,
            challenger=challenger, judge=judge,
        )

    if "mad_grpo" in configs:
        # Load GRPO-trained model with merged LoRA adapters
        grpo_v_path = args.grpo_verifier
        grpo_c_path = args.grpo_challenger
        if os.path.exists(grpo_v_path) and os.path.exists(grpo_c_path):
            logger.info("Loading GRPO model with merged LoRA from: %s", grpo_v_path)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            grpo_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            if grpo_tokenizer.pad_token is None:
                grpo_tokenizer.pad_token = grpo_tokenizer.eos_token

            # Load base model
            grpo_base = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
            )
            # Merge verifier LoRA (both agents trained on same base, use verifier adapter)
            grpo_model = PeftModel.from_pretrained(grpo_base, grpo_v_path)
            grpo_model = grpo_model.merge_and_unload()
            grpo_model.eval()

            # Create a custom LLM client wrapping the merged model
            from ehr_copilot.llm.base import LLMClient, LLMRequest, LLMResponse
            import time as _time

            class MergedLLMClient(LLMClient):
                def __init__(self, model, tokenizer):
                    self._model = model
                    self._tokenizer = tokenizer
                async def generate(self, request):
                    start = _time.perf_counter()
                    messages = [{"role": "user", "content": request.prompt}]
                    try:
                        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    except TypeError:
                        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self._model.device)
                    with torch.no_grad():
                        outputs = self._model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=max(request.temperature, 0.01), do_sample=request.temperature > 0, pad_token_id=self._tokenizer.pad_token_id)
                    resp = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    return LLMResponse(text=resp, model="grpo-merged", latency_ms=(_time.perf_counter()-start)*1000)
                async def is_available(self):
                    return True

            grpo_llm = MergedLLMClient(grpo_model, grpo_tokenizer)
            grpo_verifier = Verifier(grpo_llm)
            grpo_challenger = Challenger(grpo_llm)
            debate_grpo = DebateEngine(
                claim_extractor=ClaimExtractor(grpo_llm),
                verifier=grpo_verifier,
                challenger=grpo_challenger,
                judge=Judge(grpo_llm),
            )
            logger.info("GRPO model loaded and merged successfully")
        else:
            logger.warning("GRPO models not found at %s / %s, skipping mad_grpo", grpo_v_path, grpo_c_path)
            configs = [c for c in configs if c != "mad_grpo"]

    # Run evaluation
    all_results = {}

    for config_name in configs:
        logger.info("=" * 60)
        logger.info("Evaluating: %s", config_name)
        logger.info("=" * 60)

        start = time.time()

        if config_name == "single_critic":
            results = await evaluate_config(config_name, pairs, llm)
        elif config_name == "mad_base":
            results = await evaluate_config(config_name, pairs, llm, debate_base)
        elif config_name == "mad_grpo":
            results = await evaluate_config(config_name, pairs, llm, debate_grpo)
        else:
            continue

        elapsed = time.time() - start
        metrics = compute_detection_metrics(results)
        ci = bootstrap_ci(results)

        # Per-difficulty breakdown
        by_difficulty = defaultdict(list)
        for r in results:
            by_difficulty[r["difficulty"]].append(r)
        difficulty_metrics = {
            d: compute_detection_metrics(rs) for d, rs in by_difficulty.items()
        }

        all_results[config_name] = {
            "metrics": metrics,
            "bootstrap_ci": ci,
            "by_difficulty": difficulty_metrics,
            "elapsed_s": elapsed,
            "num_pairs": len(pairs),
        }

        logger.info("  %s: F1=%.3f [%.3f, %.3f] P=%.3f R=%.3f Acc=%.3f (%.0fs)",
                     config_name, metrics["f1"], ci["f1_ci_lower"], ci["f1_ci_upper"],
                     metrics["precision"], metrics["recall"], metrics["accuracy"], elapsed)

    # Print comparison table
    print("\n" + "=" * 80)
    print("HALLUCINATION DETECTION RESULTS (MedHallu, %d pairs)" % len(pairs))
    print("=" * 80)
    print(f"{'Config':<20} {'F1':>8} {'95% CI':>16} {'Precision':>10} {'Recall':>8} {'Accuracy':>10}")
    print("-" * 80)
    for config_name in configs:
        r = all_results.get(config_name, {})
        m = r.get("metrics", {})
        ci = r.get("bootstrap_ci", {})
        print(f"{config_name:<20} {m.get('f1',0):>8.3f} [{ci.get('f1_ci_lower',0):.3f}, {ci.get('f1_ci_upper',0):.3f}] {m.get('precision',0):>10.3f} {m.get('recall',0):>8.3f} {m.get('accuracy',0):>10.3f}")

    # Per-difficulty
    print("\nPer-Difficulty F1:")
    for config_name in configs:
        r = all_results.get(config_name, {})
        by_d = r.get("by_difficulty", {})
        parts = [f"{d}: {by_d[d]['f1']:.3f}" for d in sorted(by_d.keys())]
        print(f"  {config_name}: {', '.join(parts)}")

    print("=" * 80)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
