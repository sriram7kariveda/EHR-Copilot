"""1K MedHallu eval for MARL C3 v2 (separate LoRA adapters).

Loads verifier and challenger as separate PeftModels, runs MAD debate,
computes Detection F1 with 95% bootstrap CI and per-difficulty breakdown.

Also re-evaluates Single Critic, MAD base, and MAD+GRPO v3 for comparison.

Usage:
    python scripts/eval_marl_c3v2_1k.py \
        --configs single_critic,mad_base,mad_grpo,mad_marl_c3v2 \
        --count 1000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Data loading ───

def load_pairs(path: str, count: int) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs[:count]


def build_evidence_text(knowledge) -> str:
    if isinstance(knowledge, str):
        return knowledge[:2000]
    return "\n".join(f"[{i+1}] {k[:600]}" for i, k in enumerate(knowledge))


# ─── Prompts (same as training) ───

def verifier_prompt(answer: str, evidence: str) -> str:
    return f"""You are a STRICT clinical evidence verifier.
Answer: {answer[:500]}
Evidence: {evidence[:1500]}
Respond in JSON: {{"verdict": "supported"|"not_supported"|"partial", "confidence": 0.0-1.0}}
Return ONLY JSON:"""


def challenger_prompt(answer: str, evidence: str) -> str:
    return f"""You are an adversarial clinical auditor.
Answer: {answer[:500]}
Evidence: {evidence[:1500]}
If issues: [{{"challenge_type": "...", "challenge_text": "..."}}]
If no issues: []
Return ONLY JSON:"""


def single_critic_prompt(answer: str, evidence: str, query: str) -> str:
    return f"""You are a clinical answer critic. Evaluate if this answer is grounded in the evidence.
Query: {query}
Answer: {answer[:800]}
Evidence: {evidence[:1500]}
Respond in JSON: {{"verdict": "APPROVED"|"REVISED"|"ABSTAINED", "confidence": 0.0-1.0}}
Return ONLY JSON:"""


# ─── Generation ───

def generate(model, tokenizer, prompt: str, max_new_tokens=128, temperature=0.1) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01), do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ─── Detection logic ───

def is_flagged_verifier(output: str) -> bool:
    m = re.search(r'"verdict"\s*:\s*"(\w+)"', output)
    if not m:
        return True  # can't parse → flag as suspicious
    return m.group(1).lower() in ("not_supported", "partial")


def is_flagged_challenger(output: str) -> bool:
    return "challenge_type" in output and "[]" not in output


def is_flagged_critic(output: str) -> bool:
    m = re.search(r'"verdict"\s*:\s*"(\w+)"', output)
    if not m:
        return True
    return m.group(1).upper() in ("REVISED", "ABSTAINED")


# ─── Metrics ───

def compute_metrics(results: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for r in results:
        if r["is_hallucinated"] and r["is_flagged"]:
            tp += 1
        elif not r["is_hallucinated"] and r["is_flagged"]:
            fp += 1
        elif not r["is_hallucinated"] and not r["is_flagged"]:
            tn += 1
        else:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(results), 1)
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": len(results)}


def bootstrap_ci(results: list[dict], n_boot: int = 10000) -> dict:
    f1s = []
    n = len(results)
    for _ in range(n_boot):
        sample = [results[random.randint(0, n-1)] for _ in range(n)]
        f1s.append(compute_metrics(sample)["f1"])
    f1s.sort()
    return {
        "f1_mean": np.mean(f1s),
        "f1_ci_lower": f1s[int(0.025 * n_boot)],
        "f1_ci_upper": f1s[int(0.975 * n_boot)],
    }


# ─── Eval runners ───

def eval_single_critic(model, tokenizer, pairs):
    results = []
    for i, pair in enumerate(pairs):
        evidence = build_evidence_text(pair.get("knowledge", pair.get("evidence", "")))
        query = pair.get("question", "")

        for is_hall, answer_key in [(True, "hallucinated"), (False, "ground_truth")]:
            answer = pair.get(answer_key, pair.get("answer", ""))
            out = generate(model, tokenizer, single_critic_prompt(answer, evidence, query))
            results.append({"is_hallucinated": is_hall, "is_flagged": is_flagged_critic(out),
                            "difficulty": pair.get("difficulty", "unknown")})

        if (i + 1) % 100 == 0:
            m = compute_metrics(results)
            logger.info("  single_critic [%d/%d] F1=%.3f P=%.3f R=%.3f", i+1, len(pairs), m["f1"], m["precision"], m["recall"])
    return results


def eval_mad_separate(verifier_model, challenger_model, tokenizer, pairs, config_name="mad"):
    """Eval with separate verifier/challenger models (MARL C3 v2 or GRPO v3)."""
    results = []
    for i, pair in enumerate(pairs):
        evidence = build_evidence_text(pair.get("knowledge", pair.get("evidence", "")))

        for is_hall, answer_key in [(True, "hallucinated"), (False, "ground_truth")]:
            answer = pair.get(answer_key, pair.get("answer", ""))

            v_out = generate(verifier_model, tokenizer, verifier_prompt(answer, evidence))
            c_out = generate(challenger_model, tokenizer, challenger_prompt(answer, evidence))

            flagged = is_flagged_verifier(v_out) or is_flagged_challenger(c_out)
            results.append({"is_hallucinated": is_hall, "is_flagged": flagged,
                            "difficulty": pair.get("difficulty", "unknown"),
                            "v_out": v_out[:200], "c_out": c_out[:200]})

        if (i + 1) % 100 == 0:
            m = compute_metrics(results)
            logger.info("  %s [%d/%d] F1=%.3f P=%.3f R=%.3f", config_name, i+1, len(pairs), m["f1"], m["precision"], m["recall"])
    return results


def eval_mad_base(model, tokenizer, pairs):
    """Base model runs both verifier and challenger (no LoRA)."""
    return eval_mad_separate(model, model, tokenizer, pairs, "mad_base")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--grpo_verifier", default="models/verifier-grpo-v3")
    parser.add_argument("--grpo_challenger", default="models/challenger-grpo-v3")
    parser.add_argument("--marl_verifier", default="models/marl-c3-v2/best/verifier")
    parser.add_argument("--marl_challenger", default="models/marl-c3-v2/best/challenger")
    parser.add_argument("--data_path", default="data/medhallu_eval_2k.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--configs", default="single_critic,mad_base,mad_grpo,mad_marl_c3v2")
    parser.add_argument("--output", default="results/eval_marl_c3v2_1k.json")
    args = parser.parse_args()

    configs = args.configs.split(",")
    random.seed(42)

    # Load data
    pairs = load_pairs(args.data_path, args.count)
    logger.info("Loaded %d pairs", len(pairs))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # ─── Single Critic ───
    if "single_critic" in configs:
        logger.info("=" * 60)
        logger.info("Evaluating: single_critic (base model)")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        base_model.eval()

        t0 = time.time()
        results = eval_single_critic(base_model, tokenizer, pairs)
        elapsed = time.time() - t0
        metrics = compute_metrics(results)
        ci = bootstrap_ci(results)

        by_diff = defaultdict(list)
        for r in results:
            by_diff[r["difficulty"]].append(r)

        all_results["single_critic"] = {
            "metrics": metrics, "bootstrap_ci": ci,
            "by_difficulty": {d: compute_metrics(rs) for d, rs in by_diff.items()},
            "elapsed_s": elapsed,
        }
        logger.info("  single_critic: F1=%.3f [%.3f, %.3f] P=%.3f R=%.3f (%.0fs)",
                     metrics["f1"], ci["f1_ci_lower"], ci["f1_ci_upper"],
                     metrics["precision"], metrics["recall"], elapsed)

        # Keep base model for MAD base
        if "mad_base" not in configs:
            del base_model
            torch.cuda.empty_cache()

    # ─── MAD Base ───
    if "mad_base" in configs:
        logger.info("=" * 60)
        logger.info("Evaluating: mad_base (base model, no LoRA)")
        if "base_model" not in dir() or base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            base_model.eval()

        t0 = time.time()
        results = eval_mad_base(base_model, tokenizer, pairs)
        elapsed = time.time() - t0
        metrics = compute_metrics(results)
        ci = bootstrap_ci(results)

        by_diff = defaultdict(list)
        for r in results:
            by_diff[r["difficulty"]].append(r)

        all_results["mad_base"] = {
            "metrics": metrics, "bootstrap_ci": ci,
            "by_difficulty": {d: compute_metrics(rs) for d, rs in by_diff.items()},
            "elapsed_s": elapsed,
        }
        logger.info("  mad_base: F1=%.3f [%.3f, %.3f] P=%.3f R=%.3f (%.0fs)",
                     metrics["f1"], ci["f1_ci_lower"], ci["f1_ci_upper"],
                     metrics["precision"], metrics["recall"], elapsed)

        del base_model
        torch.cuda.empty_cache()

    # ─── MAD + GRPO v3 (separate adapters) ───
    if "mad_grpo" in configs:
        logger.info("=" * 60)
        logger.info("Evaluating: mad_grpo (GRPO v3 separate adapters)")

        grpo_v_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        grpo_v = PeftModel.from_pretrained(grpo_v_base, args.grpo_verifier)
        grpo_v = grpo_v.merge_and_unload()
        grpo_v.eval()

        grpo_c_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        grpo_c = PeftModel.from_pretrained(grpo_c_base, args.grpo_challenger)
        grpo_c = grpo_c.merge_and_unload()
        grpo_c.eval()

        logger.info("  GRPO models loaded (%.1f GB)", torch.cuda.memory_allocated() / 1024**3)

        t0 = time.time()
        results = eval_mad_separate(grpo_v, grpo_c, tokenizer, pairs, "mad_grpo")
        elapsed = time.time() - t0
        metrics = compute_metrics(results)
        ci = bootstrap_ci(results)

        by_diff = defaultdict(list)
        for r in results:
            by_diff[r["difficulty"]].append(r)

        all_results["mad_grpo"] = {
            "metrics": metrics, "bootstrap_ci": ci,
            "by_difficulty": {d: compute_metrics(rs) for d, rs in by_diff.items()},
            "elapsed_s": elapsed,
        }
        logger.info("  mad_grpo: F1=%.3f [%.3f, %.3f] P=%.3f R=%.3f (%.0fs)",
                     metrics["f1"], ci["f1_ci_lower"], ci["f1_ci_upper"],
                     metrics["precision"], metrics["recall"], elapsed)

        del grpo_v, grpo_c, grpo_v_base, grpo_c_base
        torch.cuda.empty_cache()

    # ─── MAD + MARL C3 v2 (separate adapters, warm-started from GRPO v3) ───
    if "mad_marl_c3v2" in configs:
        logger.info("=" * 60)
        logger.info("Evaluating: mad_marl_c3v2 (MARL C3 v2 best checkpoint)")

        marl_v_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        marl_v = PeftModel.from_pretrained(marl_v_base, args.marl_verifier)
        marl_v = marl_v.merge_and_unload()
        marl_v.eval()

        marl_c_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        marl_c = PeftModel.from_pretrained(marl_c_base, args.marl_challenger)
        marl_c = marl_c.merge_and_unload()
        marl_c.eval()

        logger.info("  MARL C3 v2 models loaded (%.1f GB)", torch.cuda.memory_allocated() / 1024**3)

        t0 = time.time()
        results = eval_mad_separate(marl_v, marl_c, tokenizer, pairs, "mad_marl_c3v2")
        elapsed = time.time() - t0
        metrics = compute_metrics(results)
        ci = bootstrap_ci(results)

        by_diff = defaultdict(list)
        for r in results:
            by_diff[r["difficulty"]].append(r)

        all_results["mad_marl_c3v2"] = {
            "metrics": metrics, "bootstrap_ci": ci,
            "by_difficulty": {d: compute_metrics(rs) for d, rs in by_diff.items()},
            "elapsed_s": elapsed,
        }
        logger.info("  mad_marl_c3v2: F1=%.3f [%.3f, %.3f] P=%.3f R=%.3f (%.0fs)",
                     metrics["f1"], ci["f1_ci_lower"], ci["f1_ci_upper"],
                     metrics["precision"], metrics["recall"], elapsed)

        del marl_v, marl_c, marl_v_base, marl_c_base
        torch.cuda.empty_cache()

    # ─── Print final comparison table ───
    print("\n" + "=" * 90)
    print("HALLUCINATION DETECTION RESULTS (MedHallu, %d pairs)" % len(pairs))
    print("=" * 90)
    print(f"{'Config':<20} {'F1':>8} {'95% CI':>18} {'Precision':>10} {'Recall':>8} {'Accuracy':>10} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("-" * 90)
    for cfg in configs:
        r = all_results.get(cfg, {})
        m = r.get("metrics", {})
        ci = r.get("bootstrap_ci", {})
        print(f"{cfg:<20} {m.get('f1',0):>8.3f} [{ci.get('f1_ci_lower',0):.3f}, {ci.get('f1_ci_upper',0):.3f}] "
              f"{m.get('precision',0):>10.3f} {m.get('recall',0):>8.3f} {m.get('accuracy',0):>10.3f} "
              f"{m.get('tp',0):>5} {m.get('fp',0):>5} {m.get('tn',0):>5} {m.get('fn',0):>5}")

    print("\nPer-Difficulty F1:")
    for cfg in configs:
        r = all_results.get(cfg, {})
        by_d = r.get("by_difficulty", {})
        parts = [f"{d}: {by_d[d]['f1']:.3f}" for d in sorted(by_d.keys())]
        print(f"  {cfg}: {', '.join(parts)}")
    print("=" * 90)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip v_out/c_out from saved results (too large)
    save_results = {}
    for cfg, data in all_results.items():
        save_results[cfg] = {k: v for k, v in data.items()}

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
