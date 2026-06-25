"""Full-pipeline 1K MedHallu eval for MARL C3 v2.

Uses the REAL MAD debate engine (claim extraction → verifier → challenger
challenges → verifier revises → judge scores → routing decision).

Key difference from eval_marl_c3v2_1k.py: this runs the full debate pipeline,
not raw verifier/challenger prompts. For MARL C3 v2, separate LoRA adapters
are loaded and routed to the correct agents.

Configs evaluated:
1. single_critic — base Qwen 3B, single critic prompt
2. mad_base — base Qwen 3B, full MAD debate (no LoRA)
3. mad_grpo — GRPO v3 verifier LoRA merged, full MAD debate
4. mad_marl_c3v2 — separate verifier/challenger LoRA, full MAD debate

Usage:
    python scripts/eval_marl_c3v2_full_pipeline.py \
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


# ─── Wrapped LLM client that uses a pre-loaded model ───

class PreloadedLLMClient:
    """LLM client wrapping a pre-loaded model (base or PeftModel)."""

    def __init__(self, model, tokenizer, name="preloaded"):
        self._model = model
        self._tokenizer = tokenizer
        self._name = name

    async def generate(self, request):
        from ehr_copilot.llm.base import LLMResponse
        start = time.perf_counter()

        messages = []
        if hasattr(request, 'system_prompt') and request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=max(request.temperature, 0.01),
                do_sample=request.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        response_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        elapsed = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=response_text,
            model=self._name,
            prompt_tokens=inputs["input_ids"].shape[1],
            completion_tokens=outputs.shape[1] - inputs["input_ids"].shape[1],
            latency_ms=round(elapsed, 2),
        )

    async def is_available(self):
        return True


# ─── Data ───

def load_pairs(path: str, count: int) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs[:count]


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


# ─── Single Critic ───

async def run_single_critic(llm, answer, evidence_text, query):
    from ehr_copilot.llm.base import LLMRequest
    prompt = f"""You are a clinical answer critic. Evaluate if this answer is grounded in the evidence.

Query: {query}
Answer: {answer[:800]}
Evidence: {evidence_text[:1500]}

Respond in JSON:
{{"verdict": "APPROVED" | "REVISED" | "ABSTAINED", "confidence": 0.0-1.0}}

Return ONLY the JSON:"""
    response = await llm.generate(LLMRequest(prompt=prompt, temperature=0.0, max_tokens=256))
    m = re.search(r'"verdict"\s*:\s*"(\w+)"', response.text)
    verdict = m.group(1).upper() if m else "REVISED"
    return verdict in ("REVISED", "ABSTAINED")


# ─── MAD Debate (full pipeline) ───

async def run_mad_debate(debate_engine, answer, chunks, query):
    """Run MAD debate using the full debate engine."""
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
        chunks=chunks,
    )
    context = AgentContext(session_id="eval", patient_id="eval", query_id="eval")

    result = await debate_engine.run(critic_input, context)
    is_flagged = result.output.verdict in (CriticVerdict.REVISED, CriticVerdict.ABSTAINED)

    return {
        "is_flagged": is_flagged,
        "verdict": result.output.verdict.value,
        "aggregate_score": result.metadata.get("aggregate_score", 0.5),
        "num_claims": result.metadata.get("num_claims", 0),
        "num_challenges": result.metadata.get("num_challenges", 0),
    }


# ─── Eval runner ───

async def eval_config(config_name, pairs, llm=None, debate_engine=None):
    from ehr_copilot.domain.document import DocumentChunk, ChunkMetadata, DocumentType

    results = []
    for i, pair in enumerate(pairs):
        knowledge = pair.get("knowledge", [])
        if isinstance(knowledge, str):
            evidence_text = knowledge[:2000]
            knowledge_list = [knowledge]
        else:
            evidence_text = "\n".join(f"[{j+1}] {k[:600]}" for j, k in enumerate(knowledge))
            knowledge_list = knowledge

        chunks = []
        for j, k in enumerate(knowledge_list):
            chunks.append(DocumentChunk(
                chunk_id=f"medhallu-{i}-{j}",
                text=k,
                metadata=ChunkMetadata(
                    patient_id="medhallu", document_id=f"doc-{i}",
                    document_type=DocumentType.CLINICAL_NOTE,
                ),
            ))

        query = pair.get("question", "")

        for is_hall, answer_key in [(True, "hallucinated"), (False, "ground_truth")]:
            answer = pair.get(answer_key, "")

            if config_name == "single_critic":
                flagged = await run_single_critic(llm, answer, evidence_text, query)
            else:
                r = await run_mad_debate(debate_engine, answer, chunks, query)
                flagged = r["is_flagged"]

            results.append({
                "is_hallucinated": is_hall,
                "is_flagged": flagged,
                "difficulty": pair.get("difficulty", "unknown"),
            })

        if (i + 1) % 50 == 0:
            m = compute_metrics(results)
            logger.info("  %s [%d/%d] F1=%.3f P=%.3f R=%.3f Acc=%.3f",
                        config_name, i+1, len(pairs), m["f1"], m["precision"], m["recall"], m["accuracy"])

    return results


def build_debate_engine(llm, verifier_llm=None, challenger_llm=None):
    """Build debate engine, optionally with separate LLMs for verifier/challenger."""
    from ehr_copilot.agents.mad.claim_extractor import ClaimExtractor
    from ehr_copilot.agents.mad.verifier import Verifier
    from ehr_copilot.agents.mad.challenger import Challenger
    from ehr_copilot.agents.mad.judge import Judge
    from ehr_copilot.agents.mad.debate_engine import DebateEngine

    v_llm = verifier_llm or llm
    c_llm = challenger_llm or llm

    return DebateEngine(
        claim_extractor=ClaimExtractor(llm),  # extractor uses base/shared LLM
        verifier=Verifier(v_llm),
        challenger=Challenger(c_llm),
        judge=Judge(llm),  # judge uses base/shared LLM (unbiased)
    )


def load_peft_merged(base_model_name, adapter_path, tokenizer, name="peft"):
    """Load base + LoRA adapter, merge, return PreloadedLLMClient."""
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged = merged.merge_and_unload()
    merged.eval()
    return PreloadedLLMClient(merged, tokenizer, name=name)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--grpo_verifier", default="models/verifier-grpo-v3")
    parser.add_argument("--grpo_challenger", default="models/challenger-grpo-v3")
    parser.add_argument("--marl_verifier", default="models/marl-c3-v2/best/verifier")
    parser.add_argument("--marl_challenger", default="models/marl-c3-v2/best/challenger")
    parser.add_argument("--data_path", default="data/medhallu_eval_2k.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--configs", default="single_critic,mad_base,mad_grpo,mad_marl_c3v2")
    parser.add_argument("--output", default="results/eval_full_pipeline_1k.json")
    args = parser.parse_args()

    configs = args.configs.split(",")
    random.seed(42)

    pairs = load_pairs(args.data_path, args.count)
    logger.info("Loaded %d pairs", len(pairs))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # ─── Single Critic ───
    if "single_critic" in configs:
        logger.info("=" * 60)
        logger.info("Evaluating: single_critic (base model, full pipeline)")

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        base.eval()
        base_llm = PreloadedLLMClient(base, tokenizer, "base")

        t0 = time.time()
        results = await eval_config("single_critic", pairs, llm=base_llm)
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
        del base
        torch.cuda.empty_cache()

    # ─── MAD Base (no LoRA) ───
    if "mad_base" in configs:
        logger.info("=" * 60)
        logger.info("Evaluating: mad_base (base model, full MAD debate)")

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        base.eval()
        base_llm = PreloadedLLMClient(base, tokenizer, "base")
        engine = build_debate_engine(base_llm)

        t0 = time.time()
        results = await eval_config("mad_base", pairs, debate_engine=engine)
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
        del base
        torch.cuda.empty_cache()

    # ─── MAD + GRPO v3 ───
    if "mad_grpo" in configs:
        logger.info("=" * 60)
        logger.info("Evaluating: mad_grpo (GRPO v3, separate adapters, full MAD debate)")

        # Load base for extractor + judge
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        base.eval()
        base_llm = PreloadedLLMClient(base, tokenizer, "base")

        # Load verifier and challenger with merged LoRA
        grpo_v_llm = load_peft_merged(args.base_model, args.grpo_verifier, tokenizer, "grpo-verifier")
        grpo_c_llm = load_peft_merged(args.base_model, args.grpo_challenger, tokenizer, "grpo-challenger")
        logger.info("  GRPO models loaded (%.1f GB)", torch.cuda.memory_allocated() / 1024**3)

        engine = build_debate_engine(base_llm, verifier_llm=grpo_v_llm, challenger_llm=grpo_c_llm)

        t0 = time.time()
        results = await eval_config("mad_grpo", pairs, debate_engine=engine)
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
        del base, grpo_v_llm, grpo_c_llm
        torch.cuda.empty_cache()

    # ─── MAD + MARL C3 v2 ───
    if "mad_marl_c3v2" in configs:
        logger.info("=" * 60)
        logger.info("Evaluating: mad_marl_c3v2 (MARL C3 v2 best, separate adapters, full MAD debate)")

        # Load base for extractor + judge
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        base.eval()
        base_llm = PreloadedLLMClient(base, tokenizer, "base")

        # Load MARL verifier and challenger with merged LoRA
        marl_v_llm = load_peft_merged(args.base_model, args.marl_verifier, tokenizer, "marl-verifier")
        marl_c_llm = load_peft_merged(args.base_model, args.marl_challenger, tokenizer, "marl-challenger")
        logger.info("  MARL C3 v2 models loaded (%.1f GB)", torch.cuda.memory_allocated() / 1024**3)

        engine = build_debate_engine(base_llm, verifier_llm=marl_v_llm, challenger_llm=marl_c_llm)

        t0 = time.time()
        results = await eval_config("mad_marl_c3v2", pairs, debate_engine=engine)
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
        del base, marl_v_llm, marl_c_llm
        torch.cuda.empty_cache()

    # ─── Print comparison ───
    print("\n" + "=" * 95)
    print("FULL PIPELINE HALLUCINATION DETECTION (MedHallu, %d pairs)" % len(pairs))
    print("=" * 95)
    print(f"{'Config':<20} {'F1':>8} {'95% CI':>18} {'Precision':>10} {'Recall':>8} {'Accuracy':>10} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("-" * 95)
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
    print("=" * 95)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
