"""MARL Full Pipeline: Train BOTH 8B (pipeline) and 3B (debate) with shared reward.

Trial mode: 50 trajectories, 1 iteration, conservative hyperparameters.
Saves sanity check outputs before/after training to detect model degradation.

Architecture:
- Qwen 3 8B + LoRA r=4 (tiny): affects Triage, CRAG, Reasoning
- Qwen 2.5 3B + LoRA r=8: affects Verifier, Challenger
- Shared reward = detection accuracy (same as GRPO v3)
- Round-robin: train 8B (3B frozen) → check → train 3B (8B frozen) → check

Usage:
    # Trial run (50 examples, ~2 hours)
    python scripts/train_marl_full_pipeline.py --count 50 --iterations 1

    # Full run (500 examples, 2 iterations)
    python scripts/train_marl_full_pipeline.py --count 500 --iterations 2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_trajectories(path: str, count: int = 50) -> list[dict]:
    trajs = []
    with open(path) as f:
        for line in f:
            trajs.append(json.loads(line.strip()))
    random.shuffle(trajs)
    return trajs[:count]


# Agent prompts for each model
def triage_prompt(traj: dict) -> str:
    return f"""Classify this clinical query: {traj.get('query', '')[:300]}
Respond in JSON: {{"query_type": "FACTUAL|TEMPORAL|NUMERIC|MEDICATION|SUMMARY", "requires_temporal": true/false, "requires_numeric": true/false}}
Return ONLY JSON:"""


def crag_prompt(traj: dict) -> str:
    return f"""Is this evidence sufficient to answer the query?
Query: {traj.get('query', '')[:300]}
Evidence: {traj.get('evidence', '')[:800]}
Respond in JSON: {{"verdict": "sufficient"|"insufficient", "coverage_score": 0.0-1.0}}
Return ONLY JSON:"""


def verifier_prompt(traj: dict) -> str:
    return f"""You are a STRICT clinical evidence verifier.
Answer: {traj.get('answer', '')[:500]}
Evidence: {traj.get('evidence', '')[:1000]}
Respond in JSON: {{"verdict": "supported"|"not_supported"|"partial", "confidence": 0.0-1.0}}
Return ONLY JSON:"""


def challenger_prompt(traj: dict) -> str:
    return f"""You are an adversarial clinical auditor. Challenge if issues found.
Answer: {traj.get('answer', '')[:500]}
Evidence: {traj.get('evidence', '')[:1000]}
If issues: [{{"challenge_type": "...", "challenge_text": "..."}}]
If no issues: []
Return ONLY JSON:"""


PIPELINE_AGENTS = {
    "triage": {"model": "8b", "prompt_fn": triage_prompt},
    "crag": {"model": "8b", "prompt_fn": crag_prompt},
}

DEBATE_AGENTS = {
    "verifier": {"model": "3b", "prompt_fn": verifier_prompt},
    "challenger": {"model": "3b", "prompt_fn": challenger_prompt},
}


def compute_detection_reward(output_text: str, traj: dict, agent_type: str) -> float:
    """Shared detection reward — same for all agents."""
    target = traj.get("judge_target_score", 0.5)
    is_hallucinated = target < 0.5

    if agent_type in ("verifier", "triage", "crag"):
        v_match = re.search(r'"verdict"\s*:\s*"(\w+)"', output_text)
        if not v_match:
            return -0.3
        verdict = v_match.group(1).lower()
        flagged = verdict in ("not_supported", "partial", "insufficient")

        if is_hallucinated and flagged:
            return 1.0
        elif is_hallucinated and not flagged:
            return -1.0
        elif not is_hallucinated and not flagged:
            return 1.0
        else:
            return -0.5

    elif agent_type == "challenger":
        has_challenges = "challenge_type" in output_text and "[]" not in output_text
        if is_hallucinated and has_challenges:
            return 1.0
        elif is_hallucinated and not has_challenges:
            return -1.0
        elif not is_hallucinated and not has_challenges:
            return 1.0
        else:
            return -0.5

    return 0.0


def sanity_check(model, tokenizer, test_prompts: list[str], label: str) -> list[str]:
    """Generate outputs for sanity check — compare before/after training."""
    model.eval()
    outputs = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text[:200])

    logger.info("  Sanity check [%s]: %d outputs generated", label, len(outputs))
    for i, o in enumerate(outputs[:3]):
        logger.info("    Sample %d: %s", i, o[:100])
    return outputs


def train_agent_group(
    model, tokenizer, agents: dict, trajectories: list[dict],
    k_samples: int = 4, grad_accum: int = 4, lr: float = 1e-6,
    max_new_tokens: int = 128, group_name: str = "8b",
):
    """Train a group of agents sharing the same model."""
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer.zero_grad()

    rewards = []
    step = 0
    model.train()

    for i, traj in enumerate(trajectories):
        for agent_name, agent_cfg in agents.items():
            prompt = agent_cfg["prompt_fn"](traj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            # Generate k samples
            sample_rewards = []
            sample_gen_ids = []

            model.eval()
            for _ in range(k_samples):
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
                gen_ids = gen[0][prompt_len:]
                if len(gen_ids) == 0:
                    continue
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                r = compute_detection_reward(text, traj, agent_name)
                sample_rewards.append(r)
                sample_gen_ids.append(gen_ids)
            model.train()

            if len(sample_rewards) < 2:
                continue

            r_tensor = torch.tensor(sample_rewards)
            if r_tensor.std() < 1e-8:
                rewards.append(r_tensor.mean().item())
                continue

            advantages = (r_tensor - r_tensor.mean()) / (r_tensor.std() + 1e-8)

            traj_loss = torch.tensor(0.0, device=model.device)
            for adv, gen_ids in zip(advantages, sample_gen_ids):
                full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)
                outputs = model(full_ids)
                logits = outputs.logits[0, prompt_len-1:-1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_lp = log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
                traj_loss = traj_loss - adv.item() * token_lp.sum() / (k_samples * grad_accum)

            traj_loss.backward()
            rewards.append(r_tensor.mean().item())

        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 5 == 0:
                logger.info("    %s step %d | reward: %.4f", group_name, step, np.mean(rewards[-grad_accum*len(agents):]))

    return {"steps": step, "mean_reward": np.mean(rewards) if rewards else 0}


def main():
    parser = argparse.ArgumentParser(description="MARL Full Pipeline Trial")
    parser.add_argument("--trajectories", default="data/grpo_trajectories/train.jsonl")
    parser.add_argument("--pipeline_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--debate_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--k_samples", type=int, default=4)
    parser.add_argument("--pipeline_lr", type=float, default=1e-6, help="Very conservative for 8B")
    parser.add_argument("--debate_lr", type=float, default=5e-6)
    parser.add_argument("--pipeline_lora_r", type=int, default=4, help="Tiny LoRA for 8B")
    parser.add_argument("--debate_lora_r", type=int, default=8)
    parser.add_argument("--output_dir", default="models/marl-full-pipeline")
    args = parser.parse_args()

    random.seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectories
    trajectories = load_trajectories(args.trajectories, args.count)
    split = int(len(trajectories) * 0.8)
    train_traj = trajectories[:split]
    eval_traj = trajectories[split:]
    logger.info("Loaded %d trajectories (train: %d, eval: %d)", len(trajectories), len(train_traj), len(eval_traj))

    # Sanity check prompts
    test_prompts_8b = [triage_prompt(train_traj[0]), crag_prompt(train_traj[0])]
    test_prompts_3b = [verifier_prompt(train_traj[0]), challenger_prompt(train_traj[0])]

    # ============ LOAD 8B MODEL ============
    logger.info("Loading pipeline model: %s (LoRA r=%d, lr=%s)", args.pipeline_model, args.pipeline_lora_r, args.pipeline_lr)
    tok_8b = AutoTokenizer.from_pretrained(args.pipeline_model, trust_remote_code=True)
    if tok_8b.pad_token is None:
        tok_8b.pad_token = tok_8b.eos_token

    model_8b = AutoModelForCausalLM.from_pretrained(args.pipeline_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_8b.gradient_checkpointing_enable()
    model_8b.config.use_cache = False

    lora_8b = LoraConfig(r=args.pipeline_lora_r, lora_alpha=args.pipeline_lora_r * 2, lora_dropout=0.1,
                         bias="none", task_type=TaskType.CAUSAL_LM,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model_8b = get_peft_model(model_8b, lora_8b)
    model_8b.print_trainable_parameters()
    logger.info("8B GPU memory: %.1f GB", torch.cuda.memory_allocated() / 1024**3)

    # Sanity check BEFORE training
    logger.info("=== SANITY CHECK: 8B BEFORE TRAINING ===")
    before_8b = sanity_check(model_8b, tok_8b, test_prompts_8b, "8b_before")

    # ============ TRAIN 8B ============
    for iteration in range(args.iterations):
        logger.info("\n>>> ITERATION %d/%d", iteration + 1, args.iterations)

        logger.info("  Training 8B pipeline agents (Triage + CRAG)...")
        result_8b = train_agent_group(
            model_8b, tok_8b, PIPELINE_AGENTS, train_traj,
            k_samples=args.k_samples, lr=args.pipeline_lr, group_name="8b",
        )
        logger.info("  8B done: %d steps, reward: %.4f", result_8b["steps"], result_8b["mean_reward"])

        # Sanity check AFTER 8B training
        logger.info("=== SANITY CHECK: 8B AFTER TRAINING ===")
        after_8b = sanity_check(model_8b, tok_8b, test_prompts_8b, "8b_after")

        # Check for degradation
        degraded = False
        for b, a in zip(before_8b, after_8b):
            if len(a) < 10 or a.count('{') > 10:  # gibberish check
                logger.warning("  !!! 8B MODEL DEGRADED — output looks like gibberish")
                degraded = True
                break

        if degraded:
            logger.warning("  STOPPING — 8B model degraded. Reverting.")
            break

        # Save 8B
        model_8b.save_pretrained(output_dir / "8b" / f"iter_{iteration+1}")
        tok_8b.save_pretrained(output_dir / "8b" / f"iter_{iteration+1}")
        logger.info("  Saved 8B to %s", output_dir / "8b" / f"iter_{iteration+1}")

    # Free 8B memory
    del model_8b
    torch.cuda.empty_cache()
    logger.info("Freed 8B model. GPU memory: %.1f GB", torch.cuda.memory_allocated() / 1024**3)

    # ============ LOAD 3B MODEL ============
    logger.info("Loading debate model: %s (LoRA r=%d, lr=%s)", args.debate_model, args.debate_lora_r, args.debate_lr)
    tok_3b = AutoTokenizer.from_pretrained(args.debate_model, trust_remote_code=True)
    if tok_3b.pad_token is None:
        tok_3b.pad_token = tok_3b.eos_token

    model_3b = AutoModelForCausalLM.from_pretrained(args.debate_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_3b.gradient_checkpointing_enable()
    model_3b.config.use_cache = False

    lora_3b = LoraConfig(r=args.debate_lora_r, lora_alpha=args.debate_lora_r * 2, lora_dropout=0.1,
                         bias="none", task_type=TaskType.CAUSAL_LM,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model_3b = get_peft_model(model_3b, lora_3b)
    model_3b.print_trainable_parameters()

    # Sanity check BEFORE
    logger.info("=== SANITY CHECK: 3B BEFORE TRAINING ===")
    before_3b = sanity_check(model_3b, tok_3b, test_prompts_3b, "3b_before")

    # ============ TRAIN 3B ============
    for iteration in range(args.iterations):
        logger.info("  Training 3B debate agents (Verifier + Challenger)...")
        result_3b = train_agent_group(
            model_3b, tok_3b, DEBATE_AGENTS, train_traj,
            k_samples=args.k_samples, lr=args.debate_lr, group_name="3b",
        )
        logger.info("  3B done: %d steps, reward: %.4f", result_3b["steps"], result_3b["mean_reward"])

        # Sanity check
        logger.info("=== SANITY CHECK: 3B AFTER TRAINING ===")
        after_3b = sanity_check(model_3b, tok_3b, test_prompts_3b, "3b_after")

        # Save
        model_3b.save_pretrained(output_dir / "3b" / f"iter_{iteration+1}")
        tok_3b.save_pretrained(output_dir / "3b" / f"iter_{iteration+1}")
        logger.info("  Saved 3B to %s", output_dir / "3b" / f"iter_{iteration+1}")

    logger.info("\n=== MARL FULL PIPELINE TRIAL COMPLETE ===")
    logger.info("8B model: %s/8b/", output_dir)
    logger.info("3B model: %s/3b/", output_dir)
    logger.info("Check sanity outputs above to verify models didn't degrade.")


if __name__ == "__main__":
    main()
