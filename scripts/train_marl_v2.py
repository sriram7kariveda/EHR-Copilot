"""MARL v2: Detection-Aligned Shared Reward.

Key change: shared reward = detection accuracy (same as eval metric).
Follows MMOA-RAG: train on F1, evaluate on F1.

For each trajectory:
1. Both agents process the same input
2. Shared reward = did BOTH agents correctly classify (hallucinated vs ground truth)?
3. Individual agent rewards modulated by shared outcome

Usage:
    python scripts/train_marl_v2.py --iterations 3
"""

from __future__ import annotations

import argparse
import csv
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


def load_trajectories(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line.strip()) for line in f]


AGENT_PROMPTS = {
    "verifier": lambda traj: f"""You are a STRICT clinical evidence verifier.

Answer: {traj.get('answer', '')[:500]}
Evidence: {traj.get('evidence', '')[:1500]}

Respond in JSON: {{"verdict": "supported"|"not_supported"|"partial", "confidence": 0.0-1.0}}
Return ONLY JSON:""",

    "challenger": lambda traj: f"""You are an adversarial clinical auditor.

Answer: {traj.get('answer', '')[:500]}
Evidence: {traj.get('evidence', '')[:1500]}

If issues: [{{"challenge_type": "...", "challenge_text": "..."}}]
If no issues: []
Return ONLY JSON:""",
}


def compute_shared_detection_reward(verifier_text: str, challenger_text: str, traj: dict) -> dict:
    """Compute shared detection reward based on both agents' outputs."""
    target_score = traj.get("judge_target_score", 0.5)
    is_hallucinated = target_score < 0.5

    # Parse verifier
    v_match = re.search(r'"verdict"\s*:\s*"(\w+)"', verifier_text)
    v_flagged = False
    if v_match:
        v_flagged = v_match.group(1).lower() in ("not_supported", "partial")

    # Parse challenger
    c_challenged = "challenge_type" in challenger_text and "[]" not in challenger_text

    # Combined detection: flag if EITHER agent flags
    pipeline_flagged = v_flagged or c_challenged

    # Shared outcome
    correct = (is_hallucinated and pipeline_flagged) or (not is_hallucinated and not pipeline_flagged)

    # Shared reward (continuous, not binary)
    if correct:
        base_reward = 1.0
        # Bonus for agreement (both agents align)
        if v_flagged == c_challenged:
            base_reward += 0.3  # Both agree = stronger signal
    else:
        base_reward = -1.0
        # Extra penalty for confident wrong answer
        if v_flagged == c_challenged:
            base_reward -= 0.3  # Both wrong = worse

    # Individual modulation
    v_correct = (is_hallucinated and v_flagged) or (not is_hallucinated and not v_flagged)
    c_correct = (is_hallucinated and c_challenged) or (not is_hallucinated and not c_challenged)

    return {
        "shared_reward": base_reward,
        "verifier_reward": base_reward + (0.2 if v_correct else -0.2),
        "challenger_reward": base_reward + (0.2 if c_correct else -0.2),
        "correct": correct,
        "v_correct": v_correct,
        "c_correct": c_correct,
    }


def train_one_agent(model, tokenizer, agent_name, trajectories, k_samples=4, grad_accum=4, lr=5e-6, max_new_tokens=128):
    """One round of MARL for one agent with detection-aligned reward."""
    build_prompt = AGENT_PROMPTS[agent_name]
    other_agent = "challenger" if agent_name == "verifier" else "verifier"
    other_prompt = AGENT_PROMPTS[other_agent]

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer.zero_grad()

    rewards = []
    step = 0

    model.train()
    for i, traj in enumerate(trajectories):
        prompt = build_prompt(traj)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate k samples for THIS agent
        sample_rewards = []
        sample_gen_ids = []

        model.eval()

        # First get the OTHER agent's output (frozen, single sample)
        other_p = other_prompt(traj)
        other_inputs = tokenizer(other_p, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            other_gen = model.generate(**other_inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        other_text = tokenizer.decode(other_gen[0][other_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Generate k samples for current agent
        for _ in range(k_samples):
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
            gen_ids = gen[0][prompt_len:]
            if len(gen_ids) == 0:
                continue
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Compute SHARED reward using both agents' outputs
            if agent_name == "verifier":
                reward_info = compute_shared_detection_reward(gen_text, other_text, traj)
                r = reward_info["verifier_reward"]
            else:
                reward_info = compute_shared_detection_reward(other_text, gen_text, traj)
                r = reward_info["challenger_reward"]

            sample_rewards.append(r)
            sample_gen_ids.append(gen_ids)

        model.train()

        if len(sample_rewards) < 2:
            continue

        r_tensor = torch.tensor(sample_rewards)
        std_r = r_tensor.std()
        if std_r < 1e-8:
            rewards.append(r_tensor.mean().item())
            continue

        advantages = (r_tensor - r_tensor.mean()) / (std_r + 1e-8)

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

            if step % 10 == 0:
                logger.info("    %s step %d | reward: %.4f", agent_name, step, np.mean(rewards[-grad_accum:]))

    return {"steps": step, "mean_reward": np.mean(rewards) if rewards else 0}


def eval_detection(model, tokenizer, eval_traj, max_new_tokens=128):
    """Eval: run both agents, compute detection accuracy."""
    model.eval()
    correct = 0
    total = 0

    for traj in eval_traj[:200]:
        # Run verifier
        v_prompt = AGENT_PROMPTS["verifier"](traj)
        v_inputs = tokenizer(v_prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            v_gen = model.generate(**v_inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        v_text = tokenizer.decode(v_gen[0][v_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Run challenger
        c_prompt = AGENT_PROMPTS["challenger"](traj)
        c_inputs = tokenizer(c_prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            c_gen = model.generate(**c_inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        c_text = tokenizer.decode(c_gen[0][c_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        reward_info = compute_shared_detection_reward(v_text, c_text, traj)
        if reward_info["correct"]:
            correct += 1
        total += 1

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="MARL v2 — Detection-Aligned Shared Reward")
    parser.add_argument("--trajectories", default="data/grpo_trajectories/train.jsonl")
    parser.add_argument("--eval_trajectories", default="data/grpo_trajectories/eval.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--k_samples", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--output_dir", default="models/marl-v2")
    args = parser.parse_args()

    train_traj = load_trajectories(args.trajectories)
    eval_traj = load_trajectories(args.eval_trajectories)
    logger.info("Train: %d, Eval: %d", len(train_traj), len(eval_traj))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.1, bias="none",
                             task_type=TaskType.CAUSAL_LM,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = open("logs/marl_v2.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iteration", "agent", "reward", "eval_accuracy"])

    # Initial eval
    init_acc = eval_detection(model, tokenizer, eval_traj)
    logger.info("Initial detection accuracy: %.1f%%", init_acc * 100)

    for iteration in range(args.iterations):
        logger.info("\n>>> MARL v2 Iteration %d/%d", iteration + 1, args.iterations)
        random.shuffle(train_traj)

        for agent_name in ["verifier", "challenger"]:
            logger.info("  Training: %s (other frozen)", agent_name)
            result = train_one_agent(model, tokenizer, agent_name, train_traj,
                                     k_samples=args.k_samples, lr=args.learning_rate)
            logger.info("  %s done: %d steps, reward: %.4f", agent_name, result["steps"], result["mean_reward"])

        # Eval after each iteration
        acc = eval_detection(model, tokenizer, eval_traj)
        logger.info("  >>> Iteration %d Detection Accuracy: %.1f%%", iteration + 1, acc * 100)

        csv_writer.writerow([iteration + 1, "both", f"{result['mean_reward']:.4f}", f"{acc:.4f}"])
        csv_file.flush()

        # Save
        iter_dir = output_dir / f"iter_{iteration+1}"
        iter_dir.mkdir(exist_ok=True)
        model.save_pretrained(iter_dir)
        tokenizer.save_pretrained(iter_dir)

    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    logger.info("MARL v2 complete. Final model: %s/final", output_dir)
    csv_file.close()


if __name__ == "__main__":
    main()
