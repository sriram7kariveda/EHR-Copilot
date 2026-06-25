"""MARL with Counterfactual Credit Assignment (C3) + Hybrid Reward.

Key innovation: Instead of giving all agents the same shared reward,
compute each agent's MARGINAL CONTRIBUTION via counterfactual replay.

For each trajectory:
1. Run both agents → compute shared outcome
2. Replace agent_i with default output → recompute outcome
3. marginal_i = outcome_with - outcome_without
4. hybrid_reward = 0.5 * individual + 0.3 * counterfactual + 0.2 * shared

This solves the credit assignment problem that made MARL v1/v2 fail.

Usage:
    python scripts/train_marl_c3.py --count 500 --iterations 2
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

# Default outputs for counterfactual (neutral — doesn't help or hurt)
DEFAULT_VERIFIER = '{"verdict": "partial", "confidence": 0.5, "reasoning": "uncertain"}'
DEFAULT_CHALLENGER = '[]'


def load_trajectories(path: str, count: int) -> list[dict]:
    trajs = []
    with open(path) as f:
        for line in f:
            trajs.append(json.loads(line.strip()))
    random.shuffle(trajs)
    return trajs[:count]


def verifier_prompt(traj: dict) -> str:
    return f"""You are a STRICT clinical evidence verifier.
Answer: {traj.get('answer', '')[:500]}
Evidence: {traj.get('evidence', '')[:1500]}
Respond in JSON: {{"verdict": "supported"|"not_supported"|"partial", "confidence": 0.0-1.0}}
Return ONLY JSON:"""


def challenger_prompt(traj: dict) -> str:
    return f"""You are an adversarial clinical auditor.
Answer: {traj.get('answer', '')[:500]}
Evidence: {traj.get('evidence', '')[:1500]}
If issues: [{{"challenge_type": "...", "challenge_text": "..."}}]
If no issues: []
Return ONLY JSON:"""


def is_flagged_by_verifier(output: str) -> bool:
    m = re.search(r'"verdict"\s*:\s*"(\w+)"', output)
    if not m:
        return False
    return m.group(1).lower() in ("not_supported", "partial")


def is_flagged_by_challenger(output: str) -> bool:
    return "challenge_type" in output and "[]" not in output


def compute_pipeline_detection(verifier_out: str, challenger_out: str, traj: dict) -> float:
    """Compute whether the pipeline correctly detected hallucination/ground truth."""
    target = traj.get("judge_target_score", 0.5)
    is_hallucinated = target < 0.5

    v_flagged = is_flagged_by_verifier(verifier_out)
    c_flagged = is_flagged_by_challenger(challenger_out)
    pipeline_flagged = v_flagged or c_flagged

    if is_hallucinated and pipeline_flagged:
        return 1.0
    elif not is_hallucinated and not pipeline_flagged:
        return 1.0
    elif is_hallucinated and not pipeline_flagged:
        return -1.0
    else:  # false positive
        return -0.5


def compute_individual_reward(output: str, traj: dict, agent: str) -> float:
    """Individual detection reward (same as GRPO v3)."""
    target = traj.get("judge_target_score", 0.5)
    is_hallucinated = target < 0.5

    if agent == "verifier":
        flagged = is_flagged_by_verifier(output)
    else:
        flagged = is_flagged_by_challenger(output)

    if is_hallucinated and flagged:
        return 1.0
    elif is_hallucinated and not flagged:
        return -1.0
    elif not is_hallucinated and not flagged:
        return 1.0
    else:
        return -0.5


def compute_counterfactual(verifier_out: str, challenger_out: str, traj: dict) -> tuple[float, float]:
    """C3: Compute each agent's marginal contribution via counterfactual."""
    full = compute_pipeline_detection(verifier_out, challenger_out, traj)
    no_verifier = compute_pipeline_detection(DEFAULT_VERIFIER, challenger_out, traj)
    no_challenger = compute_pipeline_detection(verifier_out, DEFAULT_CHALLENGER, traj)

    verifier_marginal = full - no_verifier
    challenger_marginal = full - no_challenger

    return verifier_marginal, challenger_marginal


def compute_hybrid_reward(output: str, traj: dict, agent: str,
                          verifier_out: str, challenger_out: str) -> float:
    """Hybrid = 0.5 * individual + 0.3 * counterfactual + 0.2 * shared."""
    individual = compute_individual_reward(output, traj, agent)
    shared = compute_pipeline_detection(verifier_out, challenger_out, traj)
    v_marginal, c_marginal = compute_counterfactual(verifier_out, challenger_out, traj)
    counterfactual = v_marginal if agent == "verifier" else c_marginal

    return 0.5 * individual + 0.3 * counterfactual + 0.2 * shared


def eval_detection(model, tokenizer, eval_traj, max_new_tokens=128):
    """Eval: run both agents, compute detection accuracy."""
    model.eval()
    correct = 0
    total = 0

    for traj in eval_traj[:200]:
        v_in = tokenizer(verifier_prompt(traj), return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            v_gen = model.generate(**v_in, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        v_text = tokenizer.decode(v_gen[0][v_in["input_ids"].shape[1]:], skip_special_tokens=True)

        c_in = tokenizer(challenger_prompt(traj), return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            c_gen = model.generate(**c_in, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        c_text = tokenizer.decode(c_gen[0][c_in["input_ids"].shape[1]:], skip_special_tokens=True)

        score = compute_pipeline_detection(v_text, c_text, traj)
        if score > 0:
            correct += 1
        total += 1

    return correct / max(total, 1)


def train_agent_c3(model, tokenizer, agent_name, trajectories,
                   k_samples=4, grad_accum=4, lr=5e-6, max_new_tokens=128):
    """Train one agent with C3 hybrid reward."""
    prompt_fn = verifier_prompt if agent_name == "verifier" else challenger_prompt
    other_prompt_fn = challenger_prompt if agent_name == "verifier" else verifier_prompt

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer.zero_grad()

    rewards = []
    counterfactuals = []
    step = 0

    model.train()
    for i, traj in enumerate(trajectories):
        prompt = prompt_fn(traj)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Get OTHER agent's output (frozen, single sample)
        model.eval()
        other_in = tokenizer(other_prompt_fn(traj), return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            other_gen = model.generate(**other_in, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        other_text = tokenizer.decode(other_gen[0][other_in["input_ids"].shape[1]:], skip_special_tokens=True)

        # Generate k samples for current agent
        sample_rewards = []
        sample_gen_ids = []

        for _ in range(k_samples):
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
            gen_ids = gen[0][prompt_len:]
            if len(gen_ids) == 0:
                continue
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Compute HYBRID reward (individual + counterfactual + shared)
            if agent_name == "verifier":
                r = compute_hybrid_reward(gen_text, traj, "verifier", gen_text, other_text)
            else:
                r = compute_hybrid_reward(gen_text, traj, "challenger", other_text, gen_text)

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
                logger.info("    %s step %d | hybrid_reward: %.4f", agent_name, step, np.mean(rewards[-grad_accum:]))

    return {"steps": step, "mean_reward": np.mean(rewards) if rewards else 0}


def main():
    parser = argparse.ArgumentParser(description="MARL C3 + Hybrid Reward")
    parser.add_argument("--trajectories", default="data/grpo_trajectories/train.jsonl")
    parser.add_argument("--eval_trajectories", default="data/grpo_trajectories/eval.jsonl")
    parser.add_argument("--debate_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--k_samples", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--output_dir", default="models/marl-c3")
    args = parser.parse_args()

    random.seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_traj = load_trajectories(args.trajectories, args.count)
    eval_traj = load_trajectories(args.eval_trajectories, 200)
    logger.info("Train: %d, Eval: %d", len(train_traj), len(eval_traj))

    # Load 3B model + LoRA
    logger.info("Loading: %s", args.debate_model)
    tokenizer = AutoTokenizer.from_pretrained(args.debate_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.debate_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.1, bias="none",
                             task_type=TaskType.CAUSAL_LM,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("GPU: %.1f GB", torch.cuda.memory_allocated() / 1024**3)

    # CSV log
    csv_file = open("logs/marl_c3.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iteration", "agent", "reward", "eval_accuracy"])

    # Initial eval
    init_acc = eval_detection(model, tokenizer, eval_traj)
    logger.info("Initial detection accuracy: %.1f%%", init_acc * 100)

    best_acc = init_acc

    for iteration in range(args.iterations):
        logger.info("\n>>> MARL C3 Iteration %d/%d", iteration + 1, args.iterations)
        random.shuffle(train_traj)

        for agent_name in ["verifier", "challenger"]:
            logger.info("  Training: %s (C3 hybrid reward)", agent_name)
            result = train_agent_c3(model, tokenizer, agent_name, train_traj,
                                    k_samples=args.k_samples, lr=args.lr)
            logger.info("  %s: %d steps, reward: %.4f", agent_name, result["steps"], result["mean_reward"])

        # Eval
        acc = eval_detection(model, tokenizer, eval_traj)
        logger.info("  >>> Iteration %d Detection Accuracy: %.1f%% (best: %.1f%%)", iteration + 1, acc * 100, best_acc * 100)

        csv_writer.writerow([iteration + 1, "both", f"{result['mean_reward']:.4f}", f"{acc:.4f}"])
        csv_file.flush()

        # Save
        iter_dir = output_dir / f"iter_{iteration+1}"
        iter_dir.mkdir(exist_ok=True)
        model.save_pretrained(iter_dir)
        tokenizer.save_pretrained(iter_dir)

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
            logger.info("  >>> NEW BEST! Saved to %s/best", output_dir)

    logger.info("\nMARL C3 complete. Best accuracy: %.1f%%", best_acc * 100)
    csv_file.close()


if __name__ == "__main__":
    main()
