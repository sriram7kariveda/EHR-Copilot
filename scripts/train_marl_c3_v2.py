"""MARL C3 v2: Separate LoRA adapters + Warm-start + Iterated Best Response.

Fixes from v1:
1. SEPARATE LoRA adapters per agent (no weight interference)
2. Warm-start from GRPO v3 checkpoints (already F1=0.657)
3. Iterated best response (train one → freeze → train other)
4. All 1800 trajectories, grad_accum=2 for more gradient steps

Architecture:
- Base: Qwen2.5-3B-Instruct (frozen)
- Verifier adapter: loaded from verifier-grpo-v3, fine-tuned
- Challenger adapter: loaded from challenger-grpo-v3, fine-tuned
- Only one adapter active at a time during training

Usage:
    python scripts/train_marl_c3_v2.py --count 1800 --iterations 3
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
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

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


def compute_pipeline_detection(v_out: str, c_out: str, traj: dict) -> float:
    target = traj.get("judge_target_score", 0.5)
    is_hall = target < 0.5
    flagged = is_flagged_by_verifier(v_out) or is_flagged_by_challenger(c_out)

    if is_hall and flagged:
        return 1.0
    elif not is_hall and not flagged:
        return 1.0
    elif is_hall and not flagged:
        return -1.0
    else:
        return -0.5


def compute_individual_reward(output: str, traj: dict, agent: str) -> float:
    target = traj.get("judge_target_score", 0.5)
    is_hall = target < 0.5
    flagged = is_flagged_by_verifier(output) if agent == "verifier" else is_flagged_by_challenger(output)

    if is_hall and flagged:
        return 1.0
    elif is_hall and not flagged:
        return -1.0
    elif not is_hall and not flagged:
        return 1.0
    else:
        return -0.5


def compute_hybrid_reward(output: str, traj: dict, agent: str,
                          v_out: str, c_out: str) -> float:
    individual = compute_individual_reward(output, traj, agent)
    shared = compute_pipeline_detection(v_out, c_out, traj)

    # Counterfactual: what's the pipeline score WITHOUT this agent?
    if agent == "verifier":
        no_me = compute_pipeline_detection(DEFAULT_VERIFIER, c_out, traj)
    else:
        no_me = compute_pipeline_detection(v_out, DEFAULT_CHALLENGER, traj)
    counterfactual = shared - no_me

    return 0.5 * individual + 0.3 * counterfactual + 0.2 * shared


def generate_text(model, tokenizer, prompt: str, max_new_tokens=128, temperature=0.7) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=temperature, do_sample=True,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_with_ids(model, tokenizer, prompt: str, max_new_tokens=128, temperature=0.7):
    """Generate and return (text, gen_ids, prompt_len)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=temperature, do_sample=True,
                             pad_token_id=tokenizer.pad_token_id)
    gen_ids = gen[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, gen_ids, prompt_len, inputs


def eval_pipeline(verifier_model, challenger_model, tokenizer, eval_traj) -> dict:
    """Eval with separate models for each agent."""
    verifier_model.eval()
    challenger_model.eval()
    correct = 0
    total = 0
    tp = fp = tn = fn = 0

    for traj in eval_traj:
        v_text = generate_text(verifier_model, tokenizer, verifier_prompt(traj),
                               max_new_tokens=128, temperature=0.1)
        c_text = generate_text(challenger_model, tokenizer, challenger_prompt(traj),
                               max_new_tokens=128, temperature=0.1)

        score = compute_pipeline_detection(v_text, c_text, traj)
        target = traj.get("judge_target_score", 0.5)
        is_hall = target < 0.5
        flagged = is_flagged_by_verifier(v_text) or is_flagged_by_challenger(c_text)

        if is_hall and flagged:
            tp += 1
        elif is_hall and not flagged:
            fn += 1
        elif not is_hall and not flagged:
            tn += 1
        else:
            fp += 1

        if score > 0:
            correct += 1
        total += 1

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def train_agent_ibr(active_model, frozen_model, tokenizer, agent_name, trajectories,
                    k_samples=4, grad_accum=2, lr=3e-6, max_new_tokens=128):
    """Iterated Best Response: train active_model while frozen_model is fixed."""
    prompt_fn = verifier_prompt if agent_name == "verifier" else challenger_prompt
    other_prompt_fn = challenger_prompt if agent_name == "verifier" else verifier_prompt

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, active_model.parameters()), lr=lr)
    optimizer.zero_grad()

    rewards = []
    step = 0

    active_model.train()
    frozen_model.eval()

    for i, traj in enumerate(trajectories):
        prompt = prompt_fn(traj)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(active_model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Get frozen agent's output (truly frozen — different model)
        other_text = generate_text(frozen_model, tokenizer, other_prompt_fn(traj),
                                   max_new_tokens=max_new_tokens, temperature=0.1)

        # Generate k samples from active agent
        sample_rewards = []
        sample_gen_ids = []

        active_model.eval()
        for _ in range(k_samples):
            text, gen_ids, _, _ = generate_with_ids(
                active_model, tokenizer, prompt,
                max_new_tokens=max_new_tokens, temperature=0.7)
            if len(gen_ids) == 0:
                continue

            if agent_name == "verifier":
                r = compute_hybrid_reward(text, traj, "verifier", text, other_text)
            else:
                r = compute_hybrid_reward(text, traj, "challenger", other_text, text)

            sample_rewards.append(r)
            sample_gen_ids.append(gen_ids)

        active_model.train()

        if len(sample_rewards) < 2:
            continue

        r_tensor = torch.tensor(sample_rewards)
        if r_tensor.std() < 1e-8:
            rewards.append(r_tensor.mean().item())
            continue

        advantages = (r_tensor - r_tensor.mean()) / (r_tensor.std() + 1e-8)

        traj_loss = torch.tensor(0.0, device=active_model.device)
        for adv, gen_ids in zip(advantages, sample_gen_ids):
            full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)
            outputs = active_model(full_ids)
            logits = outputs.logits[0, prompt_len-1:-1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
            traj_loss = traj_loss - adv.item() * token_lp.sum() / (k_samples * grad_accum)

        traj_loss.backward()
        rewards.append(r_tensor.mean().item())

        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(active_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                logger.info("    %s step %d | hybrid_reward: %.4f",
                            agent_name, step, np.mean(rewards[-grad_accum:]))

    # Final optimizer step for remaining
    if (len(trajectories)) % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(active_model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    return {"steps": step, "mean_reward": np.mean(rewards) if rewards else 0}


def main():
    parser = argparse.ArgumentParser(description="MARL C3 v2: Separate adapters + IBR + Warm-start")
    parser.add_argument("--trajectories", default="data/grpo_trajectories/train.jsonl")
    parser.add_argument("--eval_trajectories", default="data/grpo_trajectories/eval.jsonl")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--verifier_adapter", default="models/verifier-grpo-v3")
    parser.add_argument("--challenger_adapter", default="models/challenger-grpo-v3")
    parser.add_argument("--count", type=int, default=1800)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--k_samples", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--output_dir", default="models/marl-c3-v2")
    args = parser.parse_args()

    random.seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_traj = load_trajectories(args.trajectories, args.count)
    eval_traj = load_trajectories(args.eval_trajectories, 200)
    logger.info("Train: %d, Eval: %d", len(train_traj), len(eval_traj))

    # Load base model
    logger.info("Loading base: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load TWO separate models with their own LoRA adapters
    logger.info("Loading verifier from %s", args.verifier_adapter)
    verifier_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    verifier_model = PeftModel.from_pretrained(verifier_model, args.verifier_adapter, is_trainable=True)
    verifier_model.gradient_checkpointing_enable()
    verifier_model.config.use_cache = False
    verifier_model.print_trainable_parameters()
    logger.info("Verifier GPU: %.1f GB", torch.cuda.memory_allocated() / 1024**3)

    logger.info("Loading challenger from %s", args.challenger_adapter)
    challenger_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    challenger_model = PeftModel.from_pretrained(challenger_model, args.challenger_adapter, is_trainable=True)
    challenger_model.gradient_checkpointing_enable()
    challenger_model.config.use_cache = False
    challenger_model.print_trainable_parameters()
    logger.info("Total GPU: %.1f GB", torch.cuda.memory_allocated() / 1024**3)

    # CSV log
    Path("logs").mkdir(exist_ok=True)
    csv_file = open("logs/marl_c3_v2.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iteration", "agent_trained", "reward", "accuracy", "f1", "precision", "recall"])

    # Initial eval
    logger.info("Initial eval (both agents from GRPO v3)...")
    init = eval_pipeline(verifier_model, challenger_model, tokenizer, eval_traj)
    logger.info("Initial: acc=%.1f%% F1=%.3f P=%.3f R=%.3f (tp=%d fp=%d tn=%d fn=%d)",
                init["accuracy"]*100, init["f1"], init["precision"], init["recall"],
                init["tp"], init["fp"], init["tn"], init["fn"])
    csv_writer.writerow([0, "baseline", "", f"{init['accuracy']:.4f}", f"{init['f1']:.4f}",
                         f"{init['precision']:.4f}", f"{init['recall']:.4f}"])
    csv_file.flush()

    best_f1 = init["f1"]

    for iteration in range(args.iterations):
        logger.info("\n>>> IBR Iteration %d/%d", iteration + 1, args.iterations)
        random.shuffle(train_traj)

        # Phase 1: Train VERIFIER (challenger frozen)
        logger.info("  Phase 1: Training VERIFIER (challenger frozen)")
        v_result = train_agent_ibr(
            verifier_model, challenger_model, tokenizer, "verifier",
            train_traj, k_samples=args.k_samples, lr=args.lr)
        logger.info("  Verifier: %d steps, reward: %.4f", v_result["steps"], v_result["mean_reward"])

        # Mid-iteration eval
        mid = eval_pipeline(verifier_model, challenger_model, tokenizer, eval_traj)
        logger.info("  Mid-iter: acc=%.1f%% F1=%.3f", mid["accuracy"]*100, mid["f1"])

        # Phase 2: Train CHALLENGER (verifier frozen)
        logger.info("  Phase 2: Training CHALLENGER (verifier frozen)")
        c_result = train_agent_ibr(
            challenger_model, verifier_model, tokenizer, "challenger",
            train_traj, k_samples=args.k_samples, lr=args.lr)
        logger.info("  Challenger: %d steps, reward: %.4f", c_result["steps"], c_result["mean_reward"])

        # Full eval
        result = eval_pipeline(verifier_model, challenger_model, tokenizer, eval_traj)
        logger.info("  >>> Iter %d: acc=%.1f%% F1=%.3f P=%.3f R=%.3f (best F1=%.3f)",
                    iteration + 1, result["accuracy"]*100, result["f1"],
                    result["precision"], result["recall"], best_f1)

        csv_writer.writerow([iteration + 1, "both",
                             f"{(v_result['mean_reward']+c_result['mean_reward'])/2:.4f}",
                             f"{result['accuracy']:.4f}", f"{result['f1']:.4f}",
                             f"{result['precision']:.4f}", f"{result['recall']:.4f}"])
        csv_file.flush()

        # Save
        iter_dir = output_dir / f"iter_{iteration+1}"
        iter_dir.mkdir(exist_ok=True)
        verifier_model.save_pretrained(iter_dir / "verifier")
        challenger_model.save_pretrained(iter_dir / "challenger")
        tokenizer.save_pretrained(iter_dir)

        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_dir = output_dir / "best"
            best_dir.mkdir(exist_ok=True)
            verifier_model.save_pretrained(best_dir / "verifier")
            challenger_model.save_pretrained(best_dir / "challenger")
            tokenizer.save_pretrained(best_dir)
            logger.info("  >>> NEW BEST F1! Saved to %s/best", output_dir)

    logger.info("\nMARL C3 v2 complete. Best F1: %.3f", best_f1)
    csv_file.close()


if __name__ == "__main__":
    main()
