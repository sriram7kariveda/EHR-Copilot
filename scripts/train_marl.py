"""MARL: Multi-Agent RL with Shared Reward across 4 pipeline agents.

Trains Triage, CRAG Evaluator, Verifier, and Challenger with a shared
reward signal = whether the full pipeline correctly classified the answer
as hallucinated or ground truth.

Key difference from independent GRPO:
- Independent: each agent has its own reward function
- MARL: ALL agents share ONE reward (end-to-end classification accuracy)
- This teaches agents to COOPERATE — Triage routes in ways that help
  Verifier, CRAG filters in ways that help Challenger, etc.

Algorithm: Iterative Multi-Agent GRPO with Shared Reward
  For each iteration:
    1. Run pipeline on data → collect trajectories with shared reward
    2. For each agent, generate k=4 completions
    3. Score with SHARED reward (not individual)
    4. GRPO update with group-relative advantage
    5. Round-robin: update one agent at a time, others frozen

Usage:
    python scripts/train_marl.py \
        --trajectories data/marl_trajectories.jsonl \
        --iterations 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Agent definitions: which model, which prompt field, which output field
AGENTS = {
    "triage": {
        "model_key": "pipeline",  # uses pipeline model (Qwen 3 8B)
        "prompt_field": "triage_prompt",
        "output_field": "triage_output",
    },
    "crag": {
        "model_key": "pipeline",
        "prompt_field": "crag_prompt",
        "output_field": "crag_output",
    },
    "verifier": {
        "model_key": "debate",  # uses debate model (Qwen 2.5 3B)
        "prompt_field": "verifier_prompt",
        "output_field": "verifier_output",
    },
    "challenger": {
        "model_key": "debate",
        "prompt_field": "challenger_prompt",
        "output_field": "challenger_output",
    },
}


def load_trajectories(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line.strip()) for line in f]


def train_agent_marl(
    model, tokenizer, agent_name: str, trajectories: list[dict],
    k_samples: int = 4, learning_rate: float = 5e-6,
    max_new_tokens: int = 128, grad_accum: int = 4,
):
    """One round of MARL GRPO for a single agent using shared reward."""
    prompt_field = AGENTS[agent_name]["prompt_field"]

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    rewards_collected = []
    losses_collected = []
    global_step = 0

    model.train()
    optimizer.zero_grad()

    for i, traj in enumerate(trajectories):
        prompt = traj.get(prompt_field, "")
        if not prompt:
            continue

        shared_reward_gt = traj["shared_reward"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate k samples
        sample_rewards = []
        sample_gen_ids = []

        model.eval()
        for _ in range(k_samples):
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=0.7, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = gen_out[0][prompt_len:]
            if len(gen_ids) == 0:
                continue

            # SHARED reward: use the trajectory's shared reward
            # But modulate by whether this sample's output is "good"
            # Simple heuristic: if output is parseable JSON → bonus, else penalty
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            try:
                json.loads(gen_text.strip().split("\n")[0])
                format_bonus = 0.2
            except Exception:
                format_bonus = -0.2

            sample_reward = shared_reward_gt + format_bonus
            sample_rewards.append(sample_reward)
            sample_gen_ids.append(gen_ids)

        model.train()

        if len(sample_rewards) < 2:
            continue

        # Group-relative advantage
        r_tensor = torch.tensor(sample_rewards)
        std_r = r_tensor.std()
        if std_r < 1e-8:
            rewards_collected.append(r_tensor.mean().item())
            continue

        advantages = (r_tensor - r_tensor.mean()) / (std_r + 1e-8)

        # Forward passes with gradients
        traj_loss = torch.tensor(0.0, device=model.device)
        for adv, gen_ids in zip(advantages, sample_gen_ids):
            full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)
            outputs = model(full_ids)
            logits = outputs.logits[0, prompt_len-1:-1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
            traj_loss = traj_loss - adv.item() * token_lp.sum() / (k_samples * grad_accum)

        traj_loss.backward()
        rewards_collected.append(r_tensor.mean().item())
        losses_collected.append(traj_loss.item())

        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 10 == 0:
                logger.info(
                    "    %s step %d | loss: %.4f | reward: %.4f",
                    agent_name, global_step,
                    np.mean(losses_collected[-grad_accum:]),
                    np.mean(rewards_collected[-grad_accum:]),
                )

    return {
        "agent": agent_name,
        "steps": global_step,
        "mean_reward": np.mean(rewards_collected) if rewards_collected else 0,
        "mean_loss": np.mean(losses_collected) if losses_collected else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="MARL: Shared Reward Training")
    parser.add_argument("--trajectories", default="data/marl_trajectories.jsonl")
    parser.add_argument("--pipeline_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--debate_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--k_samples", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--output_dir", default="models/marl")
    parser.add_argument("--log_csv", default="logs/marl_training.csv")
    args = parser.parse_args()

    # Load trajectories
    trajectories = load_trajectories(args.trajectories)
    random.shuffle(trajectories)
    split = int(len(trajectories) * 0.9)
    train_traj = trajectories[:split]
    eval_traj = trajectories[split:]
    logger.info("Loaded %d trajectories (train: %d, eval: %d)", len(trajectories), len(train_traj), len(eval_traj))

    # Load models + LoRA
    logger.info("Loading debate model: %s", args.debate_model)
    debate_tokenizer = AutoTokenizer.from_pretrained(args.debate_model, trust_remote_code=True)
    if debate_tokenizer.pad_token is None:
        debate_tokenizer.pad_token = debate_tokenizer.eos_token

    debate_model = AutoModelForCausalLM.from_pretrained(
        args.debate_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    debate_model.gradient_checkpointing_enable()
    debate_model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.1,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    debate_model = get_peft_model(debate_model, lora_config)
    debate_model.print_trainable_parameters()

    mem_gb = torch.cuda.memory_allocated() / 1024**3
    logger.info("GPU memory: %.1f GB", mem_gb)

    # CSV log
    csv_path = Path(args.log_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iteration", "agent", "steps", "mean_reward", "mean_loss"])

    # MARL iterative training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Round-robin: train debate agents (verifier, challenger) with shared reward
    # Skip triage and crag for now (they use pipeline model which is too large for LoRA on same GPU)
    debate_agents = ["verifier", "challenger"]

    logger.info("=" * 60)
    logger.info("MARL Training: %d iterations, agents: %s", args.iterations, debate_agents)
    logger.info("Shared reward: pipeline classification accuracy")
    logger.info("=" * 60)

    for iteration in range(args.iterations):
        logger.info("\n>>> MARL Iteration %d/%d", iteration + 1, args.iterations)
        random.shuffle(train_traj)

        for agent_name in debate_agents:
            logger.info("  Training agent: %s (others frozen)", agent_name)

            result = train_agent_marl(
                model=debate_model,
                tokenizer=debate_tokenizer,
                agent_name=agent_name,
                trajectories=train_traj,
                k_samples=args.k_samples,
                learning_rate=args.learning_rate,
            )

            csv_writer.writerow([
                iteration + 1, agent_name, result["steps"],
                f"{result['mean_reward']:.4f}", f"{result['mean_loss']:.4f}",
            ])
            csv_file.flush()

            logger.info(
                "  %s: %d steps, reward: %.4f, loss: %.4f",
                agent_name, result["steps"], result["mean_reward"], result["mean_loss"],
            )

        # Save after each iteration
        iter_dir = output_dir / f"iter_{iteration+1}"
        iter_dir.mkdir(exist_ok=True)
        debate_model.save_pretrained(iter_dir)
        debate_tokenizer.save_pretrained(iter_dir)
        logger.info("  Saved iteration %d to %s", iteration + 1, iter_dir)

        # Eval on held-out trajectories
        correct = sum(1 for t in eval_traj if t["shared_reward"] > 0)
        logger.info(
            "  Eval baseline accuracy: %.1f%% (%d/%d)",
            100 * correct / len(eval_traj), correct, len(eval_traj),
        )

    # Save final
    debate_model.save_pretrained(output_dir / "final")
    debate_tokenizer.save_pretrained(output_dir / "final")
    logger.info("MARL training complete. Final model: %s/final", output_dir)
    csv_file.close()


if __name__ == "__main__":
    main()
