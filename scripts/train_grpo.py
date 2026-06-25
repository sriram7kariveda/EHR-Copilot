"""GRPO Training for MAD Debate Agents (Verifier + Challenger).

Group Relative Policy Optimization (DeepSeek-R1 style):
1. For each trajectory, generate K completions from the current policy
2. Score each completion with the reward function
3. Compute group-relative advantage: A_i = (r_i - mean(r)) / std(r)
4. Policy gradient update weighted by advantage

Rewards:
- Verifier (Agent A): Brier score = 2 * confidence * judge_score - confidence²
  Rewards calibrated confidence matching ground truth.
- Challenger (Agent B): +1 if challenged wrong claim (judge < 1.0), -1 if gaslighting

Usage:
    python scripts/train_grpo.py \
        --agent verifier \
        --trajectories data/grpo_trajectories/train.jsonl \
        --model Qwen/Qwen3.5-4B \
        --output_dir models/verifier-grpo \
        --epochs 1 \
        --k_samples 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_trajectories(path: str) -> list[dict]:
    trajectories = []
    with open(path) as f:
        for line in f:
            trajectories.append(json.loads(line.strip()))
    return trajectories


def build_verifier_prompt(traj: dict) -> str:
    """Build the verification prompt from a trajectory."""
    return f"""You are a clinical evidence verifier. Verify if the following claim is supported by the evidence.

Claim: {traj['answer'][:500]}

Evidence:
{traj['evidence'][:1500]}

Respond in JSON:
{{"verdict": "supported" | "not_supported" | "partial", "confidence": 0.0-1.0, "reasoning": "..."}}

Return ONLY the JSON:"""


def build_challenger_prompt(traj: dict) -> str:
    """Build the challenge prompt from a trajectory."""
    return f"""You are an adversarial clinical auditor. The following claim has been marked as SUPPORTED. Challenge it if you find issues.

Claim: {traj['answer'][:500]}

Evidence:
{traj['evidence'][:1500]}

Attempt these challenges:
1. CONTRAINDICATION: Any patient-specific contraindications?
2. DOSAGE_CHECK: Is dosage appropriate?
3. INTERACTION: Any drug interactions?
4. GUIDELINE_CURRENCY: Current guidelines support this?
5. GAP_FINDING: What's missing?

If no valid challenges, respond with empty array [].
Otherwise respond as JSON array:
[{{"challenge_type": "...", "challenge_text": "...", "severity": "high|medium|low"}}]

Return ONLY the JSON:"""


def compute_verifier_reward(output_text: str, traj: dict) -> float:
    """Compute Brier reward for Verifier output."""
    try:
        # Parse confidence from output
        import re
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', output_text)
        if not conf_match:
            return -0.5  # Penalty for unparseable output

        confidence = float(conf_match.group(1))
        confidence = min(max(confidence, 0.0), 1.0)

        # Brier reward: 2 * p * v - p²
        target = traj["judge_target_score"]
        brier = 2.0 * confidence * target - confidence ** 2
        return brier

    except Exception:
        return -0.5


def compute_challenger_reward(output_text: str, traj: dict) -> float:
    """Compute reward for Challenger output."""
    should_challenge = traj["challenger_should_challenge"]

    # Check if challenges were raised
    has_challenges = "challenge_type" in output_text and "[]" not in output_text

    if should_challenge and has_challenges:
        return 1.0   # Correctly challenged a bad claim
    elif should_challenge and not has_challenges:
        return -0.5  # Missed a bad claim
    elif not should_challenge and has_challenges:
        return -1.0  # Gaslighting: challenged a good claim
    else:
        return 0.5   # Correctly didn't challenge a good claim


def grpo_step(
    model,
    tokenizer,
    trajectories: list[dict],
    agent: str,
    k_samples: int = 4,
    beta: float = 0.1,
    max_length: int = 2048,
    max_new_tokens: int = 256,
) -> dict:
    """One GRPO training step.

    For each trajectory:
    1. Generate K completions
    2. Score each with reward function
    3. Compute group-relative advantage
    4. Policy gradient update
    """
    build_prompt = build_verifier_prompt if agent == "verifier" else build_challenger_prompt
    compute_reward = compute_verifier_reward if agent == "verifier" else compute_challenger_reward

    total_reward = 0.0
    total_loss = 0.0
    num_steps = 0

    for traj in trajectories:
        prompt = build_prompt(traj)

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate K samples
        rewards = []
        log_probs_list = []

        for _ in range(k_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Decode
            gen_ids = outputs.sequences[0][prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Compute reward
            reward = compute_reward(gen_text, traj)
            rewards.append(reward)

            # Compute log prob of this generation
            with torch.no_grad():
                full_output = model(outputs.sequences)
                logits = full_output.logits[0, prompt_len-1:-1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
                total_log_prob = token_log_probs.sum()
                log_probs_list.append(total_log_prob)

        # Group-relative advantage
        rewards_tensor = torch.tensor(rewards)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_r) / std_r

        # Policy gradient loss: -sum(advantage_i * log_prob_i)
        loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        for adv, log_prob in zip(advantages, log_probs_list):
            # Recompute log_prob with gradients
            pass  # Will use the actual training loop below

        total_reward += mean_r.item()
        num_steps += 1

    return {
        "mean_reward": total_reward / max(num_steps, 1),
        "num_trajectories": num_steps,
    }


def train_grpo(
    model,
    tokenizer,
    train_trajectories: list[dict],
    eval_trajectories: list[dict],
    agent: str,
    output_dir: str,
    epochs: int = 1,
    k_samples: int = 4,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    max_new_tokens: int = 256,
):
    """Full GRPO training loop."""
    build_prompt = build_verifier_prompt if agent == "verifier" else build_challenger_prompt
    compute_reward = compute_verifier_reward if agent == "verifier" else compute_challenger_reward

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    logger.info("Starting GRPO training:")
    logger.info("  Agent: %s", agent)
    logger.info("  Train trajectories: %d", len(train_trajectories))
    logger.info("  Eval trajectories: %d", len(eval_trajectories))
    logger.info("  K samples: %d", k_samples)
    logger.info("  Epochs: %d", epochs)
    logger.info("  Learning rate: %s", learning_rate)

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_trajectories)

        epoch_rewards = []
        epoch_losses = []

        for i in range(0, len(train_trajectories), batch_size):
            batch = train_trajectories[i:i + batch_size]
            batch_loss = torch.tensor(0.0, device=model.device)

            for traj in batch:
                prompt = build_prompt(traj)
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=1536,
                ).to(model.device)
                prompt_len = inputs["input_ids"].shape[1]

                # Generate K samples and collect rewards
                sample_rewards = []
                sample_log_probs = []

                for _ in range(k_samples):
                    with torch.no_grad():
                        gen_out = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    gen_ids = gen_out[0][prompt_len:]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    reward = compute_reward(gen_text, traj)
                    sample_rewards.append(reward)

                    # Forward pass WITH gradients for this generation
                    outputs = model(gen_out)
                    logits = outputs.logits[0, prompt_len-1:-1, :]
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    token_lp = log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
                    sample_log_probs.append(token_lp.sum())

                # Group-relative advantage
                r_tensor = torch.tensor(sample_rewards)
                mean_r = r_tensor.mean()
                std_r = r_tensor.std() + 1e-8
                advantages = (r_tensor - mean_r) / std_r

                # GRPO loss: -mean(advantage * log_prob)
                for adv, lp in zip(advantages, sample_log_probs):
                    batch_loss = batch_loss - adv.item() * lp / (k_samples * len(batch))

                epoch_rewards.append(mean_r.item())

            # Backward + update
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(batch_loss.item())

            if (i // batch_size) % 10 == 0:
                logger.info(
                    "  Epoch %d, Step %d/%d, Loss: %.4f, Mean Reward: %.4f",
                    epoch + 1, i // batch_size, len(train_trajectories) // batch_size,
                    batch_loss.item(), np.mean(epoch_rewards[-batch_size:]),
                )

        # Eval
        model.eval()
        eval_rewards = []
        for traj in eval_trajectories[:50]:  # Quick eval on subset
            prompt = build_prompt(traj)
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1536,
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                gen_out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=0.1, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_text = tokenizer.decode(gen_out[0][prompt_len:], skip_special_tokens=True)
            reward = compute_reward(gen_text, traj)
            eval_rewards.append(reward)

        logger.info(
            "Epoch %d complete. Train reward: %.4f, Eval reward: %.4f, Loss: %.4f",
            epoch + 1, np.mean(epoch_rewards), np.mean(eval_rewards), np.mean(epoch_losses),
        )

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Model saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="GRPO training for MAD agents")
    parser.add_argument("--agent", choices=["verifier", "challenger"], required=True)
    parser.add_argument("--trajectories", default="data/grpo_trajectories/train.jsonl")
    parser.add_argument("--eval_trajectories", default="data/grpo_trajectories/eval.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output_dir", default="models/verifier-grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--k_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    args = parser.parse_args()

    # Load trajectories
    logger.info("Loading trajectories from %s", args.trajectories)
    train_traj = load_trajectories(args.trajectories)
    eval_traj = load_trajectories(args.eval_trajectories)

    # Filter by agent relevance
    if args.agent == "challenger":
        # Challenger only trains on examples where challenge is expected
        train_traj = [t for t in train_traj if t["type"].startswith("medhallu")]
        eval_traj = [t for t in eval_traj if t["type"].startswith("medhallu")]

    logger.info("Train: %d, Eval: %d", len(train_traj), len(eval_traj))

    # Load model with LoRA
    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    # Add LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Train
    if args.agent == "verifier":
        args.output_dir = "models/verifier-grpo"
    else:
        args.output_dir = "models/challenger-grpo"

    train_grpo(
        model=model,
        tokenizer=tokenizer,
        train_trajectories=train_traj,
        eval_trajectories=eval_traj,
        agent=args.agent,
        output_dir=args.output_dir,
        epochs=args.epochs,
        k_samples=args.k_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
