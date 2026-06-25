"""GRPO v3: Detection-Aligned Reward.

Key change from v2: reward directly measures what we evaluate (detection accuracy),
not confidence calibration (Brier score). This follows MMOA-RAG's approach of
training on the same metric you evaluate on.

Reward:
- Hallucinated answer: correctly flagged = +1.0, missed = -1.0
- Ground truth answer: correctly approved = +1.0, wrongly flagged = -0.5
- Confidence bonus: high confidence correct = +0.3, low confidence correct = +0.1
- Format bonus: valid JSON = +0.2, invalid = -0.2

Usage:
    python scripts/train_grpo_v3.py --agent verifier --k_samples 4
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


def build_verifier_prompt(traj: dict) -> str:
    return f"""You are a STRICT clinical evidence verifier. Check if this answer is supported by evidence.

Answer: {traj['answer'][:500]}

Evidence:
{traj['evidence'][:1500]}

Respond in JSON:
{{"verdict": "supported" | "not_supported" | "partial", "confidence": 0.0-1.0, "reasoning": "brief"}}

RULES:
- "supported" ONLY if evidence EXPLICITLY states the same fact
- "not_supported" if the answer adds facts NOT in evidence or contradicts it
- When in doubt, say "not_supported" — patient safety first

Return ONLY JSON:"""


def build_challenger_prompt(traj: dict) -> str:
    return f"""You are an adversarial clinical auditor. Challenge this answer if you find issues.

Answer: {traj['answer'][:500]}

Evidence:
{traj['evidence'][:1500]}

If issues found, respond: [{{"challenge_type": "...", "challenge_text": "..."}}]
If no issues, respond: []
Return ONLY JSON:"""


def compute_detection_reward(output_text: str, traj: dict, agent: str) -> float:
    """Detection-aligned reward: directly measures what we evaluate (detection accuracy).

    For hallucinated answers: agent should flag them (NOT_SUPPORTED / has challenges)
    For ground truth answers: agent should approve them (SUPPORTED / no challenges)
    """
    is_hallucinated = traj.get("type", "").endswith("hallucinated") or traj.get("expected_verdict") == "not_supported"
    target_score = traj.get("judge_target_score", 0.5)
    is_hallucinated = target_score < 0.5  # 0.0 = hallucinated, 1.0 = ground truth

    if agent == "verifier":
        # Parse verdict
        verdict_match = re.search(r'"verdict"\s*:\s*"(\w+)"', output_text)
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', output_text)

        if not verdict_match:
            return -0.5  # Can't parse = bad

        verdict = verdict_match.group(1).lower()
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        confidence = min(max(confidence, 0.0), 1.0)

        # Detection reward
        flagged = verdict in ("not_supported", "partial")

        if is_hallucinated and flagged:
            reward = 1.0  # Correctly caught hallucination
            reward += 0.3 * (1 - confidence)  # Bonus for low confidence on bad answer
        elif is_hallucinated and not flagged:
            reward = -1.0  # Missed hallucination (worst case)
        elif not is_hallucinated and not flagged:
            reward = 1.0  # Correctly approved good answer
            reward += 0.3 * confidence  # Bonus for high confidence on good answer
        elif not is_hallucinated and flagged:
            reward = -0.5  # False positive (bad but less bad than missing hallucination)

        # Format bonus
        try:
            json.loads(output_text.strip().split("\n")[0])
            reward += 0.1
        except Exception:
            reward -= 0.1

        return reward

    elif agent == "challenger":
        has_challenges = "challenge_type" in output_text and "[]" not in output_text

        if is_hallucinated and has_challenges:
            reward = 1.0  # Correctly challenged bad answer
        elif is_hallucinated and not has_challenges:
            reward = -1.0  # Missed bad answer
        elif not is_hallucinated and not has_challenges:
            reward = 1.0  # Correctly left good answer alone
        elif not is_hallucinated and has_challenges:
            reward = -0.5  # False alarm

        # Format bonus
        try:
            parsed = json.loads(output_text.strip())
            if isinstance(parsed, list):
                reward += 0.1
        except Exception:
            reward -= 0.1

        return reward

    return 0.0


def run_eval(model, tokenizer, eval_traj, agent, max_new_tokens=128):
    """Eval: compute mean detection reward on held-out set."""
    model.eval()
    build_prompt = build_verifier_prompt if agent == "verifier" else build_challenger_prompt
    rewards = []

    for traj in eval_traj[:100]:
        prompt = build_prompt(traj)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rewards.append(compute_detection_reward(text, traj, agent))

    model.train()
    return np.mean(rewards) if rewards else 0.0


def train(
    model, tokenizer, train_traj, eval_traj, agent, output_dir,
    k_samples=4, grad_accum=4, learning_rate=1e-5, max_new_tokens=128,
    eval_every=50, patience=5, log_csv="logs/grpo_v3.csv",
):
    build_prompt = build_verifier_prompt if agent == "verifier" else build_challenger_prompt
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    csv_path = Path(log_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_reward", "eval_reward", "loss"])

    # Initial eval
    eval_reward = run_eval(model, tokenizer, eval_traj, agent)
    logger.info("Initial eval reward: %.4f", eval_reward)
    best_eval = eval_reward
    patience_counter = 0
    global_step = 0
    start = time.time()
    accum_rewards = []
    optimizer.zero_grad()

    model.train()
    random.shuffle(train_traj)

    for i, traj in enumerate(train_traj):
        prompt = build_prompt(traj)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
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
            reward = compute_detection_reward(text, traj, agent)
            sample_rewards.append(reward)
            sample_gen_ids.append(gen_ids)
        model.train()

        if len(sample_rewards) < 2:
            continue

        r_tensor = torch.tensor(sample_rewards)
        std_r = r_tensor.std()
        if std_r < 1e-8:
            accum_rewards.append(r_tensor.mean().item())
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
        accum_rewards.append(r_tensor.mean().item())

        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 5 == 0:
                logger.info("  Step %d | Loss: %.4f | Reward: %.4f | Time: %.0fs",
                            global_step, traj_loss.item(), np.mean(accum_rewards[-grad_accum:]), time.time() - start)

            if global_step % eval_every == 0:
                eval_reward = run_eval(model, tokenizer, eval_traj, agent)
                avg_train = np.mean(accum_rewards[-eval_every*grad_accum:]) if accum_rewards else 0
                logger.info("  >>> EVAL Step %d | Train: %.4f | Eval: %.4f | Best: %.4f | Patience: %d/%d",
                            global_step, avg_train, eval_reward, best_eval, patience_counter, patience)

                csv_writer.writerow([global_step, f"{avg_train:.4f}", f"{eval_reward:.4f}", f"{traj_loss.item():.4f}"])
                csv_file.flush()

                if eval_reward > best_eval + 0.01:
                    best_eval = eval_reward
                    patience_counter = 0
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info("  >>> NEW BEST! Saved to %s", output_dir)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("  >>> EARLY STOPPING at step %d", global_step)
                        csv_file.close()
                        return

    # Final save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete. Saved to %s", output_dir)
    csv_file.close()


def main():
    parser = argparse.ArgumentParser(description="GRPO v3 — Detection-Aligned Reward")
    parser.add_argument("--agent", choices=["verifier", "challenger"], required=True)
    parser.add_argument("--trajectories", default="data/grpo_trajectories/train.jsonl")
    parser.add_argument("--eval_trajectories", default="data/grpo_trajectories/eval.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--k_samples", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train_traj = load_trajectories(args.trajectories)
    eval_traj = load_trajectories(args.eval_trajectories)

    if args.agent == "challenger":
        train_traj = [t for t in train_traj if t["type"].startswith("medhallu")]
        eval_traj = [t for t in eval_traj if t["type"].startswith("medhallu")]

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
    logger.info("GPU memory: %.1f GB", torch.cuda.memory_allocated() / 1024**3)

    output_dir = f"models/{args.agent}-grpo-v3"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train(model, tokenizer, train_traj, eval_traj, args.agent, output_dir,
          k_samples=args.k_samples, grad_accum=args.grad_accum,
          learning_rate=args.learning_rate, eval_every=args.eval_every,
          patience=args.patience, log_csv=f"logs/grpo_{args.agent}_v3.csv")


if __name__ == "__main__":
    main()
