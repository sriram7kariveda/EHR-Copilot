"""GRPO v2 Training — QLoRA + k=4 + gradient accumulation + early stopping.

Improvements over v1:
- QLoRA (4-bit quantization): model ~2GB instead of ~6GB → k=4 fits on 40GB A100
- Gradient accumulation: effective batch without OOM
- Early stopping: stop if eval reward doesn't improve for N steps
- CSV logging: monitor training progress in real-time
- Eval every 50 steps (not just end of epoch)

Usage:
    python scripts/train_grpo_v2.py \
        --agent verifier \
        --model Qwen/Qwen2.5-3B-Instruct \
        --k_samples 4 \
        --grad_accum 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_trajectories(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line.strip()) for line in f]


def build_verifier_prompt(traj: dict) -> str:
    return f"""You are a clinical evidence verifier. Verify if the following claim is supported by the evidence.

Claim: {traj['answer'][:500]}

Evidence:
{traj['evidence'][:1500]}

Respond in JSON:
{{"verdict": "supported" | "not_supported" | "partial", "confidence": 0.0-1.0, "reasoning": "..."}}

Return ONLY the JSON:"""


def build_challenger_prompt(traj: dict) -> str:
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
    try:
        import re
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', output_text)
        if not conf_match:
            return -0.5
        confidence = min(max(float(conf_match.group(1)), 0.0), 1.0)
        target = traj["judge_target_score"]
        return 2.0 * confidence * target - confidence ** 2
    except Exception:
        return -0.5


def compute_challenger_reward(output_text: str, traj: dict) -> float:
    should_challenge = traj["challenger_should_challenge"]
    has_challenges = "challenge_type" in output_text and "[]" not in output_text
    if should_challenge and has_challenges:
        return 1.0
    elif should_challenge and not has_challenges:
        return -0.5
    elif not should_challenge and has_challenges:
        return -1.0
    else:
        return 0.5


def run_eval(model, tokenizer, eval_trajectories, build_prompt, compute_reward, max_new_tokens=256):
    """Quick eval on subset."""
    model.eval()
    rewards = []
    for traj in eval_trajectories[:50]:
        prompt = build_prompt(traj)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen_out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_text = tokenizer.decode(gen_out[0][prompt_len:], skip_special_tokens=True)
        rewards.append(compute_reward(gen_text, traj))
    model.train()
    return np.mean(rewards) if rewards else 0.0


def train_grpo_v2(
    model, tokenizer, train_traj, eval_traj, agent, output_dir,
    epochs=1, k_samples=4, batch_size=1, grad_accum=4,
    learning_rate=1e-5, max_new_tokens=128, eval_every=50,
    patience=3, log_csv="logs/grpo_training_log.csv",
):
    build_prompt = build_verifier_prompt if agent == "verifier" else build_challenger_prompt
    compute_reward = compute_verifier_reward if agent == "verifier" else compute_challenger_reward

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    # CSV logger
    csv_path = Path(log_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_reward", "eval_reward", "loss", "time_s", "unique_rewards"])

    logger.info("=" * 60)
    logger.info("GRPO v2 Training")
    logger.info("=" * 60)
    logger.info("  Agent: %s", agent)
    logger.info("  Model: QLoRA 4-bit")
    logger.info("  Train: %d, Eval: %d", len(train_traj), len(eval_traj))
    logger.info("  k_samples: %d", k_samples)
    logger.info("  batch_size: %d, grad_accum: %d (effective: %d)", batch_size, grad_accum, batch_size * grad_accum)
    logger.info("  Eval every: %d steps", eval_every)
    logger.info("  Early stopping patience: %d evals", patience)
    logger.info("  CSV log: %s", csv_path)
    logger.info("=" * 60)

    # Initial eval
    eval_reward = run_eval(model, tokenizer, eval_traj, build_prompt, compute_reward)
    logger.info("Initial eval reward: %.4f", eval_reward)
    best_eval_reward = eval_reward
    patience_counter = 0
    global_step = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_traj)
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_rewards = []

        for i, traj in enumerate(train_traj):
            prompt = build_prompt(traj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            # Generate K samples (disable grad checkpoint for fast generation)
            sample_rewards = []
            sample_gen_ids = []

            model.eval()
            if hasattr(model, 'disable_adapter_layers'):
                pass  # keep adapters active
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
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                reward = compute_reward(gen_text, traj)
                sample_rewards.append(reward)
                sample_gen_ids.append(gen_ids)

            model.train()

            # Now do forward passes WITH gradients (one at a time to save memory)
            sample_log_probs = []
            for gen_ids in sample_gen_ids:
                full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)
                outputs = model(full_ids)
                logits = outputs.logits[0, prompt_len-1:-1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_lp = log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
                sample_log_probs.append(token_lp.sum())

            if len(sample_rewards) < 2:
                continue

            # Group-relative advantage
            r_tensor = torch.tensor(sample_rewards)
            mean_r = r_tensor.mean()
            std_r = r_tensor.std()
            unique_rewards = len(set(sample_rewards))

            if std_r < 1e-8:
                # All rewards same → no gradient signal, skip
                accum_rewards.append(mean_r.item())
                continue

            advantages = (r_tensor - mean_r) / (std_r + 1e-8)

            # GRPO loss
            traj_loss = torch.tensor(0.0, device=model.device)
            for adv, lp in zip(advantages, sample_log_probs):
                traj_loss = traj_loss - adv.item() * lp / (k_samples * grad_accum)

            traj_loss.backward()
            accum_loss += traj_loss.item()
            accum_rewards.append(mean_r.item())

            # Gradient accumulation step
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log every step
                if True:
                    avg_reward = np.mean(accum_rewards[-grad_accum:]) if accum_rewards else 0
                    elapsed = time.time() - start_time
                    logger.info(
                        "  Step %d/%d | Loss: %.4f | Reward: %.4f | Unique_k: %d | Time: %.0fs",
                        global_step, len(train_traj) // grad_accum,
                        accum_loss / grad_accum, avg_reward, unique_rewards, elapsed,
                    )

                # Eval + early stopping
                if global_step % eval_every == 0:
                    eval_reward = run_eval(model, tokenizer, eval_traj, build_prompt, compute_reward)
                    elapsed = time.time() - start_time
                    avg_train = np.mean(accum_rewards[-eval_every:]) if len(accum_rewards) >= eval_every else np.mean(accum_rewards)

                    logger.info(
                        "  >>> EVAL Step %d | Train: %.4f | Eval: %.4f | Best: %.4f | Patience: %d/%d",
                        global_step, avg_train, eval_reward, best_eval_reward, patience_counter, patience,
                    )

                    csv_writer.writerow([global_step, f"{avg_train:.4f}", f"{eval_reward:.4f}", f"{accum_loss/grad_accum:.4f}", f"{elapsed:.0f}", unique_rewards])
                    csv_file.flush()

                    if eval_reward > best_eval_reward + 0.01:
                        best_eval_reward = eval_reward
                        patience_counter = 0
                        # Save best model
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        logger.info("  >>> NEW BEST! Saved to %s", output_dir)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info("  >>> EARLY STOPPING at step %d (no improvement for %d evals)", global_step, patience)
                            csv_file.close()
                            return

                accum_loss = 0.0

        # End of epoch eval
        eval_reward = run_eval(model, tokenizer, eval_traj, build_prompt, compute_reward)
        logger.info(
            "Epoch %d complete. Train: %.4f, Eval: %.4f, Best: %.4f",
            epoch + 1, np.mean(accum_rewards), eval_reward, best_eval_reward,
        )

    # Final save if not already saved
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Training complete. Model saved to %s", output_path)
    csv_file.close()


def main():
    parser = argparse.ArgumentParser(description="GRPO v2 — QLoRA + k=4 + early stopping")
    parser.add_argument("--agent", choices=["verifier", "challenger"], required=True)
    parser.add_argument("--trajectories", default="data/grpo_trajectories/train.jsonl")
    parser.add_argument("--eval_trajectories", default="data/grpo_trajectories/eval.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", default="models/verifier-grpo-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--k_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--use_qlora", action="store_true", default=False)
    args = parser.parse_args()

    # Load trajectories
    train_traj = load_trajectories(args.trajectories)
    eval_traj = load_trajectories(args.eval_trajectories)

    if args.agent == "challenger":
        train_traj = [t for t in train_traj if t["type"].startswith("medhallu")]
        eval_traj = [t for t in eval_traj if t["type"].startswith("medhallu")]

    logger.info("Train: %d, Eval: %d", len(train_traj), len(eval_traj))

    # Load model with QLoRA (4-bit)
    logger.info("Loading model: %s (QLoRA 4-bit)", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()

    model.config.use_cache = False

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.1,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    mem_gb = torch.cuda.memory_allocated() / 1024**3
    logger.info("GPU memory after model load: %.1f GB", mem_gb)

    # Output dir
    if args.agent == "verifier":
        args.output_dir = "models/verifier-grpo-v2"
    else:
        args.output_dir = "models/challenger-grpo-v2"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_grpo_v2(
        model=model, tokenizer=tokenizer,
        train_traj=train_traj, eval_traj=eval_traj,
        agent=args.agent, output_dir=args.output_dir,
        epochs=args.epochs, k_samples=args.k_samples,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every, patience=args.patience,
        log_csv=f"logs/grpo_{args.agent}_v2.csv",
    )


if __name__ == "__main__":
    main()
