#!/usr/bin/env python3
"""Train the critic agent using DPO (Direct Preference Optimization).

Fine-tunes Qwen 3.5 9B with QLoRA adapters on preference pairs from
EHR evaluation results + MedHallu dataset (~570 pairs).

Requirements (install in Colab):
    pip install torch transformers trl peft datasets bitsandbytes accelerate

Usage:
    python scripts/train_critic_dpo.py [--data_path DATA] [--output_dir OUTPUT]
                                       [--epochs EPOCHS] [--batch_size BS]
                                       [--learning_rate LR] [--lora_r RANK]

Designed to run on Google Colab Pro (T4 16GB or A100 40GB).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_DATA_PATH = "data/dpo_pairs_hf.jsonl"
DEFAULT_OUTPUT_DIR = "models/critic-dpo"

# NOTE: Qwen 3.5 9B is a hybrid architecture (Gated DeltaNet + Sparse MoE).
# LoRA targets the standard projection layers which works fine.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train critic with DPO")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1536)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter (higher = more conservative)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dpo_dataset(data_path: str) -> Dataset:
    """Load preference pairs from JSONL file."""
    records = []
    with open(data_path) as f:
        for line in f:
            record = json.loads(line.strip())
            records.append({
                "prompt": record["prompt"],
                "chosen": record["chosen"],
                "rejected": record["rejected"],
            })

    print(f"Loaded {len(records)} preference pairs from {data_path}")
    dataset = Dataset.from_list(records)

    # Train/eval split (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train'])}, Eval: {len(split['test'])}")
    return split


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_model_and_tokenizer(base_model: str, use_4bit: bool):
    """Load base model with optional 4-bit quantization + LoRA."""
    print(f"\nLoading model: {base_model}")
    print(f"  4-bit quantization: {use_4bit}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for DPO

    # Quantization config
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Print memory usage
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU memory used: {mem_gb:.1f} GB")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    """Run DPO training."""
    print("=" * 60)
    print("CRITIC DPO TRAINING")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    else:
        print("WARNING: No GPU detected. Training will be very slow.")

    use_4bit = args.use_4bit and not args.no_4bit

    # Load data
    dataset_split = load_dpo_dataset(args.data_path)

    # Load model
    model, tokenizer = setup_model_and_tokenizer(args.base_model, use_4bit)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Also need the reference model for DPO
    if use_4bit:
        ref_model = None  # TRL handles this with peft
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available(),
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb
        seed=42,
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"\nStarting DPO training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  DPO beta: {args.beta}")
    print(f"  Output: {output_dir}")

    # Train
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"\n{'=' * 60}")
    print(f"Training complete! Model saved to {final_path}")
    print(f"{'=' * 60}")

    # Log training metrics
    metrics = trainer.state.log_history
    if metrics:
        train_losses = [m["loss"] for m in metrics if "loss" in m]
        eval_losses = [m["eval_loss"] for m in metrics if "eval_loss" in m]
        if train_losses:
            print(f"  Final train loss: {train_losses[-1]:.4f}")
        if eval_losses:
            print(f"  Final eval loss: {eval_losses[-1]:.4f}")

    return str(final_path)


if __name__ == "__main__":
    args = parse_args()
    train(args)
