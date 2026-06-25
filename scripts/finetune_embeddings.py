"""Fine-tune PubMedBERT embeddings on clinical EHR query-chunk triplets.

Uses sentence-transformers with MultipleNegativesRankingLoss (InfoNCE)
to train the embedding model to rank relevant clinical chunks higher
than irrelevant ones for EHR queries.

Usage:
    python scripts/finetune_embeddings.py \
        --triplets_path data/embedding_triplets.jsonl \
        --base_model NeuML/pubmedbert-base-embeddings \
        --output_dir models/pubmedbert-ehr-finetuned \
        --epochs 5 \
        --batch_size 16

For HPC:
    sbatch hpc/slurm_finetune_embeddings.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_triplets(path: str) -> list[dict]:
    """Load triplets from JSONL file."""
    triplets = []
    with open(path) as f:
        for line in f:
            triplets.append(json.loads(line.strip()))
    return triplets


def build_train_eval_split(
    triplets: list[dict],
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split triplets into train and eval sets."""
    import random
    random.seed(seed)
    shuffled = triplets.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - eval_fraction))
    return shuffled[:split_idx], shuffled[split_idx:]


def triplets_to_examples(triplets: list[dict]) -> list[InputExample]:
    """Convert triplets to sentence-transformers InputExamples.

    For MultipleNegativesRankingLoss, we provide (anchor, positive) pairs.
    The loss function automatically uses other positives in the batch as
    in-batch negatives. We also include the hard negative as a separate pair.
    """
    examples = []
    for t in triplets:
        # Primary pair: query → positive chunk
        examples.append(InputExample(
            texts=[t["query"], t["positive"]],
        ))
    return examples


def build_evaluator(
    eval_triplets: list[dict],
) -> evaluation.TripletEvaluator:
    """Build a TripletEvaluator from eval triplets."""
    anchors = [t["query"] for t in eval_triplets]
    positives = [t["positive"] for t in eval_triplets]
    negatives = [t["negative"] for t in eval_triplets]
    return evaluation.TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name="ehr-triplet-eval",
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PubMedBERT embeddings")
    parser.add_argument(
        "--triplets_path",
        default="data/embedding_triplets.jsonl",
        help="Path to training triplets JSONL",
    )
    parser.add_argument(
        "--base_model",
        default="NeuML/pubmedbert-base-embeddings",
        help="Base embedding model to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        default="models/pubmedbert-ehr-finetuned",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--eval_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load triplets.
    logger.info("Loading triplets from %s", args.triplets_path)
    triplets = load_triplets(args.triplets_path)
    logger.info("Loaded %d triplets", len(triplets))

    # Split.
    train_triplets, eval_triplets = build_train_eval_split(
        triplets, eval_fraction=args.eval_fraction, seed=args.seed,
    )
    logger.info("Train: %d, Eval: %d", len(train_triplets), len(eval_triplets))

    # Load model.
    logger.info("Loading base model: %s", args.base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.base_model, device=device)
    logger.info("Model loaded on %s. Embedding dim: %d", device, model.get_sentence_embedding_dimension())

    # Build DataLoader.
    train_examples = triplets_to_examples(train_triplets)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    # Loss: MultipleNegativesRankingLoss (InfoNCE / contrastive loss).
    # Standard loss for retrieval embedding fine-tuning.
    # Uses in-batch negatives: each positive in the batch acts as a negative
    # for all other queries. Much more effective than TripletLoss.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Evaluator.
    evaluator = build_evaluator(eval_triplets)

    # Training.
    num_training_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(num_training_steps * args.warmup_ratio)

    logger.info("Starting training:")
    logger.info("  Epochs: %d", args.epochs)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Training steps: %d", num_training_steps)
    logger.info("  Warmup steps: %d", warmup_steps)
    logger.info("  Learning rate: %s", args.learning_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(output_dir),
        evaluation_steps=len(train_dataloader) // 2,  # eval twice per epoch
        save_best_model=True,
        show_progress_bar=True,
    )

    logger.info("Training complete. Model saved to %s", output_dir)

    # Final evaluation.
    final_score = evaluator(model)
    logger.info("Final eval score: %.4f", final_score)


if __name__ == "__main__":
    main()
