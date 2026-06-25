"""Compare base PubMedBERT vs fine-tuned PubMedBERT on retrieval quality.

Measures:
1. Recall@K: Of the chunks that were actually cited, how many appear in top-K?
2. MRR (Mean Reciprocal Rank): Where does the first relevant chunk appear?
3. NDCG@K: Normalized discounted cumulative gain (rank-aware relevance)

Usage:
    python scripts/eval_embeddings.py \
        --eval_path results/eval_results_10patients_merged.json \
        --base_model NeuML/pubmedbert-base-embeddings \
        --finetuned_model models/pubmedbert-ehr-finetuned \
        --top_k 15
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np


def load_eval_data(path: str) -> list[dict]:
    """Extract (query, all_chunks, cited_chunk_ids) from eval results."""
    with open(path) as f:
        data = json.load(f)

    examples = []
    for patient in data["patient_results"]:
        for result in patient.get("rag_results", []):
            query = result.get("query", "")
            evidence_pack = result.get("evidence_pack")
            if not query or not evidence_pack:
                continue

            source_chunks = evidence_pack.get("source_chunks", {})
            if not source_chunks or not isinstance(source_chunks, dict):
                continue

            # Cited chunk IDs (ground truth relevant)
            cited_ids = set()
            for citation in result.get("citations", []):
                for span in citation.get("evidence_spans", []):
                    cited_ids.add(span["chunk_id"])

            if not cited_ids:
                continue

            # All chunks with text
            chunks = []
            for chunk_id, chunk_data in source_chunks.items():
                text = chunk_data.get("text", "")
                if text and len(text) > 20:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": text,
                        "is_relevant": chunk_id in cited_ids,
                    })

            if chunks:
                examples.append({
                    "query": query,
                    "chunks": chunks,
                    "num_relevant": len(cited_ids),
                })

    return examples


def compute_metrics(
    model,
    examples: list[dict],
    top_k: int = 15,
) -> dict:
    """Compute retrieval metrics for a given embedding model."""
    recalls = []
    mrrs = []
    ndcgs = []

    for ex in examples:
        query = ex["query"]
        chunks = ex["chunks"]

        # Encode query and chunks
        query_emb = model.encode(query, normalize_embeddings=True)
        chunk_texts = [c["text"] for c in chunks]
        chunk_embs = model.encode(chunk_texts, normalize_embeddings=True)

        # Compute cosine similarity (already normalized = dot product)
        scores = np.dot(chunk_embs, query_emb)

        # Rank by score (descending)
        ranked_indices = np.argsort(-scores)

        # Get relevance labels in ranked order
        relevance = [1 if chunks[i]["is_relevant"] else 0 for i in ranked_indices]

        # Recall@K
        top_k_relevant = sum(relevance[:top_k])
        total_relevant = sum(1 for c in chunks if c["is_relevant"])
        recall = top_k_relevant / max(total_relevant, 1)
        recalls.append(recall)

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, rel in enumerate(relevance):
            if rel == 1:
                mrr = 1.0 / (rank + 1)
                break
        mrrs.append(mrr)

        # NDCG@K
        dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance[:top_k]))
        ideal = sorted(relevance, reverse=True)
        idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal[:top_k]))
        ndcg = dcg / max(idcg, 1e-10)
        ndcgs.append(ndcg)

    return {
        "recall@k": np.mean(recalls),
        "mrr": np.mean(mrrs),
        "ndcg@k": np.mean(ndcgs),
        "num_queries": len(examples),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument("--eval_path", default="results/eval_results_10patients_merged.json")
    parser.add_argument("--base_model", default="NeuML/pubmedbert-base-embeddings")
    parser.add_argument("--finetuned_model", default="models/pubmedbert-ehr-finetuned")
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    # Set offline mode if no internet
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

    from sentence_transformers import SentenceTransformer

    print(f"Loading eval data from {args.eval_path}")
    examples = load_eval_data(args.eval_path)
    print(f"Loaded {len(examples)} queries with relevance labels\n")

    # Base model
    print(f"Loading BASE model: {args.base_model}")
    base_model = SentenceTransformer(args.base_model)
    print("Computing base model metrics...")
    base_metrics = compute_metrics(base_model, examples, top_k=args.top_k)
    del base_model  # free memory

    # Fine-tuned model
    print(f"\nLoading FINE-TUNED model: {args.finetuned_model}")
    ft_model = SentenceTransformer(args.finetuned_model)
    print("Computing fine-tuned model metrics...")
    ft_metrics = compute_metrics(ft_model, examples, top_k=args.top_k)

    # Print comparison
    print("\n" + "=" * 60)
    print(f"EMBEDDING COMPARISON (top_k={args.top_k}, {len(examples)} queries)")
    print("=" * 60)
    print(f"{'Metric':<20} {'Base PubMedBERT':>15} {'Fine-tuned':>15} {'Delta':>10}")
    print("-" * 60)

    for metric in ["recall@k", "mrr", "ndcg@k"]:
        base_val = base_metrics[metric]
        ft_val = ft_metrics[metric]
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<20} {base_val:>15.4f} {ft_val:>15.4f} {sign}{delta:>9.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
