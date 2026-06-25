"""Generate training triplets for PubMedBERT embedding fine-tuning.

Extracts (query, positive_chunk, hard_negative_chunk) triplets from
pipeline evaluation results. Positive = chunks cited in the answer.
Hard negative = chunks retrieved but NOT cited (looked relevant to
the retriever but weren't actually useful).

Usage:
    python scripts/generate_embedding_triplets.py \
        --eval_path results/eval_results_10patients_merged.json \
        --output_path data/embedding_triplets.jsonl \
        --negatives_per_positive 5
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def extract_triplets(
    eval_path: str,
    negatives_per_positive: int = 5,
) -> list[dict]:
    """Extract training triplets from pipeline evaluation results.

    For each query, cited chunks are positives and retrieved-but-not-cited
    chunks are hard negatives. This produces triplets in the format
    expected by sentence-transformers MultipleNegativesRankingLoss.
    """
    with open(eval_path) as f:
        data = json.load(f)

    triplets = []
    skipped = 0

    for patient in data["patient_results"]:
        for result in patient.get("rag_results", []):
            query_text = result.get("query", "")
            if not query_text:
                skipped += 1
                continue

            # Get evidence pack.
            evidence_pack = result.get("evidence_pack")
            if not evidence_pack:
                skipped += 1
                continue

            source_chunks = evidence_pack.get("source_chunks", {})
            if not source_chunks or not isinstance(source_chunks, dict):
                skipped += 1
                continue

            # Collect cited chunk IDs (positives).
            cited_ids: set[str] = set()
            for citation in result.get("citations", []):
                for span in citation.get("evidence_spans", []):
                    cited_ids.add(span["chunk_id"])

            # Build positive and negative chunk lists.
            positives: list[dict] = []
            negatives: list[dict] = []

            for chunk_id, chunk_data in source_chunks.items():
                chunk_text = chunk_data.get("text", "")
                if not chunk_text or len(chunk_text) < 20:
                    continue

                if chunk_id in cited_ids:
                    positives.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    })
                else:
                    negatives.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    })

            if not positives or not negatives:
                skipped += 1
                continue

            # Generate triplets: for each positive, sample N hard negatives.
            for pos in positives:
                sampled_negs = random.sample(
                    negatives,
                    min(negatives_per_positive, len(negatives)),
                )
                for neg in sampled_negs:
                    triplets.append({
                        "query": query_text,
                        "positive": pos["text"],
                        "negative": neg["text"],
                        "positive_chunk_id": pos["chunk_id"],
                        "negative_chunk_id": neg["chunk_id"],
                    })

    print(f"Extracted {len(triplets)} triplets from {len(data['patient_results'])} patients")
    print(f"Skipped {skipped} queries (missing data)")

    return triplets


def main():
    parser = argparse.ArgumentParser(description="Generate embedding training triplets")
    parser.add_argument(
        "--eval_path",
        default="results/eval_results_10patients_merged.json",
        help="Path to pipeline evaluation results",
    )
    parser.add_argument(
        "--output_path",
        default="data/embedding_triplets.jsonl",
        help="Output JSONL path for triplets",
    )
    parser.add_argument(
        "--negatives_per_positive",
        type=int,
        default=5,
        help="Number of hard negatives per positive (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    triplets = extract_triplets(
        eval_path=args.eval_path,
        negatives_per_positive=args.negatives_per_positive,
    )

    # Shuffle and write.
    random.shuffle(triplets)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(triplets)} triplets to {output_path}")

    # Also write sentence-transformers compatible format (anchor, positive, negative).
    st_path = output_path.with_suffix(".st.jsonl")
    with open(st_path, "w") as f:
        for t in triplets:
            f.write(json.dumps({
                "anchor": t["query"],
                "positive": t["positive"],
                "negative": t["negative"],
            }) + "\n")
    print(f"Wrote sentence-transformers format to {st_path}")


if __name__ == "__main__":
    main()
