#!/usr/bin/env python3
"""Multi-model benchmark: One-shot with 5 top models vs our RAG pipeline.

Runs all models IN PARALLEL per query to minimize wall-clock time.
Uses cached embedding model for faithfulness (no LLM cost for that metric).

Usage:
    uv run python scripts/run_multi_model_benchmark.py --num-patients 3
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MIMIC_DATA_DIR = "data/mimic-fhir/mimic-iv-clinical-database-demo-on-fhir-2.1.0/fhir"

JUDGE_MODEL = "anthropic/claude-3.5-haiku"

BENCHMARK_MODELS = {
    "GPT-5": "openai/gpt-5",
    "Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "Gemini 3 Pro": "google/gemini-3-pro-preview",
    "GLM 4.6": "z-ai/glm-4.6",
    "Haiku 3.5": "anthropic/claude-3.5-haiku",
}

EVAL_QUERIES = [
    {"query": "What are the patient's diagnoses from their most recent encounter?", "type": "FACTUAL", "answerable": True},
    {"query": "What medications is this patient currently prescribed?", "type": "MEDICATION", "answerable": True},
    {"query": "What are the most recent lab results for this patient?", "type": "FACTUAL", "answerable": True},
    {"query": "What procedures has this patient undergone?", "type": "FACTUAL", "answerable": True},
    {"query": "Summarize the patient's clinical history across all encounters.", "type": "SUMMARY", "answerable": True},
    {"query": "What is the patient's genetic risk for Alzheimer's disease?", "type": "REASONING", "answerable": False},
    {"query": "What imaging studies has the patient had and what were the findings?", "type": "FACTUAL", "answerable": True},
    {"query": "Has the patient's kidney function changed over time?", "type": "TEMPORAL", "answerable": True},
]

ONE_SHOT_PROMPT = """You are a clinical assistant. Answer the following question about a patient based ONLY on the provided clinical data. If the information is not available, say so clearly.

## Patient Clinical Data
{context}

## Question
{query}

## Instructions
- Answer based ONLY on the data provided above
- Be specific with dates, values, and units
- If the data does not contain enough information to answer, say "Insufficient data"
"""

JUDGE_PROMPT = """Rate the quality of this clinical answer on a scale of 1-5.

## Question
{query}

## Answer
{answer}

## Rating Criteria
1 = Completely wrong, hallucinated, or nonsensical
2 = Partially relevant but mostly inaccurate or vague
3 = Somewhat relevant and partially accurate
4 = Mostly accurate with minor issues
5 = Accurate, specific, well-supported, and clinically useful

Respond with ONLY a single integer (1-5), nothing else."""


# ---------------------------------------------------------------------------
# Shared client + cached embedding model
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from ehr_copilot.config import get_settings
        from ehr_copilot.indexing.embedding import EmbeddingModel
        settings = get_settings()
        _embedding_model = EmbeddingModel(settings.embedding)
        print("  [Embedding model loaded]")
    return _embedding_model


async def get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(180.0),
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ehr-copilot.local",
                "X-Title": "EHR Copilot Benchmark",
            },
        )
    return _http_client


async def call_openrouter(model: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> dict:
    client = await get_client()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    t0 = time.monotonic()
    resp = await client.post(f"{OPENROUTER_BASE}/chat/completions", json=payload)
    latency_ms = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "text": choice["message"]["content"],
        "model": data.get("model", model),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "latency_ms": latency_ms,
    }


def compute_faithfulness(answer_text: str, source_texts: list[str]) -> float:
    try:
        model = get_embedding_model()
        if not source_texts:
            return -1.0
        answer_emb = model.encode([answer_text])[0]
        source_embs = model.encode(source_texts)
        sims = [float(np.dot(answer_emb, se)) for se in source_embs]
        return round(max(sims), 4) if sims else -1.0
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Per-model query: one-shot + judge (run together per model)
# ---------------------------------------------------------------------------

async def run_single_model_query(model_name: str, model_id: str, patient_context: str,
                                  query: str, text_segments: list[str], eq: dict, pid: str) -> dict:
    """Run one-shot + judge for a single model on a single query. Returns result dict."""
    try:
        prompt = ONE_SHOT_PROMPT.format(context=patient_context[:12000], query=query)
        result = await call_openrouter(model_id, prompt, max_tokens=1024)
        answer = result["text"]

        # Judge call
        judge_prompt = JUDGE_PROMPT.format(query=query, answer=answer[:2000])
        try:
            judge_resp = await call_openrouter(JUDGE_MODEL, judge_prompt, max_tokens=8, temperature=0.0)
            score = float(max(1, min(5, int(judge_resp["text"].strip()[0]))))
        except Exception:
            score = 0.0

        # Faithfulness (local embeddings, no API cost)
        faithfulness = compute_faithfulness(answer, text_segments)

        return {
            "model_name": model_name,
            "patient_id": pid,
            "query": query,
            "query_type": eq["type"],
            "answerable": eq["answerable"],
            "answer_text": answer,
            "latency_ms": result["latency_ms"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "llm_judge_score": score,
            "semantic_faithfulness": faithfulness,
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "patient_id": pid,
            "query": query,
            "query_type": eq["type"],
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def run_benchmark(num_patients: int = 3) -> dict:
    from ehr_copilot.ingestion.mimic_fhir_loader import MimicFhirLoader

    loader = MimicFhirLoader(MIMIC_DATA_DIR)
    patients = loader.list_patients()
    num_patients = min(num_patients, len(patients))
    total_queries = num_patients * len(EVAL_QUERIES)
    n_models = len(BENCHMARK_MODELS)

    print("=" * 70)
    print("MULTI-MODEL BENCHMARK: One-Shot vs RAG Pipeline")
    print("=" * 70)
    print(f"Models: {', '.join(BENCHMARK_MODELS.keys())}")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"Patients: {num_patients} | Queries/patient: {len(EVAL_QUERIES)} | Models: {n_models}")
    print(f"Total queries: {total_queries * n_models} ({n_models} models x {total_queries} queries)")
    print(f"Mode: PARALLEL (all {n_models} models run concurrently per query)")
    print()

    # Pre-load embedding model once
    get_embedding_model()

    all_results: dict[str, list[dict]] = {name: [] for name in BENCHMARK_MODELS}
    completed = 0
    start_time = time.monotonic()

    for pi in range(num_patients):
        patient = patients[pi]
        pid = patient["id"]

        # Load patient text
        _ctx, docs, _res = loader.load_patient(pid)
        full_text = "\n\n---\n\n".join(doc.text for doc in docs)
        text_segments = [full_text[i:i+500] for i in range(0, min(len(full_text), 5000), 500)]

        print(f"\n[Patient {pi+1}/{num_patients}] {patient['name']} ({len(full_text)} chars)")

        for qi, eq in enumerate(EVAL_QUERIES):
            # Launch ALL models in parallel for this query
            tasks = [
                run_single_model_query(name, mid, full_text, eq["query"], text_segments, eq, pid)
                for name, mid in BENCHMARK_MODELS.items()
            ]
            results = await asyncio.gather(*tasks)

            completed += 1
            elapsed = time.monotonic() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_queries - completed) / rate if rate > 0 else 0

            # Progress bar
            pct = completed / total_queries
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            parts = []
            for r in results:
                nm = r["model_name"][:8]
                if "error" in r:
                    parts.append(f"{nm}=ERR")
                else:
                    parts.append(f"{nm}={r['llm_judge_score']:.0f}")
            scores = " | ".join(parts)
            print(f"  [{bar}] {completed}/{total_queries} ({pct:.0%}) "
                  f"ETA {eta:.0f}s | Q{qi+1}: {scores}")

            for r in results:
                mname = r.pop("model_name")
                all_results[mname].append(r)

    # Cleanup
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None

    total_elapsed = time.monotonic() - start_time
    print(f"\nDone in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    return all_results


def compute_model_metrics(results: list[dict]) -> dict:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"count": len(results), "errors": len(results)}

    latencies = [r["latency_ms"] for r in valid]
    quality = [r["llm_judge_score"] for r in valid if r.get("llm_judge_score", 0) > 0]
    faith = [r["semantic_faithfulness"] for r in valid if r.get("semantic_faithfulness", -1) >= 0]
    answer_lens = [len(r.get("answer_text", "")) for r in valid]

    unanswerable = [r for r in valid if not r.get("answerable", True)]
    correct_abstentions = sum(
        1 for r in unanswerable
        if any(p in r.get("answer_text", "").lower()
               for p in ["insufficient data", "not available", "no information",
                          "cannot determine", "no data", "not found", "no evidence",
                          "not possible to", "cannot be determined", "no genetic"])
    )

    return {
        "count": len(results),
        "errors": len(results) - len(valid),
        "avg_latency_ms": round(sum(latencies) / len(latencies)),
        "avg_quality": round(sum(quality) / len(quality), 2) if quality else 0,
        "avg_faithfulness": round(sum(faith) / len(faith), 4) if faith else 0,
        "avg_answer_length": round(sum(answer_lens) / len(answer_lens)),
        "abstention_accuracy": round(correct_abstentions / len(unanswerable), 3) if unanswerable else 0,
        "citation_coverage": 0.0,
    }


def print_comparison_table(model_metrics: dict, rag_metrics: dict | None = None):
    print(f"\n{'='*120}")
    print("COMPARISON TABLE: One-Shot (5 Models) vs Multi-Agent RAG Pipeline")
    print(f"{'='*120}")

    cols = list(model_metrics.keys())
    if rag_metrics:
        cols.append("RAG (Ours)")

    header = f"{'Metric':<22}"
    for c in cols:
        header += f" | {c:>15}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Quality (1-5)", "avg_quality", "{:.2f}"),
        ("Faithfulness", "avg_faithfulness", "{:.4f}"),
        ("Abstention Acc.", "abstention_accuracy", "{:.1%}"),
        ("Citation Coverage", "citation_coverage", "{:.1%}"),
        ("Avg Latency (ms)", "avg_latency_ms", "{:.0f}"),
        ("Avg Answer Len", "avg_answer_length", "{:.0f}"),
        ("Errors", "errors", "{:.0f}"),
    ]

    for label, key, fmt in rows:
        row = f"{label:<22}"
        for name in model_metrics:
            val = model_metrics[name].get(key, 0)
            row += f" | {fmt.format(val):>15}"
        if rag_metrics:
            val = rag_metrics.get(key, 0)
            row += f" | {fmt.format(val):>15}"
        print(row)

    print(f"{'='*120}")


def load_rag_results(path: str = "results/eval_results_improved.json") -> dict | None:
    p = Path(path)
    if not p.exists():
        return None

    with open(p) as f:
        data = json.load(f)

    all_rag = [r for pr in data["patient_results"] for r in pr["rag_results"]]
    valid = [r for r in all_rag if "error" not in r]

    latencies = [r["latency_ms"] for r in valid]
    quality = [r["llm_judge_score"] for r in valid if r.get("llm_judge_score", 0) > 0]
    faith = [r["semantic_faithfulness"] for r in valid if r.get("semantic_faithfulness", -1) >= 0]
    answer_lens = [len(r.get("answer_text", "")) for r in valid]
    with_citations = sum(1 for r in valid if len(r.get("citations", [])) > 0)

    unanswerable_indices = [i for i, eq in enumerate(EVAL_QUERIES) if not eq["answerable"]]
    correct_abstentions = 0
    total_unanswerable = 0
    for pr in data["patient_results"]:
        for i in unanswerable_indices:
            if i < len(pr["rag_results"]):
                total_unanswerable += 1
                if pr["rag_results"][i].get("verdict") == "abstained":
                    correct_abstentions += 1

    return {
        "count": len(all_rag),
        "errors": len(all_rag) - len(valid),
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "avg_quality": round(sum(quality) / len(quality), 2) if quality else 0,
        "avg_faithfulness": round(sum(faith) / len(faith), 4) if faith else 0,
        "avg_answer_length": round(sum(answer_lens) / len(answer_lens)) if answer_lens else 0,
        "abstention_accuracy": round(correct_abstentions / total_unanswerable, 3) if total_unanswerable else 0,
        "citation_coverage": round(with_citations / len(valid), 3) if valid else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Benchmark")
    parser.add_argument("--num-patients", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/multi_model_benchmark.json")
    parser.add_argument("--rag-results", type=str, default="results/eval_results_improved.json")
    args = parser.parse_args()

    all_results = asyncio.run(run_benchmark(num_patients=args.num_patients))

    model_metrics = {name: compute_model_metrics(results) for name, results in all_results.items()}
    rag_metrics = load_rag_results(args.rag_results)

    print_comparison_table(model_metrics, rag_metrics)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark_config": {"models": BENCHMARK_MODELS, "judge_model": JUDGE_MODEL},
        "model_metrics": model_metrics,
        "rag_metrics": rag_metrics,
        "raw_results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
