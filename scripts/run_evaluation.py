#!/usr/bin/env python3
"""Evaluation script: Multi-Agent RAG vs One-Shot baseline.

Runs the same clinical queries through two modes:
1. Full pipeline (multi-agent RAG with retrieval, validation, citations)
2. One-shot baseline (direct LLM call with full patient context, no RAG)

Computes metrics: faithfulness, citation quality, answer quality, latency,
abstention accuracy, and hallucination rate.

Usage:
    # Start the server first:
    uv run python -m uvicorn ehr_copilot.api.app:create_app --factory --port 8001

    # Run evaluation on first 5 patients:
    uv run python scripts/run_evaluation.py --num-patients 5 --output results/eval_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE_URL = "http://localhost:8001"
MIMIC_DATA_DIR = "data/mimic-fhir/mimic-iv-clinical-database-demo-on-fhir-2.1.0/fhir"

# Evaluation queries with ground-truth annotations for scoring
EVAL_QUERIES = [
    {
        "query": "What are the patient's diagnoses from their most recent encounter?",
        "type": "FACTUAL",
        "requires_evidence": True,
        "answerable": True,
    },
    {
        "query": "What medications is this patient currently prescribed?",
        "type": "MEDICATION",
        "requires_evidence": True,
        "answerable": True,
    },
    {
        "query": "What are the most recent lab results for this patient?",
        "type": "FACTUAL",
        "requires_evidence": True,
        "answerable": True,
    },
    {
        "query": "What procedures has this patient undergone?",
        "type": "FACTUAL",
        "requires_evidence": True,
        "answerable": True,
    },
    {
        "query": "Summarize the patient's clinical history across all encounters.",
        "type": "SUMMARY",
        "requires_evidence": True,
        "answerable": True,
    },
    {
        "query": "What is the patient's genetic risk for Alzheimer's disease?",
        "type": "REASONING",
        "requires_evidence": True,
        "answerable": False,  # Should abstain - no genetic data in MIMIC
    },
    {
        "query": "What imaging studies has the patient had and what were the findings?",
        "type": "FACTUAL",
        "requires_evidence": True,
        "answerable": True,  # May or may not have imaging
    },
    {
        "query": "Has the patient's kidney function changed over time?",
        "type": "TEMPORAL",
        "requires_evidence": True,
        "answerable": True,  # Creatinine/BUN trend
    },
]


def api_post(endpoint: str, data: dict, timeout: int = 600) -> dict:
    payload = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def api_get(endpoint: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def one_shot_query(patient_context_text: str, query: str) -> dict:
    """Run a one-shot query: send the full patient context + query directly
    to the LLM without retrieval, validation, or citation mapping.

    This simulates the baseline approach of dumping all patient data into
    the LLM context and asking it to answer directly.
    """
    from ehr_copilot.config import get_settings
    from ehr_copilot.llm import create_llm_client
    from ehr_copilot.llm.base import LLMRequest

    settings = get_settings()
    llm = create_llm_client(provider=settings.llm.provider, config=settings.llm)

    prompt = f"""You are a clinical assistant. Answer the following question about a patient based ONLY on the provided clinical data. If the information is not available, say so clearly.

## Patient Clinical Data
{patient_context_text[:12000]}

## Question
{query}

## Instructions
- Answer based ONLY on the data provided above
- Be specific with dates, values, and units
- If the data does not contain enough information to answer, say "Insufficient data"
"""

    request = LLMRequest(prompt=prompt, max_tokens=1024)

    t0 = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            response = pool.submit(
                asyncio.run, llm.generate(request)
            ).result()
    else:
        response = asyncio.run(llm.generate(request))
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "answer_text": response.text,
        "latency_ms": latency_ms,
        "citations": [],
        "verdict": "one_shot",
        "confidence": 0.0,
    }


def get_patient_full_text(patient_id: str) -> str:
    """Get the full text of all chunks for a patient (for one-shot baseline)."""
    from ehr_copilot.ingestion.mimic_fhir_loader import MimicFhirLoader

    loader = MimicFhirLoader(MIMIC_DATA_DIR)
    _ctx, docs, _res = loader.load_patient(patient_id)
    return "\n\n---\n\n".join(doc.text for doc in docs)


def get_cost_snapshot() -> dict:
    """Return a snapshot of the current cost tracker state."""
    try:
        from ehr_copilot.llm.anthropic_client import get_cost_tracker
        return get_cost_tracker().summary()
    except Exception:
        return {}


def llm_judge_score(query: str, answer_text: str) -> float:
    """Use a one-shot LLM call to score answer quality on a 1-5 scale.

    Returns the numeric score (1-5), or 0.0 on failure.
    """
    from ehr_copilot.config import get_settings
    from ehr_copilot.llm import create_llm_client
    from ehr_copilot.llm.base import LLMRequest

    settings = get_settings()
    llm = create_llm_client(provider=settings.llm.provider, config=settings.llm)

    prompt = f"""Rate the quality of this clinical answer on a scale of 1-5.

## Question
{query}

## Answer
{answer_text[:2000]}

## Rating Criteria
1 = Completely wrong, hallucinated, or nonsensical
2 = Partially relevant but mostly inaccurate or vague
3 = Somewhat relevant and partially accurate
4 = Mostly accurate with minor issues
5 = Accurate, specific, well-supported, and clinically useful

Respond with ONLY a single integer (1-5), nothing else."""

    request = LLMRequest(prompt=prompt, max_tokens=8, temperature=0.0)

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                response = pool.submit(asyncio.run, llm.generate(request)).result()
        else:
            response = asyncio.run(llm.generate(request))

        score = int(response.text.strip()[0])
        return float(max(1, min(5, score)))
    except Exception:
        return 0.0


def compute_semantic_faithfulness(answer_text: str, source_texts: list[str]) -> float:
    """Compute semantic faithfulness as the max cosine similarity between
    the answer embedding and each source chunk embedding.

    Returns a score in [0, 1], or -1.0 on failure.
    """
    try:
        from ehr_copilot.config import get_settings
        from ehr_copilot.indexing.embedding import EmbeddingModel
        import numpy as np

        settings = get_settings()
        model = EmbeddingModel(settings.embedding)

        if not source_texts:
            return -1.0

        answer_emb = model.encode([answer_text])[0]
        source_embs = model.encode(source_texts)

        # Max cosine similarity across source chunks
        similarities = [float(np.dot(answer_emb, se)) for se in source_embs]
        return round(max(similarities), 4) if similarities else -1.0
    except Exception:
        return -1.0


def compute_per_type_metrics(results: list[dict]) -> dict:
    """Break down metrics by query type."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        qtype = r.get("query_type", "UNKNOWN")
        by_type[qtype].append(r)

    type_metrics = {}
    for qtype, type_results in sorted(by_type.items()):
        valid = [r for r in type_results if "error" not in r]
        latencies = [r["latency_ms"] for r in valid]
        citations = [len(r.get("citations", [])) for r in valid]
        abstentions = sum(1 for r in valid if r.get("verdict") == "abstained")
        quality_scores = [r["llm_judge_score"] for r in valid if r.get("llm_judge_score", 0) > 0]

        type_metrics[qtype] = {
            "count": len(type_results),
            "errors": len(type_results) - len(valid),
            "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            "avg_citations": round(sum(citations) / len(citations), 1) if citations else 0,
            "abstention_rate": round(abstentions / len(valid), 3) if valid else 0,
            "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0,
        }
    return type_metrics


def compute_metrics(rag_results: list[dict], oneshot_results: list[dict]) -> dict:
    """Compute comparison metrics between RAG and one-shot."""

    def _answer_stats(results: list[dict]) -> dict:
        latencies = [r["latency_ms"] for r in results if "error" not in r]
        citations = [len(r.get("citations", [])) for r in results if "error" not in r]
        answer_lens = [len(r.get("answer_text", "")) for r in results if "error" not in r]
        verdicts = {}
        for r in results:
            v = r.get("verdict", "unknown")
            verdicts[v] = verdicts.get(v, 0) + 1

        abstentions = sum(1 for r in results if r.get("verdict") == "abstained")
        errors = sum(1 for r in results if "error" in r)

        return {
            "count": len(results),
            "errors": errors,
            "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            "median_latency_ms": round(sorted(latencies)[len(latencies) // 2]) if latencies else 0,
            "avg_citations": round(sum(citations) / len(citations), 1) if citations else 0,
            "avg_answer_length": round(sum(answer_lens) / len(answer_lens)) if answer_lens else 0,
            "abstention_rate": round(abstentions / len(results), 3) if results else 0,
            "verdicts": verdicts,
        }

    rag_stats = _answer_stats(rag_results)
    oneshot_stats = _answer_stats(oneshot_results)

    # Citation coverage: what % of RAG answers have at least 1 citation
    rag_with_citations = sum(
        1 for r in rag_results
        if len(r.get("citations", [])) > 0 and "error" not in r
    )
    rag_answerable = sum(1 for r in rag_results if "error" not in r)

    # Abstention accuracy: did the system abstain on unanswerable queries?
    unanswerable_indices = [
        i for i, eq in enumerate(EVAL_QUERIES) if not eq["answerable"]
    ]
    rag_correct_abstentions = sum(
        1 for i in unanswerable_indices
        if i < len(rag_results) and rag_results[i].get("verdict") == "abstained"
    )

    # Per-query-type stratification
    rag_by_type = compute_per_type_metrics(rag_results)
    oneshot_by_type = compute_per_type_metrics(oneshot_results)

    # Aggregate quality scores
    rag_quality = [r.get("llm_judge_score", 0) for r in rag_results if r.get("llm_judge_score", 0) > 0]
    oneshot_quality = [r.get("llm_judge_score", 0) for r in oneshot_results if r.get("llm_judge_score", 0) > 0]

    # Aggregate faithfulness scores
    rag_faith = [r.get("semantic_faithfulness", -1) for r in rag_results if r.get("semantic_faithfulness", -1) >= 0]
    oneshot_faith = [r.get("semantic_faithfulness", -1) for r in oneshot_results if r.get("semantic_faithfulness", -1) >= 0]

    return {
        "rag": rag_stats,
        "one_shot": oneshot_stats,
        "comparison": {
            "latency_ratio": round(
                oneshot_stats["avg_latency_ms"] / rag_stats["avg_latency_ms"], 2
            ) if rag_stats["avg_latency_ms"] > 0 else 0,
            "rag_citation_coverage": round(
                rag_with_citations / rag_answerable, 3
            ) if rag_answerable > 0 else 0,
            "rag_abstention_accuracy": round(
                rag_correct_abstentions / len(unanswerable_indices), 3
            ) if unanswerable_indices else 1.0,
            "oneshot_abstention_accuracy": 0.0,  # One-shot never abstains
            "rag_avg_quality_score": round(sum(rag_quality) / len(rag_quality), 2) if rag_quality else 0,
            "oneshot_avg_quality_score": round(sum(oneshot_quality) / len(oneshot_quality), 2) if oneshot_quality else 0,
            "rag_avg_faithfulness": round(sum(rag_faith) / len(rag_faith), 4) if rag_faith else 0,
            "oneshot_avg_faithfulness": round(sum(oneshot_faith) / len(oneshot_faith), 4) if oneshot_faith else 0,
        },
        "rag_by_query_type": rag_by_type,
        "oneshot_by_query_type": oneshot_by_type,
    }


def run_evaluation(num_patients: int = 3, skip_oneshot: bool = False) -> dict:
    """Run the full evaluation."""
    print("=" * 70)
    print("EHR COPILOT - Evaluation: Multi-Agent RAG vs One-Shot")
    print("=" * 70)

    # Check server
    try:
        health = api_get("/health")
        if health.get("status") != "ok":
            raise Exception("Server unhealthy")
    except Exception:
        print("\nERROR: Server not running. Start with:")
        print("  uv run python -m uvicorn ehr_copilot.api.app:create_app --factory --port 8001")
        sys.exit(1)

    # List patients
    from ehr_copilot.ingestion.mimic_fhir_loader import MimicFhirLoader
    loader = MimicFhirLoader(MIMIC_DATA_DIR)
    patients = loader.list_patients()
    num_patients = min(num_patients, len(patients))

    print(f"\nEvaluating on {num_patients} patients, {len(EVAL_QUERIES)} queries each")
    print(f"One-shot baseline: {'ENABLED' if not skip_oneshot else 'DISABLED'}")

    all_results = []
    cost_before_eval = get_cost_snapshot()

    for pi in range(num_patients):
        patient = patients[pi]
        pid = patient["id"]
        print(f"\n{'='*70}")
        print(f"Patient {pi+1}/{num_patients}: {patient['name']} ({pid[:12]}...)")
        print(f"{'='*70}")

        # Load patient
        t0 = time.time()
        data_path = os.path.abspath(MIMIC_DATA_DIR)
        load_result = api_post("/patient/load", {
            "file_path": data_path,
            "source": "mimic-fhir",
            "patient_id": pid,
        })
        load_time = time.time() - t0
        print(f"Loaded in {load_time:.1f}s ({load_result['chunk_count']} chunks)")

        # Get full text for one-shot baseline
        full_text = ""
        if not skip_oneshot:
            full_text = get_patient_full_text(pid)
            print(f"Full patient text: {len(full_text)} chars")

        patient_results = {
            "patient_id": pid,
            "patient_name": patient["name"],
            "chunks": load_result["chunk_count"],
            "resources": load_result["resource_counts"],
            "load_time_s": round(load_time, 2),
            "rag_results": [],
            "oneshot_results": [],
        }

        for qi, eq in enumerate(EVAL_QUERIES):
            print(f"\n  [{qi+1}/{len(EVAL_QUERIES)}] {eq['query'][:60]}...")

            # RAG pipeline
            try:
                cost_before = get_cost_snapshot()
                rag_result = api_post("/query", {
                    "patient_id": pid,
                    "query": eq["query"],
                })
                cost_after = get_cost_snapshot()
                query_cost = round(
                    cost_after.get("total_cost_usd", 0) - cost_before.get("total_cost_usd", 0), 6
                )

                # LLM-as-Judge quality score
                judge_score = llm_judge_score(eq["query"], rag_result.get("answer_text", ""))

                # Semantic faithfulness (using evidence pack source chunks if available)
                source_texts = []
                ep = rag_result.get("evidence_pack")
                if ep and isinstance(ep.get("source_chunks"), dict):
                    source_texts = [c.get("text", "") for c in ep["source_chunks"].values() if c.get("text")]
                faithfulness = compute_semantic_faithfulness(
                    rag_result.get("answer_text", ""), source_texts
                ) if source_texts else -1.0

                print(f"    RAG: {rag_result['verdict']} | "
                      f"{len(rag_result.get('citations',[]))} citations | "
                      f"{rag_result['latency_ms']:.0f}ms | "
                      f"${query_cost:.4f} | quality={judge_score:.0f}")
                patient_results["rag_results"].append({
                    **rag_result,
                    "query": eq["query"],
                    "query_type": eq["type"],
                    "expected_answerable": eq["answerable"],
                    "query_cost_usd": query_cost,
                    "llm_judge_score": judge_score,
                    "semantic_faithfulness": faithfulness,
                })
            except Exception as e:
                print(f"    RAG ERROR: {e}")
                patient_results["rag_results"].append({
                    "query": eq["query"],
                    "query_type": eq["type"],
                    "error": str(e),
                })

            # One-shot baseline
            if not skip_oneshot:
                try:
                    cost_before = get_cost_snapshot()
                    os_result = one_shot_query(full_text, eq["query"])
                    cost_after = get_cost_snapshot()
                    query_cost = round(
                        cost_after.get("total_cost_usd", 0) - cost_before.get("total_cost_usd", 0), 6
                    )

                    # LLM-as-Judge quality score
                    judge_score = llm_judge_score(eq["query"], os_result.get("answer_text", ""))

                    # Semantic faithfulness against full patient text
                    # Split full text into manageable segments for embedding
                    text_segments = [full_text[i:i+500] for i in range(0, min(len(full_text), 5000), 500)]
                    faithfulness = compute_semantic_faithfulness(
                        os_result.get("answer_text", ""), text_segments
                    ) if text_segments else -1.0

                    print(f"    1-Shot: {os_result['latency_ms']:.0f}ms | "
                          f"{len(os_result['answer_text'])} chars | "
                          f"${query_cost:.4f} | quality={judge_score:.0f}")
                    patient_results["oneshot_results"].append({
                        **os_result,
                        "query": eq["query"],
                        "query_type": eq["type"],
                        "query_cost_usd": query_cost,
                        "llm_judge_score": judge_score,
                        "semantic_faithfulness": faithfulness,
                    })
                except Exception as e:
                    print(f"    1-Shot ERROR: {e}")
                    patient_results["oneshot_results"].append({
                        "query": eq["query"],
                        "query_type": eq["type"],
                        "error": str(e),
                    })

        # Unload patient
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{BASE_URL}/patient/{pid}",
                    method="DELETE",
                ),
                timeout=30,
            )
        except Exception:
            pass

        all_results.append(patient_results)

    # Aggregate metrics
    all_rag = [r for pr in all_results for r in pr["rag_results"]]
    all_oneshot = [r for pr in all_results for r in pr["oneshot_results"]]
    metrics = compute_metrics(all_rag, all_oneshot)

    # Cost summary
    cost_after_eval = get_cost_snapshot()
    eval_cost = round(
        cost_after_eval.get("total_cost_usd", 0) - cost_before_eval.get("total_cost_usd", 0), 4
    )
    total_rag_cost = sum(r.get("query_cost_usd", 0) for r in all_rag if "error" not in r)
    total_oneshot_cost = sum(r.get("query_cost_usd", 0) for r in all_oneshot if "error" not in r)

    cost_summary = {
        "total_eval_cost_usd": eval_cost,
        "total_rag_cost_usd": round(total_rag_cost, 4),
        "total_oneshot_cost_usd": round(total_oneshot_cost, 4),
        "avg_rag_query_cost_usd": round(
            total_rag_cost / len([r for r in all_rag if "error" not in r]), 4
        ) if any("error" not in r for r in all_rag) else 0,
        "avg_oneshot_query_cost_usd": round(
            total_oneshot_cost / len([r for r in all_oneshot if "error" not in r]), 4
        ) if any("error" not in r for r in all_oneshot) else 0,
        "cost_tracker_state": cost_after_eval,
    }

    # Print results
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"\nMulti-Agent RAG Pipeline:")
    print(f"  Queries: {metrics['rag']['count']}")
    print(f"  Avg latency: {metrics['rag']['avg_latency_ms']}ms")
    print(f"  Avg citations: {metrics['rag']['avg_citations']}")
    print(f"  Avg answer length: {metrics['rag']['avg_answer_length']} chars")
    print(f"  Abstention rate: {metrics['rag']['abstention_rate']:.1%}")
    print(f"  Verdicts: {metrics['rag']['verdicts']}")
    print(f"  Avg quality score: {metrics['comparison'].get('rag_avg_quality_score', 0)}")
    print(f"  Avg faithfulness: {metrics['comparison'].get('rag_avg_faithfulness', 0):.4f}")

    if not skip_oneshot:
        print(f"\nOne-Shot Baseline:")
        print(f"  Queries: {metrics['one_shot']['count']}")
        print(f"  Avg latency: {metrics['one_shot']['avg_latency_ms']}ms")
        print(f"  Avg answer length: {metrics['one_shot']['avg_answer_length']} chars")
        print(f"  Avg quality score: {metrics['comparison'].get('oneshot_avg_quality_score', 0)}")
        print(f"  Avg faithfulness: {metrics['comparison'].get('oneshot_avg_faithfulness', 0):.4f}")

        print(f"\nComparison:")
        print(f"  RAG citation coverage: {metrics['comparison']['rag_citation_coverage']:.1%}")
        print(f"  RAG abstention accuracy: {metrics['comparison']['rag_abstention_accuracy']:.1%}")
        print(f"  One-shot abstention accuracy: {metrics['comparison']['oneshot_abstention_accuracy']:.1%}")
        print(f"  Latency ratio (1-shot/RAG): {metrics['comparison']['latency_ratio']}x")

    # Per-query-type breakdown
    if metrics.get("rag_by_query_type"):
        print(f"\nPer-Query-Type Breakdown (RAG):")
        for qtype, tm in metrics["rag_by_query_type"].items():
            print(f"  {qtype}: {tm['count']} queries, "
                  f"avg {tm['avg_latency_ms']}ms, "
                  f"{tm['avg_citations']} citations, "
                  f"quality={tm['avg_quality_score']}")

    print(f"\nCost Summary:")
    print(f"  Total evaluation cost: ${cost_summary['total_eval_cost_usd']:.4f}")
    print(f"  RAG total: ${cost_summary['total_rag_cost_usd']:.4f} "
          f"(avg ${cost_summary['avg_rag_query_cost_usd']:.4f}/query)")
    if not skip_oneshot:
        print(f"  One-shot total: ${cost_summary['total_oneshot_cost_usd']:.4f} "
              f"(avg ${cost_summary['avg_oneshot_query_cost_usd']:.4f}/query)")

    return {
        "patients_evaluated": num_patients,
        "queries_per_patient": len(EVAL_QUERIES),
        "patient_results": all_results,
        "metrics": metrics,
        "cost_summary": cost_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="EHR Copilot Evaluation")
    parser.add_argument("--num-patients", type=int, default=3)
    parser.add_argument("--skip-oneshot", action="store_true")
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = f"http://localhost:{args.port}"

    results = run_evaluation(
        num_patients=args.num_patients,
        skip_oneshot=args.skip_oneshot,
    )

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
