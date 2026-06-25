#!/usr/bin/env python3
"""Re-run failed patients for both one-shot benchmark and RAG eval.

Merges results back into the existing JSON files.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MIMIC_DATA_DIR = "data/mimic-fhir/mimic-iv-clinical-database-demo-on-fhir-2.1.0/fhir"
RAG_SERVER = "http://localhost:8001"

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_patient_full_text(patient_id: str) -> str:
    from ehr_copilot.ingestion.mimic_fhir_loader import MimicFhirLoader
    loader = MimicFhirLoader(Path(MIMIC_DATA_DIR))
    _ctx, docs, _res = loader.load_patient(patient_id)
    return "\n\n---\n\n".join(doc.text for doc in docs)


async def one_shot_query(client: httpx.AsyncClient, model_id: str, context: str, query: str) -> dict:
    prompt = ONE_SHOT_PROMPT.format(context=context, query=query)
    t0 = time.time()
    try:
        resp = await client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.1,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        latency_ms = (time.time() - t0) * 1000
        return {"answer_text": text, "latency_ms": latency_ms}
    except Exception as e:
        return {"error": str(e), "latency_ms": (time.time() - t0) * 1000}


def api_post(path: str, body: dict, timeout: int = 600) -> dict:
    import urllib.request
    req = urllib.request.Request(
        f"{RAG_SERVER}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_get(path: str) -> dict:
    import urllib.request
    req = urllib.request.Request(f"{RAG_SERVER}{path}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# One-shot re-run
# ---------------------------------------------------------------------------

async def rerun_oneshot(failed_patients: list[dict], benchmark_path: str):
    """Re-run one-shot benchmark for failed patients and merge results."""
    print(f"\n{'='*70}")
    print(f"RE-RUNNING ONE-SHOT BENCHMARK FOR {len(failed_patients)} PATIENTS")
    print(f"{'='*70}")

    with open(benchmark_path) as f:
        existing = json.load(f)

    async with httpx.AsyncClient() as client:
        for pi, patient in enumerate(failed_patients):
            pid = patient["id"]
            pname = patient["name"]
            full_text = get_patient_full_text(pid)
            print(f"\n[Patient {pi+1}/{len(failed_patients)}] {pname} ({len(full_text)} chars)")

            for qi, eq in enumerate(EVAL_QUERIES):
                print(f"  [{qi+1}/8] {eq['query'][:55]}...", end=" ", flush=True)

                # Run all models in parallel
                tasks = {}
                for model_name, model_id in BENCHMARK_MODELS.items():
                    tasks[model_name] = one_shot_query(client, model_id, full_text, eq["query"])

                results = {}
                for model_name, coro in tasks.items():
                    results[model_name] = await coro

                # Print status
                statuses = []
                for mn, r in results.items():
                    if "error" in r:
                        statuses.append(f"{mn[:8]}=ERR")
                    else:
                        statuses.append(f"{mn[:8]}={len(r['answer_text'])}c")
                print(" | ".join(statuses))

                # Merge into existing results
                for model_name, r in results.items():
                    if "error" not in r:
                        # Find and replace/add in raw_results
                        entry = {
                            "patient_id": pid,
                            "patient_name": pname,
                            "query": eq["query"],
                            "query_type": eq["type"],
                            "answerable": eq["answerable"],
                            "answer_text": r["answer_text"],
                            "latency_ms": r["latency_ms"],
                        }
                        # Remove any existing error entry for this patient+query
                        existing["raw_results"][model_name] = [
                            x for x in existing["raw_results"][model_name]
                            if not (x.get("patient_id") == pid and x.get("query") == eq["query"])
                        ]
                        existing["raw_results"][model_name].append(entry)

    # Save merged results
    with open(benchmark_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\nMerged results saved to {benchmark_path}")


# ---------------------------------------------------------------------------
# RAG re-run
# ---------------------------------------------------------------------------

def rerun_rag(rag_patients: list[dict], rag_path: str):
    """Run RAG eval for patients and save/merge results."""
    print(f"\n{'='*70}")
    print(f"RUNNING RAG EVAL FOR {len(rag_patients)} PATIENTS")
    print(f"{'='*70}")

    # Load existing results if they exist
    if os.path.exists(rag_path):
        with open(rag_path) as f:
            existing = json.load(f)
    else:
        existing = {"patient_results": []}

    # Remove any entries for patients we're re-running
    rag_pids = {p["id"] for p in rag_patients}
    existing["patient_results"] = [
        pr for pr in existing["patient_results"]
        if pr["patient_id"] not in rag_pids
    ]

    for pi, patient in enumerate(rag_patients):
        pid = patient["id"]
        pname = patient["name"]

        print(f"\n{'='*70}")
        print(f"Patient {pi+1}/{len(rag_patients)}: {pname} ({pid[:12]}...)")
        print(f"{'='*70}")

        # Load patient into RAG
        t0 = time.time()
        data_path = os.path.abspath(MIMIC_DATA_DIR)
        try:
            load_result = api_post("/patient/load", {
                "file_path": data_path,
                "source": "mimic-fhir",
                "patient_id": pid,
            })
            load_time = time.time() - t0
            print(f"Loaded in {load_time:.1f}s ({load_result['chunk_count']} chunks)")
        except Exception as e:
            print(f"ERROR loading patient: {e}")
            continue

        patient_results = {
            "patient_id": pid,
            "patient_name": pname,
            "chunks": load_result["chunk_count"],
            "resources": load_result.get("resource_counts", {}),
            "load_time_s": round(load_time, 2),
            "rag_results": [],
            "oneshot_results": [],
        }

        for qi, eq in enumerate(EVAL_QUERIES):
            print(f"\n  [{qi+1}/8] {eq['query'][:55]}...")

            try:
                t0 = time.time()
                result = api_post("/query", {
                    "patient_id": pid,
                    "query": eq["query"],
                })
                latency = (time.time() - t0) * 1000

                answer_text = result.get("answer_text", "")
                print(f"    RAG: {latency:.0f}ms | {len(answer_text)} chars | verdict={result.get('verdict','?')}")

                patient_results["rag_results"].append({
                    "answer_id": result.get("answer_id", ""),
                    "query_id": result.get("query_id", ""),
                    "patient_id": pid,
                    "answer_text": answer_text,
                    "citations": result.get("citations", []),
                    "verdict": result.get("verdict", ""),
                    "confidence": result.get("confidence", 0),
                    "abstention_reason": result.get("abstention_reason", ""),
                    "latency_ms": latency,
                    "evidence_pack": result.get("evidence_pack", {}),
                    "query": eq["query"],
                    "query_type": eq["type"],
                    "expected_answerable": eq["answerable"],
                })
            except Exception as e:
                print(f"    RAG ERROR: {e}")
                patient_results["rag_results"].append({
                    "query": eq["query"],
                    "query_type": eq["type"],
                    "expected_answerable": eq["answerable"],
                    "error": str(e),
                })

        existing["patient_results"].append(patient_results)

        # Save after each patient (in case of crash)
        with open(rag_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        print(f"\nSaved progress ({len(existing['patient_results'])} patients so far)")

    print(f"\nRAG eval complete. Results saved to {rag_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Re-run failed patients")
    parser.add_argument("--mode", choices=["oneshot", "rag", "both"], default="both")
    parser.add_argument("--benchmark-path", default="results/multi_model_benchmark_10patients.json")
    parser.add_argument("--rag-path", default="results/eval_results_10patients_rag.json")
    args = parser.parse_args()

    # List all patients
    from ehr_copilot.ingestion.mimic_fhir_loader import MimicFhirLoader
    loader = MimicFhirLoader(Path(MIMIC_DATA_DIR))
    all_patients = loader.list_patients()[:10]

    # Determine which patients need one-shot re-run
    oneshot_failed = []
    if os.path.exists(args.benchmark_path):
        with open(args.benchmark_path) as f:
            bench = json.load(f)
        model1 = list(bench["raw_results"].keys())[0]
        for patient in all_patients:
            pid = patient["id"]
            ok = sum(1 for r in bench["raw_results"][model1]
                     if r.get("patient_id") == pid and "error" not in r)
            if ok < 8:
                oneshot_failed.append(patient)
                print(f"  One-shot needs re-run: {patient['name']} ({ok}/8 ok)")
    else:
        oneshot_failed = all_patients

    # Determine which patients need RAG
    # Patients 1-3 already done in eval_results_minimax_m25.json
    done_rag_pids = set()
    old_rag = Path("results/eval_results_minimax_m25.json")
    if old_rag.exists():
        with open(old_rag) as f:
            old_data = json.load(f)
        for pr in old_data["patient_results"]:
            ok = sum(1 for r in pr["rag_results"] if r.get("answer_text", ""))
            if ok >= 6:  # at least 6/8 good answers
                done_rag_pids.add(pr["patient_id"])
                print(f"  RAG already done: {pr['patient_name']} ({ok}/8 ok)")

    rag_needed = [p for p in all_patients if p["id"] not in done_rag_pids]
    print(f"\nOne-shot to re-run: {len(oneshot_failed)} patients")
    print(f"RAG to run: {len(rag_needed)} patients")

    if args.mode in ("oneshot", "both") and oneshot_failed:
        asyncio.run(rerun_oneshot(oneshot_failed, args.benchmark_path))

    if args.mode in ("rag", "both") and rag_needed:
        rerun_rag(rag_needed, args.rag_path)


if __name__ == "__main__":
    main()
