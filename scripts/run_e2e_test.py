#!/usr/bin/env python3
"""End-to-end test script for the EHR Copilot.

Usage:
    # Start the server first:
    uv run python -m uvicorn ehr_copilot.api.app:create_app --factory --port 8001

    # Then run this script:
    uv run python scripts/run_e2e_test.py

    # Or test with a specific patient:
    uv run python scripts/run_e2e_test.py --patient-index 5

    # Run in one-shot mode (no RAG, for comparison):
    uv run python scripts/run_e2e_test.py --one-shot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

BASE_URL = "http://localhost:8001"
MIMIC_DATA_DIR = "data/mimic-fhir/mimic-iv-clinical-database-demo-on-fhir-2.1.0/fhir"

# Clinical queries covering different query types
TEST_QUERIES = [
    {
        "query": "What are the patient's active diagnoses?",
        "type": "FACTUAL",
        "description": "Factual query about conditions",
    },
    {
        "query": "What medications is this patient currently taking?",
        "type": "MEDICATION",
        "description": "Medication list query",
    },
    {
        "query": "What are the most recent laboratory results?",
        "type": "FACTUAL",
        "description": "Recent lab results",
    },
    {
        "query": "Has the patient's hemoglobin changed over time?",
        "type": "TEMPORAL",
        "description": "Temporal trend query",
    },
    {
        "query": "What procedures has the patient undergone?",
        "type": "FACTUAL",
        "description": "Procedure history",
    },
    {
        "query": "Are there any abnormal lab values that need attention?",
        "type": "NUMERIC",
        "description": "Numeric reasoning about lab values",
    },
    {
        "query": "Summarize the patient's most recent hospital encounter.",
        "type": "SUMMARY",
        "description": "Encounter summary",
    },
    {
        "query": "What is the patient's cardiac risk based on available data?",
        "type": "REASONING",
        "description": "Clinical reasoning query (may abstain)",
    },
]


def api_call(endpoint: str, data: dict, timeout: int = 120) -> dict:
    """Make a POST request to the API."""
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
    """Make a GET request to the API."""
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def check_health() -> bool:
    """Check if the server is running."""
    try:
        result = api_get("/health")
        return result.get("status") == "ok"
    except Exception:
        return False


def list_mimic_patients() -> list[dict]:
    """List available MIMIC-FHIR patients."""
    from ehr_copilot.ingestion.mimic_fhir_loader import MimicFhirLoader

    loader = MimicFhirLoader(MIMIC_DATA_DIR)
    return loader.list_patients()


def load_patient(patient_id: str) -> dict:
    """Load a MIMIC patient via the API."""
    import os

    data_path = os.path.abspath(MIMIC_DATA_DIR)
    return api_call("/patient/load", {
        "file_path": data_path,
        "source": "mimic-fhir",
        "patient_id": patient_id,
    })


def query_patient(patient_id: str, query: str) -> dict:
    """Run a query against a loaded patient."""
    return api_call("/query", {
        "patient_id": patient_id,
        "query": query,
    })


def get_audit(session_id: str) -> dict:
    """Get audit trail for a session."""
    return api_get(f"/audit/{session_id}")


def run_e2e_test(patient_index: int = 0) -> dict:
    """Run the full end-to-end test."""
    results = {
        "patient": None,
        "queries": [],
        "audit": None,
        "summary": {},
    }

    # 1. Check health
    print("=" * 70)
    print("EHR COPILOT - End-to-End Test")
    print("=" * 70)

    if not check_health():
        print("\nERROR: Server is not running at", BASE_URL)
        print("Start it with: uv run python -m uvicorn ehr_copilot.api.app:create_app --factory --port 8001")
        sys.exit(1)
    print("\n[OK] Server is healthy")

    # 2. List patients
    print("\nListing MIMIC-FHIR patients...")
    patients = list_mimic_patients()
    print(f"Found {len(patients)} patients")

    if patient_index >= len(patients):
        print(f"ERROR: Patient index {patient_index} out of range (0-{len(patients)-1})")
        sys.exit(1)

    target = patients[patient_index]
    pid = target["id"]
    print(f"\nSelected patient: {target['name']} ({target['gender']}, born {target['birthDate']})")
    print(f"Patient ID: {pid}")

    # 3. Load patient
    print("\n--- Loading Patient ---")
    t0 = time.time()
    load_result = load_patient(pid)
    load_time = time.time() - t0

    results["patient"] = load_result
    print(f"Loaded in {load_time:.1f}s")
    print(f"  Chunks: {load_result['chunk_count']}")
    print(f"  Resources: {load_result['resource_counts']}")

    session_id = load_result["session_id"]

    # 4. Run queries
    print("\n--- Running Queries ---")
    total_latency = 0
    verdicts = {"approved": 0, "revised": 0, "abstained": 0}
    total_citations = 0

    for i, tq in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] {tq['description']}")
        print(f"  Q: {tq['query']}")

        t0 = time.time()
        try:
            result = query_patient(pid, tq["query"])
            wall_time = time.time() - t0

            verdict = result["verdict"]
            n_citations = len(result.get("citations", []))
            confidence = result.get("confidence", 0)
            pipeline_ms = result.get("latency_ms", 0)

            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            total_citations += n_citations
            total_latency += pipeline_ms

            print(f"  Verdict: {verdict} | Confidence: {confidence:.2f} | Citations: {n_citations}")
            print(f"  Pipeline: {pipeline_ms:.0f}ms | Wall: {wall_time:.1f}s")
            print(f"  Answer: {result['answer_text'][:200]}...")

            results["queries"].append({
                "query": tq["query"],
                "type": tq["type"],
                "verdict": verdict,
                "confidence": confidence,
                "citations": n_citations,
                "pipeline_ms": pipeline_ms,
                "wall_time_s": round(wall_time, 2),
                "answer_length": len(result["answer_text"]),
                "answer_preview": result["answer_text"][:300],
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results["queries"].append({
                "query": tq["query"],
                "type": tq["type"],
                "error": str(e),
            })

    # 5. Audit trail
    print("\n--- Audit Trail ---")
    try:
        audit = get_audit(session_id)
        n_entries = len(audit.get("entries", []))
        chain_valid = audit.get("chain_valid", False)
        print(f"  Entries: {n_entries}")
        print(f"  Hash chain valid: {chain_valid}")

        event_types = [e["event_type"] for e in audit.get("entries", [])]
        print(f"  Event types: {', '.join(event_types)}")

        results["audit"] = {
            "entries": n_entries,
            "chain_valid": chain_valid,
            "event_types": event_types,
        }
    except Exception as e:
        print(f"  ERROR: {e}")

    # 6. Summary
    n_queries = len([q for q in results["queries"] if "error" not in q])
    avg_latency = total_latency / n_queries if n_queries else 0
    avg_citations = total_citations / n_queries if n_queries else 0

    results["summary"] = {
        "patient": target["name"],
        "chunks": load_result["chunk_count"],
        "load_time_s": round(load_time, 2),
        "queries_run": n_queries,
        "avg_pipeline_ms": round(avg_latency),
        "avg_citations": round(avg_citations, 1),
        "verdicts": verdicts,
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Patient: {target['name']} ({load_result['chunk_count']} chunks)")
    print(f"  Load time: {load_time:.1f}s")
    print(f"  Queries: {n_queries} run")
    print(f"  Avg pipeline latency: {avg_latency:.0f}ms")
    print(f"  Avg citations per answer: {avg_citations:.1f}")
    print(f"  Verdicts: {verdicts}")
    print(f"  Audit entries: {results['audit']['entries'] if results['audit'] else 'N/A'}")
    print(f"  Hash chain valid: {results['audit']['chain_valid'] if results['audit'] else 'N/A'}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="EHR Copilot End-to-End Test")
    parser.add_argument("--patient-index", type=int, default=0,
                        help="Index of patient to test (0-99)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--port", type=int, default=8001,
                        help="API server port")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = f"http://localhost:{args.port}"

    results = run_e2e_test(patient_index=args.patient_index)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
