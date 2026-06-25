"""Simulate the entity verifier on existing RAG results.

This script loads the existing eval results, extracts entities from each answer,
checks them against the citation evidence spans, and reports what would be
removed. It then re-runs the ground truth scoring on the "cleaned" answers
to show the projected improvement in hallucination rate.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ehr_copilot.agents.entity_verifier import verify_entities, extract_entities

RESULTS_PATH = "results/eval_results_10patients_merged.json"


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    total_entities = 0
    total_grounded = 0
    total_removed = 0
    removed_details = []

    cleaned_results = json.loads(json.dumps(data))  # deep copy

    for pi, patient in enumerate(data["patient_results"]):
        patient_name = patient["patient_name"]
        for qi, result in enumerate(patient.get("rag_results", [])):
            answer_text = result.get("answer_text", "")
            if not answer_text:
                continue

            # Collect evidence texts from citations
            evidence_texts = []
            for citation in result.get("citations", []):
                for span in citation.get("evidence_spans", []):
                    if span.get("text"):
                        evidence_texts.append(span["text"])

            if not evidence_texts:
                continue

            # Run entity verifier
            cleaned_text, grounded, removed = verify_entities(answer_text, evidence_texts)

            entities = extract_entities(answer_text)
            total_entities += len(entities)
            total_grounded += len(grounded)
            total_removed += len(removed)

            if removed:
                removed_details.append({
                    "patient": patient_name,
                    "query": result.get("query_text", f"query_{qi}"),
                    "removed": removed,
                    "kept": grounded,
                })

            # Update the cleaned results
            cleaned_results["patient_results"][pi]["rag_results"][qi]["answer_text"] = cleaned_text

    print("=" * 70)
    print("ENTITY VERIFICATION SIMULATION")
    print("=" * 70)
    print(f"Total entities extracted:  {total_entities}")
    print(f"Grounded (kept):           {total_grounded}")
    print(f"Hallucinated (removed):    {total_removed}")
    if total_entities > 0:
        print(f"Removal rate:              {total_removed/total_entities:.1%}")
    print()

    if removed_details:
        print(f"Affected answers: {len(removed_details)}")
        print("-" * 70)
        for d in removed_details:
            print(f"\n  Patient: {d['patient']}")
            query_short = d['query'][:60] + "..." if len(d['query']) > 60 else d['query']
            print(f"  Query:   {query_short}")
            print(f"  Removed: {d['removed']}")
    else:
        print("No entities would be removed!")

    # Save cleaned results for rescoring
    out_path = "results/eval_results_10patients_verified.json"
    with open(out_path, "w") as f:
        json.dump(cleaned_results, f, indent=2)
    print(f"\nCleaned results saved to {out_path}")
    print("Run rescore_ground_truth.py with this file to see metric changes.")


if __name__ == "__main__":
    main()
