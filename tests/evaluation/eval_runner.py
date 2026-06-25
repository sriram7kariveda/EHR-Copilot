"""Evaluation framework runner for the EHR Copilot.

Loads a QA dataset, runs each question through the pipeline with a mock
LLM, computes quality metrics, and outputs results.

Usage (from project root)::

    python -m tests.evaluation.eval_runner
    python -m tests.evaluation.eval_runner --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from ehr_copilot.agents.critic import CriticAgent
from ehr_copilot.agents.numeric_validator import NumericValidatorAgent
from ehr_copilot.agents.pipeline import CopilotPipeline
from ehr_copilot.agents.reasoning import ReasoningAgent
from ehr_copilot.agents.retrieval import RetrievalAgent
from ehr_copilot.agents.router import RouterAgent
from ehr_copilot.agents.temporal_validator import TemporalValidatorAgent
from ehr_copilot.config import AgentsConfig, ChunkingConfig
from ehr_copilot.domain.document import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    NoteSection,
)
from ehr_copilot.domain.query import ClinicalQuery
from ehr_copilot.ingestion.chunker import SectionAwareChunker
from ehr_copilot.ingestion.fhir_parser import FHIRBundleParser
from ehr_copilot.llm.mock_client import MockLLMClient
from ehr_copilot.llm.prompt_engine import PromptEngine

from tests.evaluation.metrics import (
    abstention_accuracy,
    citation_precision,
    citation_recall,
    compute_faithfulness,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

QA_DATASET_PATH = Path(__file__).parent / "qa_dataset.json"
FHIR_BUNDLE_PATH = Path(__file__).parent.parent / "fixtures" / "synthea_bundle_small.json"


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _build_mock_pipeline(
    chunks: list[DocumentChunk],
) -> CopilotPipeline:
    """Build a CopilotPipeline backed by a MockLLMClient and in-memory
    retrieval that returns the provided chunks."""
    router_resp = json.dumps({
        "query_type": "FACTUAL",
        "requires_temporal": False,
        "requires_numeric": False,
        "key_entities": [],
        "confidence": 0.85,
    })
    reasoning_resp = (
        "<reasoning>Analyzing the evidence chunks for the answer.</reasoning>\n"
        "<answer>Based on the available records, the patient's data shows "
        "Type 2 diabetes mellitus with Hemoglobin A1c values of 7.2% and 6.8%. "
        "The patient takes Metformin hydrochloride 500 MG twice daily and "
        "Lisinopril 10 MG once daily.</answer>\n"
        "<source_chunks>1, 2</source_chunks>"
    )
    critic_resp = json.dumps({
        "verdict": "APPROVED",
        "issues": [],
        "revised_text": None,
        "abstention_reason": None,
    })
    abstain_resp = json.dumps({
        "verdict": "ABSTAINED",
        "issues": ["No relevant evidence found."],
        "revised_text": None,
        "abstention_reason": "The patient records do not contain information to answer this question.",
    })

    client = MockLLMClient(
        default_response=critic_resp,
        responses={
            "clinical query classifier": router_resp,
            "classify": router_resp,
            "reasoning assistant": reasoning_resp,
            "think step by step": reasoning_resp,
            "answer critic": critic_resp,
            "faithfulness": critic_resp,
            "temporal": json.dumps({"valid": True, "issues": [], "corrections": []}),
            "numeric": json.dumps({"valid": True, "issues": [], "corrections": []}),
            "ejection fraction": abstain_resp,
            "colonoscopy": abstain_resp,
        },
    )

    engine = PromptEngine()

    # Build a mock retriever that returns the provided chunks
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [(c, 0.8) for c in chunks[:3]]

    router = RouterAgent(llm_client=client, prompt_engine=engine)
    retrieval = RetrievalAgent(retriever=mock_retriever, top_k=3)
    reasoning = ReasoningAgent(llm_client=client, prompt_engine=engine)
    temporal = TemporalValidatorAgent(llm_client=client, prompt_engine=engine)
    numeric = NumericValidatorAgent(llm_client=client, prompt_engine=engine)
    critic = CriticAgent(llm_client=client, prompt_engine=engine)
    config = AgentsConfig(max_retry_loops=0)

    pipeline = CopilotPipeline(
        router=router,
        retrieval=retrieval,
        reasoning=reasoning,
        temporal_validator=temporal,
        numeric_validator=numeric,
        critic=critic,
        config=config,
    )

    return pipeline


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_evaluation(output_path: str | None = None) -> dict:
    """Run the full evaluation and return metrics."""

    # Load QA dataset
    with open(QA_DATASET_PATH) as f:
        qa_pairs = json.load(f)

    # Load and chunk the FHIR bundle
    parser = FHIRBundleParser()
    patient_ctx, documents, resources = parser.parse(FHIR_BUNDLE_PATH)

    chunker = SectionAwareChunker(
        ChunkingConfig(max_chunk_tokens=200, overlap_tokens=20, min_chunk_tokens=10)
    )
    all_chunks: list[DocumentChunk] = []
    for doc in documents:
        all_chunks.extend(chunker.chunk(doc))

    # Build mock pipeline
    pipeline = _build_mock_pipeline(all_chunks)
    mock_retriever = pipeline._retrieval._retriever

    # Run each question
    results: list[dict] = []
    abstention_preds: list[bool] = []
    abstention_labels: list[bool] = []
    faithfulness_scores: list[float] = []

    for i, qa in enumerate(qa_pairs):
        query = ClinicalQuery(
            query_id=f"eval-q-{i:03d}",
            patient_id=patient_ctx.patient_id.value,
            session_id="eval-session",
            text=qa["question"],
        )

        answer = await pipeline.run(query, mock_retriever)

        # Check expected_answer_contains
        contains_matches = sum(
            1 for term in qa["expected_answer_contains"]
            if term.lower() in answer.text.lower()
        )
        total_expected = len(qa["expected_answer_contains"])
        answer_relevance = (
            contains_matches / total_expected if total_expected > 0 else 1.0
        )

        # Abstention tracking
        abstention_preds.append(answer.is_abstention)
        abstention_labels.append(qa["should_abstain"])

        # Faithfulness
        evidence_texts = [c.text for c in all_chunks[:3]]
        faith_score = compute_faithfulness(answer.text, evidence_texts)
        faithfulness_scores.append(faith_score)

        result = {
            "question": qa["question"],
            "query_type": qa["query_type"],
            "answer_text": answer.text,
            "verdict": answer.verdict.value,
            "is_abstention": answer.is_abstention,
            "should_abstain": qa["should_abstain"],
            "answer_relevance": round(answer_relevance, 4),
            "faithfulness": round(faith_score, 4),
            "latency_ms": answer.latency_ms,
        }
        results.append(result)

    # Aggregate metrics
    avg_relevance = (
        sum(r["answer_relevance"] for r in results) / len(results) if results else 0.0
    )
    avg_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    )
    abs_acc = abstention_accuracy(abstention_preds, abstention_labels)

    metrics = {
        "evaluation_timestamp": datetime.utcnow().isoformat(),
        "total_questions": len(qa_pairs),
        "aggregate_metrics": {
            "avg_answer_relevance": round(avg_relevance, 4),
            "avg_faithfulness": round(avg_faithfulness, 4),
            "abstention_accuracy": round(abs_acc, 4),
        },
        "per_question_results": results,
    }

    # Console output
    print("\n" + "=" * 60)
    print("EHR Copilot Evaluation Results")
    print("=" * 60)
    print(f"Total questions:       {metrics['total_questions']}")
    print(f"Avg answer relevance:  {metrics['aggregate_metrics']['avg_answer_relevance']:.4f}")
    print(f"Avg faithfulness:      {metrics['aggregate_metrics']['avg_faithfulness']:.4f}")
    print(f"Abstention accuracy:   {metrics['aggregate_metrics']['abstention_accuracy']:.4f}")
    print("-" * 60)
    for r in results:
        verdict_marker = "ABSTAIN" if r["is_abstention"] else r["verdict"]
        print(
            f"  [{verdict_marker:>8s}] relevance={r['answer_relevance']:.2f}  "
            f"{r['question'][:60]}"
        )
    print("=" * 60)

    # File output
    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults written to {output_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run EHR Copilot evaluation")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to write JSON results file",
    )
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.output))


if __name__ == "__main__":
    main()
