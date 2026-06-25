"""Test the Multi-Agent Debate on a single query from eval data.

Loads Qwen 3.5 4B locally on GPU and runs the full debate pipeline
on one query to verify the architecture works end-to-end.

Usage:
    python scripts/test_mad_debate.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


async def main():
    # Load a test query from eval results
    with open("results/eval_results_10patients_merged.json") as f:
        data = json.load(f)

    # Find a query with good citation data
    test_result = None
    for patient in data["patient_results"]:
        for result in patient.get("rag_results", []):
            if result.get("citations") and result.get("evidence_pack"):
                test_result = result
                break
        if test_result:
            break

    if not test_result:
        print("No suitable test query found")
        sys.exit(1)

    query_text = test_result["query"]
    answer_text = test_result["answer_text"]
    verdict = test_result.get("verdict", "unknown")

    print(f"Query: {query_text[:100]}...")
    print(f"Answer: {answer_text[:200]}...")
    print(f"Original verdict: {verdict}")
    print()

    # Build mock DocumentChunks from evidence_pack
    from pydantic import BaseModel
    from ehr_copilot.domain.document import DocumentChunk, ChunkMetadata, DocumentType, NoteSection

    chunks = []
    source_chunks = test_result["evidence_pack"].get("source_chunks", {})
    for chunk_id, chunk_data in source_chunks.items():
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            text=chunk_data.get("text", ""),
            metadata=ChunkMetadata(
                patient_id=test_result.get("patient_id", "test"),
                document_id="test-doc",
                document_type=DocumentType.CLINICAL_NOTE,
            ),
        ))

    print(f"Evidence chunks: {len(chunks)}")
    print()

    # Initialize LLM client
    from ehr_copilot.llm.local_client import LocalLLMClient

    print("Loading Qwen 3.5 4B...")
    llm = LocalLLMClient(model_name="Qwen/Qwen3.5-4B")

    # Initialize MAD components
    from ehr_copilot.agents.mad.claim_extractor import ClaimExtractor
    from ehr_copilot.agents.mad.verifier import Verifier
    from ehr_copilot.agents.mad.challenger import Challenger
    from ehr_copilot.agents.mad.judge import Judge
    from ehr_copilot.agents.mad.debate_engine import DebateEngine
    from ehr_copilot.agents.mad.storage import DebateStorage
    from ehr_copilot.agents.critic import CriticInput
    from ehr_copilot.agents.base import AgentContext
    from ehr_copilot.domain.answer import DraftAnswer

    extractor = ClaimExtractor(llm)
    verifier = Verifier(llm)
    challenger = Challenger(llm)
    judge = Judge(llm)
    storage = DebateStorage(db_path="data/debate_test.db")

    engine = DebateEngine(
        claim_extractor=extractor,
        verifier=verifier,
        challenger=challenger,
        judge=judge,
        storage=storage,
    )

    # Build input
    critic_input = CriticInput(
        query_text=query_text,
        draft_answer=DraftAnswer(
            text=answer_text,
            reasoning_trace="",
            source_chunk_ids=[c.chunk_id for c in chunks[:5]],
            confidence=0.0,
        ),
        chunks=chunks,
    )

    context = AgentContext(
        session_id="test-session",
        patient_id=test_result.get("patient_id", "test"),
        query_id=test_result.get("query_id", "test-query"),
    )

    # Run debate
    print("=" * 60)
    print("RUNNING MULTI-AGENT DEBATE")
    print("=" * 60)

    result = await engine.run(critic_input, context)

    print()
    print("=" * 60)
    print("DEBATE RESULT")
    print("=" * 60)
    print(f"Verdict: {result.output.verdict.value}")
    print(f"Issues: {result.output.issues}")
    if result.output.abstention_reason:
        print(f"Abstention reason: {result.output.abstention_reason}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    print(f"Metadata: {result.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
