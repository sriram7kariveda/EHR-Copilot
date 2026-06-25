"""Query endpoint -- the core clinical Q&A flow."""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from ehr_copilot.api.schemas import QueryRequest, QueryResponse
from ehr_copilot.citations.evidence_pack import EvidencePack
from ehr_copilot.domain.audit import AuditEventType
from ehr_copilot.domain.query import ClinicalQuery

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def answer_query(body: QueryRequest, request: Request) -> QueryResponse:
    """Run a clinical query through the full CopilotPipeline.

    Steps:
    1. Resolve session and query IDs.
    2. Build a ``ClinicalQuery`` domain object.
    3. Obtain the patient's ``HybridRetriever``.
    4. Run the pipeline (router -> retrieval -> reasoning -> validation -> critic).
    5. Build an ``EvidencePack`` with inline citations.
    6. Audit the query and answer events.
    7. Return a ``QueryResponse``.
    """
    settings = request.app.state.settings
    index_registry = request.app.state.index_registry
    pipeline = request.app.state.pipeline
    audit_logger = request.app.state.audit_logger
    patient_contexts = request.app.state.patient_contexts

    patient_id = body.patient_id

    # Resolve session_id -- prefer the one from the request, otherwise fall
    # back to the session attached to the patient context, or generate fresh.
    patient_ctx = patient_contexts.get(patient_id)
    if body.session_id:
        session_id = body.session_id
    elif patient_ctx:
        session_id = patient_ctx.session_id
    else:
        session_id = str(uuid4())

    query_id = str(uuid4())

    # Build the domain query.
    clinical_query = ClinicalQuery(
        query_id=query_id,
        patient_id=patient_id,
        session_id=session_id,
        text=body.query,
    )

    # Obtain the patient-specific retriever.
    patient_index = index_registry.get_index(patient_id)
    if patient_index is None:
        raise HTTPException(
            status_code=422,
            detail=f"Patient '{patient_id}' is not loaded",
        )

    retriever = patient_index.get_retriever(settings.indexing.retrieval)

    # Audit the incoming query.
    try:
        await audit_logger.log(
            session_id=session_id,
            patient_id=patient_id,
            event_type=AuditEventType.QUERY_RECEIVED,
            data={
                "query_id": query_id,
                "query_text": body.query,
            },
        )
    except Exception:
        logger.warning("Failed to write query audit log", exc_info=True)

    # Execute the pipeline.
    try:
        answer = await pipeline.run(clinical_query, retriever)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)

        # Audit the error.
        try:
            await audit_logger.log(
                session_id=session_id,
                patient_id=patient_id,
                event_type=AuditEventType.ERROR,
                data={
                    "query_id": query_id,
                    "error": str(exc),
                },
            )
        except Exception:
            pass

        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # Build the evidence pack from the answer and source chunks.
    evidence_pack_dict: dict | None = None
    citation_dicts: list[dict] = []
    try:
        # Gather the source chunks from the retriever for citation mapping.
        source_chunks_results = retriever.retrieve(body.query, top_k=settings.indexing.retrieval.final_top_k)
        source_chunks = [chunk for chunk, _score in source_chunks_results]
        embedding_model = request.app.state.embedding_model
        evidence_pack = EvidencePack.build(
            answer.text, source_chunks, embedding_model=embedding_model,
        )
        evidence_pack_dict = evidence_pack.to_dict()
        citation_dicts = [cit.model_dump() for cit in evidence_pack.citations]

        # Audit the citation mapping.
        try:
            await audit_logger.log(
                session_id=session_id,
                patient_id=patient_id,
                event_type=AuditEventType.CITATION_MAPPED,
                data={
                    "query_id": query_id,
                    "num_citations": len(evidence_pack.citations),
                    "citation_ids": [c.citation_id for c in evidence_pack.citations],
                },
            )
        except Exception:
            logger.warning("Failed to write citation audit log", exc_info=True)
    except Exception:
        logger.warning("Failed to build evidence pack", exc_info=True)

    # Audit the answer.
    try:
        await audit_logger.log(
            session_id=session_id,
            patient_id=patient_id,
            event_type=AuditEventType.ANSWER_RETURNED,
            data={
                "answer_id": answer.answer_id,
                "query_id": query_id,
                "verdict": answer.verdict.value,
                "confidence": answer.confidence,
                "latency_ms": answer.latency_ms,
            },
        )
    except Exception:
        logger.warning("Failed to write answer audit log", exc_info=True)

    return QueryResponse(
        answer_id=answer.answer_id,
        query_id=query_id,
        patient_id=patient_id,
        answer_text=answer.text,
        citations=citation_dicts,
        verdict=answer.verdict.value,
        confidence=answer.confidence,
        abstention_reason=answer.abstention_reason,
        latency_ms=answer.latency_ms,
        evidence_pack=evidence_pack_dict,
    )
