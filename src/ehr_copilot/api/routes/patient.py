"""Patient management endpoints -- load, unload, list."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ehr_copilot.api.schemas import PatientListItem, PatientLoadRequest, PatientLoadResponse
from ehr_copilot.domain.audit import AuditEventType
from ehr_copilot.ingestion.chunker import SectionAwareChunker
from ehr_copilot.ingestion.fhir_parser import FHIRBundleParser
from ehr_copilot.ingestion.mimic_fhir_loader import MimicFhirLoader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patient", tags=["patient"])


@router.post("/load", response_model=PatientLoadResponse)
async def load_patient(body: PatientLoadRequest, request: Request) -> PatientLoadResponse:
    """Parse a FHIR bundle, chunk it, and index it for querying.

    The patient context is stored on ``app.state.patient_contexts`` keyed by
    patient ID for subsequent query and audit operations.
    """
    settings = request.app.state.settings
    embedding_model = request.app.state.embedding_model
    index_registry = request.app.state.index_registry
    audit_logger = request.app.state.audit_logger

    file_path = Path(body.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {body.file_path}")

    # 1. Parse source data based on source type.
    if body.source == "mimic-fhir":
        if not body.patient_id:
            raise HTTPException(
                status_code=422,
                detail="patient_id is required for mimic-fhir source",
            )
        try:
            loader = MimicFhirLoader(file_path)
            patient_ctx, documents, resources = loader.load_patient(body.patient_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error("Failed to load MIMIC-FHIR patient: %s", exc, exc_info=True)
            raise HTTPException(status_code=422, detail=f"Failed to load MIMIC-FHIR: {exc}")
    else:
        # Default: Synthea FHIR Bundle
        parser = FHIRBundleParser()
        try:
            patient_ctx, documents, resources = parser.parse(file_path)
        except Exception as exc:
            logger.error("Failed to parse FHIR bundle: %s", exc, exc_info=True)
            raise HTTPException(status_code=422, detail=f"Failed to parse FHIR bundle: {exc}")

    patient_id = patient_ctx.patient_id.value

    # 2. Chunk the documents.
    chunker = SectionAwareChunker(settings.chunking)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk(doc))

    if not all_chunks:
        raise HTTPException(
            status_code=422,
            detail="No chunks produced from the FHIR bundle. The bundle may be empty.",
        )

    # 3. Index the chunks.
    index_registry.create_index(
        patient_id=patient_id,
        chunks=all_chunks,
        embedding_model=embedding_model,
        config=settings.indexing,
    )

    # 4. Store the patient context for later use.
    request.app.state.patient_contexts[patient_id] = patient_ctx

    # 5. Audit the load event.
    try:
        await audit_logger.log(
            session_id=patient_ctx.session_id,
            patient_id=patient_id,
            event_type=AuditEventType.PATIENT_LOADED,
            data={
                "file_path": str(file_path),
                "source": body.source,
                "chunk_count": len(all_chunks),
                "resource_counts": patient_ctx.resource_counts,
            },
        )
    except Exception:
        logger.warning("Failed to write audit log for patient load", exc_info=True)

    logger.info(
        "Patient %s loaded: %d documents, %d chunks",
        patient_id,
        len(documents),
        len(all_chunks),
    )

    return PatientLoadResponse(
        patient_id=patient_id,
        display_name=patient_ctx.demographics.full_name,
        chunk_count=len(all_chunks),
        resource_counts=patient_ctx.resource_counts,
        session_id=patient_ctx.session_id,
    )


@router.delete("/{patient_id}")
async def unload_patient(patient_id: str, request: Request) -> dict:
    """Destroy the patient index and remove the context."""
    index_registry = request.app.state.index_registry
    audit_logger = request.app.state.audit_logger

    destroyed = index_registry.destroy_index(patient_id)
    if not destroyed:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' is not loaded")

    # Remove from cached contexts.
    patient_ctx = request.app.state.patient_contexts.pop(patient_id, None)
    session_id = patient_ctx.session_id if patient_ctx else "unknown"

    # Audit the unload event.
    try:
        await audit_logger.log(
            session_id=session_id,
            patient_id=patient_id,
            event_type=AuditEventType.PATIENT_UNLOADED,
            data={"patient_id": patient_id},
        )
    except Exception:
        logger.warning("Failed to write audit log for patient unload", exc_info=True)

    logger.info("Patient %s unloaded", patient_id)

    return {"status": "ok", "patient_id": patient_id, "message": "Patient unloaded"}


@router.get("/list", response_model=list[PatientListItem])
async def list_patients(request: Request) -> list[PatientListItem]:
    """List all currently loaded patients."""
    index_registry = request.app.state.index_registry
    patient_contexts = request.app.state.patient_contexts

    items: list[PatientListItem] = []
    for pid in index_registry.list_patients():
        ctx = patient_contexts.get(pid)
        patient_index = index_registry.get_index(pid)
        chunk_count = patient_index.chunk_count if patient_index else 0

        items.append(
            PatientListItem(
                patient_id=pid,
                display_name=ctx.demographics.full_name if ctx else pid,
                session_id=ctx.session_id if ctx else "",
                chunk_count=chunk_count,
            )
        )

    return items
