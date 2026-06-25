"""API request and response models for the EHR Copilot REST layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming clinical query request."""

    patient_id: str
    query: str
    session_id: str | None = None  # auto-generated if not provided


class QueryResponse(BaseModel):
    """Full response to a clinical query."""

    answer_id: str
    query_id: str
    patient_id: str
    answer_text: str
    citations: list[dict] = Field(default_factory=list)
    verdict: str  # approved / revised / abstained
    confidence: float
    abstention_reason: str | None = None
    latency_ms: float
    evidence_pack: dict | None = None


class PatientLoadRequest(BaseModel):
    """Request to ingest and index a FHIR bundle for a patient.

    For ``source="synthea"``, ``file_path`` should point to a single
    FHIR Bundle JSON file.

    For ``source="mimic-fhir"``, ``file_path`` should point to the
    directory containing ``*.ndjson.gz`` files, and ``patient_id`` must
    be provided (the FHIR Patient resource UUID).
    """

    file_path: str
    source: str = "synthea"
    patient_id: str | None = None  # Required for mimic-fhir


class PatientLoadResponse(BaseModel):
    """Confirmation that a patient was loaded successfully."""

    patient_id: str
    display_name: str
    chunk_count: int
    resource_counts: dict[str, int]
    session_id: str


class PatientListItem(BaseModel):
    """Summary of a loaded patient."""

    patient_id: str
    display_name: str
    session_id: str
    chunk_count: int


class HealthResponse(BaseModel):
    """Service health check."""

    status: str
    llm_available: bool
    loaded_patients: list[str]
    version: str


class AuditResponse(BaseModel):
    """Audit trail for a session."""

    session_id: str
    entries: list[dict]
    chain_valid: bool
