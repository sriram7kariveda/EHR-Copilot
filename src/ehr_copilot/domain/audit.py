"""Audit and provenance models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    PATIENT_LOADED = "patient_loaded"
    PATIENT_UNLOADED = "patient_unloaded"
    QUERY_RECEIVED = "query_received"
    ROUTE_CLASSIFIED = "route_classified"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    REASONING_COMPLETED = "reasoning_completed"
    VALIDATION_COMPLETED = "validation_completed"
    CRITIC_VERDICT = "critic_verdict"
    CITATION_MAPPED = "citation_mapped"
    ANSWER_RETURNED = "answer_returned"
    ERROR = "error"


class AuditEntry(BaseModel):
    """A single audit log entry with hash chain."""

    entry_id: str
    session_id: str
    patient_id: str
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict = Field(default_factory=dict)
    previous_hash: str = ""
    entry_hash: str = ""


class ProvenanceRecord(BaseModel):
    """Full provenance trail for an answer."""

    answer_id: str
    query_id: str
    session_id: str
    patient_id: str
    events: list[AuditEntry] = Field(default_factory=list)

    @property
    def event_chain(self) -> list[str]:
        return [e.event_type.value for e in self.events]
