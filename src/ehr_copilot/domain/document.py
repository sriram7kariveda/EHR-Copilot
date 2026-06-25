"""Clinical document and chunk models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    CLINICAL_NOTE = "clinical_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    LAB_REPORT = "lab_report"
    RADIOLOGY_REPORT = "radiology_report"
    PATHOLOGY_REPORT = "pathology_report"
    ENCOUNTER_SUMMARY = "encounter_summary"
    STRUCTURED_DATA = "structured_data"


class NoteSection(str, Enum):
    """Standard clinical note sections."""

    CHIEF_COMPLAINT = "chief_complaint"
    HISTORY_PRESENT_ILLNESS = "history_present_illness"
    PAST_MEDICAL_HISTORY = "past_medical_history"
    MEDICATIONS = "medications"
    ALLERGIES = "allergies"
    SOCIAL_HISTORY = "social_history"
    FAMILY_HISTORY = "family_history"
    REVIEW_OF_SYSTEMS = "review_of_systems"
    PHYSICAL_EXAM = "physical_exam"
    ASSESSMENT_PLAN = "assessment_plan"
    LABS_RESULTS = "labs_results"
    IMAGING = "imaging"
    PROCEDURES = "procedures"
    OTHER = "other"


class ChunkMetadata(BaseModel):
    """Metadata attached to each document chunk for provenance."""

    patient_id: str
    document_id: str
    document_type: DocumentType
    section: NoteSection = NoteSection.OTHER
    encounter_id: str | None = None
    encounter_date: datetime | None = None
    provider: str | None = None
    source_file: str | None = None
    char_start: int = 0
    char_end: int = 0
    chunk_index: int = 0
    total_chunks: int = 1


class DocumentChunk(BaseModel):
    """A chunk of a clinical document, ready for indexing."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    token_count: int = 0

    @property
    def display_source(self) -> str:
        dt = self.metadata.document_type.value.replace("_", " ").title()
        section = self.metadata.section.value.replace("_", " ").title()
        date_str = ""
        if self.metadata.encounter_date:
            date_str = f" ({self.metadata.encounter_date.strftime('%Y-%m-%d')})"
        return f"{dt} > {section}{date_str}"


class ClinicalDocument(BaseModel):
    """A full clinical document before chunking."""

    document_id: str
    patient_id: str
    document_type: DocumentType
    title: str = ""
    text: str = ""
    encounter_id: str | None = None
    encounter_date: datetime | None = None
    provider: str | None = None
    source_file: str | None = None
    sections: dict[NoteSection, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
