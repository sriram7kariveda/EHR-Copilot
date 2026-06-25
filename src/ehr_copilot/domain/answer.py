"""Answer, citation, and evidence models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CriticVerdict(str, Enum):
    APPROVED = "approved"
    REVISED = "revised"
    ABSTAINED = "abstained"


class EvidenceSpan(BaseModel):
    """A span of text in a source document that supports a claim."""

    chunk_id: str
    text: str
    char_start: int = 0
    char_end: int = 0
    relevance_score: float = 0.0
    document_source: str = ""


class Citation(BaseModel):
    """A citation linking an answer claim to source evidence."""

    citation_id: int
    claim_text: str
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = 0.0

    @property
    def marker(self) -> str:
        return f"[{self.citation_id}]"


class DraftAnswer(BaseModel):
    """Intermediate answer from the reasoning agent before validation."""

    text: str
    reasoning_trace: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ValidationResult(BaseModel):
    """Result from a validator agent."""

    valid: bool
    issues: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    details: dict[str, object] = Field(default_factory=dict)


class CopilotAnswer(BaseModel):
    """Final answer returned to the user."""

    answer_id: str
    query_id: str
    patient_id: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    verdict: CriticVerdict = CriticVerdict.APPROVED
    confidence: float = 0.0
    reasoning_trace: str = ""
    temporal_validation: ValidationResult | None = None
    numeric_validation: ValidationResult | None = None
    abstention_reason: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0

    @property
    def is_abstention(self) -> bool:
        return self.verdict == CriticVerdict.ABSTAINED
