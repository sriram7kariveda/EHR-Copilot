"""Query and intent models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Classification of query types for routing."""

    FACTUAL = "factual"                    # e.g. "What is the patient's blood type?"
    TEMPORAL = "temporal"                  # e.g. "When was the last A1c test?"
    TEMPORAL_NUMERIC = "temporal_numeric"  # e.g. "What is the A1c trend over time?"
    NUMERIC = "numeric"                    # e.g. "What is the latest creatinine?"
    MEDICATION = "medication"              # e.g. "What medications is the patient on?"
    SUMMARY = "summary"                    # e.g. "Summarize recent encounters"
    COMPARISON = "comparison"              # e.g. "Compare labs from Jan vs Feb"
    UNKNOWN = "unknown"


class QueryIntent(BaseModel):
    """Parsed intent from the router agent."""

    query_type: QueryType
    requires_temporal: bool = False
    requires_numeric: bool = False
    time_range: str | None = None
    key_entities: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ClinicalQuery(BaseModel):
    """A clinical question submitted by a user."""

    query_id: str
    patient_id: str
    session_id: str
    text: str
    intent: QueryIntent | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
