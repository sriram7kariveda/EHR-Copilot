"""Timeline event models for temporal reasoning."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TimelineEventType(str, Enum):
    ENCOUNTER = "encounter"
    OBSERVATION = "observation"
    CONDITION_ONSET = "condition_onset"
    CONDITION_RESOLVED = "condition_resolved"
    MEDICATION_START = "medication_start"
    MEDICATION_END = "medication_end"
    PROCEDURE = "procedure"
    LAB_RESULT = "lab_result"


class TimelineEvent(BaseModel):
    """A single event on a patient's clinical timeline."""

    event_id: str
    patient_id: str
    event_type: TimelineEventType
    timestamp: datetime
    description: str
    code: str | None = None
    code_display: str | None = None
    value: str | None = None
    unit: str | None = None
    source_chunk_id: str | None = None
    encounter_id: str | None = None


class PatientTimeline(BaseModel):
    """Ordered collection of a patient's clinical events."""

    patient_id: str
    events: list[TimelineEvent] = Field(default_factory=list)
    built_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def sorted_events(self) -> list[TimelineEvent]:
        return sorted(self.events, key=lambda e: e.timestamp)

    def events_in_range(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> list[TimelineEvent]:
        events = self.sorted_events
        if start:
            events = [e for e in events if e.timestamp >= start]
        if end:
            events = [e for e in events if e.timestamp <= end]
        return events

    def events_by_type(self, event_type: TimelineEventType) -> list[TimelineEvent]:
        return [e for e in self.sorted_events if e.event_type == event_type]

    def events_by_code(self, code: str) -> list[TimelineEvent]:
        return [e for e in self.sorted_events if e.code == code]
