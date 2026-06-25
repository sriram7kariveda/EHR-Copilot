"""Clinical resource models: encounters, observations, conditions, medications."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ClinicalStatus(str, Enum):
    ACTIVE = "active"
    RECURRENCE = "recurrence"
    RELAPSE = "relapse"
    INACTIVE = "inactive"
    REMISSION = "remission"
    RESOLVED = "resolved"


class EncounterClass(str, Enum):
    AMBULATORY = "ambulatory"
    EMERGENCY = "emergency"
    INPATIENT = "inpatient"
    OUTPATIENT = "outpatient"
    WELLNESS = "wellness"
    OTHER = "other"


class CodingEntry(BaseModel):
    """A code from a terminology system."""

    system: str
    code: str
    display: str = ""


class Encounter(BaseModel):
    """A clinical encounter."""

    resource_id: str
    patient_id: str
    encounter_class: EncounterClass = EncounterClass.OTHER
    type_code: CodingEntry | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    reason_codes: list[CodingEntry] = Field(default_factory=list)
    provider: str | None = None


class Observation(BaseModel):
    """A clinical observation/lab result."""

    resource_id: str
    patient_id: str
    code: CodingEntry
    effective_date: datetime | None = None
    value_quantity: float | None = None
    value_unit: str | None = None
    value_string: str | None = None
    reference_range_low: float | None = None
    reference_range_high: float | None = None
    encounter_id: str | None = None

    @property
    def display_value(self) -> str:
        if self.value_quantity is not None:
            unit = self.value_unit or ""
            return f"{self.value_quantity} {unit}".strip()
        return self.value_string or "N/A"


class Condition(BaseModel):
    """A clinical condition/diagnosis."""

    resource_id: str
    patient_id: str
    code: CodingEntry
    clinical_status: ClinicalStatus = ClinicalStatus.ACTIVE
    onset_date: datetime | None = None
    abatement_date: datetime | None = None
    encounter_id: str | None = None


class MedicationRequest(BaseModel):
    """A medication prescription/order."""

    resource_id: str
    patient_id: str
    medication_code: CodingEntry
    status: str = "active"
    authored_on: datetime | None = None
    dosage_text: str | None = None
    reason_codes: list[CodingEntry] = Field(default_factory=list)
    encounter_id: str | None = None
