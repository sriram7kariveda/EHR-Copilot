"""Patient context and identifier models."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class PatientIdentifier(BaseModel):
    """Unique patient identifier within the system."""

    system: str = "urn:ehr-copilot"
    value: str
    display: str | None = None

    @property
    def scoped_id(self) -> str:
        return f"{self.system}|{self.value}"


class PatientDemographics(BaseModel):
    """Basic patient demographic information."""

    family_name: str
    given_names: list[str] = Field(default_factory=list)
    birth_date: date | None = None
    gender: Gender = Gender.UNKNOWN
    deceased: bool = False
    deceased_date: date | None = None

    @property
    def full_name(self) -> str:
        given = " ".join(self.given_names)
        return f"{given} {self.family_name}".strip()

    @property
    def age_description(self) -> str | None:
        if self.birth_date is None:
            return None
        ref_date = self.deceased_date or date.today()
        years = (ref_date - self.birth_date).days // 365
        return f"{years} years old"


class PatientContext(BaseModel):
    """Full patient context for a loaded session."""

    patient_id: PatientIdentifier
    demographics: PatientDemographics
    session_id: str
    loaded_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = "unknown"  # e.g. "synthea", "mimic-iii"
    resource_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def display_summary(self) -> str:
        age = self.demographics.age_description or "unknown age"
        gender = self.demographics.gender.value
        return f"{self.demographics.full_name} ({age}, {gender})"
