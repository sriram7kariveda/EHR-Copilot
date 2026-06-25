"""Unit tests for the domain model layer."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from ehr_copilot.domain.answer import (
    Citation,
    CopilotAnswer,
    CriticVerdict,
    EvidenceSpan,
)
from ehr_copilot.domain.clinical import CodingEntry, Observation
from ehr_copilot.domain.document import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    NoteSection,
)
from ehr_copilot.domain.patient import (
    Gender,
    PatientContext,
    PatientDemographics,
    PatientIdentifier,
)
from ehr_copilot.domain.query import ClinicalQuery, QueryType
from ehr_copilot.domain.timeline import (
    PatientTimeline,
    TimelineEvent,
    TimelineEventType,
)


# ---------------------------------------------------------------------------
# PatientIdentifier
# ---------------------------------------------------------------------------


class TestPatientIdentifier:
    def test_scoped_id_format(self):
        pid = PatientIdentifier(system="urn:synthea", value="abc-123")
        assert pid.scoped_id == "urn:synthea|abc-123"

    def test_scoped_id_default_system(self):
        pid = PatientIdentifier(value="xyz")
        assert pid.scoped_id == "urn:ehr-copilot|xyz"


# ---------------------------------------------------------------------------
# PatientDemographics
# ---------------------------------------------------------------------------


class TestPatientDemographics:
    def test_full_name(self):
        demo = PatientDemographics(
            family_name="Smith",
            given_names=["John", "Michael"],
        )
        assert demo.full_name == "John Michael Smith"

    def test_full_name_no_given(self):
        demo = PatientDemographics(family_name="Smith")
        assert demo.full_name == "Smith"

    def test_age_description_living(self):
        demo = PatientDemographics(
            family_name="Smith",
            given_names=["John"],
            birth_date=date(1965, 3, 15),
        )
        age = demo.age_description
        assert age is not None
        assert "years old" in age
        # Should be approximately 60-61 years old in 2026
        years = int(age.split()[0])
        assert 60 <= years <= 62

    def test_age_description_deceased(self):
        demo = PatientDemographics(
            family_name="Doe",
            given_names=["Jane"],
            birth_date=date(1950, 1, 1),
            deceased=True,
            deceased_date=date(2020, 6, 15),
        )
        age = demo.age_description
        assert age is not None
        assert "70 years old" in age

    def test_age_description_no_birthdate(self):
        demo = PatientDemographics(family_name="Unknown")
        assert demo.age_description is None


# ---------------------------------------------------------------------------
# PatientContext
# ---------------------------------------------------------------------------


class TestPatientContext:
    def test_display_summary(self, sample_patient_context):
        summary = sample_patient_context.display_summary
        assert "John Michael Smith" in summary
        assert "male" in summary
        assert "years old" in summary


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class TestObservation:
    def test_display_value_with_quantity(self):
        obs = Observation(
            resource_id="obs-1",
            patient_id="p1",
            code=CodingEntry(system="http://loinc.org", code="4548-4", display="A1c"),
            value_quantity=7.2,
            value_unit="%",
        )
        assert obs.display_value == "7.2 %"

    def test_display_value_quantity_no_unit(self):
        obs = Observation(
            resource_id="obs-2",
            patient_id="p1",
            code=CodingEntry(system="http://loinc.org", code="0000", display="Test"),
            value_quantity=42.0,
        )
        assert obs.display_value == "42.0"

    def test_display_value_string(self):
        obs = Observation(
            resource_id="obs-3",
            patient_id="p1",
            code=CodingEntry(system="http://loinc.org", code="0000", display="Test"),
            value_string="Positive",
        )
        assert obs.display_value == "Positive"

    def test_display_value_missing(self):
        obs = Observation(
            resource_id="obs-4",
            patient_id="p1",
            code=CodingEntry(system="http://loinc.org", code="0000", display="Test"),
        )
        assert obs.display_value == "N/A"


# ---------------------------------------------------------------------------
# DocumentChunk
# ---------------------------------------------------------------------------


class TestDocumentChunk:
    def test_display_source_with_date(self):
        chunk = DocumentChunk(
            chunk_id="c1",
            text="test",
            metadata=ChunkMetadata(
                patient_id="p1",
                document_id="d1",
                document_type=DocumentType.ENCOUNTER_SUMMARY,
                section=NoteSection.ASSESSMENT_PLAN,
                encounter_date=datetime(2024, 1, 15),
            ),
        )
        ds = chunk.display_source
        assert "Encounter Summary" in ds
        assert "Assessment Plan" in ds
        assert "2024-01-15" in ds

    def test_display_source_without_date(self):
        chunk = DocumentChunk(
            chunk_id="c2",
            text="test",
            metadata=ChunkMetadata(
                patient_id="p1",
                document_id="d1",
                document_type=DocumentType.CLINICAL_NOTE,
                section=NoteSection.MEDICATIONS,
            ),
        )
        ds = chunk.display_source
        assert "Clinical Note" in ds
        assert "Medications" in ds
        assert "(" not in ds  # no date parenthetical


# ---------------------------------------------------------------------------
# ClinicalQuery
# ---------------------------------------------------------------------------


class TestClinicalQuery:
    def test_creation(self, sample_clinical_query):
        assert sample_clinical_query.query_id == "query-001"
        assert sample_clinical_query.patient_id == "patient-001"
        assert sample_clinical_query.text.startswith("What are")
        assert sample_clinical_query.intent is None

    def test_intent_assignment(self, sample_clinical_query):
        from ehr_copilot.domain.query import QueryIntent

        intent = QueryIntent(query_type=QueryType.FACTUAL, confidence=0.9)
        sample_clinical_query.intent = intent
        assert sample_clinical_query.intent.query_type == QueryType.FACTUAL


# ---------------------------------------------------------------------------
# CopilotAnswer
# ---------------------------------------------------------------------------


class TestCopilotAnswer:
    def test_is_abstention_true(self):
        answer = CopilotAnswer(
            answer_id="a1",
            query_id="q1",
            patient_id="p1",
            text="Cannot answer.",
            verdict=CriticVerdict.ABSTAINED,
            abstention_reason="Insufficient evidence.",
        )
        assert answer.is_abstention is True

    def test_is_abstention_false(self):
        answer = CopilotAnswer(
            answer_id="a2",
            query_id="q1",
            patient_id="p1",
            text="The A1c is 7.2%.",
            verdict=CriticVerdict.APPROVED,
        )
        assert answer.is_abstention is False


# ---------------------------------------------------------------------------
# QueryType enum
# ---------------------------------------------------------------------------


class TestQueryType:
    def test_all_expected_values(self):
        expected = {
            "factual", "temporal", "temporal_numeric", "numeric",
            "medication", "summary", "comparison", "unknown",
        }
        actual = {qt.value for qt in QueryType}
        assert actual == expected


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class TestCitation:
    def test_marker(self):
        cit = Citation(
            citation_id=3,
            claim_text="The A1c is 7.2%.",
            evidence_spans=[],
            confidence=0.85,
        )
        assert cit.marker == "[3]"


# ---------------------------------------------------------------------------
# PatientTimeline
# ---------------------------------------------------------------------------


class TestPatientTimeline:
    @pytest.fixture
    def timeline(self) -> PatientTimeline:
        events = [
            TimelineEvent(
                event_id="e3",
                patient_id="p1",
                event_type=TimelineEventType.LAB_RESULT,
                timestamp=datetime(2024, 6, 20, 9, 30),
                description="A1c 6.8%",
                code="4548-4",
                value="6.8",
                unit="%",
            ),
            TimelineEvent(
                event_id="e1",
                patient_id="p1",
                event_type=TimelineEventType.CONDITION_ONSET,
                timestamp=datetime(2018, 5, 10),
                description="Type 2 diabetes mellitus",
                code="44054006",
            ),
            TimelineEvent(
                event_id="e2",
                patient_id="p1",
                event_type=TimelineEventType.ENCOUNTER,
                timestamp=datetime(2024, 1, 15, 10, 0),
                description="Encounter for check up",
            ),
        ]
        return PatientTimeline(patient_id="p1", events=events)

    def test_sorted_events(self, timeline):
        sorted_evts = timeline.sorted_events
        timestamps = [e.timestamp for e in sorted_evts]
        assert timestamps == sorted(timestamps)
        assert sorted_evts[0].event_id == "e1"  # 2018 is earliest
        assert sorted_evts[-1].event_id == "e3"  # 2024-06 is latest

    def test_events_in_range(self, timeline):
        start = datetime(2024, 1, 1)
        end = datetime(2024, 2, 1)
        results = timeline.events_in_range(start=start, end=end)
        assert len(results) == 1
        assert results[0].event_id == "e2"

    def test_events_in_range_open_start(self, timeline):
        end = datetime(2020, 1, 1)
        results = timeline.events_in_range(end=end)
        assert len(results) == 1
        assert results[0].description == "Type 2 diabetes mellitus"

    def test_events_by_type(self, timeline):
        labs = timeline.events_by_type(TimelineEventType.LAB_RESULT)
        assert len(labs) == 1
        assert labs[0].value == "6.8"

        encounters = timeline.events_by_type(TimelineEventType.ENCOUNTER)
        assert len(encounters) == 1
