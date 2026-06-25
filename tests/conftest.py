"""Shared test fixtures for the EHR Copilot test suite."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from ehr_copilot.agents.base import AgentContext
from ehr_copilot.config import Settings
from ehr_copilot.domain.answer import Citation, CopilotAnswer, CriticVerdict, EvidenceSpan
from ehr_copilot.domain.clinical import CodingEntry, Observation
from ehr_copilot.domain.document import (
    ChunkMetadata,
    ClinicalDocument,
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
from ehr_copilot.domain.query import ClinicalQuery, QueryIntent, QueryType
from ehr_copilot.domain.timeline import PatientTimeline, TimelineEvent, TimelineEventType
from ehr_copilot.llm.mock_client import MockLLMClient


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_fhir_bundle_path() -> Path:
    """Path to the small Synthea FHIR bundle fixture."""
    return FIXTURES_DIR / "synthea_bundle_small.json"


@pytest.fixture
def expected_timeline_path() -> Path:
    """Path to the expected timeline JSON fixture."""
    return FIXTURES_DIR / "expected_timeline.json"


# ---------------------------------------------------------------------------
# Mock LLM Client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """MockLLMClient with canned responses for all agent stages.

    The keyword-matching logic in MockLLMClient matches on prompt substrings,
    so we map recognisable keywords to realistic structured responses.
    """
    router_response = json.dumps({
        "query_type": "FACTUAL",
        "requires_temporal": False,
        "requires_numeric": False,
        "key_entities": ["hemoglobin", "A1c"],
        "confidence": 0.92,
    })

    temporal_router_response = json.dumps({
        "query_type": "TEMPORAL",
        "requires_temporal": True,
        "requires_numeric": False,
        "key_entities": ["encounter", "visit"],
        "confidence": 0.88,
    })

    medication_router_response = json.dumps({
        "query_type": "MEDICATION",
        "requires_temporal": False,
        "requires_numeric": False,
        "key_entities": ["metformin", "lisinopril"],
        "confidence": 0.95,
    })

    reasoning_response = (
        "<reasoning>\n"
        "Looking at the evidence chunks, I can see that the patient has a "
        "Hemoglobin A1c of 7.2% recorded on 2024-01-15 from chunk [1], "
        "and a subsequent A1c of 6.8% on 2024-06-20 from chunk [2]. "
        "The patient is also on Metformin 500mg twice daily.\n"
        "</reasoning>\n\n"
        "<answer>\n"
        "The patient's most recent Hemoglobin A1c is 6.8%, "
        "measured on 2024-06-20 [1]. This shows improvement from the "
        "previous reading of 7.2% on 2024-01-15 [2]. The patient is "
        "currently taking Metformin hydrochloride 500 MG twice daily "
        "for Type 2 diabetes mellitus management.\n"
        "</answer>\n\n"
        "<source_chunks>\n"
        "1, 2\n"
        "</source_chunks>"
    )

    critic_approved_response = json.dumps({
        "verdict": "APPROVED",
        "issues": [],
        "revised_text": None,
        "abstention_reason": None,
    })

    critic_abstained_response = json.dumps({
        "verdict": "ABSTAINED",
        "issues": ["Insufficient evidence to answer the question."],
        "revised_text": None,
        "abstention_reason": "There is not enough evidence in the provided records to answer this question reliably.",
    })

    critic_revised_response = json.dumps({
        "verdict": "REVISED",
        "issues": ["Minor date inaccuracy corrected."],
        "revised_text": "The patient's most recent Hemoglobin A1c is 6.8%, measured on 2024-06-20.",
        "abstention_reason": None,
    })

    temporal_validation_response = json.dumps({
        "valid": True,
        "issues": [],
        "corrections": [],
    })

    numeric_validation_response = json.dumps({
        "valid": True,
        "issues": [],
        "corrections": [],
    })

    return MockLLMClient(
        default_response=critic_approved_response,
        responses={
            # Router responses -- matched by keyword in the rendered prompt
            "classify the following patient query": router_response,
            "clinical query classifier": router_response,
            # Reasoning responses
            "reasoning assistant": reasoning_response,
            "think step by step": reasoning_response,
            # Critic responses
            "answer critic": critic_approved_response,
            "faithfulness": critic_approved_response,
            # Temporal validation
            "temporal": temporal_validation_response,
            # Numeric validation
            "numeric": numeric_validation_response,
            # Specific keyword overrides for special test scenarios
            "insufficient evidence": critic_abstained_response,
            "abstain": critic_abstained_response,
            "revise": critic_revised_response,
        },
    )


# ---------------------------------------------------------------------------
# Patient Context
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_patient_context() -> PatientContext:
    """PatientContext for the fixture patient John Michael Smith."""
    return PatientContext(
        patient_id=PatientIdentifier(
            system="urn:synthea",
            value="patient-001",
            display="John Michael Smith",
        ),
        demographics=PatientDemographics(
            family_name="Smith",
            given_names=["John", "Michael"],
            birth_date=date(1965, 3, 15),
            gender=Gender.MALE,
            deceased=False,
        ),
        session_id="test-session-001",
        source="synthea",
        resource_counts={
            "encounters": 2,
            "conditions": 2,
            "observations": 4,
            "medications": 2,
        },
    )


# ---------------------------------------------------------------------------
# Document Chunks
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """List of 4 DocumentChunks with realistic clinical content."""
    base_meta = dict(
        patient_id="patient-001",
        document_id="enc-summary-encounter-001",
        document_type=DocumentType.ENCOUNTER_SUMMARY,
        encounter_id="encounter-001",
        encounter_date=datetime(2024, 1, 15, 10, 0, 0),
        provider=None,
        source_file="synthea_bundle_small.json",
    )

    chunks = [
        DocumentChunk(
            chunk_id="chunk-001",
            text=(
                "Ambulatory Encounter For Check Up on 2024-01-15. "
                "Conditions: Type 2 diabetes mellitus [active] (onset: 2018-05-10). "
                "Essential hypertension [active] (onset: 2019-08-22)."
            ),
            metadata=ChunkMetadata(
                **base_meta,
                section=NoteSection.ASSESSMENT_PLAN,
                char_start=0,
                char_end=200,
                chunk_index=0,
                total_chunks=4,
            ),
            token_count=45,
        ),
        DocumentChunk(
            chunk_id="chunk-002",
            text=(
                "Observations: Hemoglobin A1c/Hemoglobin.total in Blood: 7.2 %. "
                "Glucose [Mass/volume] in Blood: 145 mg/dL. "
                "These values were recorded during the encounter on 2024-01-15."
            ),
            metadata=ChunkMetadata(
                **base_meta,
                section=NoteSection.LABS_RESULTS,
                char_start=200,
                char_end=400,
                chunk_index=1,
                total_chunks=4,
            ),
            token_count=50,
        ),
        DocumentChunk(
            chunk_id="chunk-003",
            text=(
                "Medications: Metformin hydrochloride 500 MG Oral Tablet "
                "(Take 500mg twice daily with meals). "
                "Lisinopril 10 MG Oral Tablet (Take 10mg once daily)."
            ),
            metadata=ChunkMetadata(
                **base_meta,
                section=NoteSection.MEDICATIONS,
                char_start=400,
                char_end=550,
                chunk_index=2,
                total_chunks=4,
            ),
            token_count=35,
        ),
        DocumentChunk(
            chunk_id="chunk-004",
            text=(
                "Ambulatory Encounter For Check Up on 2024-06-20. "
                "Observations: Hemoglobin A1c/Hemoglobin.total in Blood: 6.8 %. "
                "Blood pressure systolic and diastolic: N/A."
            ),
            metadata=ChunkMetadata(
                patient_id="patient-001",
                document_id="enc-summary-encounter-002",
                document_type=DocumentType.ENCOUNTER_SUMMARY,
                section=NoteSection.LABS_RESULTS,
                encounter_id="encounter-002",
                encounter_date=datetime(2024, 6, 20, 9, 0, 0),
                provider=None,
                source_file="synthea_bundle_small.json",
                char_start=0,
                char_end=200,
                chunk_index=3,
                total_chunks=4,
            ),
            token_count=42,
        ),
    ]

    return chunks


# ---------------------------------------------------------------------------
# Clinical Query
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_clinical_query() -> ClinicalQuery:
    """A sample factual clinical query about the patient."""
    return ClinicalQuery(
        query_id="query-001",
        patient_id="patient-001",
        session_id="test-session-001",
        text="What are the patient's active conditions and latest A1c result?",
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@pytest.fixture
def test_settings() -> Settings:
    """Settings loaded with test profile defaults (mock LLM, in-memory audit)."""
    return Settings(
        app={"name": "EHR Copilot Test", "debug": True},
        llm={"provider": "mock", "timeout_seconds": 10},
        embedding={
            "model": "NeuML/pubmedbert-base-embeddings",
            "dimension": 768,
            "batch_size": 8,
            "device": "cpu",
        },
        indexing={
            "retrieval": {
                "top_k_dense": 5,
                "top_k_sparse": 5,
                "final_top_k": 3,
            }
        },
        audit={"db_path": ":memory:", "enabled": True},
        logging={"level": "DEBUG"},
    )


# ---------------------------------------------------------------------------
# Agent Context
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_context() -> AgentContext:
    """An AgentContext wired to the test session and patient."""
    return AgentContext(
        session_id="test-session-001",
        patient_id="patient-001",
        query_id="query-001",
    )


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------

@pytest.fixture
async def audit_logger(tmp_path):
    """AuditLogger backed by a temporary SQLite database file.

    Uses a file-based database because each aiosqlite.connect(\":memory:\")
    call creates a separate in-memory DB, so the table created during
    initialize() would not be visible to subsequent calls.
    """
    from ehr_copilot.audit.logger import AuditLogger

    db_path = str(tmp_path / "conftest_audit.db")
    logger = AuditLogger(db_path=db_path)
    await logger.initialize()
    return logger
