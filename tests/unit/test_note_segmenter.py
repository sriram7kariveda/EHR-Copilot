"""Unit tests for the clinical note segmenter (ingestion/note_segmenter.py)."""

from __future__ import annotations

import pytest

from ehr_copilot.domain.document import NoteSection
from ehr_copilot.ingestion.note_segmenter import NoteSegmenter


@pytest.fixture
def segmenter() -> NoteSegmenter:
    return NoteSegmenter()


MULTI_SECTION_NOTE = """\
Patient Name: John Smith
MRN: 12345

Chief Complaint
Chest pain for 3 days.

History of Present Illness
Patient is a 59-year-old male presenting with substernal chest pain
that started 3 days ago. Pain is described as a tightness, rated 6/10,
non-radiating. No associated shortness of breath, nausea, or diaphoresis.

Past Medical History
Type 2 diabetes mellitus, diagnosed 2018.
Essential hypertension, diagnosed 2019.

Medications
Metformin 500mg twice daily.
Lisinopril 10mg once daily.

Physical Examination
Vitals: BP 138/88, HR 78, Temp 37.0 C, SpO2 98%.
Heart: Regular rate and rhythm, no murmurs.
Lungs: Clear to auscultation bilaterally.

Assessment and Plan
1. Chest pain - likely musculoskeletal. Order troponin and ECG.
2. Continue Metformin and Lisinopril.
3. Follow up in 1 week.
"""


class TestNoteSegmenter:
    def test_multi_section_note(self, segmenter):
        """Segmenting a multi-section clinical note returns expected sections."""
        sections = segmenter.segment(MULTI_SECTION_NOTE)

        assert NoteSection.CHIEF_COMPLAINT in sections
        assert "chest pain" in sections[NoteSection.CHIEF_COMPLAINT].lower()

        assert NoteSection.HISTORY_PRESENT_ILLNESS in sections
        assert "substernal" in sections[NoteSection.HISTORY_PRESENT_ILLNESS].lower()

        assert NoteSection.PAST_MEDICAL_HISTORY in sections
        assert "diabetes" in sections[NoteSection.PAST_MEDICAL_HISTORY].lower()

        assert NoteSection.MEDICATIONS in sections
        assert "Metformin" in sections[NoteSection.MEDICATIONS]

        assert NoteSection.PHYSICAL_EXAM in sections
        assert "BP 138/88" in sections[NoteSection.PHYSICAL_EXAM]

        assert NoteSection.ASSESSMENT_PLAN in sections
        assert "musculoskeletal" in sections[NoteSection.ASSESSMENT_PLAN].lower()

    def test_missing_sections_not_invented(self, segmenter):
        """Sections that are not present in the note should not appear."""
        sections = segmenter.segment(MULTI_SECTION_NOTE)
        # No social history, family history, or labs section in this note
        assert NoteSection.SOCIAL_HISTORY not in sections
        assert NoteSection.FAMILY_HISTORY not in sections

    def test_section_header_detection(self, segmenter):
        """Individual standard headers should be recognized."""
        note = "Review of Systems\nNo fevers, chills, or weight loss."
        sections = segmenter.segment(note)
        assert NoteSection.REVIEW_OF_SYSTEMS in sections
        assert "fevers" in sections[NoteSection.REVIEW_OF_SYSTEMS].lower()

    def test_single_section_note(self, segmenter):
        """A note with no recognizable headers goes under OTHER."""
        text = "The patient was seen today and appears to be doing well overall."
        sections = segmenter.segment(text)
        assert NoteSection.OTHER in sections
        assert len(sections) == 1
        assert "doing well" in sections[NoteSection.OTHER]

    def test_empty_string(self, segmenter):
        assert segmenter.segment("") == {}
        assert segmenter.segment("   ") == {}

    def test_preamble_captured(self, segmenter):
        """Text before the first header goes under OTHER."""
        note = "Patient: John Smith\n\nChief Complaint\nHeadache."
        sections = segmenter.segment(note)
        assert NoteSection.OTHER in sections
        assert "John Smith" in sections[NoteSection.OTHER]
        assert NoteSection.CHIEF_COMPLAINT in sections
