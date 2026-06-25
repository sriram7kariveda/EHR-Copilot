"""Clinical note section segmentation.

Splits free-text clinical notes into labelled sections by detecting
standard section headers commonly found in progress notes, discharge
summaries, and H&P reports.
"""

from __future__ import annotations

import re

from ehr_copilot.domain.document import NoteSection

# ---------------------------------------------------------------------------
# Header-to-NoteSection mapping
# ---------------------------------------------------------------------------

# Each entry maps a regex pattern (case-insensitive) that matches a section
# header to the corresponding ``NoteSection`` enum value.  Patterns are
# tried in order; the first match wins for a given header line.

_HEADER_PATTERNS: list[tuple[re.Pattern, NoteSection]] = [
    # Chief complaint
    (re.compile(r"chief\s+complaint", re.IGNORECASE), NoteSection.CHIEF_COMPLAINT),
    (re.compile(r"cc\s*:", re.IGNORECASE), NoteSection.CHIEF_COMPLAINT),
    (re.compile(r"reason\s+for\s+(visit|consultation|admission)", re.IGNORECASE), NoteSection.CHIEF_COMPLAINT),

    # History of present illness
    (re.compile(r"history\s+of\s+present\s+illness", re.IGNORECASE), NoteSection.HISTORY_PRESENT_ILLNESS),
    (re.compile(r"hpi\s*:", re.IGNORECASE), NoteSection.HISTORY_PRESENT_ILLNESS),
    (re.compile(r"history\s+of\s+the\s+present\s+illness", re.IGNORECASE), NoteSection.HISTORY_PRESENT_ILLNESS),
    (re.compile(r"present\s+illness", re.IGNORECASE), NoteSection.HISTORY_PRESENT_ILLNESS),

    # Past medical history
    (re.compile(r"past\s+medical\s+history", re.IGNORECASE), NoteSection.PAST_MEDICAL_HISTORY),
    (re.compile(r"pmh\s*:", re.IGNORECASE), NoteSection.PAST_MEDICAL_HISTORY),
    (re.compile(r"medical\s+history", re.IGNORECASE), NoteSection.PAST_MEDICAL_HISTORY),
    (re.compile(r"past\s+history", re.IGNORECASE), NoteSection.PAST_MEDICAL_HISTORY),

    # Medications
    (re.compile(r"medications?\s*(on\s+admission|at\s+home|list)?", re.IGNORECASE), NoteSection.MEDICATIONS),
    (re.compile(r"current\s+medications?", re.IGNORECASE), NoteSection.MEDICATIONS),
    (re.compile(r"home\s+medications?", re.IGNORECASE), NoteSection.MEDICATIONS),
    (re.compile(r"discharge\s+medications?", re.IGNORECASE), NoteSection.MEDICATIONS),

    # Allergies
    (re.compile(r"allergi(es|c\s+reactions?)", re.IGNORECASE), NoteSection.ALLERGIES),
    (re.compile(r"drug\s+allergies", re.IGNORECASE), NoteSection.ALLERGIES),
    (re.compile(r"nkda", re.IGNORECASE), NoteSection.ALLERGIES),

    # Social history
    (re.compile(r"social\s+history", re.IGNORECASE), NoteSection.SOCIAL_HISTORY),
    (re.compile(r"sh\s*:", re.IGNORECASE), NoteSection.SOCIAL_HISTORY),

    # Family history
    (re.compile(r"family\s+history", re.IGNORECASE), NoteSection.FAMILY_HISTORY),
    (re.compile(r"fh\s*:", re.IGNORECASE), NoteSection.FAMILY_HISTORY),
    (re.compile(r"fhx\s*:", re.IGNORECASE), NoteSection.FAMILY_HISTORY),

    # Review of systems
    (re.compile(r"review\s+of\s+systems?", re.IGNORECASE), NoteSection.REVIEW_OF_SYSTEMS),
    (re.compile(r"ros\s*:", re.IGNORECASE), NoteSection.REVIEW_OF_SYSTEMS),

    # Physical exam
    (re.compile(r"physical\s+exam(ination)?", re.IGNORECASE), NoteSection.PHYSICAL_EXAM),
    (re.compile(r"pe\s*:", re.IGNORECASE), NoteSection.PHYSICAL_EXAM),
    (re.compile(r"exam\s*:", re.IGNORECASE), NoteSection.PHYSICAL_EXAM),
    (re.compile(r"vital\s+signs?\s*", re.IGNORECASE), NoteSection.PHYSICAL_EXAM),

    # Assessment and plan
    (re.compile(r"assessment\s+(and|&)\s+plan", re.IGNORECASE), NoteSection.ASSESSMENT_PLAN),
    (re.compile(r"a\s*/\s*p\s*:", re.IGNORECASE), NoteSection.ASSESSMENT_PLAN),
    (re.compile(r"a&p\s*:", re.IGNORECASE), NoteSection.ASSESSMENT_PLAN),
    (re.compile(r"assessment\s*:", re.IGNORECASE), NoteSection.ASSESSMENT_PLAN),
    (re.compile(r"plan\s*:", re.IGNORECASE), NoteSection.ASSESSMENT_PLAN),
    (re.compile(r"impression\s+(and|&)\s+plan", re.IGNORECASE), NoteSection.ASSESSMENT_PLAN),
    (re.compile(r"impression\s*:", re.IGNORECASE), NoteSection.ASSESSMENT_PLAN),

    # Labs / results
    (re.compile(r"lab(oratory)?\s+results?", re.IGNORECASE), NoteSection.LABS_RESULTS),
    (re.compile(r"labs?\s*:", re.IGNORECASE), NoteSection.LABS_RESULTS),
    (re.compile(r"results?\s*:", re.IGNORECASE), NoteSection.LABS_RESULTS),
    (re.compile(r"pertinent\s+(labs?|results?)", re.IGNORECASE), NoteSection.LABS_RESULTS),

    # Imaging
    (re.compile(r"imaging\s*(results?|studies|findings)?", re.IGNORECASE), NoteSection.IMAGING),
    (re.compile(r"radiology\s*(results?|findings)?", re.IGNORECASE), NoteSection.IMAGING),
    (re.compile(r"diagnostic\s+imaging", re.IGNORECASE), NoteSection.IMAGING),

    # Procedures
    (re.compile(r"procedures?\s*(performed)?", re.IGNORECASE), NoteSection.PROCEDURES),
    (re.compile(r"operative\s+(note|findings|report)", re.IGNORECASE), NoteSection.PROCEDURES),
    (re.compile(r"surgical\s+history", re.IGNORECASE), NoteSection.PROCEDURES),
]

# Pattern for lines that look like section headers:
# - Starts at beginning of line (possibly after whitespace)
# - Followed by a colon, or is ALL-CAPS, or is on its own line
_HEADER_LINE_RE = re.compile(
    r"^[ \t]*"
    r"(?P<header>[A-Z][A-Za-z &/\-]+)"
    r"[ \t]*:?[ \t]*$",
    re.MULTILINE,
)


class NoteSegmenter:
    """Splits a clinical note into labelled sections.

    Usage::

        segmenter = NoteSegmenter()
        sections = segmenter.segment(note_text)
        # sections -> dict[NoteSection, str]
    """

    def __init__(self) -> None:
        self._header_patterns = _HEADER_PATTERNS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, text: str) -> dict[NoteSection, str]:
        """Segment *text* into clinical note sections.

        Returns a mapping from ``NoteSection`` to the body text of that
        section.  Text that appears before the first detected header is
        placed under ``NoteSection.OTHER``.
        """
        if not text or not text.strip():
            return {}

        # Find all candidate header positions
        header_hits = self._find_headers(text)

        if not header_hits:
            # No sections detected -- treat the whole note as OTHER
            return {NoteSection.OTHER: text.strip()}

        # Sort by position
        header_hits.sort(key=lambda h: h[0])

        sections: dict[NoteSection, str] = {}

        # Text before the first header
        preamble = text[: header_hits[0][0]].strip()
        if preamble:
            sections[NoteSection.OTHER] = preamble

        # Extract section bodies
        for idx, (start, _end, section_enum) in enumerate(header_hits):
            body_start = _end
            if idx + 1 < len(header_hits):
                body_end = header_hits[idx + 1][0]
            else:
                body_end = len(text)

            body = text[body_start:body_end].strip()
            if body:
                # If the same section appears more than once, append text
                if section_enum in sections:
                    sections[section_enum] += "\n\n" + body
                else:
                    sections[section_enum] = body

        return sections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_headers(self, text: str) -> list[tuple[int, int, NoteSection]]:
        """Return (start, end, NoteSection) for every detected header line."""
        hits: list[tuple[int, int, NoteSection]] = []

        for match in _HEADER_LINE_RE.finditer(text):
            header_text = match.group("header").strip()
            section = self._classify_header(header_text)
            if section is not None:
                hits.append((match.start(), match.end(), section))
                continue

            # Also try the full matched line with colon for abbreviated
            # headers like "CC:" or "ROS:"
            full_line = match.group(0).strip()
            section = self._classify_header(full_line)
            if section is not None:
                hits.append((match.start(), match.end(), section))

        return hits

    def _classify_header(self, header_text: str) -> NoteSection | None:
        """Match a header string against known patterns."""
        for pattern, section in self._header_patterns:
            if pattern.search(header_text):
                return section
        return None
