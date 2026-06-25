"""Query-type-aware chunk filtering and reranking.

The Router classifies query intent; this module applies branch-specific
post-retrieval filtering so each tree branch operates on a tailored
evidence set.  This is what makes the pipeline a genuine tree architecture
rather than a sequential chain with conditional validators.
"""

from __future__ import annotations

import logging
from datetime import datetime

from ehr_copilot.domain.document import DocumentChunk, DocumentType, NoteSection
from ehr_copilot.domain.query import QueryType

logger = logging.getLogger(__name__)

# Sections most relevant to each query type
_TEMPORAL_SECTIONS = {
    NoteSection.HISTORY_PRESENT_ILLNESS,
    NoteSection.ASSESSMENT_PLAN,
    NoteSection.CHIEF_COMPLAINT,
    NoteSection.LABS_RESULTS,
    NoteSection.PROCEDURES,
}

_NUMERIC_SECTIONS = {
    NoteSection.LABS_RESULTS,
    NoteSection.IMAGING,
    NoteSection.PROCEDURES,
    NoteSection.ASSESSMENT_PLAN,
}

_NUMERIC_DOC_TYPES = {
    DocumentType.LAB_REPORT,
    DocumentType.RADIOLOGY_REPORT,
    DocumentType.PATHOLOGY_REPORT,
    DocumentType.STRUCTURED_DATA,
}

_MEDICATION_SECTIONS = {
    NoteSection.MEDICATIONS,
    NoteSection.ALLERGIES,
    NoteSection.ASSESSMENT_PLAN,
    NoteSection.HISTORY_PRESENT_ILLNESS,
}

# How many chunks each branch should target
_BRANCH_TOP_K = {
    QueryType.FACTUAL: 15,
    QueryType.TEMPORAL: 15,
    QueryType.NUMERIC: 15,
    QueryType.TEMPORAL_NUMERIC: 20,
    QueryType.MEDICATION: 15,
    QueryType.SUMMARY: 25,       # broader retrieval for summaries
    QueryType.COMPARISON: 20,    # need data from multiple time periods
    QueryType.UNKNOWN: 15,
}


def get_branch_top_k(query_type: QueryType) -> int:
    """Return the retrieval top-k for a given query type branch."""
    return _BRANCH_TOP_K.get(query_type, 15)


def filter_and_rerank_chunks(
    chunks: list[DocumentChunk],
    query_type: QueryType,
) -> list[DocumentChunk]:
    """Apply branch-specific filtering and reranking to retrieved chunks.

    Each branch of the tree architecture operates on a different evidence set:

    - TEMPORAL: Sort by encounter date (most recent first), boost date-rich sections
    - NUMERIC: Boost lab reports and numeric-heavy sections
    - MEDICATION: Boost medication sections, deprioritize unrelated sections
    - SUMMARY: Diversify across sections and document types
    - COMPARISON: Sort by date, ensure coverage of multiple time periods
    - FACTUAL/UNKNOWN: No modification (standard retrieval order)
    """
    if query_type in (QueryType.FACTUAL, QueryType.UNKNOWN):
        return chunks

    if query_type == QueryType.TEMPORAL:
        return _filter_temporal(chunks)

    if query_type == QueryType.NUMERIC:
        return _filter_numeric(chunks)

    if query_type == QueryType.TEMPORAL_NUMERIC:
        return _filter_temporal_numeric(chunks)

    if query_type == QueryType.MEDICATION:
        return _filter_medication(chunks)

    if query_type == QueryType.SUMMARY:
        return _filter_summary(chunks)

    if query_type == QueryType.COMPARISON:
        return _filter_comparison(chunks)

    return chunks


def _filter_temporal(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Sort by encounter date (most recent first), boost temporal sections."""
    scored: list[tuple[float, DocumentChunk]] = []
    now = datetime.utcnow()

    for chunk in chunks:
        score = 0.0
        # Boost chunks from temporal-relevant sections
        if chunk.metadata.section in _TEMPORAL_SECTIONS:
            score += 2.0
        # Boost chunks that have encounter dates (more useful for temporal)
        if chunk.metadata.encounter_date:
            # Recency boost: more recent = higher score
            days_ago = (now - chunk.metadata.encounter_date).days
            score += max(0, 5.0 - (days_ago / 365.0))  # up to 5 pts for recent
        else:
            score -= 1.0  # penalize undated chunks
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [chunk for _, chunk in scored]
    logger.info("Temporal filter: reranked %d chunks (boost date-rich sections)", len(result))
    return result


def _filter_numeric(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Boost lab reports and numeric-heavy sections to the top."""
    scored: list[tuple[float, DocumentChunk]] = []

    for chunk in chunks:
        score = 0.0
        # Boost lab-related document types
        if chunk.metadata.document_type in _NUMERIC_DOC_TYPES:
            score += 3.0
        # Boost numeric-relevant sections
        if chunk.metadata.section in _NUMERIC_SECTIONS:
            score += 2.0
        # Simple heuristic: chunks with more digits likely contain numeric data
        digit_density = sum(c.isdigit() for c in chunk.text) / max(len(chunk.text), 1)
        score += digit_density * 10.0  # up to ~1-2 pts typically
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [chunk for _, chunk in scored]
    logger.info("Numeric filter: reranked %d chunks (boost lab/numeric sections)", len(result))
    return result


def _filter_temporal_numeric(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Combine temporal and numeric boosting for trend-over-time queries."""
    scored: list[tuple[float, DocumentChunk]] = []
    now = datetime.utcnow()

    for chunk in chunks:
        score = 0.0
        # Temporal boost
        if chunk.metadata.section in _TEMPORAL_SECTIONS:
            score += 1.5
        if chunk.metadata.encounter_date:
            days_ago = (now - chunk.metadata.encounter_date).days
            score += max(0, 3.0 - (days_ago / 365.0))
        # Numeric boost
        if chunk.metadata.document_type in _NUMERIC_DOC_TYPES:
            score += 2.0
        if chunk.metadata.section in _NUMERIC_SECTIONS:
            score += 1.5
        digit_density = sum(c.isdigit() for c in chunk.text) / max(len(chunk.text), 1)
        score += digit_density * 8.0
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [chunk for _, chunk in scored]
    logger.info("Temporal+Numeric filter: reranked %d chunks", len(result))
    return result


def _filter_medication(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Boost medication sections to the top, keep others as fallback."""
    primary: list[DocumentChunk] = []
    secondary: list[DocumentChunk] = []

    for chunk in chunks:
        if chunk.metadata.section in _MEDICATION_SECTIONS:
            primary.append(chunk)
        else:
            secondary.append(chunk)

    # Sort primary by date (most recent first)
    primary.sort(
        key=lambda c: c.metadata.encounter_date or datetime.min,
        reverse=True,
    )
    result = primary + secondary
    logger.info(
        "Medication filter: %d primary (med sections) + %d secondary",
        len(primary), len(secondary),
    )
    return result


def _filter_summary(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Diversify chunks across sections and document types for broad coverage."""
    # Group by section
    by_section: dict[NoteSection, list[DocumentChunk]] = {}
    for chunk in chunks:
        by_section.setdefault(chunk.metadata.section, []).append(chunk)

    # Round-robin across sections for diversity
    result: list[DocumentChunk] = []
    seen_ids: set[str] = set()
    max_rounds = max((len(v) for v in by_section.values()), default=0)

    for round_idx in range(max_rounds):
        for section in NoteSection:
            section_chunks = by_section.get(section, [])
            if round_idx < len(section_chunks):
                chunk = section_chunks[round_idx]
                if chunk.chunk_id not in seen_ids:
                    result.append(chunk)
                    seen_ids.add(chunk.chunk_id)

    logger.info(
        "Summary filter: diversified %d chunks across %d sections",
        len(result), len(by_section),
    )
    return result


def _filter_comparison(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Sort by date to enable comparison across time periods."""
    # Separate dated and undated
    dated = [c for c in chunks if c.metadata.encounter_date]
    undated = [c for c in chunks if not c.metadata.encounter_date]

    # Sort dated chunks chronologically (oldest first for comparison)
    dated.sort(key=lambda c: c.metadata.encounter_date)  # type: ignore[arg-type]

    result = dated + undated
    logger.info(
        "Comparison filter: %d dated (chronological) + %d undated",
        len(dated), len(undated),
    )
    return result
