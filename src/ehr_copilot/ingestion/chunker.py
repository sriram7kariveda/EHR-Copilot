"""Section-aware semantic chunking for clinical documents.

Splits ``ClinicalDocument`` objects into ``DocumentChunk`` objects suitable
for embedding and retrieval, respecting section boundaries and token limits.
"""

from __future__ import annotations

import hashlib
import logging

from ehr_copilot.config import ChunkingConfig
from ehr_copilot.domain.document import (
    ChunkMetadata,
    ClinicalDocument,
    DocumentChunk,
    NoteSection,
)
from ehr_copilot.ingestion.base import ChunkerBase

logger = logging.getLogger(__name__)

# Approximate tokens per whitespace-delimited word.  Clinical text tends to
# have abbreviations and short tokens, so 1.3 is a reasonable estimate.
_TOKENS_PER_WORD = 1.3


def _estimate_tokens(text: str) -> int:
    """Estimate the token count of *text* using whitespace splitting."""
    return int(len(text.split()) * _TOKENS_PER_WORD)


def _generate_chunk_id(patient_id: str, document_id: str, chunk_index: int) -> str:
    """Generate a deterministic unique chunk id."""
    raw = f"{patient_id}::{document_id}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SectionAwareChunker(ChunkerBase):
    """Chunk clinical documents respecting section boundaries.

    Strategy:
    1. If the document has pre-identified sections, chunk each section
       independently so that a chunk never crosses a section boundary.
    2. Within a section (or the full document if no sections exist),
       split by token count using ``max_chunk_tokens`` with
       ``overlap_tokens`` overlap between consecutive chunks.
    3. Discard any chunk smaller than ``min_chunk_tokens``.

    Parameters
    ----------
    config:
        Chunking hyper-parameters.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: ClinicalDocument) -> list[DocumentChunk]:
        """Chunk a ``ClinicalDocument`` into ``DocumentChunk`` objects."""

        section_texts: list[tuple[NoteSection, str]] = []

        if document.sections:
            # Use the pre-segmented sections
            for section_enum, section_body in document.sections.items():
                if section_body.strip():
                    section_texts.append((section_enum, section_body.strip()))
        elif document.text.strip():
            # Treat the whole document as a single OTHER section
            section_texts.append((NoteSection.OTHER, document.text.strip()))
        else:
            return []

        all_chunks: list[DocumentChunk] = []
        global_chunk_index = 0

        for section_enum, section_body in section_texts:
            text_spans = self._split_by_tokens(section_body)

            for span_text, char_start, char_end in text_spans:
                token_count = _estimate_tokens(span_text)

                # Skip chunks that are too small (unless it is the only chunk
                # for the entire document)
                if (
                    token_count < self.config.min_chunk_tokens
                    and len(text_spans) > 1
                ):
                    continue

                chunk_id = _generate_chunk_id(
                    document.patient_id,
                    document.document_id,
                    global_chunk_index,
                )

                metadata = ChunkMetadata(
                    patient_id=document.patient_id,
                    document_id=document.document_id,
                    document_type=document.document_type,
                    section=section_enum,
                    encounter_id=document.encounter_id,
                    encounter_date=document.encounter_date,
                    provider=document.provider,
                    source_file=document.source_file,
                    char_start=char_start,
                    char_end=char_end,
                    chunk_index=global_chunk_index,
                    # total_chunks is set after we know the final count
                )

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=span_text,
                        metadata=metadata,
                        token_count=token_count,
                    )
                )
                global_chunk_index += 1

        # Back-fill total_chunks
        total = len(all_chunks)
        for ch in all_chunks:
            ch.metadata.total_chunks = total

        return all_chunks

    # ------------------------------------------------------------------
    # Internal splitting
    # ------------------------------------------------------------------

    def _split_by_tokens(
        self, text: str
    ) -> list[tuple[str, int, int]]:
        """Split *text* into spans of at most ``max_chunk_tokens`` tokens.

        Returns a list of ``(span_text, char_start, char_end)`` tuples.
        Consecutive spans overlap by ``overlap_tokens`` tokens.
        """
        words = text.split()
        if not words:
            return []

        max_words = max(1, int(self.config.max_chunk_tokens / _TOKENS_PER_WORD))
        overlap_words = max(0, int(self.config.overlap_tokens / _TOKENS_PER_WORD))

        # Pre-compute character offsets for each word
        word_starts: list[int] = []
        pos = 0
        for word in words:
            idx = text.index(word, pos)
            word_starts.append(idx)
            pos = idx + len(word)

        spans: list[tuple[str, int, int]] = []
        start_word = 0

        while start_word < len(words):
            end_word = min(start_word + max_words, len(words))

            char_start = word_starts[start_word]
            # char_end is the end of the last word in the span
            last_word = words[end_word - 1]
            char_end = word_starts[end_word - 1] + len(last_word)

            span_text = text[char_start:char_end]
            spans.append((span_text, char_start, char_end))

            # Advance, accounting for overlap
            step = max_words - overlap_words
            if step < 1:
                step = 1
            start_word += step

            # Avoid creating a tiny trailing chunk that is identical to
            # the overlap of the previous chunk
            if start_word >= len(words):
                break

        return spans
