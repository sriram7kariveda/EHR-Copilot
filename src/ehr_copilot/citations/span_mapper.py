"""Map answer sentences to evidence spans in source document chunks."""

from __future__ import annotations

import re

import numpy as np
from rapidfuzz import fuzz

from ehr_copilot.domain.answer import Citation, EvidenceSpan
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.indexing.embedding import EmbeddingModel

# Sentences shorter than this character count are treated as transitional
# connective tissue ("In summary,", "Additionally,", etc.) and skipped.
_MIN_SENTENCE_LENGTH = 25

# Regex that splits on sentence-ending punctuation followed by whitespace.
# We keep the delimiter attached to the preceding sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")


class SpanMapper:
    """Maps individual sentences of an answer to the best-matching spans in
    the source ``DocumentChunk`` objects using dual scoring:

    - **Lexical** (50%): ``rapidfuzz.fuzz.partial_ratio`` for exact term matching.
    - **Semantic** (50%): Embedding cosine similarity for paraphrase matching.
    """

    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self._embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_citations(
        self,
        answer_text: str,
        source_chunks: list[DocumentChunk],
        threshold: float = 55.0,
    ) -> list[Citation]:
        """Return a list of :class:`Citation` objects that link each
        substantive answer sentence to the best matching source span.

        Parameters
        ----------
        answer_text:
            The full answer text produced by the reasoning agent.
        source_chunks:
            The document chunks that were retrieved for this query.
        threshold:
            Minimum combined score (0-100) required to accept a match.

        Returns
        -------
        list[Citation]
            Citations with sequential ``citation_id`` values starting at 1.
        """
        sentences = self._split_sentences(answer_text)
        citations: list[Citation] = []
        citation_id = 1

        for sentence in sentences:
            if self._is_transitional(sentence):
                continue

            best_span: EvidenceSpan | None = None
            best_score: float = 0.0

            for chunk in source_chunks:
                span_text, score = self.find_best_span(sentence, chunk)
                if score > best_score:
                    best_score = score
                    best_span = EvidenceSpan(
                        chunk_id=chunk.chunk_id,
                        text=span_text,
                        char_start=chunk.text.find(span_text),
                        char_end=chunk.text.find(span_text) + len(span_text),
                        relevance_score=round(score / 100.0, 4),
                        document_source=chunk.display_source,
                    )

            if best_score >= threshold and best_span is not None:
                citations.append(
                    Citation(
                        citation_id=citation_id,
                        claim_text=sentence,
                        evidence_spans=[best_span],
                        confidence=round(best_score / 100.0, 4),
                    )
                )
                citation_id += 1

        return citations

    def find_best_span(
        self,
        sentence: str,
        chunk: DocumentChunk,
        window_size: int = 500,
    ) -> tuple[str, float]:
        """Slide a window over *chunk.text* and return the window that best
        matches *sentence* using dual scoring (lexical + semantic).

        Parameters
        ----------
        sentence:
            The answer sentence to match.
        chunk:
            A source document chunk.
        window_size:
            The character width of the sliding window.

        Returns
        -------
        tuple[str, float]
            ``(best_matching_text, combined_score)`` where *score* is 0-100.
        """
        text = chunk.text
        if not text:
            return ("", 0.0)

        # If the chunk is shorter than the window, compare the whole chunk.
        if len(text) <= window_size:
            score = self._dual_score(sentence, text)
            return (text, score)

        best_window = ""
        best_score: float = 0.0
        step = max(1, window_size // 4)

        for start in range(0, len(text) - window_size + 1, step):
            window = text[start : start + window_size]
            score = self._dual_score(sentence, window)
            if score > best_score:
                best_score = score
                best_window = window

        return (best_window, best_score)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _dual_score(self, sentence: str, candidate: str) -> float:
        """Compute a combined score from lexical (rapidfuzz) and semantic
        (embedding cosine similarity) matching.

        Returns a value in the range 0-100.
        """
        # Lexical score: rapidfuzz partial_ratio (0-100)
        lexical_score = fuzz.partial_ratio(sentence, candidate)

        # If no embedding model available, use lexical-only scoring.
        if self._embedding_model is None:
            return lexical_score

        # Semantic score: embedding cosine similarity (0-1) → scale to 0-100
        try:
            embeddings = self._embedding_model.encode([sentence, candidate])
            cosine_sim = float(np.dot(embeddings[0], embeddings[1]))
            # Embeddings are already L2-normalized, so dot product = cosine sim.
            # Clamp to [0, 1] to handle numerical edge cases.
            cosine_sim = max(0.0, min(1.0, cosine_sim))
            semantic_score = cosine_sim * 100.0
        except Exception:
            # Fallback to lexical-only if embedding fails.
            return lexical_score

        # Dual scoring: 50% lexical + 50% semantic
        return 0.5 * lexical_score + 0.5 * semantic_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split *text* into sentences on `. `, `? `, or `! ` boundaries."""
        parts = _SENTENCE_SPLIT_RE.split(text)
        return [s.strip() for s in parts if s.strip()]

    @staticmethod
    def _is_transitional(sentence: str) -> bool:
        """Return ``True`` if the sentence is too short or looks like
        purely connective/transitional text with no medical content."""
        stripped = sentence.strip().rstrip(".")
        if len(stripped) < _MIN_SENTENCE_LENGTH:
            return True

        # Common transitional phrases that carry no clinical content.
        transitional_phrases = [
            "in summary",
            "to summarize",
            "in conclusion",
            "additionally",
            "furthermore",
            "moreover",
            "however",
            "based on the above",
            "as noted above",
            "as mentioned",
            "overall",
            "in general",
        ]
        lower = stripped.lower()
        if lower in transitional_phrases:
            return True

        return False
