"""Format answer text with inline citation markers and a references section."""

from __future__ import annotations

import re
from collections import defaultdict

from ehr_copilot.domain.answer import Citation

# Must stay in sync with the splitting logic in SpanMapper.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")

# Maximum excerpt length shown in the references section.
_MAX_EXCERPT_LENGTH = 120


class CitationFormatter:
    """Insert inline citation markers into answer text and generate a
    formatted references block."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_answer(self, answer_text: str, citations: list[Citation]) -> str:
        """Return *answer_text* with inline citation markers appended to the
        sentences that have citations.

        If a sentence matches multiple citations the markers are concatenated,
        e.g. ``[1][2]``.

        Parameters
        ----------
        answer_text:
            The original answer text (no markers yet).
        citations:
            The citations produced by :class:`SpanMapper`.

        Returns
        -------
        str
            Answer text with ``[N]`` markers inserted.
        """
        if not citations:
            return answer_text

        # Build a mapping: normalised claim text -> list of markers.
        claim_to_markers: dict[str, list[str]] = defaultdict(list)
        for cit in citations:
            key = cit.claim_text.strip()
            claim_to_markers[key].append(cit.marker)

        # Walk through the answer sentence by sentence and append markers.
        sentences = _SENTENCE_SPLIT_RE.split(answer_text)
        result_parts: list[str] = []

        for sentence in sentences:
            stripped = sentence.strip()
            markers = claim_to_markers.get(stripped)
            if markers:
                # Attach markers right after the sentence-ending punctuation.
                marker_str = "".join(markers)
                result_parts.append(f"{stripped}{marker_str}")
            else:
                result_parts.append(stripped)

        return " ".join(result_parts)

    def format_references(self, citations: list[Citation]) -> str:
        """Generate a "References" section listing each citation.

        Format per line::

            [1] Source: Clinical Note > Assessment Plan (2024-01-15) | "excerpt..."

        Parameters
        ----------
        citations:
            The citation objects (must have at least one evidence span each).

        Returns
        -------
        str
            A multi-line references block.
        """
        if not citations:
            return ""

        lines: list[str] = ["References", "----------"]
        for cit in citations:
            if not cit.evidence_spans:
                lines.append(f"[{cit.citation_id}] (no source span)")
                continue

            span = cit.evidence_spans[0]
            excerpt = self._truncate(span.text, _MAX_EXCERPT_LENGTH)
            source = span.document_source or "Unknown source"
            lines.append(
                f"[{cit.citation_id}] Source: {source} | \"{excerpt}\""
            )

        return "\n".join(lines)

    def format_full(self, answer_text: str, citations: list[Citation]) -> str:
        """Return the formatted answer followed by the references section.

        Parameters
        ----------
        answer_text:
            The original answer text.
        citations:
            Citations from :class:`SpanMapper`.

        Returns
        -------
        str
            ``<annotated answer>\\n\\n<references block>``
        """
        annotated = self.format_answer(answer_text, citations)
        references = self.format_references(citations)
        if references:
            return f"{annotated}\n\n{references}"
        return annotated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Truncate *text* to *max_len* characters, adding ellipsis."""
        text = text.replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."
