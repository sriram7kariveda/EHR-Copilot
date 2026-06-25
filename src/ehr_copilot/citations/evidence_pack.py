"""Bundle citations, source chunks, and formatted text for API responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ehr_copilot.domain.answer import Citation
from ehr_copilot.domain.document import DocumentChunk

from .formatter import CitationFormatter
from .span_mapper import SpanMapper

if TYPE_CHECKING:
    from ehr_copilot.indexing.embedding import EmbeddingModel


class EvidencePack:
    """Immutable bundle of everything the API needs to return alongside an
    answer: the citation objects, source chunks, and pre-formatted text.

    Use the :meth:`build` class method to construct an instance from raw
    answer text and source chunks.

    Attributes
    ----------
    citations : list[Citation]
        Ordered citations with sequential IDs.
    source_chunks : dict[str, DocumentChunk]
        Mapping of ``chunk_id`` to the original :class:`DocumentChunk`.
    formatted_answer : str
        Answer text with inline ``[N]`` markers.
    formatted_references : str
        The "References" block listing each citation's source.
    """

    def __init__(
        self,
        citations: list[Citation],
        source_chunks: dict[str, DocumentChunk],
        formatted_answer: str,
        formatted_references: str,
    ) -> None:
        self.citations = citations
        self.source_chunks = source_chunks
        self.formatted_answer = formatted_answer
        self.formatted_references = formatted_references

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        answer_text: str,
        source_chunks: list[DocumentChunk],
        threshold: float = 55.0,
        embedding_model: EmbeddingModel | None = None,
    ) -> EvidencePack:
        """Build an :class:`EvidencePack` from an answer and its source chunks.

        Parameters
        ----------
        answer_text:
            The full answer text produced by the reasoning agent.
        source_chunks:
            The retrieved document chunks used to generate the answer.
        threshold:
            Minimum combined score (0-100) for a sentence to receive a
            citation.  Passed through to :meth:`SpanMapper.map_citations`.
        embedding_model:
            Optional embedding model for dual scoring (lexical + semantic).
            When ``None``, only lexical (rapidfuzz) scoring is used.

        Returns
        -------
        EvidencePack
        """
        mapper = SpanMapper(embedding_model=embedding_model)
        formatter = CitationFormatter()

        citations = mapper.map_citations(answer_text, source_chunks, threshold)
        formatted_answer = formatter.format_answer(answer_text, citations)
        formatted_references = formatter.format_references(citations)

        chunks_by_id = {chunk.chunk_id: chunk for chunk in source_chunks}

        return cls(
            citations=citations,
            source_chunks=chunks_by_id,
            formatted_answer=formatted_answer,
            formatted_references=formatted_references,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the evidence pack into a plain dictionary suitable for
        JSON API responses.

        Returns
        -------
        dict
            Keys: ``citations``, ``source_chunks``, ``formatted_answer``,
            ``formatted_references``.
        """
        return {
            "citations": [cit.model_dump() for cit in self.citations],
            "source_chunks": {
                cid: chunk.model_dump() for cid, chunk in self.source_chunks.items()
            },
            "formatted_answer": self.formatted_answer,
            "formatted_references": self.formatted_references,
        }
