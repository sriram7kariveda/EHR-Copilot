"""Per-patient index lifecycle management."""

from __future__ import annotations

import logging

from ehr_copilot.config import FAISSConfig, BM25Config, RetrievalConfig
from ehr_copilot.domain.document import DocumentChunk
from ehr_copilot.indexing.embedding import EmbeddingModel
from ehr_copilot.indexing.hybrid_retriever import HybridRetriever
from ehr_copilot.indexing.sparse_store import BM25SparseStore
from ehr_copilot.indexing.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class PatientIndex:
    """Holds a dense and sparse index for a single patient's documents.

    Typical lifecycle::

        idx = PatientIndex(faiss_cfg, bm25_cfg, dimension=768)
        idx.build(chunks, embedding_model)
        retriever = idx.get_retriever(retrieval_cfg)
        results = retriever.retrieve("latest A1c?")
        idx.destroy()
    """

    def __init__(
        self,
        faiss_config: FAISSConfig,
        bm25_config: BM25Config,
        dimension: int,
    ) -> None:
        self._vector_store = FAISSVectorStore(faiss_config, dimension)
        self._sparse_store = BM25SparseStore(bm25_config)
        self._embedding_model: EmbeddingModel | None = None
        self._built = False

    # ------------------------------------------------------------------
    # Build / destroy
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: list[DocumentChunk],
        embedding_model: EmbeddingModel,
    ) -> None:
        """Embed all chunks and insert them into both dense and sparse stores.

        Parameters
        ----------
        chunks:
            Pre-chunked clinical document chunks for this patient.
        embedding_model:
            Shared embedding model instance used to compute dense vectors.
        """
        if not chunks:
            logger.warning("build() called with an empty chunk list; skipping.")
            return

        self._embedding_model = embedding_model

        # Compute dense embeddings for all chunks.
        texts = [c.text for c in chunks]
        embeddings = embedding_model.encode(texts)

        # Populate both indices.
        self._vector_store.add(chunks, embeddings=embeddings)
        self._sparse_store.add(chunks)

        self._built = True
        logger.info(
            "PatientIndex built with %d chunks (%d dense, %d sparse).",
            len(chunks),
            len(self._vector_store),
            len(self._sparse_store),
        )

    def get_retriever(self, config: RetrievalConfig) -> HybridRetriever:
        """Create a :class:`HybridRetriever` wired to this patient's stores.

        Parameters
        ----------
        config:
            Retrieval hyper-parameters (top-k values, RRF k, etc.).

        Raises
        ------
        RuntimeError
            If the index has not been built yet.
        """
        if not self._built or self._embedding_model is None:
            raise RuntimeError(
                "PatientIndex has not been built yet.  Call build() first."
            )
        return HybridRetriever(
            vector_store=self._vector_store,
            sparse_store=self._sparse_store,
            embedding_model=self._embedding_model,
            config=config,
        )

    def destroy(self) -> None:
        """Clear both indices and release references."""
        self._vector_store.clear()
        self._sparse_store.clear()
        self._embedding_model = None
        self._built = False
        logger.info("PatientIndex destroyed.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently held in the dense index."""
        return len(self._vector_store)

    @property
    def is_built(self) -> bool:
        """Whether :meth:`build` has been called successfully."""
        return self._built
