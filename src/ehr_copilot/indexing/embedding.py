"""Embedding model wrapper with lazy loading and unit-vector normalisation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from ehr_copilot.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wraps a sentence-transformers model behind a lazy-loading facade.

    Embeddings are L2-normalised so that inner-product equals cosine similarity.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence-transformers model on first use."""
        if self._model is None:
            logger.info(
                "Loading embedding model %s on %s",
                self._config.model,
                self._config.device,
            )
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._config.model,
                device=self._config.device,
            )
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into unit-normalised embeddings.

        Parameters
        ----------
        texts:
            Arbitrary-length list of strings.

        Returns
        -------
        np.ndarray of shape ``(len(texts), dimension)`` with dtype float32,
        where each row has unit L2 norm.
        """
        model = self._load_model()
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)
        return self._normalize(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string and return a 1-D unit vector.

        Returns
        -------
        np.ndarray of shape ``(dimension,)`` with dtype float32.
        """
        vec = self.encode([query])  # shape (1, dim)
        return vec[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality (from config, no model load needed)."""
        return self._config.dimension

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise each row in-place and return the array."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Guard against zero-length vectors.
        norms = np.maximum(norms, 1e-12)
        vectors /= norms
        return vectors
