"""Sentence-Transformers embedding loader with thread-safe caching and error handling."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingModelError(RuntimeError):
    """Raised when the sentence embedding model cannot be loaded or invoked."""


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str | None = None):
    """Load and cache the SentenceTransformer model."""
    target_model = model_name or settings.embedding_model
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(target_model)
    except Exception as exc:
        logger.warning("Failed to load embedding model %s: %s", target_model, exc)
        raise EmbeddingModelError(
            f"Could not load embedding model '{target_model}': {exc}"
        ) from exc


def encode_texts(texts: list[str], model_name: Optional[str] = None) -> np.ndarray:
    """Encode a list of texts into normalized L2 embedding vectors."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = load_embedding_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)
