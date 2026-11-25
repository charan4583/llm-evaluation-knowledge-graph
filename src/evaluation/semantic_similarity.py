"""Semantic similarity calculation using pretrained sentence embeddings and cosine similarity."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.nlp.embeddings import EmbeddingModelError, encode_texts


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors safely."""
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    score = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, score))


def interpret_similarity(score: float) -> str:
    """Return a concise human-readable interpretation of cosine similarity."""
    if score >= 0.85:
        return "Very high semantic alignment — both texts convey closely matching meaning."
    if score >= 0.70:
        return "Moderate-to-high semantic alignment — core ideas overlap substantially."
    if score >= 0.50:
        return "Partial semantic alignment — some shared concepts, noticeable divergence."
    return "Low semantic alignment — meanings differ significantly."


def compute_semantic_similarity(
    reference: str,
    generated: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Calculate cosine similarity between reference and generated answer embeddings."""
    if not reference.strip() or not generated.strip():
        return {
            "score": 0.0,
            "percentage": 0.0,
            "interpretation": "Cannot compute semantic similarity because one or both inputs are empty.",
            "error": None,
            "disclaimer": (
                "Semantic similarity measures closeness of meaning in embedding space, "
                "not factual truth."
            ),
        }

    try:
        vectors = encode_texts([reference.strip(), generated.strip()], model_name=model_name)
        score = cosine_similarity(vectors[0], vectors[1])
        return {
            "score": round(score, 4),
            "percentage": round(score * 100.0, 2),
            "interpretation": interpret_similarity(score),
            "error": None,
            "disclaimer": (
                "Semantic similarity measures closeness of meaning in embedding space, "
                "not factual truth. A fluent but inaccurate answer can still score high."
            ),
        }
    except EmbeddingModelError as exc:
        return {
            "score": 0.0,
            "percentage": 0.0,
            "interpretation": "Embedding model unavailable.",
            "error": str(exc),
            "disclaimer": (
                "Semantic similarity measures closeness of meaning in embedding space, "
                "not factual truth."
            ),
        }
