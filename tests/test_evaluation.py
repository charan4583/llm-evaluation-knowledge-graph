"""Unit tests for evaluation orchestrator and semantic similarity helpers."""

import numpy as np

from src.evaluation.evaluator import evaluate_response
from src.evaluation.semantic_similarity import compute_semantic_similarity, cosine_similarity


def test_cosine_similarity_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == 1.0
    assert cosine_similarity(a, c) == 0.0
    assert cosine_similarity(np.array([]), b) == 0.0


def test_semantic_similarity_empty_inputs():
    res = compute_semantic_similarity("", "Some answer")
    assert res["score"] == 0.0
    assert res["percentage"] == 0.0


def test_evaluate_response_without_semantic():
    result = evaluate_response(
        question="What is Python?",
        reference_answer="Python is a high-level programming language.",
        generated_answer="Python is a high-level programming language.",
        include_semantic=False,
    )
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0
    assert result.exact_match == 1.0
    assert result.keyword_coverage == 1.0
