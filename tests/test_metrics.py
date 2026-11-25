"""Unit tests for Precision, Recall, F1, Exact Match, Keyword Coverage, and edge cases."""

from src.evaluation.metrics import (
    compute_exact_match,
    compute_keyword_coverage,
    compute_precision_recall_f1,
)


def test_identical_answers():
    ref = "Python is a programming language used for machine learning and automation."
    gen = "Python is a programming language used for machine learning and automation."
    res = compute_precision_recall_f1(ref, gen)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1_score"] == 1.0
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert compute_exact_match(ref, gen) == 1.0


def test_exact_match_normalization():
    ref = "Python is great!"
    gen = "  python   is great  "
    assert compute_exact_match(ref, gen) == 1.0


def test_completely_different_answers():
    ref = "Photosynthesis converts sunlight into chemical energy in plants."
    gen = "TCP/IP governs packet routing across computer networks."
    res = compute_precision_recall_f1(ref, gen)
    assert res["true_positives"] == 0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1_score"] == 0.0
    assert compute_exact_match(ref, gen) == 0.0


def test_partial_match():
    ref = "Python is used for automation, web development, and machine learning."
    gen = "Python is used for automation and robotics."
    res = compute_precision_recall_f1(ref, gen)
    assert 0.0 < res["precision"] < 1.0
    assert 0.0 < res["recall"] < 1.0
    assert 0.0 < res["f1_score"] < 1.0
    assert res["true_positives"] > 0


def test_empty_reference():
    res = compute_precision_recall_f1("", "Python is a language.")
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1_score"] == 0.0
    assert compute_exact_match("", "Python") == 0.0


def test_empty_generated():
    res = compute_precision_recall_f1("Python is a language.", "")
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1_score"] == 0.0
    assert compute_exact_match("Python", "") == 0.0


def test_both_empty_division_by_zero():
    res = compute_precision_recall_f1("", "")
    assert res["true_positives"] == 0
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1_score"] == 0.0
    assert compute_exact_match("", "") == 1.0


def test_keyword_coverage():
    ref = "Python supports automation, machine learning, and data analysis."
    gen = "Python is popular for machine learning and automation."
    cov = compute_keyword_coverage(ref, gen)
    assert "python" in cov["matched_keywords"]
    assert "automation" in cov["matched_keywords"]
    assert "data" in cov["missing_keywords"]
    assert 0.5 <= cov["coverage_score"] < 1.0
