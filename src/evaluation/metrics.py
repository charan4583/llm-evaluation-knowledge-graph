"""Quantitative lexical evaluation metrics: Precision, Recall, F1, Exact Match, and Keyword Coverage."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.nlp.text_processing import extract_keywords, normalize_text, tokenize


def compute_exact_match(reference: str, generated: str) -> float:
    """Return 1.0 if normalized reference and generated answers match exactly, else 0.0."""
    norm_ref = normalize_text(reference)
    norm_gen = normalize_text(generated)
    if not norm_ref and not norm_gen:
        return 1.0
    if not norm_ref or not norm_gen:
        return 0.0
    return 1.0 if norm_ref == norm_gen else 0.0


def compute_precision_recall_f1(
    reference: str,
    generated: str,
    *,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
) -> dict[str, Any]:
    """Compute token-level Precision, Recall, and F1-score using multiset overlap.

    Definitions:
        TP = sum of overlapping token frequencies between reference and generated
        FP = tokens in generated not matched in reference
        FN = tokens in reference not matched in generated
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * Precision * Recall / (Precision + Recall)
    """
    ref_tokens = tokenize(reference, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    gen_tokens = tokenize(generated, remove_stopwords=remove_stopwords, lemmatize=lemmatize)

    ref_counts = Counter(ref_tokens)
    gen_counts = Counter(gen_tokens)

    overlap = ref_counts & gen_counts
    fp_counts = gen_counts - ref_counts
    fn_counts = ref_counts - gen_counts

    tp = sum(overlap.values())
    fp = sum(fp_counts.values())
    fn = sum(fn_counts.values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "reference_token_count": len(ref_tokens),
        "generated_token_count": len(gen_tokens),
        "matched_tokens": sorted(overlap.keys()),
        "missing_tokens": sorted(fn_counts.keys()),
        "additional_tokens": sorted(fp_counts.keys()),
    }


def compute_keyword_coverage(
    reference: str,
    generated: str,
    *,
    top_n: int = 15,
) -> dict[str, Any]:
    """Measure what fraction of key terms from the reference answer appear in the generated answer."""
    ref_keywords_ranked = extract_keywords(reference, top_n=top_n)
    ref_keywords = [kw for kw, _ in ref_keywords_ranked]

    if not ref_keywords:
        return {
            "reference_keywords": [],
            "matched_keywords": [],
            "missing_keywords": [],
            "coverage_score": 0.0,
        }

    gen_tokens = set(tokenize(generated, remove_stopwords=True, lemmatize=False))
    gen_lemmas = set(tokenize(generated, remove_stopwords=True, lemmatize=True))

    matched: list[str] = []
    missing: list[str] = []
    for kw in ref_keywords:
        if kw in gen_tokens or kw in gen_lemmas:
            matched.append(kw)
        else:
            missing.append(kw)

    coverage = len(matched) / len(ref_keywords)
    return {
        "reference_keywords": ref_keywords,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "coverage_score": round(coverage, 4),
    }
