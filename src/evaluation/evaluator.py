"""High-level evaluation orchestrator combining lexical metrics, semantic similarity, and diffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.evaluation.metrics import (
    compute_exact_match,
    compute_keyword_coverage,
    compute_precision_recall_f1,
)
from src.evaluation.semantic_similarity import compute_semantic_similarity
from src.nlp.text_processing import segment_sentences


@dataclass
class EvaluationResult:
    """Structured output of a complete response evaluation."""

    question: str
    reference_answer: str
    generated_answer: str
    precision: float
    recall: float
    f1_score: float
    exact_match: float
    semantic_similarity: float
    keyword_coverage: float
    prf_details: dict[str, Any] = field(default_factory=dict)
    keyword_details: dict[str, Any] = field(default_factory=dict)
    semantic_details: dict[str, Any] = field(default_factory=dict)
    missing_information: list[str] = field(default_factory=list)
    additional_information: list[str] = field(default_factory=list)
    claim_analysis: list[dict[str, Any]] = field(default_factory=list)
    ai_evaluation: Optional[dict[str, Any]] = None

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "exact_match": self.exact_match,
            "semantic_similarity": self.semantic_similarity,
            "keyword_coverage": self.keyword_coverage,
            "ai_score": self.ai_evaluation.get("overall_score") if self.ai_evaluation else None,
        }


def _identify_missing_and_additional_info(
    reference: str,
    generated: str,
    keyword_details: dict[str, Any],
    prf_details: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Identify missing reference concepts and additional generated concepts."""
    missing_items: list[str] = []
    additional_items: list[str] = []

    missing_kw = keyword_details.get("missing_keywords", [])
    if missing_kw:
        missing_items.append(f"Missing reference keywords: {', '.join(missing_kw)}")

    ref_sentences = segment_sentences(reference)
    gen_lower = generated.lower()
    for sent in ref_sentences:
        sent_words = [w for w in sent.lower().split() if len(w) > 4]
        if sent_words:
            overlap_count = sum(1 for w in sent_words if w.strip(".,!?;:") in gen_lower)
            if overlap_count / len(sent_words) < 0.3:
                missing_items.append(f"Reference point not clearly covered: \"{sent}\"")

    add_tokens = prf_details.get("additional_tokens", [])
    if add_tokens:
        preview = ", ".join(add_tokens[:15])
        suffix = "..." if len(add_tokens) > 15 else ""
        additional_items.append(f"Additional terms not in reference: {preview}{suffix}")

    return missing_items, additional_items


def evaluate_response(
    question: str,
    reference_answer: str,
    generated_answer: str,
    *,
    include_semantic: bool = True,
    embedding_model: str | None = None,
) -> EvaluationResult:
    """Run all quantitative evaluation metrics on a generated answer against a reference answer."""
    prf = compute_precision_recall_f1(reference_answer, generated_answer)
    exact = compute_exact_match(reference_answer, generated_answer)
    kw = compute_keyword_coverage(reference_answer, generated_answer)

    if include_semantic:
        sem = compute_semantic_similarity(
            reference_answer, generated_answer, model_name=embedding_model
        )
    else:
        sem = {
            "score": 0.0,
            "percentage": 0.0,
            "interpretation": "Semantic similarity skipped.",
            "error": None,
            "disclaimer": "Semantic similarity measures closeness of meaning, not factual truth.",
        }

    missing_info, additional_info = _identify_missing_and_additional_info(
        reference_answer, generated_answer, kw, prf
    )

    return EvaluationResult(
        question=question.strip(),
        reference_answer=reference_answer.strip(),
        generated_answer=generated_answer.strip(),
        precision=prf["precision"],
        recall=prf["recall"],
        f1_score=prf["f1_score"],
        exact_match=exact,
        semantic_similarity=sem["score"],
        keyword_coverage=kw["coverage_score"],
        prf_details=prf,
        keyword_details=kw,
        semantic_details=sem,
        missing_information=missing_info,
        additional_information=additional_info,
    )
