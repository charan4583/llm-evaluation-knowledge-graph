"""Claim-level verification against reference answers (rule-based baseline + optional AI)."""

from __future__ import annotations

from typing import Any

from src.nlp.text_processing import segment_sentences, tokenize

CLAIM_DISCLAIMER = (
    "Reference-based claim analysis checks whether statements are supported by the provided "
    "reference answer. 'Not found in reference' does not prove a claim is factually false."
)


def analyze_claims_rule_based(
    reference_answer: str,
    generated_answer: str,
) -> list[dict[str, Any]]:
    """Segment the generated answer into claims and compare each against reference sentences."""
    gen_sentences = segment_sentences(generated_answer)
    if not gen_sentences:
        return []

    ref_sentences = segment_sentences(reference_answer)
    ref_token_sets = [
        set(tokenize(sent, remove_stopwords=True, lemmatize=True))
        for sent in ref_sentences
    ]
    all_ref_tokens = set().union(*ref_token_sets) if ref_token_sets else set()

    results: list[dict[str, Any]] = []
    for claim in gen_sentences:
        claim_tokens = set(tokenize(claim, remove_stopwords=True, lemmatize=True))
        if not claim_tokens:
            results.append(
                {
                    "claim_text": claim,
                    "classification": "Requires verification",
                    "evidence": None,
                    "explanation": "Sentence contains too few content terms for lexical comparison.",
                    "confidence": None,
                }
            )
            continue

        best_ratio = 0.0
        best_sentence = None
        for ref_sent, ref_tokens in zip(ref_sentences, ref_token_sets):
            if not ref_tokens:
                continue
            overlap = len(claim_tokens & ref_tokens)
            ratio = overlap / len(claim_tokens)
            if ratio > best_ratio:
                best_ratio = ratio
                best_sentence = ref_sent

        overall_overlap = len(claim_tokens & all_ref_tokens) / len(claim_tokens) if all_ref_tokens else 0.0
        effective_score = max(best_ratio, overall_overlap)

        if effective_score >= 0.75:
            classification = "Supported"
            explanation = "Most key concepts in this claim align with the reference answer."
        elif effective_score >= 0.35:
            classification = "Partially supported"
            explanation = (
                "Some concepts appear in the reference, but additional or unverified details are present."
            )
        elif effective_score > 0.0:
            classification = "Requires verification"
            explanation = (
                "Minimal lexical overlap with the reference; verify whether this adds valid context or unsupported claims."
            )
        else:
            classification = "Not found in reference"
            explanation = (
                "No supporting evidence found in the reference answer. "
                "This does not automatically mean the claim is factually incorrect."
            )

        results.append(
            {
                "claim_text": claim,
                "classification": classification,
                "evidence": best_sentence if best_ratio > 0 else None,
                "explanation": explanation,
                "confidence": round(effective_score, 2),
            }
        )

    return results
