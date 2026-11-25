"""Entity extraction combining spaCy NER with domain-specific AI/tech rules and a pure-Python fallback."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from config.settings import settings
from src.nlp.text_processing import segment_sentences

logger = logging.getLogger(__name__)

# Domain dictionaries for reliable AI/ML/Software entity typing
_PROGRAMMING_LANGUAGES = {
    "python", "java", "javascript", "typescript", "c++", "c#", "rust", "go", "golang",
    "r", "sql", "scala", "julia", "ruby", "php", "swift", "kotlin",
}
_AI_MODELS = {
    "gemini", "gpt", "gpt-3", "gpt-4", "gpt-4o", "chatgpt", "claude", "llama", "llama 3",
    "mistral", "bert", "roberta", "t5", "all-minilm-l6-v2", "whisper", "dall-e", "palm",
}
_TECHNOLOGIES = {
    "generative ai", "machine learning", "deep learning", "natural language processing",
    "nlp", "artificial intelligence", "ai", "computer vision", "transformers",
    "sentence transformers", "knowledge graph", "knowledge graphs", "rag",
    "retrieval augmented generation", "neural network", "neural networks",
    "streamlit", "sqlalchemy", "sqlite", "networkx", "pyvis", "plotly", "spacy",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "docker", "fastapi",
}
_ORGANIZATIONS = {
    "google", "openai", "anthropic", "microsoft", "meta", "amazon", "aws",
    "nvidia", "hugging face", "ibm", "cgi", "apple", "deepmind",
}
_DATA_TYPES = {
    "text", "images", "image", "audio", "video", "code", "json", "csv", "pdf", "docx",
}
_DATASETS = {
    "imagenet", "squad", "glue", "mmlu", "gsm8k", "humaneval", "wikipedia", "common crawl",
}
_METHODS = {
    "cosine similarity", "named entity recognition", "ner", "tokenization",
    "lemmatization", "embeddings", "semantic similarity", "fine-tuning", "prompting",
    "rag", "exact match", "f1-score", "precision", "recall",
}

_SPACY_LABEL_MAP = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "PRODUCT": "Product",
    "WORK_OF_ART": "Product",
    "FAC": "Location",
    "NORP": "Concept",
    "LANGUAGE": "Programming Language",
}


@lru_cache(maxsize=2)
def _load_spacy_nlp(model_name: str):
    """Attempt to load spaCy model; return None if unavailable."""
    try:
        import spacy

        return spacy.load(model_name)
    except Exception as exc:
        logger.warning("spaCy model '%s' unavailable (%s); using rule-based fallback.", model_name, exc)
        return None


def is_spacy_available() -> bool:
    """Return True if the configured spaCy model is installed and loadable."""
    return _load_spacy_nlp(settings.spacy_model) is not None


def _classify_domain_entity(text: str, default_type: str = "Other") -> str:
    lower = text.strip().lower()
    if lower in _ORGANIZATIONS:
        return "Organization"
    if lower in _PROGRAMMING_LANGUAGES:
        return "Programming Language"
    if lower in _AI_MODELS:
        return "AI Model"
    if lower in _DATASETS:
        return "Dataset"
    if lower in _METHODS:
        return "Method"
    if lower in _DATA_TYPES:
        return "Data Type"
    if lower in _TECHNOLOGIES:
        return "Technology"
    return default_type


def _find_occurrences(text: str, entity_name: str) -> list[dict[str, Any]]:
    """Find character positions and supporting sentences for an entity in text."""
    occurrences: list[dict[str, Any]] = []
    sentences = segment_sentences(text)
    pattern = re.compile(rf"\b{re.escape(entity_name)}\b", re.IGNORECASE)
    for idx, sent in enumerate(sentences):
        if pattern.search(sent):
            occurrences.append({"sentence_index": idx, "sentence": sent})
    return occurrences


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Extract, normalize, and deduplicate entities from text.

    Uses spaCy NER when available plus domain-specific pattern matching,
    and falls back gracefully to rule-based extraction if spaCy is missing.
    """
    if not text or not text.strip():
        return []

    raw_candidates: list[tuple[str, str]] = []

    nlp = _load_spacy_nlp(settings.spacy_model)
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            ent_text = ent.text.strip()
            if len(ent_text) < 2:
                continue
            mapped_type = _SPACY_LABEL_MAP.get(ent.label_, "Other")
            mapped_type = _classify_domain_entity(ent_text, mapped_type)
            raw_candidates.append((ent_text, mapped_type))

    # Domain dictionary matching (ensures technical terms like 'Gemini', 'Python', 'Text', 'Images' are captured)
    all_domain_terms = (
        _ORGANIZATIONS
        | _PROGRAMMING_LANGUAGES
        | _AI_MODELS
        | _TECHNOLOGIES
        | _DATA_TYPES
        | _DATASETS
        | _METHODS
    )
    for term in sorted(all_domain_terms, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            matched_str = match.group(0).strip()
            ent_type = _classify_domain_entity(matched_str, "Concept")
            raw_candidates.append((matched_str, ent_type))

    # Fallback capitalized multi-word / proper-noun extraction when spaCy is missing or sparse
    cap_pattern = re.compile(r"\b([A-Z][a-zA-Z0-9\-\+]+(?:\s+[A-Z][a-zA-Z0-9\-\+]+){0,2})\b")
    for match in cap_pattern.finditer(text):
        phrase = match.group(1).strip()
        if len(phrase) < 2 or phrase.lower() in {"the", "this", "that", "these", "those", "what", "when", "where"}:
            continue
        # Only add via capitalized heuristic if spaCy is unavailable or term is domain-known
        domain_type = _classify_domain_entity(phrase, "")
        if domain_type:
            raw_candidates.append((phrase, domain_type))
        elif nlp is None and match.start() > 0:
            raw_candidates.append((phrase, "Concept"))

    # Deduplicate by normalized lowercase name, preserving best casing & type
    merged: dict[str, dict[str, Any]] = {}
    for name, ent_type in raw_candidates:
        clean_name = re.sub(r"\s+", " ", name).strip(" .,;:!?\"'()")
        if len(clean_name) < 2:
            continue
        key = clean_name.lower()
        if key not in merged:
            occurrences = _find_occurrences(text, clean_name)
            freq = max(1, len(occurrences))
            merged[key] = {
                "entity_name": clean_name,
                "entity_type": ent_type,
                "frequency": freq,
                "occurrences": occurrences,
                "description": f"{ent_type} mentioned {freq} time(s) in source text.",
            }
        else:
            # Prefer more specific type over 'Other' or 'Concept'
            existing_type = merged[key]["entity_type"]
            if existing_type in {"Other", "Concept"} and ent_type not in {"Other", "Concept"}:
                merged[key]["entity_type"] = ent_type
                freq = merged[key]["frequency"]
                merged[key]["description"] = f"{ent_type} mentioned {freq} time(s) in source text."

    results = sorted(merged.values(), key=lambda x: (-x["frequency"], x["entity_name"].lower()))
    return results
