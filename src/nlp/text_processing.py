"""Text normalization, tokenization, stopword filtering, lemmatization, and keyword extraction."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

# Curated English stopword set so core NLP works reliably without external corpus downloads.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
        "could", "did", "do", "does", "doing", "done", "for", "from", "had", "has",
        "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
        "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "may", "me", "might", "more", "most", "must", "my", "myself", "no", "nor",
        "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
        "ourselves", "out", "over", "own", "same", "shall", "she", "should", "so",
        "some", "such", "than", "that", "the", "their", "theirs", "them",
        "themselves", "then", "there", "these", "they", "this", "those", "through",
        "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
        "when", "where", "which", "while", "who", "whom", "why", "will", "with",
        "would", "you", "your", "yours", "yourself", "yourselves",
    }
)

_PUNCT_TRANS = str.maketrans("", "", string.punctuation)


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Normalize text by collapsing whitespace, optionally lowercasing and stripping punctuation."""
    if not text:
        return ""
    cleaned = text.strip()
    if lowercase:
        cleaned = cleaned.lower()
    if remove_punctuation:
        cleaned = cleaned.translate(_PUNCT_TRANS)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def simple_lemmatize(token: str) -> str:
    """Lightweight deterministic suffix normalization for baseline lexical matching."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses") or token.endswith("ches") or token.endswith("shes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) >= 3 and stem[-1] == stem[-2]:
            return stem[:-1]
        return stem
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    return token


def tokenize(
    text: str,
    *,
    remove_stopwords: bool = True,
    lemmatize: bool = False,
) -> list[str]:
    """Tokenize normalized text into word tokens with optional stopword removal and lemmatization."""
    normalized = normalize_text(text, lowercase=True, remove_punctuation=True)
    if not normalized:
        return []
    tokens = [tok for tok in normalized.split(" ") if tok]
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in STOPWORDS]
    if lemmatize:
        tokens = [simple_lemmatize(tok) for tok in tokens]
    return tokens


def segment_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences."""
    if not text or not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def extract_keywords(text: str, top_n: int = 15) -> list[tuple[str, int]]:
    """Extract meaningful content keywords ranked by frequency, filtering stopwords and 1-char noise."""
    tokens = tokenize(text, remove_stopwords=True, lemmatize=False)
    meaningful = [tok for tok in tokens if len(tok) > 1 and not tok.isdigit()]
    if not meaningful:
        return []
    counts = Counter(meaningful)
    return counts.most_common(top_n)


def compute_text_stats(text: str) -> dict[str, Any]:
    """Compute basic descriptive statistics for a given text."""
    if not text or not text.strip():
        return {
            "word_count": 0,
            "char_count": 0,
            "char_count_no_spaces": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "unique_word_count": 0,
            "top_keywords": [],
        }

    words = tokenize(text, remove_stopwords=False, lemmatize=False)
    sentences = segment_sentences(text)
    sentence_count = len(sentences)
    word_count = len(words)
    avg_sentence_length = round(word_count / sentence_count, 2) if sentence_count > 0 else 0.0

    return {
        "word_count": word_count,
        "char_count": len(text),
        "char_count_no_spaces": len(re.sub(r"\s+", "", text)),
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "unique_word_count": len(set(words)),
        "top_keywords": extract_keywords(text, top_n=15),
    }
