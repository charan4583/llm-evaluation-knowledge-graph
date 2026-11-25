"""Unit tests for text normalization, tokenization, stopword filtering, and keyword extraction."""

from src.nlp.text_processing import (
    compute_text_stats,
    extract_keywords,
    normalize_text,
    segment_sentences,
    tokenize,
)


def test_normalize_text_basic():
    raw = "  Hello,   World!  Python 3.12 is GREAT. "
    assert normalize_text(raw) == "hello world python 312 is great"


def test_normalize_text_empty():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_tokenize_stopword_removal():
    tokens = tokenize("Python is a high-level language for data analysis.")
    assert "is" not in tokens
    assert "a" not in tokens
    assert "for" not in tokens
    assert "python" in tokens
    assert "data" in tokens
    assert "analysis" in tokens


def test_tokenize_keep_stopwords():
    tokens = tokenize("Python is great", remove_stopwords=False)
    assert tokens == ["python", "is", "great"]


def test_segment_sentences():
    text = "Google developed Gemini. Gemini processes text and images! Does it support code?"
    sents = segment_sentences(text)
    assert len(sents) == 3


def test_extract_keywords_ranking():
    text = "Python supports machine learning. Machine learning uses Python libraries."
    keywords = dict(extract_keywords(text, top_n=5))
    assert keywords["python"] == 2
    assert keywords["machine"] == 2
    assert keywords["learning"] == 2


def test_compute_text_stats():
    stats = compute_text_stats("Python is fast. Python is readable.")
    assert stats["word_count"] == 6
    assert stats["sentence_count"] == 2
    assert stats["avg_sentence_length"] == 3.0
