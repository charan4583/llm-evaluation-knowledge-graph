"""Unit tests for claim analysis, mocked AI judge, and document upload validation."""

from unittest.mock import patch

import pytest

from src.ai.ai_judge import evaluate_with_ai_judge
from src.ai.hallucination_analysis import analyze_claims_rule_based
from src.document_processing.file_handler import FileValidationError, extract_text_from_upload


def test_claim_analysis_classifications():
    ref = "Python is a high-level programming language used for automation, web development, machine learning, and data analysis."
    gen = (
        "Python is a high-level programming language used for automation and data analysis. "
        "It was invented on Mars in 3050."
    )
    claims = analyze_claims_rule_based(ref, gen)
    assert len(claims) == 2
    assert claims[0]["classification"] == "Supported"
    assert claims[1]["classification"] == "Not found in reference"


def test_ai_judge_mocked_success():
    mock_payload = {
        "overall_score": 88,
        "correctness": 90,
        "relevance": 95,
        "completeness": 80,
        "clarity": 90,
        "consistency": 90,
        "verdict": "Strong response with minor omission",
        "strengths": ["Accurate core definition"],
        "missing_information": ["Web development"],
        "unsupported_claims": [],
        "improvement_suggestions": ["Mention web development"],
        "explanation": "Mostly accurate and clear.",
    }
    with patch("src.ai.ai_judge.generate_structured_json", return_value=mock_payload):
        result, err = evaluate_with_ai_judge("What is Python?", "Ref", "Gen")
    assert err is None
    assert result is not None
    assert result.overall_score == 88.0


def test_file_upload_validation():
    text = extract_text_from_upload("notes.txt", b"Hello world from TXT.")
    assert text == "Hello world from TXT."

    with pytest.raises(FileValidationError):
        extract_text_from_upload("malware.exe", b"123")

    with pytest.raises(FileValidationError):
        extract_text_from_upload("empty.txt", b"")
