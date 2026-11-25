"""Unit tests for CSV, JSON, and PDF report exports."""

import json

from src.export.csv_exporter import export_evaluations_to_csv
from src.export.json_exporter import export_evaluation_to_json
from src.export.pdf_report import generate_pdf_report


def test_csv_and_json_export():
    sample = {
        "id": 1,
        "created_at": "2026-11-25",
        "question": "What is Python?",
        "reference_answer": "A programming language.",
        "generated_answer": "Python is a language.",
        "precision": 0.8,
        "recall": 0.7,
        "f1_score": 0.75,
        "exact_match": 0.0,
        "semantic_similarity": 0.88,
        "keyword_coverage": 0.8,
        "ai_score": None,
        "entity_count": 1,
        "relationship_count": 0,
    }
    csv_bytes = export_evaluations_to_csv([sample])
    csv_text = csv_bytes.decode("utf-8")
    assert "What is Python?" in csv_text
    assert "0.75" in csv_text

    json_bytes = export_evaluation_to_json(sample)
    parsed = json.loads(json_bytes.decode("utf-8"))
    assert parsed["question"] == "What is Python?"
    assert parsed["f1_score"] == 0.75


def test_pdf_report_generation():
    pdf_bytes = generate_pdf_report(
        {
            "id": 1,
            "created_at": "2026-11-25",
            "question": "What is Python?",
            "reference_answer": "Python is a programming language.",
            "generated_answer": "Python is a high-level programming language.",
            "precision": 0.8,
            "recall": 1.0,
            "f1_score": 0.8889,
            "exact_match": 0.0,
            "semantic_similarity": 0.92,
            "keyword_coverage": 1.0,
        }
    )
    assert pdf_bytes.startswith(b"%PDF")
