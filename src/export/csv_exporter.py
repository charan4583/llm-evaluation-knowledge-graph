"""CSV exporter for evaluation records and summary tables."""

from __future__ import annotations

import csv
import io
from typing import Any


def export_evaluations_to_csv(records: list[dict[str, Any]]) -> bytes:
    """Serialize a list of evaluation dictionaries into UTF-8 CSV bytes."""
    output = io.StringIO()
    fieldnames = [
        "id",
        "created_at",
        "question",
        "reference_answer",
        "generated_answer",
        "precision",
        "recall",
        "f1_score",
        "exact_match",
        "semantic_similarity",
        "keyword_coverage",
        "ai_score",
        "entity_count",
        "relationship_count",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in records:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")
