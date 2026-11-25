"""Page 7 — Report Export (CSV, JSON, and formatted PDF)."""

from __future__ import annotations

import json

import streamlit as st

from src.database.connection import get_session, init_db
from src.database.repository import EvaluationRepository
from src.export.csv_exporter import export_evaluations_to_csv
from src.export.json_exporter import export_evaluation_to_json
from src.export.pdf_report import generate_pdf_report
from src.ui.theme import apply_theme, render_page_header

st.set_page_config(page_title="Reports | LLM Eval & KG", page_icon="📄", layout="wide")
init_db()
apply_theme()

render_page_header(
    "Report Export",
    "Download evaluation results, knowledge graph extractions, and analysis reports in CSV, JSON, or PDF format.",
)

with get_session() as session:
    records = EvaluationRepository(session).list_evaluations(limit=100)

if not records:
    st.info("No saved evaluations found. Evaluate and save a response first to generate reports.")
    st.page_link("pages/2_Evaluate_Response.py", label="Go to Evaluate Response", icon="🧪")
    st.stop()

# Bulk CSV export of all saved records
all_csv_rows = [
    {
        "id": r.id,
        "created_at": r.created_at.isoformat() if r.created_at else "",
        "question": r.question,
        "reference_answer": r.reference_answer,
        "generated_answer": r.generated_answer,
        "precision": r.precision,
        "recall": r.recall,
        "f1_score": r.f1_score,
        "exact_match": r.exact_match,
        "semantic_similarity": r.semantic_similarity,
        "keyword_coverage": r.keyword_coverage,
        "ai_score": r.ai_score,
        "entity_count": len(r.entities),
        "relationship_count": len(r.relationships),
    }
    for r in records
]

st.subheader("Export All Saved Evaluations")
st.download_button(
    label="Download All Evaluations (CSV)",
    data=export_evaluations_to_csv(all_csv_rows),
    file_name="llm_evaluations_summary.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Export Individual Evaluation Report")
options = {f"#{r.id} — {r.question[:75]}": r for r in records}
selected_label = st.selectbox("Select evaluation record", list(options.keys()))
rec = options[selected_label]

meta = json.loads(rec.metadata_json) if rec.metadata_json else {}
ai_fb = json.loads(rec.ai_feedback_json) if rec.ai_feedback_json else None

payload = {
    "id": rec.id,
    "created_at": rec.created_at.strftime("%Y-%m-%d %H:%M UTC") if rec.created_at else "",
    "question": rec.question,
    "reference_answer": rec.reference_answer,
    "generated_answer": rec.generated_answer,
    "precision": rec.precision,
    "recall": rec.recall,
    "f1_score": rec.f1_score,
    "exact_match": rec.exact_match,
    "semantic_similarity": rec.semantic_similarity,
    "keyword_coverage": rec.keyword_coverage,
    "ai_score": rec.ai_score,
    "ai_feedback": ai_fb,
    "missing_information": meta.get("missing_information", []),
    "claim_analysis": meta.get("claim_analysis", []),
    "entities": [
        {
            "entity_name": e.entity_name,
            "entity_type": e.entity_type,
            "frequency": e.frequency,
            "description": e.description,
        }
        for e in rec.entities
    ],
    "relationships": [
        {
            "source_entity": rel.source_entity,
            "relationship_type": rel.relationship_type,
            "target_entity": rel.target_entity,
            "evidence_sentence": rel.evidence_sentence,
            "confidence": rel.confidence,
            "extraction_method": rel.extraction_method,
        }
        for rel in rec.relationships
    ],
}

c1, c2, c3 = st.columns(3)
c1.download_button(
    label="Download CSV (Single Record)",
    data=export_evaluations_to_csv([{**payload, "entity_count": len(rec.entities), "relationship_count": len(rec.relationships)}]),
    file_name=f"evaluation_{rec.id}.csv",
    mime="text/csv",
    use_container_width=True,
)
c2.download_button(
    label="Download JSON (Full Payload)",
    data=export_evaluation_to_json(payload),
    file_name=f"evaluation_{rec.id}.json",
    mime="application/json",
    use_container_width=True,
)
c3.download_button(
    label="Download PDF Report",
    data=generate_pdf_report(payload),
    file_name=f"evaluation_report_{rec.id}.pdf",
    mime="application/pdf",
    use_container_width=True,
)
