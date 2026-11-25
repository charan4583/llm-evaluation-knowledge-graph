"""Page 6 — Evaluation History backed by SQLite + SQLAlchemy."""

from __future__ import annotations

import json
from datetime import datetime, time, timezone

import streamlit as st

from src.database.connection import get_session, init_db
from src.database.repository import EvaluationRepository
from src.ui.theme import apply_theme, render_page_header

st.set_page_config(page_title="History | LLM Eval & KG", page_icon="🗂️", layout="wide")
init_db()
apply_theme()

render_page_header(
    "Evaluation History",
    "Browse, search, filter, compare, and manage saved evaluations stored in your local SQLite database.",
)

f1, f2, f3 = st.columns([3, 1.5, 1.5])
search_q = f1.text_input("Search question or answers", placeholder="Type keyword...")
use_date_filter = f2.checkbox("Filter by date range", value=False)
start_dt = None
end_dt = None
if use_date_filter:
    d_start = f2.date_input("From date")
    d_end = f3.date_input("To date")
    start_dt = datetime.combine(d_start, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(d_end, time.max, tzinfo=timezone.utc)

with get_session() as session:
    repo = EvaluationRepository(session)
    records = repo.list_evaluations(search_query=search_q, start_date=start_dt, end_date=end_dt, limit=100)

if not records:
    st.info("No saved evaluations match your criteria.")
    st.stop()

st.write(f"Showing **{len(records)}** saved evaluation(s).")

# Quick Compare 2 Saved Evaluations
if len(records) >= 2:
    with st.expander("Compare Two Saved Evaluations"):
        options_map = {f"#{r.id}: {r.question[:60]}": r for r in records}
        c_col1, c_col2 = st.columns(2)
        sel_a = c_col1.selectbox("Evaluation A", list(options_map.keys()), index=0)
        sel_b = c_col2.selectbox("Evaluation B", list(options_map.keys()), index=1)
        ev_a, ev_b = options_map[sel_a], options_map[sel_b]
        st.table(
            [
                {"Metric": "Precision", "Evaluation A": f"{ev_a.precision:.1%}", "Evaluation B": f"{ev_b.precision:.1%}"},
                {"Metric": "Recall", "Evaluation A": f"{ev_a.recall:.1%}", "Evaluation B": f"{ev_b.recall:.1%}"},
                {"Metric": "F1-Score", "Evaluation A": f"{ev_a.f1_score:.1%}", "Evaluation B": f"{ev_b.f1_score:.1%}"},
                {"Metric": "Semantic Similarity", "Evaluation A": f"{ev_a.semantic_similarity:.1%}", "Evaluation B": f"{ev_b.semantic_similarity:.1%}"},
                {"Metric": "Keyword Coverage", "Evaluation A": f"{ev_a.keyword_coverage:.1%}", "Evaluation B": f"{ev_b.keyword_coverage:.1%}"},
            ]
        )

for ev in records:
    ts = ev.created_at.strftime("%Y-%m-%d %H:%M UTC") if ev.created_at else ""
    header = f"#{ev.id} | {ev.question[:80]} | F1: {ev.f1_score:.1%} | Semantic: {ev.semantic_similarity:.1%} | {ts}"
    with st.expander(header):
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Precision", f"{ev.precision:.1%}")
        m2.metric("Recall", f"{ev.recall:.1%}")
        m3.metric("F1-Score", f"{ev.f1_score:.1%}")
        m4.metric("Exact Match", f"{ev.exact_match:.0%}")
        m5.metric("Semantic", f"{ev.semantic_similarity:.1%}")
        m6.metric("Keywords", f"{ev.keyword_coverage:.1%}")

        st.write("**Question:**", ev.question)
        ca, cb = st.columns(2)
        ca.write("**Reference Answer:**")
        ca.write(ev.reference_answer)
        cb.write("**Generated Answer:**")
        cb.write(ev.generated_answer)

        if ev.ai_feedback_json:
            st.write("**AI Judge Feedback:**")
            st.json(json.loads(ev.ai_feedback_json))

        st.write(f"**Entities ({len(ev.entities)}):** " + (", ".join(e.entity_name for e in ev.entities) or "None"))
        st.write(
            f"**Relationships ({len(ev.relationships)}):** "
            + (", ".join(f"{r.source_entity}→{r.relationship_type}→{r.target_entity}" for r in ev.relationships) or "None")
        )

        act1, act2, _ = st.columns([1.5, 1.5, 4])
        if act1.button("Open Graph in Explorer", key=f"open_kg_{ev.id}"):
            st.session_state["kg_entities"] = [
                {
                    "entity_name": e.entity_name,
                    "entity_type": e.entity_type,
                    "frequency": e.frequency,
                    "description": e.description,
                    "occurrences": [],
                }
                for e in ev.entities
            ]
            st.session_state["kg_relationships"] = [
                {
                    "source_entity": r.source_entity,
                    "relationship_type": r.relationship_type,
                    "target_entity": r.target_entity,
                    "evidence_sentence": r.evidence_sentence,
                    "confidence": r.confidence,
                    "extraction_method": r.extraction_method,
                }
                for r in ev.relationships
            ]
            st.switch_page("pages/4_Knowledge_Graph.py")

        if act2.button("Delete Record", key=f"del_ev_{ev.id}"):
            with get_session() as session:
                EvaluationRepository(session).delete_evaluation(ev.id)
            st.success(f"Deleted evaluation #{ev.id}.")
            st.rerun()
