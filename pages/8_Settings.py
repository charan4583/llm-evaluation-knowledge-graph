"""Page 8 — Application Settings, API Status, Data Management, and Privacy Notice."""

from __future__ import annotations

import streamlit as st

from config.settings import settings
from src.database.connection import get_session, init_db
from src.database.repository import EvaluationRepository
from src.nlp.entity_extraction import is_spacy_available
from src.ui.theme import apply_theme, render_page_header

st.set_page_config(page_title="Settings | LLM Eval & KG", page_icon="⚙️", layout="wide")
init_db()
apply_theme()

render_page_header(
    "Settings & System Status",
    "Inspect model configuration, API readiness, privacy guidelines, and local SQLite data management.",
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("API & Model Configuration")
    gemini_badge = "Configured" if settings.is_gemini_configured else "Not Configured (Offline Mode)"
    st.write(f"- **Gemini API Status:** {gemini_badge}")
    st.write(f"- **Gemini Model:** `{settings.gemini_model}`")
    st.write(f"- **Embedding Model:** `{settings.embedding_model}`")
    st.write(f"- **spaCy Model (`{settings.spacy_model}`):** {'Loaded' if is_spacy_available() else 'Fallback Mode (Rule-based)'}")
    st.caption("API keys are read strictly from environment variables (`.env`) and are never displayed in the UI.")

with c2:
    st.subheader("Evaluation Thresholds")
    if "similarity_threshold" not in st.session_state:
        st.session_state["similarity_threshold"] = settings.default_similarity_threshold
    thresh = st.slider(
        "Semantic Similarity Alert Threshold",
        min_value=0.1,
        max_value=0.95,
        value=float(st.session_state["similarity_threshold"]),
        step=0.05,
        help="Reference threshold used to highlight high vs. low semantic alignment.",
    )
    st.session_state["similarity_threshold"] = thresh
    st.write(f"- **Max Upload Size:** {settings.max_upload_size_mb} MB")

st.divider()
st.subheader("Data Management")
with get_session() as session:
    total = EvaluationRepository(session).get_dashboard_summary()["total_evaluations"]
st.write(f"Current saved evaluations in SQLite: **{total}**")

confirm_clear = st.checkbox("I understand this will permanently delete all saved evaluations, entities, and relationships.")
if st.button("Clear All Evaluation History", type="secondary", disabled=not confirm_clear or total == 0):
    with get_session() as session:
        deleted = EvaluationRepository(session).clear_all()
    st.success(f"Cleared {deleted} evaluation record(s) from the database.")
    st.rerun()

st.divider()
st.subheader("Privacy & Disclaimer")
st.info(
    "Core quantitative metrics, embeddings, entity extraction, and knowledge graph construction run locally on your machine. "
    "When optional Gemini AI Judge evaluation is enabled, submitted prompts and answers are sent to the Gemini API. "
    "Do not submit confidential or sensitive personal data to external APIs."
)
