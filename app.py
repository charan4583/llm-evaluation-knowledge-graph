"""Main Streamlit entrypoint for LLM Evaluation & Knowledge Graph Explorer."""

from __future__ import annotations

import streamlit as st

from config.settings import settings
from src.database.connection import init_db
from src.ui.theme import apply_theme, render_page_header

st.set_page_config(
    page_title=settings.app_title,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
apply_theme()

render_page_header(
    settings.app_title,
    "Quantitative LLM response evaluation, semantic similarity, claim verification, and interactive knowledge graph analytics.",
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="app-card">
            <div class="app-card-title">1. Quantitative & Semantic Evaluation</div>
            <p class="app-muted">
                Compute Precision, Recall, F1-score, Exact Match, Keyword Coverage, and Sentence-Transformer
                semantic similarity against reference answers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="app-card">
            <div class="app-card-title">2. Claim & AI-Assisted Analysis</div>
            <p class="app-muted">
                Inspect sentence-level claims against reference evidence, identify missing concepts,
                and optionally run structured Gemini AI Judge evaluations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div class="app-card">
            <div class="app-card-title">3. Knowledge Graph Explorer</div>
            <p class="app-muted">
                Extract entities and directed relationships from answers or uploaded PDF/DOCX/TXT documents
                and explore them in an interactive NetworkX + PyVis graph.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Quick Navigation")
nav_cols = st.columns(4)
with nav_cols[0]:
    st.page_link("pages/1_Dashboard.py", label="Overview Dashboard", icon="📊")
with nav_cols[1]:
    st.page_link("pages/2_Evaluate_Response.py", label="Evaluate Response", icon="🧪")
with nav_cols[2]:
    st.page_link("pages/3_Compare_Responses.py", label="Compare Responses", icon="⚖️")
with nav_cols[3]:
    st.page_link("pages/4_Knowledge_Graph.py", label="Knowledge Graph", icon="🕸️")

nav_cols_2 = st.columns(4)
with nav_cols_2[0]:
    st.page_link("pages/5_Text_Analysis.py", label="Text Analysis", icon="📝")
with nav_cols_2[1]:
    st.page_link("pages/6_History.py", label="Evaluation History", icon="🗂️")
with nav_cols_2[2]:
    st.page_link("pages/7_Reports.py", label="Report Export", icon="📄")
with nav_cols_2[3]:
    st.page_link("pages/8_Settings.py", label="Settings", icon="⚙️")

with st.sidebar:
    st.divider()
    gemini_status = "Configured" if settings.is_gemini_configured else "Not Configured (Offline Mode)"
    st.caption(f"Gemini API: **{gemini_status}**")
    st.caption(f"Embedding Model: `{settings.embedding_model}`")
