"""Page 5 — Text Analysis & Side-by-Side Comparison Explorer."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.document_processing.file_handler import FileValidationError, extract_text_from_upload
from src.evaluation.evaluator import evaluate_response
from src.nlp.entity_extraction import extract_entities
from src.nlp.text_processing import compute_text_stats, tokenize
from src.ui.theme import apply_theme, render_page_header

st.set_page_config(page_title="Text Analysis | LLM Eval & KG", page_icon="📝", layout="wide")
apply_theme()

render_page_header(
    "Text Analysis & Comparison Explorer",
    "Analyze single texts or uploaded documents, or compare two texts at lexical, keyword, and semantic levels.",
)

mode = st.radio("Mode", options=["Single Text / Document Analysis", "Two-Text Comparison"], horizontal=True)
plotly_template = "plotly_dark" if st.session_state.get("ui_theme", "dark") == "dark" else "plotly_white"

if mode == "Single Text / Document Analysis":
    src_opt = st.radio("Input method", ["Paste Text", "Upload File (PDF/DOCX/TXT)"], horizontal=True)
    text_input = ""
    if src_opt == "Paste Text":
        text_input = st.text_area("Text to analyze", height=180, placeholder="Paste text here...")
    else:
        up = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], key="ta_upload")
        if up is not None:
            try:
                text_input = extract_text_from_upload(up.name, up.getvalue())
                with st.expander("Extracted Text Preview", expanded=False):
                    st.text(text_input[:3000] + ("..." if len(text_input) > 3000 else ""))
            except FileValidationError as exc:
                st.error(str(exc))

    if st.button("Analyze Text", type="primary"):
        if not text_input.strip():
            st.error("Please provide non-empty text.")
        else:
            stats = compute_text_stats(text_input)
            entities = extract_entities(text_input)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Word Count", stats["word_count"])
            m2.metric("Character Count", stats["char_count"])
            m3.metric("Sentence Count", stats["sentence_count"])
            m4.metric("Avg Sentence Length", f"{stats['avg_sentence_length']} words")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Top Keywords")
                if stats["top_keywords"]:
                    kw_df = pd.DataFrame(stats["top_keywords"], columns=["Keyword", "Frequency"])
                    fig = px.bar(kw_df, x="Frequency", y="Keyword", orientation="h", template=plotly_template)
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No meaningful keywords found.")
            with c2:
                st.subheader("Entity Distribution")
                if entities:
                    ent_df = pd.DataFrame(
                        [{"Entity": e["entity_name"], "Type": e["entity_type"], "Mentions": e["frequency"]} for e in entities]
                    )
                    st.dataframe(ent_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No entities extracted.")

else:
    col_a, col_b = st.columns(2)
    with col_a:
        text_a = st.text_area("Reference Text (Text A)", height=180, placeholder="Paste reference text...")
    with col_b:
        text_b = st.text_area("Generated / Comparison Text (Text B)", height=180, placeholder="Paste comparison text...")

    if st.button("Compare Texts", type="primary"):
        if not text_a.strip() or not text_b.strip():
            st.error("Please enter both texts to compare.")
        else:
            with st.spinner("Computing lexical and semantic comparison..."):
                res = evaluate_response("Comparison", text_a, text_b, include_semantic=True)
                toks_a = set(tokenize(text_a, remove_stopwords=True, lemmatize=False))
                toks_b = set(tokenize(text_b, remove_stopwords=True, lemmatize=False))
                shared = sorted(toks_a & toks_b)
                unique_a = sorted(toks_a - toks_b)
                unique_b = sorted(toks_b - toks_a)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Semantic Similarity", f"{res.semantic_similarity:.1%}")
                m2.metric("F1-Score", f"{res.f1_score:.1%}")
                m3.metric("Keyword Coverage", f"{res.keyword_coverage:.1%}")
                m4.metric("Shared Key Terms", len(shared))

                k1, k2, k3 = st.columns(3)
                k1.subheader("Shared Keywords")
                k1.write(", ".join(shared) or "None")
                k2.subheader("Unique to Reference (Missing)")
                k2.write(", ".join(unique_a) or "None")
                k3.subheader("Unique to Comparison (Additional)")
                k3.write(", ".join(unique_b) or "None")

                st.subheader("Missing Reference Points")
                for item in res.missing_information:
                    st.write(f"- {item}")
                if not res.missing_information:
                    st.success("All major reference points appear covered.")
