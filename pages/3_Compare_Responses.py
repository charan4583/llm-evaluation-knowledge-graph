"""Page 3 — Multi-Response Benchmarking and Side-by-Side Comparison."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.database.connection import get_session, init_db
from src.database.repository import EvaluationRepository
from src.evaluation.evaluator import evaluate_response
from src.ui.theme import apply_theme, render_page_header

st.set_page_config(page_title="Compare Responses | LLM Eval & KG", page_icon="⚖️", layout="wide")
init_db()
apply_theme()

render_page_header(
    "Multi-Response Comparison & Benchmarking",
    "Compare 2 to 5 model responses against the same reference answer and rank them by any selected metric.",
)

question = st.text_input("Question / Prompt", placeholder="e.g., Explain retrieval-augmented generation (RAG).")
reference = st.text_area("Reference Answer (Ground Truth)", height=130, placeholder="Paste the reference answer...")

num_responses = st.slider("Number of responses to compare", min_value=2, max_value=5, value=3)
default_labels = ["Gemini", "ChatGPT", "Claude", "Llama", "Custom"]

entries: list[tuple[str, str]] = []
cols = st.columns(num_responses)
for idx in range(num_responses):
    with cols[idx]:
        label = st.text_input(f"Model/Label {idx + 1}", value=default_labels[idx], key=f"cmp_label_{idx}")
        resp_text = st.text_area(
            f"Response {idx + 1}",
            height=150,
            key=f"cmp_text_{idx}",
            placeholder=f"Paste {label} response...",
        )
        entries.append((label.strip() or f"Model {idx + 1}", resp_text))

rank_metric = st.selectbox(
    "Rank responses by",
    options=["F1-Score", "Semantic Similarity", "Precision", "Recall", "Keyword Coverage", "Exact Match"],
)

if st.button("Compare Responses", type="primary"):
    valid_entries = [(lbl, txt) for lbl, txt in entries if txt.strip()]
    if not reference.strip() or len(valid_entries) < 2:
        st.error("Please provide a Reference Answer and at least 2 non-empty responses.")
    else:
        with st.spinner("Evaluating all responses..."):
            rows = []
            detailed_results = {}
            for lbl, txt in valid_entries:
                res = evaluate_response(
                    question=question or "Benchmark Question",
                    reference_answer=reference,
                    generated_answer=txt,
                    include_semantic=True,
                )
                rows.append(
                    {
                        "Model": lbl,
                        "Precision": res.precision,
                        "Recall": res.recall,
                        "F1-Score": res.f1_score,
                        "Semantic Similarity": res.semantic_similarity,
                        "Keyword Coverage": res.keyword_coverage,
                        "Exact Match": res.exact_match,
                        "Response": txt,
                    }
                )
                detailed_results[lbl] = res

            df = pd.DataFrame(rows).sort_values(by=rank_metric, ascending=False).reset_index(drop=True)
            df.insert(0, "Rank", range(1, len(df) + 1))
            st.session_state["compare_df"] = df
            st.session_state["compare_details"] = detailed_results
            st.session_state["compare_meta"] = {
                "question": question or "Benchmark Question",
                "reference": reference,
            }

if "compare_df" in st.session_state:
    df = st.session_state["compare_df"]
    details = st.session_state["compare_details"]
    winner = df.iloc[0]

    st.success(
        f"**Top Ranked ({rank_metric}):** {winner['Model']} "
        f"({rank_metric}: {winner[rank_metric]:.1%})"
    )

    st.subheader("Ranking Table")
    st.dataframe(
        df[["Rank", "Model", "Precision", "Recall", "F1-Score", "Semantic Similarity", "Keyword Coverage", "Exact Match"]],
        use_container_width=True,
        hide_index=True,
    )

    plotly_template = "plotly_dark" if st.session_state.get("ui_theme", "dark") == "dark" else "plotly_white"
    melted = df.melt(
        id_vars=["Model"],
        value_vars=["Precision", "Recall", "F1-Score", "Semantic Similarity", "Keyword Coverage"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        template=plotly_template,
        title="Model Comparison Across Evaluation Metrics",
    )
    fig.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Save Benchmark to History"):
        meta = st.session_state["compare_meta"]
        top_res = details[winner["Model"]]
        benchmarks = [
            {
                "model_name": row["Model"],
                "generated_answer": row["Response"],
                "precision": row["Precision"],
                "recall": row["Recall"],
                "f1_score": row["F1-Score"],
                "semantic_similarity": row["Semantic Similarity"],
                "exact_match": row["Exact Match"],
                "keyword_coverage": row["Keyword Coverage"],
            }
            for _, row in df.iterrows()
        ]
        with get_session() as session:
            repo = EvaluationRepository(session)
            saved = repo.create_evaluation(
                question=meta["question"],
                reference_answer=meta["reference"],
                generated_answer=winner["Response"],
                precision=top_res.precision,
                recall=top_res.recall,
                f1_score=top_res.f1_score,
                exact_match=top_res.exact_match,
                semantic_similarity=top_res.semantic_similarity,
                keyword_coverage=top_res.keyword_coverage,
                benchmarks=benchmarks,
            )
            st.success(f"Saved benchmark comparison as evaluation #{saved.id}.")

    st.subheader("Detailed Response Inspection")
    for _, row in df.iterrows():
        res = details[row["Model"]]
        with st.expander(f"#{row['Rank']} — {row['Model']} (F1: {row['F1-Score']:.1%}, Semantic: {row['Semantic Similarity']:.1%})"):
            st.write("**Response Text:**")
            st.write(row["Response"])
            st.write("**Missing Reference Keywords:**", ", ".join(res.keyword_details.get("missing_keywords", [])) or "None")
            st.write("**Matched Keywords:**", ", ".join(res.keyword_details.get("matched_keywords", [])) or "None")
