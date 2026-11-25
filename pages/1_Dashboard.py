"""Page 1 — Overview Dashboard calculated from real SQLite evaluation records."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.database.connection import get_session, init_db
from src.database.repository import EvaluationRepository
from src.ui.theme import apply_theme, render_page_header

st.set_page_config(page_title="Dashboard | LLM Eval & KG", page_icon="📊", layout="wide")
init_db()
apply_theme()

render_page_header(
    "Overview Dashboard",
    "Real-time analytics computed strictly from saved evaluations, extracted entities, and relationships.",
)

with get_session() as session:
    repo = EvaluationRepository(session)
    summary = repo.get_dashboard_summary()
    recent = repo.list_evaluations(limit=50)

if summary["total_evaluations"] == 0:
    st.markdown(
        """
        <div class="app-card">
            <div class="app-card-title">No Evaluations Saved Yet</div>
            <p class="app-muted">
                This dashboard displays live metrics from your SQLite database. Run your first
                LLM response evaluation and click <b>Save Evaluation</b> to populate trends,
                entity counts, and comparison summaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Evaluate_Response.py", label="Go to Evaluate Response", icon="🧪")
    st.stop()

# Top KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Evaluations", summary["total_evaluations"])
c2.metric("Average Precision", f"{summary['avg_precision']:.1%}")
c3.metric("Average Recall", f"{summary['avg_recall']:.1%}")
c4.metric("Average F1-Score", f"{summary['avg_f1']:.1%}")

c5, c6, c7, _ = st.columns(4)
c5.metric("Avg Semantic Similarity", f"{summary['avg_semantic_similarity']:.1%}")
c6.metric("Total Entities Extracted", summary["total_entities"])
c7.metric("Total Relationships", summary["total_relationships"])

# Prepare DataFrame from actual records
records_data = []
entity_type_counts: dict[str, int] = {}
for ev in reversed(recent):
    records_data.append(
        {
            "ID": ev.id,
            "Date": ev.created_at.strftime("%Y-%m-%d %H:%M") if ev.created_at else "",
            "Question": ev.question[:70] + ("..." if len(ev.question) > 70 else ""),
            "Precision": ev.precision,
            "Recall": ev.recall,
            "F1-Score": ev.f1_score,
            "Semantic Similarity": ev.semantic_similarity,
            "Keyword Coverage": ev.keyword_coverage,
            "Entities": len(ev.entities),
            "Relationships": len(ev.relationships),
        }
    )
    for ent in ev.entities:
        entity_type_counts[ent.entity_type] = entity_type_counts.get(ent.entity_type, 0) + 1

df = pd.DataFrame(records_data)
plotly_template = "plotly_dark" if st.session_state.get("ui_theme", "dark") == "dark" else "plotly_white"

chart_col1, chart_col2 = st.columns([3, 2])
with chart_col1:
    st.subheader("Evaluation Score Trends")
    trend_df = df.melt(
        id_vars=["ID", "Date"],
        value_vars=["Precision", "Recall", "F1-Score", "Semantic Similarity"],
        var_name="Metric",
        value_name="Score",
    )
    fig_trend = px.line(
        trend_df,
        x="ID",
        y="Score",
        color="Metric",
        markers=True,
        template=plotly_template,
    )
    fig_trend.update_layout(yaxis_range=[0, 1.05], xaxis_title="Evaluation ID", yaxis_title="Score")
    st.plotly_chart(fig_trend, use_container_width=True)

with chart_col2:
    st.subheader("Entity Extraction Breakdown")
    if entity_type_counts:
        ent_df = pd.DataFrame(
            [{"Entity Type": k, "Count": v} for k, v in sorted(entity_type_counts.items(), key=lambda x: -x[1])]
        )
        fig_ent = px.bar(
            ent_df,
            x="Count",
            y="Entity Type",
            orientation="h",
            template=plotly_template,
        )
        st.plotly_chart(fig_ent, use_container_width=True)
    else:
        st.info("No entities have been saved with evaluations yet.")

st.subheader("Recent Evaluations")
st.dataframe(
    df.iloc[::-1][["ID", "Date", "Question", "F1-Score", "Semantic Similarity", "Keyword Coverage", "Entities", "Relationships"]],
    use_container_width=True,
    hide_index=True,
)
