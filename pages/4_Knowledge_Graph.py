"""Page 4 — Interactive Knowledge Graph Explorer with NetworkX analytics and PyVis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.document_processing.file_handler import FileValidationError, extract_text_from_upload
from src.graph.graph_analysis import analyze_graph
from src.graph.graph_builder import build_knowledge_graph
from src.graph.visualization import filter_graph, render_pyvis_html
from src.nlp.entity_extraction import extract_entities, is_spacy_available
from src.nlp.relationship_extraction import extract_relationships
from src.ui.theme import apply_theme, render_page_header

st.set_page_config(page_title="Knowledge Graph | LLM Eval & KG", page_icon="🕸️", layout="wide")
apply_theme()

render_page_header(
    "Knowledge Graph Explorer",
    "Extract entities and directed relationships from text or documents and explore them interactively.",
)

if not is_spacy_available():
    st.info(
        "spaCy model `en_core_web_sm` is not loaded; running with domain-specific rule-based fallback. "
        "Install with: `python -m spacy download en_core_web_sm`"
    )

source_mode = st.radio(
    "Input Source",
    options=["Custom Text", "Question & Answer", "Upload Document (PDF/DOCX/TXT)"],
    horizontal=True,
)

input_text = ""
if source_mode == "Custom Text":
    input_text = st.text_area(
        "Enter text to analyze",
        height=150,
        placeholder="Example: Google developed Gemini. Gemini is a generative AI model that can process text and images.",
    )
elif source_mode == "Question & Answer":
    q_txt = st.text_input("Question")
    a_txt = st.text_area("Answer / Reference", height=130)
    input_text = f"{q_txt}\n{a_txt}".strip()
else:
    uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    if uploaded is not None:
        try:
            extracted = extract_text_from_upload(uploaded.name, uploaded.getvalue())
            with st.expander("Extracted Document Preview", expanded=False):
                st.text(extracted[:3000] + ("..." if len(extracted) > 3000 else ""))
            input_text = extracted
        except FileValidationError as exc:
            st.error(str(exc))

if st.button("Extract Entities & Build Knowledge Graph", type="primary"):
    if not input_text.strip():
        st.error("Please provide non-empty text or upload a valid document.")
    else:
        with st.spinner("Extracting entities and relationships..."):
            entities = extract_entities(input_text)
            relationships = extract_relationships(input_text, entities)
            st.session_state["kg_entities"] = entities
            st.session_state["kg_relationships"] = relationships
            st.session_state["kg_source_text"] = input_text

entities = st.session_state.get("kg_entities", [])
relationships = st.session_state.get("kg_relationships", [])

if entities or relationships:
    # Sidebar / inline graph filters
    all_ent_types = sorted({e.get("entity_type", "Other") for e in entities})
    all_rel_types = sorted({r.get("relationship_type", "RELATED_TO") for r in relationships})

    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    selected_ent_types = f1.multiselect("Filter Entity Types", options=all_ent_types, default=all_ent_types)
    selected_rel_types = f2.multiselect("Filter Relationship Types", options=all_rel_types, default=all_rel_types)
    search_node = f3.text_input("Search Entity / Highlight", placeholder="Type entity name...")
    hide_isolated = f4.checkbox("Hide Isolated", value=False)

    full_graph = build_knowledge_graph(entities, relationships)
    filtered_graph = filter_graph(
        full_graph,
        allowed_entity_types=selected_ent_types,
        allowed_rel_types=selected_rel_types,
        search_query=search_node,
        hide_isolated=hide_isolated,
    )
    stats = analyze_graph(filtered_graph)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Nodes (Entities)", stats["total_nodes"])
    k2.metric("Edges (Relationships)", stats["total_edges"])
    k3.metric("Connected Components", stats["connected_components"])
    top_ent = stats["most_connected_entity"]
    k4.metric("Most Connected", top_ent["entity"] if top_ent else "N/A")

    view_tab, ent_tab, rel_tab, analytics_tab, edit_tab = st.tabs(
        ["Interactive Graph", "Entity Table", "Relationship Table", "Graph Analytics", "Edit Graph"]
    )

    with view_tab:
        if filtered_graph.number_of_nodes() == 0:
            st.warning("No nodes match the current filters.")
        else:
            dark_mode = st.session_state.get("ui_theme", "dark") == "dark"
            html_str = render_pyvis_html(
                filtered_graph,
                dark_mode=dark_mode,
                height_px=580,
                highlight_query=search_node,
            )
            components.html(html_str, height=600, scrolling=False)
            st.caption("Tip: Drag nodes, scroll to zoom, hover over nodes/edges for details, or use navigation buttons.")

    with ent_tab:
        ent_search = st.text_input("Filter entity table", key="ent_tbl_search", placeholder="Search entity name...")
        filtered_ents = [
            e for e in entities
            if e["entity_type"] in selected_ent_types
            and (not ent_search.strip() or ent_search.lower() in e["entity_name"].lower())
        ]
        if filtered_ents:
            st.dataframe(
                [{"Entity": e["entity_name"], "Type": e["entity_type"], "Mentions": e["frequency"]} for e in filtered_ents],
                use_container_width=True,
                hide_index=True,
            )
            selected_ent_name = st.selectbox("Inspect occurrences for entity", [e["entity_name"] for e in filtered_ents])
            chosen = next((e for e in filtered_ents if e["entity_name"] == selected_ent_name), None)
            if chosen and chosen.get("occurrences"):
                for occ in chosen["occurrences"]:
                    st.write(f"- Sentence {occ['sentence_index'] + 1}: {occ['sentence']}")
        else:
            st.info("No entities match the filter.")

    with rel_tab:
        filtered_rels = [
            r for r in relationships
            if r["relationship_type"] in selected_rel_types
        ]
        if filtered_rels:
            st.dataframe(
                [
                    {
                        "Source": r["source_entity"],
                        "Relationship": r["relationship_type"],
                        "Target": r["target_entity"],
                        "Confidence": r.get("confidence"),
                        "Method": r.get("extraction_method", "rule_based"),
                        "Evidence": r.get("evidence_sentence", ""),
                    }
                    for r in filtered_rels
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No relationships extracted or matching filter.")

    with analytics_tab:
        plotly_template = "plotly_dark" if st.session_state.get("ui_theme", "dark") == "dark" else "plotly_white"
        ac1, ac2 = st.columns(2)
        with ac1:
            if stats["entity_type_distribution"]:
                df_et = pd.DataFrame(
                    [{"Type": k, "Count": v} for k, v in stats["entity_type_distribution"].items()]
                )
                st.plotly_chart(px.pie(df_et, names="Type", values="Count", title="Entity Type Distribution", template=plotly_template), use_container_width=True)
        with ac2:
            if stats["relationship_distribution"]:
                df_rt = pd.DataFrame(
                    [{"Relationship": k, "Count": v} for k, v in stats["relationship_distribution"].items()]
                )
                st.plotly_chart(px.bar(df_rt, x="Relationship", y="Count", title="Relationship Distribution", template=plotly_template), use_container_width=True)

        st.subheader("Degree Centrality")
        if stats["degree_centrality"]:
            st.dataframe(stats["degree_centrality"], use_container_width=True, hide_index=True)
        st.write(f"**Isolated Nodes ({len(stats['isolated_nodes'])}):** {', '.join(stats['isolated_nodes']) or 'None'}")

    with edit_tab:
        st.subheader("Manually Add Entity or Relationship")
        ec1, ec2 = st.columns(2)
        with ec1:
            with st.form("add_entity_form"):
                new_ent_name = st.text_input("Entity Name")
                new_ent_type = st.selectbox(
                    "Entity Type",
                    ["Organization", "AI Model", "Technology", "Programming Language", "Person", "Location", "Product", "Concept", "Dataset", "Method", "Data Type", "Other"],
                )
                if st.form_submit_button("Add Entity"):
                    if new_ent_name.strip():
                        entities.append({"entity_name": new_ent_name.strip(), "entity_type": new_ent_type, "frequency": 1, "occurrences": []})
                        st.session_state["kg_entities"] = entities
                        st.rerun()
        with ec2:
            with st.form("add_rel_form"):
                src_e = st.text_input("Source Entity")
                rel_t = st.text_input("Relationship Type (e.g., USES)")
                tgt_e = st.text_input("Target Entity")
                ev_s = st.text_input("Supporting Sentence")
                if st.form_submit_button("Add Relationship"):
                    if src_e.strip() and rel_t.strip() and tgt_e.strip() and src_e.strip().lower() != tgt_e.strip().lower():
                        relationships.append(
                            {
                                "source_entity": src_e.strip(),
                                "relationship_type": rel_t.strip().upper(),
                                "target_entity": tgt_e.strip(),
                                "evidence_sentence": ev_s.strip() or "Manually added.",
                                "confidence": 1.0,
                                "extraction_method": "manual",
                            }
                        )
                        st.session_state["kg_relationships"] = relationships
                        st.rerun()
else:
    st.info("Enter text or upload a document above, then click **Extract Entities & Build Knowledge Graph**.")
