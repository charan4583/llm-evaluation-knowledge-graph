"""Interactive PyVis HTML visualization for NetworkX knowledge graphs."""

from __future__ import annotations

import html
from typing import Iterable, Optional

import networkx as nx
from pyvis.network import Network

ENTITY_TYPE_COLORS: dict[str, str] = {
    "Organization": "#38bdf8",
    "AI Model": "#a78bfa",
    "Technology": "#34d399",
    "Programming Language": "#fbbf24",
    "Person": "#f472b6",
    "Location": "#fb923c",
    "Product": "#60a5fa",
    "Concept": "#94a3b8",
    "Dataset": "#2dd4bf",
    "Method": "#c084fc",
    "Data Type": "#facc15",
    "Other": "#cbd5e1",
}


def filter_graph(
    graph: nx.DiGraph,
    *,
    allowed_entity_types: Optional[Iterable[str]] = None,
    allowed_rel_types: Optional[Iterable[str]] = None,
    search_query: str = "",
    hide_isolated: bool = False,
) -> nx.DiGraph:
    """Return a filtered subgraph based on entity types, relationship types, and search term."""
    filtered = nx.DiGraph()
    ent_filter = set(allowed_entity_types) if allowed_entity_types is not None else None
    rel_filter = set(allowed_rel_types) if allowed_rel_types is not None else None
    query = search_query.strip().lower()

    for node, data in graph.nodes(data=True):
        ent_type = data.get("entity_type", "Other")
        if ent_filter is not None and ent_type not in ent_filter:
            continue
        filtered.add_node(node, **data)

    for src, tgt, data in graph.edges(data=True):
        if src not in filtered or tgt not in filtered:
            continue
        rel_type = data.get("relationship_type", "RELATED_TO")
        if rel_filter is not None and rel_type not in rel_filter:
            continue
        filtered.add_edge(src, tgt, **data)

    if query:
        matched_nodes = {n for n in filtered.nodes if query in n.lower()}
        # Keep matched nodes and their immediate 1-hop neighbors so context is preserved
        keep_nodes = set(matched_nodes)
        for n in matched_nodes:
            keep_nodes.update(filtered.predecessors(n))
            keep_nodes.update(filtered.successors(n))
        filtered = filtered.subgraph(keep_nodes).copy()

    if hide_isolated:
        non_isolated = [n for n in filtered.nodes if filtered.degree(n) > 0]
        filtered = filtered.subgraph(non_isolated).copy()

    return filtered


def render_pyvis_html(
    graph: nx.DiGraph,
    *,
    dark_mode: bool = True,
    height_px: int = 580,
    highlight_query: str = "",
) -> str:
    """Generate self-contained interactive HTML for the knowledge graph using PyVis."""
    bg_color = "#0b1120" if dark_mode else "#ffffff"
    font_color = "#f8fafc" if dark_mode else "#0f172a"
    edge_color = "#64748b" if dark_mode else "#94a3b8"

    net = Network(
        height=f"{height_px}px",
        width="100%",
        directed=True,
        bgcolor=bg_color,
        font_color=font_color,
    )

    net.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true,
            "dragView": true,
            "dragNodes": true
          },
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -3500,
              "centralGravity": 0.25,
              "springLength": 140,
              "springConstant": 0.04,
              "damping": 0.12,
              "avoidOverlap": 0.6
            },
            "stabilization": {
              "iterations": 150
            }
          },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.8 } },
            "smooth": { "type": "dynamic" },
            "font": { "size": 11, "align": "middle" }
          },
          "nodes": {
            "font": { "size": 14, "face": "Inter, Segoe UI, sans-serif" },
            "borderWidth": 2
          }
        }
        """
    )

    q = highlight_query.strip().lower()
    for node, data in graph.nodes(data=True):
        ent_type = data.get("entity_type", "Other")
        freq = int(data.get("frequency", 1))
        color = ENTITY_TYPE_COLORS.get(ent_type, "#cbd5e1")
        is_match = bool(q and q in node.lower())
        size = min(38, 16 + freq * 3) + (6 if is_match else 0)
        border_color = "#ffffff" if is_match else color

        tooltip = (
            f"<b>{html.escape(node)}</b><br>"
            f"Type: {html.escape(ent_type)}<br>"
            f"Mentions: {freq}<br>"
            f"Connections: {graph.degree(node)}"
        )
        net.add_node(
            node,
            label=node,
            title=tooltip,
            color={"background": color, "border": border_color},
            size=size,
            shape="dot",
        )

    for src, tgt, data in graph.edges(data=True):
        rel_type = data.get("relationship_type", "RELATED_TO")
        evidence = data.get("evidence_sentence", "")
        conf = data.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "N/A"
        tooltip = (
            f"<b>{html.escape(src)} → {html.escape(rel_type)} → {html.escape(tgt)}</b><br>"
            f"Confidence: {conf_str}<br>"
            f"Evidence: {html.escape(evidence)}"
        )
        net.add_edge(
            src,
            tgt,
            label=rel_type,
            title=tooltip,
            color=edge_color,
        )

    return net.generate_html()
