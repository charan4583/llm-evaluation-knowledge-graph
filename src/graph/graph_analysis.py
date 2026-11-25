"""Graph analytics computed from actual NetworkX knowledge graphs."""

from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx


def analyze_graph(graph: nx.DiGraph) -> dict[str, Any]:
    """Compute structural metrics, centrality, and distributions for a knowledge graph."""
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()

    if total_nodes == 0:
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "connected_components": 0,
            "isolated_nodes": [],
            "degree_centrality": [],
            "most_connected_entity": None,
            "entity_type_distribution": {},
            "relationship_distribution": {},
        }

    centrality = nx.degree_centrality(graph)
    ranked_centrality = sorted(
        (
            {
                "entity": node,
                "entity_type": graph.nodes[node].get("entity_type", "Other"),
                "degree": graph.degree(node),
                "centrality": round(score, 4),
            }
            for node, score in centrality.items()
        ),
        key=lambda x: (-x["degree"], -x["centrality"], x["entity"].lower()),
    )

    isolated = sorted(list(nx.isolates(graph)))
    components = nx.number_weakly_connected_components(graph)

    entity_types = Counter(
        graph.nodes[n].get("entity_type", "Other") for n in graph.nodes
    )
    rel_types = Counter(
        data.get("relationship_type", "RELATED_TO") for _, _, data in graph.edges(data=True)
    )

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "connected_components": components,
        "isolated_nodes": isolated,
        "degree_centrality": ranked_centrality,
        "most_connected_entity": ranked_centrality[0] if ranked_centrality else None,
        "entity_type_distribution": dict(entity_types),
        "relationship_distribution": dict(rel_types),
    }
