"""NetworkX directed knowledge graph builder from extracted entities and relationships."""

from __future__ import annotations

from typing import Any

import networkx as nx


def build_knowledge_graph(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> nx.DiGraph:
    """Construct a NetworkX DiGraph from validated entity and relationship lists."""
    graph = nx.DiGraph()

    canonical_map: dict[str, str] = {}
    for ent in entities:
        name = str(ent.get("entity_name", "")).strip()
        if not name:
            continue
        canonical_map[name.lower()] = name
        graph.add_node(
            name,
            entity_type=str(ent.get("entity_type", "Other")),
            frequency=int(ent.get("frequency", 1)),
            description=str(ent.get("description", "")),
        )

    for rel in relationships:
        src = str(rel.get("source_entity", "")).strip()
        tgt = str(rel.get("target_entity", "")).strip()
        rel_type = str(rel.get("relationship_type", "")).strip()
        if not src or not tgt or not rel_type or src.lower() == tgt.lower():
            continue

        src_node = canonical_map.get(src.lower(), src)
        tgt_node = canonical_map.get(tgt.lower(), tgt)

        if src_node not in graph:
            graph.add_node(src_node, entity_type="Other", frequency=1, description="")
            canonical_map[src.lower()] = src_node
        if tgt_node not in graph:
            graph.add_node(tgt_node, entity_type="Other", frequency=1, description="")
            canonical_map[tgt.lower()] = tgt_node

        graph.add_edge(
            src_node,
            tgt_node,
            relationship_type=rel_type,
            evidence_sentence=str(rel.get("evidence_sentence", "")),
            confidence=rel.get("confidence"),
            extraction_method=str(rel.get("extraction_method", "rule_based")),
        )

    return graph
