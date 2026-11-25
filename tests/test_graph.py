"""Unit tests for knowledge graph construction, filtering, and analytics."""

from src.graph.graph_analysis import analyze_graph
from src.graph.graph_builder import build_knowledge_graph
from src.graph.visualization import filter_graph, render_pyvis_html


def test_build_and_analyze_graph():
    entities = [
        {"entity_name": "Google", "entity_type": "Organization", "frequency": 1},
        {"entity_name": "Gemini", "entity_type": "AI Model", "frequency": 2},
        {"entity_name": "Text", "entity_type": "Data Type", "frequency": 1},
        {"entity_name": "Python", "entity_type": "Programming Language", "frequency": 1},
    ]
    relationships = [
        {
            "source_entity": "Google",
            "relationship_type": "DEVELOPED",
            "target_entity": "Gemini",
            "evidence_sentence": "Google developed Gemini.",
            "confidence": 0.85,
        },
        {
            "source_entity": "Gemini",
            "relationship_type": "PROCESSES",
            "target_entity": "Text",
            "evidence_sentence": "Gemini processes Text.",
            "confidence": 0.85,
        },
    ]
    graph = build_knowledge_graph(entities, relationships)
    stats = analyze_graph(graph)
    assert stats["total_nodes"] == 4
    assert stats["total_edges"] == 2
    assert stats["connected_components"] == 2
    assert stats["isolated_nodes"] == ["Python"]
    assert stats["most_connected_entity"]["entity"] == "Gemini"

    filtered = filter_graph(graph, hide_isolated=True)
    assert filtered.number_of_nodes() == 3
    assert "Python" not in filtered.nodes

    html_output = render_pyvis_html(graph, dark_mode=True)
    assert "<html" in html_output.lower()
    assert "Gemini" in html_output
