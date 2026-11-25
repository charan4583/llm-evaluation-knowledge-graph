"""Unit tests for entity extraction and relationship extraction."""

from src.nlp.entity_extraction import extract_entities
from src.nlp.relationship_extraction import extract_relationships


def test_entity_extraction_and_deduplication():
    text = (
        "Google developed Gemini. Gemini is a generative AI model that processes text and images. "
        "Gemini also supports code."
    )
    entities = extract_entities(text)
    names_lower = [e["entity_name"].lower() for e in entities]
    assert "google" in names_lower
    assert "gemini" in names_lower
    assert "text" in names_lower
    assert "images" in names_lower
    # Deduplication: Gemini appears multiple times in text but once in entity list
    assert names_lower.count("gemini") == 1
    gemini_ent = next(e for e in entities if e["entity_name"].lower() == "gemini")
    assert gemini_ent["frequency"] >= 2
    assert gemini_ent["entity_type"] == "AI Model"


def test_relationship_extraction():
    text = "Google developed Gemini. Gemini processes text and images."
    entities = extract_entities(text)
    rels = extract_relationships(text, entities)
    triples = {(r["source_entity"].lower(), r["relationship_type"], r["target_entity"].lower()) for r in rels}
    assert ("google", "DEVELOPED", "gemini") in triples
    assert ("gemini", "PROCESSES", "text") in triples
    assert ("gemini", "PROCESSES", "images") in triples


def test_empty_inputs_nlp():
    assert extract_entities("") == []
    assert extract_relationships("") == []
