"""Rule-based and dependency-assisted relationship extraction between entities."""

from __future__ import annotations

import re
from typing import Any, Optional

from src.nlp.entity_extraction import extract_entities
from src.nlp.text_processing import segment_sentences

# Ordered from more specific patterns to general patterns
_RELATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("DEVELOPED", re.compile(r"\b(?:developed|created|built|designed|released|trained)\b", re.IGNORECASE)),
    ("PROCESSES", re.compile(r"\b(?:processes|handles|analyzes|accepts|parses|reads)\b", re.IGNORECASE)),
    ("SUPPORTS", re.compile(r"\b(?:supports|enables|provides|offers|allows|includes)\b", re.IGNORECASE)),
    ("USES", re.compile(r"\b(?:uses|utilizes|relies on|depends on|powered by|built with)\b", re.IGNORECASE)),
    ("APPLIED_IN", re.compile(r"\b(?:used for|applied in|used in|helpful for)\b", re.IGNORECASE)),
    ("PART_OF", re.compile(r"\b(?:part of|component of|family of|subset of|belongs to)\b", re.IGNORECASE)),
    ("TYPE", re.compile(r"\b(?:is a|is an|are|represents a)\b", re.IGNORECASE)),
    ("TRAINED_ON", re.compile(r"\b(?:trained on|fine-tuned on|evaluated on)\b", re.IGNORECASE)),
    ("CONVERTS_TO", re.compile(r"\b(?:converts|transforms|translates)\b", re.IGNORECASE)),
]


def _match_relation_label(between_text: str) -> Optional[str]:
    for rel_label, pattern in _RELATION_PATTERNS:
        if pattern.search(between_text):
            return rel_label
    return None


def extract_relationships(
    text: str,
    entities: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Extract directed relationships between entities grounded in source sentences.

    Prevents duplicate edges, empty entities, self-loops, and ungrounded triples.
    """
    if not text or not text.strip():
        return []

    entity_list = entities if entities is not None else extract_entities(text)
    if len(entity_list) < 2:
        return []

    entity_names = [e["entity_name"] for e in entity_list if e.get("entity_name", "").strip()]
    sentences = segment_sentences(text)

    seen_triples: set[tuple[str, str, str]] = set()
    relationships: list[dict[str, Any]] = []

    for sent in sentences:
        # Find entities appearing in this sentence with their first character position
        present: list[tuple[int, int, str]] = []
        for name in entity_names:
            match = re.search(rf"\b{re.escape(name)}\b", sent, re.IGNORECASE)
            if match:
                present.append((match.start(), match.end(), name))

        present.sort(key=lambda item: item[0])
        if len(present) < 2:
            continue

        # Check ordered pairs in the sentence
        for i in range(len(present)):
            src_start, src_end, src_name = present[i]
            for j in range(i + 1, len(present)):
                tgt_start, _, tgt_name = present[j]
                if src_name.lower() == tgt_name.lower():
                    continue

                between = sent[src_end:tgt_start].strip()
                # Avoid matching across overly distant clauses if no clear connector
                if len(between) > 120:
                    continue

                rel_type = _match_relation_label(between)
                if rel_type is None:
                    # Handle comma/conjunction lists like "process text and images" after a verb
                    prefix = sent[:src_start]
                    if re.search(r"\b(?:and|or|,)\b", between) and i > 0:
                        prev_between = sent[present[i - 1][1]:src_start]
                        inherited_rel = _match_relation_label(prev_between)
                        if inherited_rel:
                            # Link the subject before the list to tgt_name as well
                            subj_name = present[i - 1][2]
                            if subj_name.lower() != tgt_name.lower():
                                triple_key = (subj_name.lower(), inherited_rel, tgt_name.lower())
                                if triple_key not in seen_triples:
                                    seen_triples.add(triple_key)
                                    relationships.append(
                                        {
                                            "source_entity": subj_name,
                                            "relationship_type": inherited_rel,
                                            "target_entity": tgt_name,
                                            "evidence_sentence": sent,
                                            "confidence": 0.75,
                                            "extraction_method": "rule_based_conjunction",
                                        }
                                    )
                    continue

                triple_key = (src_name.lower(), rel_type, tgt_name.lower())
                if triple_key in seen_triples:
                    continue
                seen_triples.add(triple_key)

                relationships.append(
                    {
                        "source_entity": src_name,
                        "relationship_type": rel_type,
                        "target_entity": tgt_name,
                        "evidence_sentence": sent,
                        "confidence": 0.85,
                        "extraction_method": "rule_based",
                    }
                )

    return relationships
