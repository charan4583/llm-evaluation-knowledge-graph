"""JSON exporter for full evaluation payloads including entities, relationships, and AI feedback."""

from __future__ import annotations

import json
from typing import Any


def export_evaluation_to_json(payload: dict[str, Any]) -> bytes:
    """Serialize a complete evaluation record dictionary into formatted UTF-8 JSON bytes."""
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
