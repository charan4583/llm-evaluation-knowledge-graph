"""Safe Gemini API client wrapper with timeout, rate-limit, and JSON error handling."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiClientError(RuntimeError):
    """User-safe error raised when a Gemini API call fails."""


def generate_structured_json(
    prompt: str,
    *,
    system_instruction: str,
    timeout_seconds: Optional[int] = None,
) -> dict[str, Any]:
    """Call Gemini API requesting a JSON response and parse it safely.

    Never exposes API keys in logs or exceptions.
    """
    api_key = settings.gemini_api_key
    if not api_key:
        raise GeminiClientError(
            "Gemini API key is not configured. Set GEMINI_API_KEY in your .env file to enable AI-assisted evaluation."
        )

    timeout = timeout_seconds or settings.gemini_timeout_seconds

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout * 1000),
        )
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg:
            logger.warning("Gemini API rate limit or quota reached.")
            raise GeminiClientError(
                "Gemini API rate limit or quota exceeded. Please wait and try again."
            ) from exc
        if "timeout" in msg or "timed out" in msg or "deadline" in msg:
            logger.warning("Gemini API request timed out.")
            raise GeminiClientError(
                f"Gemini API request timed out after {timeout}s."
            ) from exc
        logger.warning("Gemini API call failed: %s", type(exc).__name__)
        raise GeminiClientError(
            "Gemini API request failed. Check your network connection and API configuration."
        ) from exc

    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raise GeminiClientError("Gemini API returned an empty response.")

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse Gemini JSON response.")
        raise GeminiClientError("Gemini API returned invalid JSON output.") from exc

    if not isinstance(parsed, dict):
        raise GeminiClientError("Gemini API returned unexpected JSON structure.")

    return parsed
