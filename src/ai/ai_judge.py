"""Optional AI Judge powered by Gemini API with Pydantic output validation."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from src.ai.gemini_client import GeminiClientError, generate_structured_json


class AIJudgeOutput(BaseModel):
    """Validated structured response from the optional AI Judge."""

    overall_score: float = Field(ge=0.0, le=100.0, default=0.0)
    correctness: float = Field(ge=0.0, le=100.0, default=0.0)
    relevance: float = Field(ge=0.0, le=100.0, default=0.0)
    completeness: float = Field(ge=0.0, le=100.0, default=0.0)
    clarity: float = Field(ge=0.0, le=100.0, default=0.0)
    consistency: float = Field(ge=0.0, le=100.0, default=0.0)
    verdict: str = Field(default="")
    strengths: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    explanation: str = Field(default="")
    disclaimer: str = Field(
        default=(
            "AI-assisted evaluation: an AI judge can make mistakes and should not be "
            "treated as absolute truth."
        )
    )


_AI_JUDGE_SYSTEM_INSTRUCTION = """You are an objective AI evaluation judge.
Compare the generated answer strictly against the question and reference answer.
Rules:
1. Do NOT invent facts or assume external truth beyond the reference answer.
2. Clearly distinguish evidence supported by the reference from unverified or missing claims.
3. Return ONLY valid JSON matching the exact schema requested.
4. Scores must be numbers between 0 and 100.
"""


def evaluate_with_ai_judge(
    question: str,
    reference_answer: str,
    generated_answer: str,
) -> tuple[Optional[AIJudgeOutput], Optional[str]]:
    """Run optional AI-assisted evaluation. Returns (AIJudgeOutput, error_message)."""
    if not generated_answer.strip() or not reference_answer.strip():
        return None, "Both reference answer and generated answer are required for AI Judge evaluation."

    prompt = f"""Evaluate the generated answer against the reference answer for the given question.

Question:
{question.strip()}

Reference Answer:
{reference_answer.strip()}

Generated Answer:
{generated_answer.strip()}

Return JSON with these exact keys:
{{
  "overall_score": 0,
  "correctness": 0,
  "relevance": 0,
  "completeness": 0,
  "clarity": 0,
  "consistency": 0,
  "verdict": "",
  "strengths": [],
  "missing_information": [],
  "unsupported_claims": [],
  "improvement_suggestions": [],
  "explanation": ""
}}"""

    try:
        raw_json = generate_structured_json(
            prompt,
            system_instruction=_AI_JUDGE_SYSTEM_INSTRUCTION,
        )
        validated = AIJudgeOutput.model_validate(raw_json)
        return validated, None
    except GeminiClientError as exc:
        return None, str(exc)
    except ValidationError as exc:
        return None, f"AI Judge returned an invalid schema: {exc.errors()[0].get('msg', 'validation error')}"
