"""Application configuration and environment settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class AppSettings:
    """Central configuration loaded from environment variables with safe defaults."""

    app_title: str = "LLM Evaluation & Knowledge Graph Explorer"
    app_version: str = "1.0.0"
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(BASE_DIR / 'data' / 'llm_eval_kg.db').as_posix()}",
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_timeout_seconds: int = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    default_similarity_threshold: float = float(
        os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.70")
    )

    @property
    def gemini_api_key(self) -> str | None:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key or key == "your_api_key_here":
            return None
        return key

    @property
    def is_gemini_configured(self) -> bool:
        return self.gemini_api_key is not None


settings = AppSettings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
