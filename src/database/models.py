"""SQLAlchemy ORM models for evaluations, entities, relationships, and benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all ORM models."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Evaluation(Base):
    """Stores a single LLM response evaluation against a reference answer."""

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    reference_answer: Mapped[str] = mapped_column(Text, nullable=False)
    generated_answer: Mapped[str] = mapped_column(Text, nullable=False)

    precision: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    recall: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    exact_match: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    semantic_similarity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    keyword_coverage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    ai_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ai_feedback_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False, index=True
    )

    entities: Mapped[list["Entity"]] = relationship(
        "Entity",
        back_populates="evaluation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    relationships: Mapped[list["Relationship"]] = relationship(
        "Relationship",
        back_populates="evaluation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    benchmark_responses: Mapped[list["BenchmarkResponse"]] = relationship(
        "BenchmarkResponse",
        back_populates="evaluation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Entity(Base):
    """Stores an extracted entity linked to an evaluation or standalone graph."""

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    frequency: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    evaluation: Mapped["Evaluation"] = relationship("Evaluation", back_populates="entities")


class Relationship(Base):
    """Stores a directed relationship triple extracted from text."""

    __tablename__ = "relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_entity: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    relationship_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    target_entity: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    evidence_sentence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extraction_method: Mapped[str] = mapped_column(
        String(50), nullable=False, default="rule_based"
    )

    evaluation: Mapped["Evaluation"] = relationship(
        "Evaluation", back_populates="relationships"
    )


class BenchmarkResponse(Base):
    """Stores a compared model/provider response for multi-response benchmarking."""

    __tablename__ = "benchmark_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    generated_answer: Mapped[str] = mapped_column(Text, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    recall: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    semantic_similarity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    exact_match: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    keyword_coverage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    ai_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    evaluation: Mapped["Evaluation"] = relationship(
        "Evaluation", back_populates="benchmark_responses"
    )


Index("ix_eval_created_f1", Evaluation.created_at, Evaluation.f1_score)
