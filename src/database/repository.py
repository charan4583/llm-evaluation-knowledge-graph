"""Repository pattern for CRUD operations on evaluations, entities, relationships, and benchmarks."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import Session

from src.database.models import BenchmarkResponse, Entity, Evaluation, Relationship


class EvaluationRepository:
    """Encapsulates database queries and mutations for evaluation records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_evaluation(
        self,
        *,
        question: str,
        reference_answer: str,
        generated_answer: str,
        precision: float,
        recall: float,
        f1_score: float,
        exact_match: float,
        semantic_similarity: float,
        keyword_coverage: float,
        ai_score: Optional[float] = None,
        ai_feedback: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        entities: Optional[list[dict[str, Any]]] = None,
        relationships: Optional[list[dict[str, Any]]] = None,
        benchmarks: Optional[list[dict[str, Any]]] = None,
    ) -> Evaluation:
        """Persist an evaluation along with its optional entities, relationships, and benchmarks."""
        evaluation = Evaluation(
            question=question.strip(),
            reference_answer=reference_answer.strip(),
            generated_answer=generated_answer.strip(),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
            exact_match=float(exact_match),
            semantic_similarity=float(semantic_similarity),
            keyword_coverage=float(keyword_coverage),
            ai_score=float(ai_score) if ai_score is not None else None,
            ai_feedback_json=json.dumps(ai_feedback) if ai_feedback is not None else None,
            metadata_json=json.dumps(metadata) if metadata is not None else None,
        )

        for ent in entities or []:
            evaluation.entities.append(
                Entity(
                    entity_name=str(ent["entity_name"])[:255],
                    entity_type=str(ent.get("entity_type", "Other"))[:100],
                    frequency=int(ent.get("frequency", 1)),
                    description=ent.get("description"),
                )
            )

        for rel in relationships or []:
            evaluation.relationships.append(
                Relationship(
                    source_entity=str(rel["source_entity"])[:255],
                    relationship_type=str(rel["relationship_type"])[:100],
                    target_entity=str(rel["target_entity"])[:255],
                    evidence_sentence=rel.get("evidence_sentence"),
                    confidence=rel.get("confidence"),
                    extraction_method=str(rel.get("extraction_method", "rule_based"))[:50],
                )
            )

        for bench in benchmarks or []:
            evaluation.benchmark_responses.append(
                BenchmarkResponse(
                    model_name=str(bench["model_name"])[:150],
                    generated_answer=str(bench["generated_answer"]),
                    precision=float(bench.get("precision", 0.0)),
                    recall=float(bench.get("recall", 0.0)),
                    f1_score=float(bench.get("f1_score", 0.0)),
                    semantic_similarity=float(bench.get("semantic_similarity", 0.0)),
                    exact_match=float(bench.get("exact_match", 0.0)),
                    keyword_coverage=float(bench.get("keyword_coverage", 0.0)),
                    ai_score=float(bench["ai_score"]) if bench.get("ai_score") is not None else None,
                )
            )

        self.session.add(evaluation)
        self.session.flush()
        return evaluation

    def get_by_id(self, evaluation_id: int) -> Optional[Evaluation]:
        """Fetch a single evaluation by primary key."""
        return self.session.get(Evaluation, evaluation_id)

    def list_evaluations(
        self,
        *,
        search_query: str = "",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Evaluation]:
        """List evaluations ordered newest first with optional keyword and date filters."""
        stmt = select(Evaluation).order_by(desc(Evaluation.created_at), desc(Evaluation.id))
        if search_query.strip():
            pattern = f"%{search_query.strip()}%"
            stmt = stmt.where(
                or_(
                    Evaluation.question.ilike(pattern),
                    Evaluation.reference_answer.ilike(pattern),
                    Evaluation.generated_answer.ilike(pattern),
                )
            )
        if start_date is not None:
            stmt = stmt.where(Evaluation.created_at >= start_date)
        if end_date is not None:
            stmt = stmt.where(Evaluation.created_at <= end_date)
        stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt).all())

    def delete_evaluation(self, evaluation_id: int) -> bool:
        """Delete an evaluation and its cascaded children. Returns True if deleted."""
        evaluation = self.get_by_id(evaluation_id)
        if evaluation is None:
            return False
        self.session.delete(evaluation)
        self.session.flush()
        return True

    def clear_all(self) -> int:
        """Delete all evaluations and return the number of deleted rows."""
        evaluations = list(self.session.scalars(select(Evaluation)).all())
        count = len(evaluations)
        for ev in evaluations:
            self.session.delete(ev)
        self.session.flush()
        return count

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Compute aggregate statistics directly from persisted database records."""
        total_evaluations = self.session.scalar(select(func.count(Evaluation.id))) or 0
        if total_evaluations == 0:
            return {
                "total_evaluations": 0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_f1": 0.0,
                "avg_semantic_similarity": 0.0,
                "total_entities": 0,
                "total_relationships": 0,
            }

        row = self.session.execute(
            select(
                func.avg(Evaluation.precision),
                func.avg(Evaluation.recall),
                func.avg(Evaluation.f1_score),
                func.avg(Evaluation.semantic_similarity),
            )
        ).one()

        total_entities = self.session.scalar(select(func.count(Entity.id))) or 0
        total_relationships = self.session.scalar(select(func.count(Relationship.id))) or 0

        return {
            "total_evaluations": int(total_evaluations),
            "avg_precision": float(row[0] or 0.0),
            "avg_recall": float(row[1] or 0.0),
            "avg_f1": float(row[2] or 0.0),
            "avg_semantic_similarity": float(row[3] or 0.0),
            "total_entities": int(total_entities),
            "total_relationships": int(total_relationships),
        }
