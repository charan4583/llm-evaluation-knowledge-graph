"""Unit tests for SQLite + SQLAlchemy database persistence and repository operations."""

from sqlalchemy.orm import sessionmaker

from src.database.connection import create_db_engine, get_session, init_db
from src.database.repository import EvaluationRepository


def test_database_crud_and_cascade(tmp_path):
    db_file = tmp_path / "test_eval.db"
    engine = create_db_engine(f"sqlite:///{db_file.as_posix()}")
    init_db(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    with get_session(session_factory) as session:
        repo = EvaluationRepository(session)
        created = repo.create_evaluation(
            question="What is Gemini?",
            reference_answer="Gemini is a multimodal AI model developed by Google.",
            generated_answer="Google developed Gemini for text and images.",
            precision=0.75,
            recall=0.60,
            f1_score=0.6667,
            exact_match=0.0,
            semantic_similarity=0.82,
            keyword_coverage=0.70,
            ai_score=85.0,
            ai_feedback={"verdict": "Good", "overall_score": 85},
            entities=[{"entity_name": "Google", "entity_type": "Organization", "frequency": 1}],
            relationships=[
                {
                    "source_entity": "Google",
                    "relationship_type": "DEVELOPED",
                    "target_entity": "Gemini",
                    "evidence_sentence": "Google developed Gemini.",
                    "confidence": 0.85,
                }
            ],
        )
        eval_id = created.id

    with get_session(session_factory) as session:
        repo = EvaluationRepository(session)
        fetched = repo.get_by_id(eval_id)
        assert fetched is not None
        assert fetched.question == "What is Gemini?"
        assert len(fetched.entities) == 1
        assert len(fetched.relationships) == 1

        summary = repo.get_dashboard_summary()
        assert summary["total_evaluations"] == 1
        assert summary["total_entities"] == 1
        assert summary["total_relationships"] == 1

        assert repo.delete_evaluation(eval_id) is True
        assert repo.get_by_id(eval_id) is None
