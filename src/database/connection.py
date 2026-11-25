"""SQLite database engine, session factory, and safe initialization."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from src.database.models import Base


def _configure_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine with SQLite foreign-key enforcement."""
    url = database_url or settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _configure_sqlite_pragmas)
    return engine


_engine = create_db_engine()
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db(engine: Engine | None = None) -> None:
    """Create all database tables if they do not already exist."""
    target_engine = engine or _engine
    Base.metadata.create_all(bind=target_engine)


@contextmanager
def get_session(session_factory: sessionmaker = SessionLocal) -> Iterator[Session]:
    """Provide a transactional database session scope."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
