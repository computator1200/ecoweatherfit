"""SQLAlchemy engine, session factory, and one-time schema initialisation.

SQLite-specific gotcha: foreign-key constraints (and the ON DELETE CASCADE
behaviour our schema declares) are **off** by default in SQLite. We turn
them on for every new connection with a ``PRAGMA foreign_keys = ON`` event
listener — without this, deleting a User row leaves orphan rows in
user_preferences / wardrobe_items / forecast_history, which then collide
with a UNIQUE constraint on the next signup that re-uses the same id.
"""

import os
from pathlib import Path
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, declarative_base

import config.settings

# Fallbacks in case settings hasn't been populated yet
DATABASE_URL = getattr(
    config.settings,
    "DATABASE_URL",
    f"sqlite:///{getattr(config.settings, 'PROJECT_ROOT', os.getcwd())}/data/users.db",
)


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
    """Turn on FK enforcement for every SQLite connection.

    Other dialects ignore this hook because their `cursor.execute("PRAGMA …")`
    would no-op or fail — we guard with a class check so the listener stays
    inert on Postgres/MySQL connections.
    """
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
        finally:
            cursor.close()


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    """Create any missing tables. Idempotent. Never drops."""
    if DATABASE_URL.startswith("sqlite:///"):
        db_path_str = DATABASE_URL.replace("sqlite:///", "")
        if db_path_str and db_path_str != ":memory:":
            db_path = Path(db_path_str).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(engine)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def configure_engine(url: str):
    """Test hook: swap the engine to point at a fresh DB URL."""
    global engine, SessionLocal
    engine = create_engine(url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
