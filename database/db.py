"""
database/db.py
Handles SQLite logging of every prediction, feedback, and handoff event.
Includes automated column migration and session management.
"""

from datetime import datetime, timezone
import logging
import os
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, text
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

try:
    from backend.config import DATABASE_URL
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "logs.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"

# Thread-safe SQLite engine
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id              = Column(Integer, primary_key=True, index=True)
    query           = Column(Text, nullable=False)
    intent          = Column(String(100), nullable=True)
    confidence      = Column(Float, nullable=True)
    entities        = Column(Text, nullable=True)          # JSON string
    routing         = Column(String(50), nullable=True)    # "auto" | "llm" | "human"
    department      = Column(String(100), nullable=True)   # Assigned routing department
    action          = Column(String(200), nullable=True)
    routing_reason  = Column(Text, nullable=True)          # Explainability rationale
    status          = Column(String(50), default="pending_review") # auto_routed | pending_review | reviewed | resolved
    llm_response    = Column(Text, nullable=True)          # JSON string
    feedback        = Column(String(20), nullable=True)    # "correct" | "incorrect" | None
    reviewed_intent = Column(String(100), nullable=True)   # Corrected label by reviewer
    review_notes    = Column(Text, nullable=True)          # Notes provided by reviewer
    reviewed_at     = Column(DateTime, nullable=True)
    timestamp       = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    """Initializes tables and performs non-destructive schema migrations.

    Migration errors are logged and propagated so startup cannot succeed with
    an incomplete schema.
    """
    Base.metadata.create_all(bind=engine)

    # Perform lightweight migration for SQLite tables if columns are missing
    with engine.connect() as conn:
        try:
            result = conn.execute(text("PRAGMA table_info(prediction_logs)"))
            existing_cols = {row[1] for row in result.fetchall()}

            desired_columns = {
                "department": "VARCHAR(100)",
                "routing_reason": "TEXT",
                "status": "VARCHAR(50) DEFAULT 'pending_review'",
                "reviewed_intent": "VARCHAR(100)",
                "review_notes": "TEXT",
                "reviewed_at": "DATETIME",
            }

            for col_name, col_def in desired_columns.items():
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE prediction_logs ADD COLUMN {col_name} {col_def}"))

            conn.commit()
        except SQLAlchemyError:
            logger.exception("Failed to migrate the prediction_logs database schema.")
            raise


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
