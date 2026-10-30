import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from database import db


class DatabaseMigrationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        with self.engine.begin() as connection:
            connection.execute(text("""
                CREATE TABLE prediction_logs (
                    id INTEGER PRIMARY KEY,
                    query TEXT NOT NULL,
                    intent VARCHAR(100),
                    confidence FLOAT,
                    entities TEXT,
                    routing VARCHAR(50),
                    action VARCHAR(200),
                    llm_response TEXT,
                    feedback VARCHAR(20),
                    timestamp DATETIME
                )
            """))
            connection.execute(text(
                "INSERT INTO prediction_logs (id, query) VALUES (1, 'existing record')"
            ))

    def tearDown(self):
        self.engine.dispose()

    def test_init_db_migrates_legacy_schema_without_losing_records(self):
        with patch.object(db, "engine", self.engine):
            db.init_db()

        with self.engine.connect() as connection:
            columns = {
                row[1]
                for row in connection.execute(text("PRAGMA table_info(prediction_logs)"))
            }
            record_count = connection.execute(
                text("SELECT COUNT(*) FROM prediction_logs")
            ).scalar_one()

        self.assertTrue({
            "department",
            "routing_reason",
            "status",
            "reviewed_intent",
            "review_notes",
            "reviewed_at",
        }.issubset(columns))
        self.assertEqual(record_count, 1)

    def test_init_db_logs_and_propagates_migration_errors(self):
        with (
            patch.object(db, "engine", self.engine),
            patch.object(db, "text", side_effect=SQLAlchemyError("migration failed")),
            self.assertLogs("database.db", level="ERROR"),
        ):
            with self.assertRaises(SQLAlchemyError):
                db.init_db()


if __name__ == "__main__":
    unittest.main()
