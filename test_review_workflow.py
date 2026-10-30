import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import main
from database.db import Base, PredictionLog


class HumanReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
        )

        def override_get_db():
            session = self.session_factory()
            try:
                yield session
            finally:
                session.close()

        self.previous_override = main.app.dependency_overrides.get(main.get_db)
        main.app.dependency_overrides[main.get_db] = override_get_db
        self.startup_patch = patch.object(main, "init_db")
        self.startup_patch.start()
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()
        self.pending_id = self._create_ticket()
        self.completed_id = self._create_ticket(status="resolved")

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.startup_patch.stop()
        if self.previous_override is None:
            main.app.dependency_overrides.pop(main.get_db, None)
        else:
            main.app.dependency_overrides[main.get_db] = self.previous_override
        self.engine.dispose()

    def _create_ticket(self, status="pending_review"):
        with self.session_factory() as session:
            ticket = PredictionLog(
                query="My monthly charge was duplicated.",
                intent="billing issue",
                confidence=0.58,
                entities='{"amounts": ["$25"]}',
                routing="human",
                department="Human Agent Queue",
                action="Triage and verify the billing issue.",
                routing_reason="Low classifier confidence.",
                status=status,
                timestamp=datetime.now(timezone.utc),
            )
            session.add(ticket)
            session.commit()
            session.refresh(ticket)
            return ticket.id

    def test_queue_and_detail_provide_review_context(self):
        queue = self.client.get("/review/queue")
        detail = self.client.get(f"/review/{self.pending_id}")
        completed_detail = self.client.get(f"/review/{self.completed_id}")

        self.assertEqual(queue.status_code, 200)
        self.assertEqual([ticket["id"] for ticket in queue.json()], [self.pending_id])
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["query"], "My monthly charge was duplicated.")
        self.assertEqual(detail.json()["predicted_intent"], "billing issue")
        self.assertEqual(detail.json()["entities"], {"amounts": ["$25"]})
        self.assertTrue(detail.json()["reviewable"])
        self.assertFalse(completed_detail.json()["reviewable"])

    def test_review_persists_correction_notes_status_and_feedback(self):
        response = self.client.post(
            "/review/submit",
            json={
                "log_id": self.pending_id,
                "reviewed_intent": "refund request",
                "notes": "  Duplicate transaction should be investigated as a refund.  ",
                "status": "reviewed",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["feedback"], "incorrect")
        self.assertEqual(response.json()["reviewed_intent"], "refund request")
        self.assertEqual(response.json()["reviewed_department"], "Billing & Refunds")
        with self.session_factory() as session:
            ticket = session.get(PredictionLog, self.pending_id)
            self.assertEqual(ticket.intent, "billing issue")
            self.assertEqual(ticket.department, "Human Agent Queue")
            self.assertEqual(ticket.reviewed_intent, "refund request")
            self.assertEqual(ticket.review_notes, "Duplicate transaction should be investigated as a refund.")
            self.assertEqual(ticket.status, "reviewed")
            self.assertEqual(ticket.feedback, "incorrect")
            self.assertIsNotNone(ticket.reviewed_at)

        self.assertEqual(self.client.get("/review/queue").json(), [])
        self.assertEqual(
            self.client.get("/review/queue", params={"status_filter": "reviewed"}).json()[0]["id"],
            self.pending_id,
        )
        self.assertEqual(
            self.client.get(f"/review/{self.pending_id}").json()["reviewed_intent"],
            "refund request",
        )
        self.assertEqual(
            self.client.get(f"/review/{self.pending_id}").json()["reviewed_department"],
            "Billing & Refunds",
        )

    def test_matching_human_label_records_correct_feedback_and_resolves(self):
        response = self.client.post(
            "/review/submit",
            json={
                "log_id": self.pending_id,
                "reviewed_intent": "billing issue",
                "status": "resolved",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["feedback"], "correct")
        self.assertEqual(response.json()["new_status"], "resolved")
        with self.session_factory() as session:
            ticket = session.get(PredictionLog, self.pending_id)
            self.assertEqual(ticket.status, "resolved")
            self.assertEqual(ticket.feedback, "correct")
            self.assertIsNone(ticket.review_notes)

    def test_only_pending_tickets_can_be_reviewed_once(self):
        payload = {
            "log_id": self.completed_id,
            "reviewed_intent": "billing issue",
            "status": "reviewed",
        }
        response = self.client.post("/review/submit", json=payload)

        self.assertEqual(response.status_code, 409)
        with self.session_factory() as session:
            ticket = session.get(PredictionLog, self.completed_id)
            self.assertIsNone(ticket.reviewed_intent)
            self.assertIsNone(ticket.feedback)

    def test_invalid_labels_statuses_and_queue_filters_are_rejected(self):
        invalid_intent = self.client.post(
            "/review/submit",
            json={
                "log_id": self.pending_id,
                "reviewed_intent": "not a supported intent",
                "status": "reviewed",
            },
        )
        invalid_status = self.client.post(
            "/review/submit",
            json={
                "log_id": self.pending_id,
                "reviewed_intent": "billing issue",
                "status": "pending_review",
            },
        )
        invalid_filter = self.client.get(
            "/review/queue",
            params={"status_filter": "unknown"},
        )

        self.assertEqual(invalid_intent.status_code, 422)
        self.assertEqual(invalid_status.status_code, 422)
        self.assertEqual(invalid_filter.status_code, 422)

    def test_binary_feedback_is_persisted_but_cannot_overwrite_human_review(self):
        feedback = self.client.post(
            "/feedback",
            json={"log_id": self.pending_id, "feedback": "incorrect"},
        )
        self.assertEqual(feedback.status_code, 200)
        with self.session_factory() as session:
            self.assertEqual(session.get(PredictionLog, self.pending_id).feedback, "incorrect")

        reviewed = self.client.post(
            "/review/submit",
            json={
                "log_id": self.pending_id,
                "reviewed_intent": "billing issue",
                "status": "reviewed",
            },
        )
        overwrite = self.client.post(
            "/feedback",
            json={"log_id": self.pending_id, "feedback": "incorrect"},
        )

        self.assertEqual(reviewed.status_code, 200)
        self.assertEqual(overwrite.status_code, 409)
        with self.session_factory() as session:
            self.assertEqual(session.get(PredictionLog, self.pending_id).feedback, "correct")

    def test_missing_ticket_returns_not_found(self):
        response = self.client.get("/review/99999")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
