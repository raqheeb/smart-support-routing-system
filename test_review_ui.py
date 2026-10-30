import unittest
from unittest.mock import MagicMock, patch

from frontend import app


class HumanReviewInterfaceTests(unittest.TestCase):
    class MockResponse:
        def __init__(self, data, status_error=None):
            self.data = data
            self.status_error = status_error

        def raise_for_status(self):
            if self.status_error:
                raise self.status_error

        def json(self):
            return self.data

    def _client(self, method, response):
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = None
        getattr(client, method).return_value = response
        return client

    def test_queue_returns_empty_state_and_dropdown_values(self):
        ticket = {
            "id": 18,
            "query": "Please check this refund request.",
            "predicted_intent": "refund request",
            "confidence": 0.62,
            "routing": "human",
            "status": "pending_review",
        }
        client = self._client("get", self.MockResponse([ticket]))
        with patch.object(app.httpx, "Client", return_value=client):
            rows, selector, message = app.load_review_queue()

        self.assertEqual(rows[0][0], "18")
        self.assertEqual(selector["choices"], [("#18 - refund request", "18")])
        self.assertEqual(selector["value"], "18")
        self.assertIn("1 ticket", message)

        client = self._client("get", self.MockResponse([]))
        with patch.object(app.httpx, "Client", return_value=client):
            rows, selector, message = app.load_review_queue()
        self.assertEqual(rows, [])
        self.assertIsNone(selector["value"])
        self.assertIn("No tickets", message)

    def test_detail_loads_full_message_and_enables_pending_review(self):
        ticket = {
            "id": 18,
            "query": "The complete customer message.",
            "predicted_intent": "refund request",
            "confidence": 0.62,
            "routing": "human",
            "department": "Human Agent Queue",
            "routing_reason": "Low confidence.",
            "entities": {"amounts": ["$15"]},
            "status": "pending_review",
            "reviewable": True,
            "timestamp": "2026-10-30T12:00:00+00:00",
        }
        client = self._client("get", self.MockResponse(ticket))
        with patch.object(app.httpx, "Client", return_value=client):
            query, details, intent, status, team, notes, submit = app.load_review_detail("18")

        self.assertEqual(query, ticket["query"])
        self.assertEqual(details["assigned_team"], "Human Agent Queue")
        self.assertEqual(intent["value"], "refund request")
        self.assertTrue(intent["interactive"])
        self.assertEqual(status["value"], "reviewed")
        self.assertIn("Billing & Refunds", team)
        self.assertTrue(submit["interactive"])

    def test_closed_ticket_details_are_read_only(self):
        ticket = {
            "id": 18,
            "query": "Already reviewed.",
            "predicted_intent": "billing issue",
            "status": "resolved",
            "reviewable": False,
        }
        client = self._client("get", self.MockResponse(ticket))
        with patch.object(app.httpx, "Client", return_value=client):
            _, details, intent, status, _, _, submit = app.load_review_detail("18")

        self.assertFalse(details["reviewable"])
        self.assertFalse(intent["interactive"])
        self.assertFalse(status["interactive"])
        self.assertFalse(submit["interactive"])

    def test_corrected_intent_previews_new_destination(self):
        self.assertEqual(
            app.preview_review_department("refund request"),
            "Destination team for verified intent: Billing & Refunds",
        )

    def test_submit_action_sends_review_and_surfaces_server_conflicts(self):
        client = self._client("post", self.MockResponse({
            "reviewed_intent": "billing issue",
            "reviewed_department": "Billing & Refunds",
            "new_status": "reviewed",
            "feedback": "correct",
        }))
        with patch.object(app.httpx, "Client", return_value=client):
            message = app.submit_review_action("18", "billing issue", "Verified.", "reviewed")
        self.assertIn("Ticket #18 reviewed", message)
        self.assertEqual(
            client.post.call_args.kwargs["json"],
            {
                "log_id": 18,
                "reviewed_intent": "billing issue",
                "notes": "Verified.",
                "status": "reviewed",
            },
        )

        import httpx

        conflict = httpx.Response(
            409,
            json={"detail": "Ticket is already resolved."},
            request=httpx.Request("POST", "http://testserver/review/submit"),
        )
        failing_client = self._client("post", self.MockResponse(
            {},
            status_error=httpx.HTTPStatusError(
                "conflict",
                request=conflict.request,
                response=conflict,
            ),
        ))
        with patch.object(app.httpx, "Client", return_value=failing_client):
            error = app.submit_review_action("18", "billing issue", "", "reviewed")
        self.assertIn("already resolved", error)


if __name__ == "__main__":
    unittest.main()
