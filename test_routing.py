import asyncio
import json
import math
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from backend import classifier, decision, llm
from backend.classifier import INTENT_LABELS


def _classification(intent, confidence, margin):
    remaining = [label for label in INTENT_LABELS if label != intent]
    return {
        "intent": intent,
        "confidence": confidence,
        "confidence_margin": margin,
        "all_scores": {intent: confidence, remaining[0]: confidence - margin},
    }


def _valid_llm_response(**overrides):
    response = {
        "intent": "refund request",
        "confidence": 0.82,
        "tone": "neutral",
        "key_issue": "The customer asks about a refund.",
        "suggested_action": "Review the refund request.",
        "priority": "medium",
    }
    response.update(overrides)
    return response


class ClassifierTests(unittest.TestCase):
    def test_classifier_returns_top_score_and_top_two_margin(self):
        fake_classifier = MagicMock(return_value={
            "labels": ["refund request", "billing issue", "general inquiry"],
            "scores": [0.72, 0.18, 0.10],
        })
        with patch.object(classifier, "load_classifier", return_value=fake_classifier):
            result = classifier.classify_intent("Please check the refund on my order")

        self.assertEqual(result["intent"], "refund request")
        self.assertEqual(result["confidence"], 0.72)
        self.assertEqual(result["confidence_margin"], 0.54)
        self.assertEqual(result["all_scores"]["billing issue"], 0.18)

    def test_classifier_rejects_non_finite_scores(self):
        fake_classifier = MagicMock(return_value={
            "labels": ["refund request"],
            "scores": [math.nan],
        })
        with patch.object(classifier, "load_classifier", return_value=fake_classifier):
            with self.assertRaises(RuntimeError):
                classifier.classify_intent("Please check my refund request")


class RoutingDecisionTests(unittest.TestCase):
    def test_high_confidence_and_margin_auto_route_to_the_mapped_team(self):
        result = decision.decide(
            "order tracking", 0.91, {}, "Please check my recent order",
            confidence_margin=0.25,
        )
        self.assertEqual(result["routing"], "auto")
        self.assertEqual(result["department"], "Logistics Team")
        self.assertFalse(result["human_review_required"])
        self.assertIn("account profile", result["action"])

    def test_high_score_with_small_margin_requires_human_review(self):
        result = decision.decide(
            "refund request", 0.91, {}, "Please check my recent order",
            confidence_margin=0.03,
        )
        self.assertEqual(result["routing"], "human")
        self.assertTrue(result["human_review_required"])
        self.assertIn("top-two score margin", result["routing_reason"])

    def test_missing_margin_does_not_auto_route(self):
        result = decision.decide(
            "order tracking", 0.95, {}, "Please check my recent order",
        )
        self.assertEqual(result["routing"], "human")
        self.assertTrue(result["human_review_required"])

    def test_medium_confidence_uses_llm_only_when_enabled(self):
        args = ("billing issue", 0.72, {}, "Please review this billing issue")
        enabled = decision.decide(*args, confidence_margin=0.20, llm_enabled=True)
        disabled = decision.decide(*args, confidence_margin=0.20, llm_enabled=False)

        self.assertTrue(enabled["needs_llm"])
        self.assertTrue(enabled["human_review_required"])
        self.assertEqual(enabled["routing"], "llm")
        self.assertFalse(disabled["needs_llm"])
        self.assertEqual(disabled["routing"], "human")

    def test_low_confidence_and_ambiguous_messages_require_review(self):
        low = decision.decide(
            "general inquiry", 0.40, {}, "Please help with my account",
            confidence_margin=0.20,
        )
        ambiguous = decision.decide(
            "general inquiry", 0.99, {}, "Need help",
            confidence_margin=0.80,
        )

        self.assertEqual(low["routing"], "human")
        self.assertTrue(low["human_review_required"])
        self.assertEqual(ambiguous["routing"], "human")
        self.assertIn("insufficient context", ambiguous["routing_reason"])

    def test_security_cues_override_confidence(self):
        result = decision.decide(
            "card issue", 0.99, {"urgency": ["stolen"]},
            "My card was stolen and I need help",
            confidence_margin=0.80,
        )
        self.assertEqual(result["department"], "Security & Fraud Ops")
        self.assertTrue(result["human_review_required"])

    def test_all_intents_have_expected_support_team(self):
        expected = {
            "order tracking": "Logistics Team",
            "refund request": "Billing & Refunds",
            "billing issue": "Billing & Refunds",
            "account problem": "Account Support",
            "product complaint": "Customer Care",
            "payment failure": "Payments Team",
            "card issue": "Cards Team",
            "general inquiry": "General Support",
        }
        for intent, department_name in expected.items():
            with self.subTest(intent=intent):
                result = decision.decide(
                    intent, 0.95, {}, "Please review this customer message",
                    confidence_margin=0.30,
                )
                self.assertEqual(result["department"], department_name)


class LLMFallbackTests(unittest.TestCase):
    class MockResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {"content": [{"text": self.content}]}

    class MockAsyncClient:
        def __init__(self, response=None, error=None, **_kwargs):
            self.response = response
            self.error = error

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            if self.error:
                raise self.error
            return self.response

    def _call_with_mocked_client(self, response=None, error=None):
        client_factory = lambda **kwargs: self.MockAsyncClient(response, error, **kwargs)
        with (
            patch.object(llm, "ANTHROPIC_ENABLED", True),
            patch.object(llm, "ANTHROPIC_API_KEY", "test-key"),
            patch.object(llm.httpx, "AsyncClient", side_effect=client_factory),
        ):
            return asyncio.run(llm.call_claude(
                "Please review this refund request",
                predicted_intent="refund request",
                confidence=0.72,
                all_scores={"refund request": 0.72, "billing issue": 0.52},
            ))

    def test_timeout_returns_local_fallback(self):
        result = self._call_with_mocked_client(error=httpx.TimeoutException("timeout"))
        self.assertEqual(result["source"], "local_heuristic_engine")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["confidence_source"], "not_calibrated")

    def test_invalid_llm_response_returns_local_fallback(self):
        response_text = json.dumps(_valid_llm_response(confidence=2.0))
        result = self._call_with_mocked_client(self.MockResponse(response_text))
        self.assertEqual(result["source"], "local_heuristic_engine")
        self.assertIn("Malformed", result["reason"])

    def test_invalid_anthropic_content_structure_returns_local_fallback(self):
        response = self.MockResponse("")
        response.json = lambda: {"content": [{"text": 123}]}
        result = self._call_with_mocked_client(response)

        self.assertEqual(result["source"], "local_heuristic_engine")
        self.assertIn("Malformed Anthropic API response", result["reason"])

    def test_valid_llm_response_passes_validation(self):
        response_text = json.dumps(_valid_llm_response())
        result = self._call_with_mocked_client(self.MockResponse(response_text))
        self.assertEqual(result["source"], "claude_api")
        self.assertEqual(result["intent"], "refund request")

    def test_api_calls_are_disabled_by_default_even_with_a_key(self):
        with (
            patch.object(llm, "ANTHROPIC_ENABLED", False),
            patch.object(llm, "ANTHROPIC_API_KEY", "test-key"),
            patch.object(llm.httpx, "AsyncClient", side_effect=AssertionError("network call attempted")),
        ):
            result = asyncio.run(llm.call_claude("Please review my refund"))
        self.assertEqual(result["source"], "local_heuristic_engine")


class AnalyzeEndpointTests(unittest.TestCase):
    def _post_analyze(self, classification, enable_llm=False, llm_result=None):
        from fastapi.testclient import TestClient
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from backend import main
        from database.db import Base

        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        test_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

        def override_get_db():
            db_session = test_session()
            try:
                yield db_session
            finally:
                db_session.close()

        main.app.dependency_overrides[main.get_db] = override_get_db
        llm_result = llm_result or {**_valid_llm_response(), "source": "claude_api"}
        try:
            with (
                patch.object(main, "init_db"),
                patch.object(main, "classify_intent", return_value=classification),
                patch.object(main, "extract_entities", return_value={}),
                patch.object(main, "ANTHROPIC_ENABLED", enable_llm),
                patch.object(main, "ANTHROPIC_API_KEY", "test-key" if enable_llm else ""),
                patch.object(main, "call_claude", new=AsyncMock(return_value=llm_result)) as mock_llm,
                TestClient(main.app) as client,
            ):
                response = client.post(
                    "/analyze",
                    json={"query": "Please review this customer support request"},
                )
            stored = test_session().query(main.PredictionLog).one()
            return response, json.loads(stored.llm_response), mock_llm
        finally:
            main.app.dependency_overrides.pop(main.get_db, None)
            engine.dispose()

    def test_high_confidence_response_has_explanation_and_no_fallback(self):
        response, saved, mock_llm = self._post_analyze(
            _classification("order tracking", 0.91, 0.25)
        )
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["predicted_intent"], "order tracking")
        self.assertEqual(body["department"], "Logistics Team")
        self.assertEqual(body["confidence"], 0.91)
        self.assertFalse(body["llm_attempted"])
        self.assertFalse(body["fallback_used"])
        self.assertFalse(body["human_review_required"])
        self.assertFalse(saved["fallback_used"])
        self.assertFalse(saved["llm_attempted"])
        mock_llm.assert_not_awaited()

    def test_medium_confidence_uses_mocked_llm_and_preserves_human_review(self):
        response, saved, mock_llm = self._post_analyze(
            _classification("refund request", 0.72, 0.20),
            enable_llm=True,
        )
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["intent"], "refund request")
        self.assertEqual(body["department"], "Billing & Refunds")
        self.assertEqual(body["routing"], "llm")
        self.assertTrue(body["llm_attempted"])
        self.assertFalse(body["fallback_used"])
        self.assertTrue(body["human_review_required"])
        self.assertTrue(saved["llm_attempted"])
        self.assertFalse(saved["fallback_used"])
        self.assertTrue(saved["human_review_required"])
        mock_llm.assert_awaited_once()

    def test_medium_confidence_does_not_call_llm_when_disabled(self):
        response, saved, mock_llm = self._post_analyze(
            _classification("refund request", 0.72, 0.20),
        )
        body = response.json()

        self.assertEqual(body["routing"], "human")
        self.assertTrue(body["human_review_required"])
        self.assertFalse(body["llm_attempted"])
        self.assertFalse(body["fallback_used"])
        self.assertFalse(saved["llm_attempted"])
        mock_llm.assert_not_awaited()

    def test_api_records_local_fallback_when_optional_llm_fails(self):
        local_fallback = {
            **_valid_llm_response(),
            "confidence": 0.0,
            "confidence_source": "not_calibrated",
            "source": "local_heuristic_engine",
        }
        response, saved, mock_llm = self._post_analyze(
            _classification("refund request", 0.72, 0.20),
            enable_llm=True,
            llm_result=local_fallback,
        )
        body = response.json()

        self.assertEqual(body["routing"], "llm")
        self.assertTrue(body["llm_attempted"])
        self.assertTrue(body["fallback_used"])
        self.assertTrue(body["human_review_required"])
        self.assertTrue(saved["fallback_used"])
        mock_llm.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
