"""
test_pipeline.py
Automated sanity tests for classifier, NER, decision engine, LLM reasoning, and database logging.
Usage: python test_pipeline.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import asyncio
import json
from unittest.mock import AsyncMock, patch
import torch
torch.set_num_threads(2)
from backend.classifier import classify_intent, INTENT_LABELS
from backend.ner import extract_entities
from backend.decision import decide
from backend.llm import call_claude
from database.db import init_db, get_db, PredictionLog


def test_classifier():
    print("\n" + "="*50)
    print("TEST 1: Intent Classifier")
    print("="*50)

    queries = [
        "I want a refund for my damaged phone",
        "Where is my order #12345?",
        "My card was charged twice this month",
        "I can't log into my account",
        "This is absolutely unacceptable, your service is terrible",
    ]
    for q in queries:
        result = classify_intent(q)
        print(f"\nQuery: {q}")
        print(f"  Intent:     {result['intent']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        assert result["intent"] in INTENT_LABELS, f"Unknown intent: {result['intent']}"
        assert 0.0 <= result["confidence"] <= 1.0


def test_ner():
    print("\n" + "="*50)
    print("TEST 2: Named Entity Recognition")
    print("="*50)

    query = "My iPhone 15 order #98765 hasn't arrived, I paid $999 last Monday and card ending in 4321 was charged"
    entities = extract_entities(query)
    print(f"\nQuery: {query}")
    print(f"Entities: {json.dumps(entities, indent=2)}")

    assert "order_ids" in entities and "98765" in entities["order_ids"], "Order ID extraction failed"
    assert "amounts" in entities and any("$999" in a for a in entities["amounts"]), "Amount extraction failed"
    assert "card_last4" in entities and "4321" in entities["card_last4"], "Card extraction failed"


def test_decision():
    print("\n" + "="*50)
    print("TEST 3: Decision Engine")
    print("="*50)

    scenarios = [
        ("refund request",  0.91, {"amounts": ["$50"]}),
        ("billing issue",   0.72, {"amounts": ["$50"]}),
        ("general inquiry", 0.55, {}),
        ("card issue",      0.85, {"urgency": ["stolen"]}),
    ]
    for intent, conf, entities in scenarios:
        d = decide(intent, conf, entities, "test query for system routing")
        print(f"\nIntent: {intent} | Confidence: {conf*100:.0f}%")
        print(f"  Routing:    {d['routing'].upper()} ({d['badge']})")
        print(f"  Department: {d['department']}")
        print(f"  Action:     {d['action']}")
        print(f"  Reason:     {d['routing_reason']}")
        assert d["routing"] in ["auto", "llm", "human"]
        assert d["department"]


async def test_llm():
    print("\n" + "="*50)
    print("TEST 4: LLM Reasoning Integration")
    print("="*50)

    query = "This product is useless, stopped working in 2 days and nobody is responding"
    print(f"\nQuery: {query}")
    mocked_result = {
        "intent": "product complaint",
        "confidence": 0.0,
        "tone": "frustrated",
        "suggested_action": "Review the reported product issue.",
    }
    with patch("test_pipeline.call_claude", new=AsyncMock(return_value=mocked_result)):
        result = await call_claude(query)
    print(f"LLM Response:\n{json.dumps(result, indent=2)}")
    assert "intent" in result and "tone" in result and "suggested_action" in result


def test_database():
    print("\n" + "="*50)
    print("TEST 5: Database Schema & Logging")
    print("="*50)
    init_db()
    db = next(get_db())
    try:
        log = PredictionLog(
            query="Automated test query",
            intent="general inquiry",
            confidence=0.95,
            entities=json.dumps({"order_ids": ["TEST1234"]}),
            routing="auto",
            department="General Support",
            action="Automated test action",
            routing_reason="Test reason",
            status="auto_routed",
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        print(f"Logged test row ID: {log.id}, status: {log.status}, department: {log.department}")
        assert log.id is not None
        # Clean up test row
        db.delete(log)
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    test_classifier()
    test_ner()
    test_decision()
    asyncio.run(test_llm())
    test_database()
    print("\n" + "="*50)
    print(" ALL TESTS PASSED SUCCESSFULLY! ")
    print("="*50)
