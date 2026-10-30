"""
backend/main.py
FastAPI backend — exposes REST endpoints for the frontend and external clients.
Provides intent classification, entity extraction, routing explainability,
human review workflows, analytics, and health status.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import logging
from typing import List, Literal, Optional

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.config import ANTHROPIC_ENABLED, ANTHROPIC_API_KEY, API_PORT, API_HOST, LOG_LEVEL
from backend.classifier import classify_intent, INTENT_LABELS, get_department
from backend.ner import extract_entities
from backend.decision import decide
from backend.llm import call_claude
from database.db import init_db, get_db, PredictionLog

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smart_support")

app = FastAPI(
    title="Smart Support Routing API",
    version="2.0.0",
    description="Intelligent Customer Support Intent Recognition, Routing & Human Review Platform",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    logger.info("Initializing database schema...")
    init_db()
    logger.info("Smart Support Routing System initialized successfully.")


# ------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Customer inquiry text")

class FeedbackRequest(BaseModel):
    log_id: int = Field(..., ge=1, description="Database ID of the prediction log")
    feedback: str = Field(..., pattern="^(correct|incorrect)$", description="'correct' or 'incorrect'")

class ReviewSubmissionRequest(BaseModel):
    log_id: int = Field(..., ge=1, description="Database ID of the ticket")
    reviewed_intent: Literal[
        "order tracking",
        "refund request",
        "billing issue",
        "account problem",
        "product complaint",
        "payment failure",
        "card issue",
        "general inquiry",
    ] = Field(..., description="Human-verified intent category")
    notes: Optional[str] = Field(None, max_length=1000, description="Reviewer notes or reason for correction")
    status: Literal["reviewed", "resolved"] = Field(
        "reviewed",
        description="Review completion status",
    )


# ------------------------------------------------------------------
# Core Endpoints
# ------------------------------------------------------------------
@app.get("/", tags=["System"])
def root():
    return {
        "status": "online",
        "service": "Smart Support Routing System",
        "version": "2.0.0",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Healthcheck endpoint verifying DB connectivity and system readiness."""
    try:
        db_count = db.query(PredictionLog).count()
        return {
            "status": "healthy",
            "database": "connected",
            "total_records": db_count,
            "intents_supported": len(INTENT_LABELS),
        }
    except Exception as e:
        logger.error(f"Health check failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unreachable: {str(e)}"
        )


@app.post("/analyze", tags=["Pipeline"])
async def analyze(req: QueryRequest, db: Session = Depends(get_db)):
    """
    Main End-to-End Pipeline:
    1. Zero-shot intent classification
    2. spaCy domain entity extraction
    3. Multi-tier confidence & urgency routing decision
    4. Guarded AI reasoning escalation (if moderate confidence)
    5. Persistence with explainability reasoning
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty or pure whitespace."
        )

    # 1. Classification
    classification = classify_intent(query)
    intent = classification["intent"]
    confidence = classification["confidence"]
    confidence_margin = classification["confidence_margin"]

    # 2. Entity Extraction
    entities = extract_entities(query)

    # 3. Decision Engine
    decision = decide(
        intent=intent,
        confidence=confidence,
        entities=entities,
        query=query,
        confidence_margin=confidence_margin,
        llm_enabled=ANTHROPIC_ENABLED and bool(ANTHROPIC_API_KEY),
    )

    # 4. LLM Escalation if required
    llm_data = None
    llm_attempted = decision["needs_llm"]
    if decision["needs_llm"]:
        logger.info(f"Escalating query to LLM reasoning (Confidence: {confidence:.2f})")
        llm_data = await call_claude(
            query,
            entities=entities,
            predicted_intent=intent,
            confidence=confidence,
            all_scores=classification["all_scores"],
        )
        if llm_data:
            decision["action"] = llm_data["suggested_action"]
            decision["routing_reason"] += (
                f" Optional analysis ({llm_data.get('source', 'unknown')}) suggested "
                f"'{llm_data['intent']}' and identified: {llm_data['key_issue']} "
                f"The original prediction and destination are unchanged; human review is required."
            )
    fallback_used = bool(llm_data and llm_data.get("source") != "claude_api")
    human_review_required = decision["human_review_required"]

    # 5. Persist to Database
    log = PredictionLog(
        query=query,
        intent=intent,
        confidence=confidence,
        entities=json.dumps(entities),
        routing=decision["routing"],
        department=decision["department"],
        action=decision["action"],
        routing_reason=decision["routing_reason"],
        status=decision["status"],
        llm_response=json.dumps({
            "llm_attempted": llm_attempted,
            "fallback_used": fallback_used,
            "human_review_required": human_review_required,
            "llm_analysis": llm_data,
        }),
        feedback=None,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return {
        "log_id":         log.id,
        "query":          query,
        "intent":         intent,
        "predicted_intent": intent,
        "confidence":     confidence,
        "confidence_margin": confidence_margin,
        "all_scores":     classification["all_scores"],
        "entities":       entities,
        "routing":        decision["routing"],
        "department":     decision["department"],
        "action":         decision["action"],
        "badge":          decision["badge"],
        "status":         decision["status"],
        "routing_reason": decision["routing_reason"],
        "llm_attempted": llm_attempted,
        "fallback_used": fallback_used,
        "human_review_required": human_review_required,
        "llm_analysis":   llm_data,
        "timestamp":      log.timestamp.isoformat() if log.timestamp else None,
    }


@app.post("/feedback", tags=["Review & Feedback"])
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Records binary feedback ('correct' or 'incorrect') for a routing prediction."""
    log = db.query(PredictionLog).filter(PredictionLog.id == req.log_id).first()
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction log ID {req.log_id} not found."
        )

    if log.reviewed_intent is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Feedback is already recorded by the human review; it cannot be overwritten.",
        )

    log.feedback = req.feedback
    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to persist feedback for prediction log %s.", req.log_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feedback could not be saved.",
        ) from exc
    return {
        "status": "success",
        "message": f"Feedback '{req.feedback}' recorded for Log #{req.log_id}",
        "log_id": req.log_id
    }


@app.get("/review/queue", tags=["Review & Feedback"])
def get_review_queue(
    status_filter: Literal["pending_review", "reviewed", "resolved", "auto_routed", "all"] = Query(
        "pending_review",
        description="pending_review | reviewed | resolved | auto_routed | all",
    ),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Retrieves tickets requiring human evaluation or manual routing adjustment."""
    query = db.query(PredictionLog)
    if status_filter != "all":
        query = query.filter(PredictionLog.status == status_filter)
    
    logs = query.order_by(PredictionLog.id.desc()).limit(limit).all()
    return [
        {
            "id":              l.id,
            "query":           l.query,
            "predicted_intent": l.intent,
            "confidence":      l.confidence,
            "routing":         l.routing,
            "department":      l.department,
            "action":          l.action,
            "routing_reason":  l.routing_reason,
            "status":          l.status,
            "feedback":        l.feedback,
            "reviewed_intent": l.reviewed_intent,
            "reviewed_department": get_department(l.reviewed_intent) if l.reviewed_intent else None,
            "review_notes":    l.review_notes,
            "timestamp":       l.timestamp.isoformat() if l.timestamp else None,
            "reviewed_at":     l.reviewed_at.isoformat() if l.reviewed_at else None,
        }
        for l in logs
    ]


@app.post("/review/submit", tags=["Review & Feedback"])
def submit_human_review(req: ReviewSubmissionRequest, db: Session = Depends(get_db)):
    """Submits human agent review, recording corrected intent, notes, and updated status."""
    log = db.query(PredictionLog).filter(PredictionLog.id == req.log_id).first()
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket ID {req.log_id} not found."
        )

    if log.status != "pending_review":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ticket #{req.log_id} is already in status '{log.status}' and is not awaiting review.",
        )

    log.reviewed_intent = req.reviewed_intent
    log.review_notes = req.notes.strip() if req.notes and req.notes.strip() else None
    log.status = req.status
    log.reviewed_at = datetime.now(timezone.utc)
    log.feedback = "incorrect" if req.reviewed_intent != log.intent else "correct"

    try:
        db.commit()
        db.refresh(log)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to persist human review for prediction log %s.", req.log_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Human review could not be saved.",
        ) from exc

    return {
        "status": "success",
        "message": f"Ticket #{req.log_id} updated with reviewed intent '{req.reviewed_intent}'",
        "log_id": req.log_id,
        "new_status": log.status,
        "reviewed_intent": log.reviewed_intent,
        "reviewed_department": get_department(log.reviewed_intent),
        "feedback": log.feedback,
        "reviewed_at": log.reviewed_at.isoformat() if log.reviewed_at else None,
    }


@app.get("/review/{log_id}", tags=["Review & Feedback"])
def get_review_detail(log_id: int, db: Session = Depends(get_db)):
    """Returns the complete prediction and review context for one queue item."""
    log = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction log ID {log_id} not found.",
        )

    try:
        entities = json.loads(log.entities) if log.entities else {}
    except json.JSONDecodeError:
        logger.warning("Prediction log %s contains malformed entity data.", log_id)
        entities = {}

    return {
        "id": log.id,
        "query": log.query,
        "predicted_intent": log.intent,
        "confidence": log.confidence,
        "routing": log.routing,
        "department": log.department,
        "action": log.action,
        "routing_reason": log.routing_reason,
        "entities": entities,
        "status": log.status,
        "feedback": log.feedback,
        "reviewed_intent": log.reviewed_intent,
        "reviewed_department": get_department(log.reviewed_intent) if log.reviewed_intent else None,
        "review_notes": log.review_notes,
        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
        "reviewed_at": log.reviewed_at.isoformat() if log.reviewed_at else None,
        "reviewable": log.status == "pending_review",
    }


@app.get("/analytics", tags=["Analytics"])
def get_analytics(db: Session = Depends(get_db)):
    """
    Returns rich aggregate statistics for operational dashboards:
    - Volume, average confidence, routing breakdown
    - Confidence distribution buckets
    - Human-review frequency & correction rates
    - Feedback summary
    """
    logs = db.query(PredictionLog).all()
    if not logs:
        return {
            "total": 0,
            "avg_confidence": 0.0,
            "intents": {},
            "routing_breakdown": {},
            "status_breakdown": {},
            "confidence_distribution": {"<50%": 0, "50-65%": 0, "65-80%": 0, ">=80%": 0},
            "feedback_summary": {},
            "correction_rate": 0.0,
        }

    total = len(logs)
    intents = Counter(l.intent for l in logs if l.intent)
    routing = Counter(l.routing for l in logs if l.routing)
    statuses = Counter(l.status for l in logs if l.status)
    feedbacks = Counter(l.feedback for l in logs if l.feedback)

    conf_values = [l.confidence for l in logs if l.confidence is not None]
    avg_conf = round(sum(conf_values) / len(conf_values), 4) if conf_values else 0.0

    # Confidence distribution buckets
    buckets = {"<50%": 0, "50-65%": 0, "65-80%": 0, ">=80%": 0}
    for c in conf_values:
        if c < 0.50:
            buckets["<50%"] += 1
        elif c < 0.65:
            buckets["50-65%"] += 1
        elif c < 0.80:
            buckets["65-80%"] += 1
        else:
            buckets[">=80%"] += 1

    # Reviewed tickets correction count
    reviewed_count = sum(1 for l in logs if l.reviewed_intent is not None)
    corrections = sum(1 for l in logs if l.reviewed_intent and l.reviewed_intent != l.intent)
    correction_rate = round(corrections / reviewed_count, 4) if reviewed_count > 0 else 0.0

    return {
        "total":                   total,
        "avg_confidence":          avg_conf,
        "intents":                 dict(intents),
        "routing_breakdown":       dict(routing),
        "status_breakdown":        dict(statuses),
        "confidence_distribution": buckets,
        "feedback_summary":        dict(feedbacks),
        "human_review_count":      statuses.get("pending_review", 0) + statuses.get("reviewed", 0),
        "corrected_tickets":       corrections,
        "correction_rate":         correction_rate,
    }


@app.get("/logs", tags=["Analytics"])
def get_logs(limit: int = Query(25, ge=1, le=200), db: Session = Depends(get_db)):
    """Returns recent prediction records with full routing & review metadata."""
    logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
    return [
        {
            "id":              l.id,
            "query":           l.query,
            "intent":          l.intent,
            "confidence":      l.confidence,
            "routing":         l.routing,
            "department":      l.department,
            "action":          l.action,
            "routing_reason":  l.routing_reason,
            "status":          l.status,
            "feedback":        l.feedback,
            "reviewed_intent": l.reviewed_intent,
            "timestamp":       l.timestamp.isoformat() if l.timestamp else None,
        }
        for l in logs
    ]
