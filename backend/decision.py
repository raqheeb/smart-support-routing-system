"""
Confidence and safety rules for assigning support messages to a destination.
"""

import math
from typing import Any, Dict, Optional

from backend.classifier import get_department
from backend.config import (
    ANTHROPIC_ENABLED,
    AUTO_ROUTE_MIN_MARGIN,
    AUTO_ROUTE_THRESHOLD,
    LLM_ASSIST_THRESHOLD,
)

SECURITY_CUES = {"stolen", "fraud", "unauthorized", "compromised", "hacked"}


def _decision(
    routing: str,
    department: str,
    action: str,
    badge: str,
    status: str,
    reason: str,
    needs_llm: bool = False,
) -> Dict[str, Any]:
    return {
        "routing": routing,
        "department": department,
        "action": action,
        "badge": badge,
        "status": status,
        "needs_llm": needs_llm,
        "human_review_required": status == "pending_review",
        "routing_reason": reason,
    }


def decide(
    intent: str,
    confidence: float,
    entities: dict,
    query: str,
    confidence_margin: Optional[float] = None,
    llm_enabled: bool = ANTHROPIC_ENABLED,
) -> Dict[str, Any]:
    """Choose a route using the classifier's top score and top-two margin."""
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("Confidence must be a finite value between 0 and 1.")
    if confidence_margin is not None and (
        isinstance(confidence_margin, bool)
        or not isinstance(confidence_margin, (int, float))
        or not math.isfinite(confidence_margin)
        or not 0.0 <= confidence_margin <= 1.0
    ):
        raise ValueError("Confidence margin must be a finite value between 0 and 1.")

    department = get_department(intent)
    words = query.split()
    urgency_terms = entities.get("urgency", [])
    security_terms = [term for term in urgency_terms if term.lower() in SECURITY_CUES]
    if security_terms:
        return _decision(
            "human",
            "Security & Fraud Ops",
            "Priority escalation: security flag detected. Route to Fraud Ops.",
            "red",
            "pending_review",
            f"Security cues ({', '.join(security_terms)}) require human review regardless of classifier confidence.",
        )

    if len(words) <= 2:
        return _decision(
            "human",
            "General Support",
            "Request clarification before assigning a specialist.",
            "red",
            "pending_review",
            f"The message contains only {len(words)} word(s), so there is insufficient context for reliable routing.",
        )

    if confidence >= AUTO_ROUTE_THRESHOLD:
        if confidence_margin is None or confidence_margin < AUTO_ROUTE_MIN_MARGIN:
            margin_text = "unavailable" if confidence_margin is None else f"{confidence_margin:.1%}"
            return _decision(
                "human",
                "Human Agent Queue",
                "Route to a human specialist to resolve competing intent predictions.",
                "red",
                "pending_review",
                f"Top intent confidence is {confidence:.1%}, but the top-two score margin "
                f"({margin_text}) is below the required {AUTO_ROUTE_MIN_MARGIN:.1%}.",
            )

        return _decision(
            "auto",
            department,
            _generate_auto_action(intent, entities),
            "green",
            "auto_routed",
            f"Top intent '{intent}' scored {confidence:.1%} (threshold {AUTO_ROUTE_THRESHOLD:.1%}) "
            f"with a {confidence_margin:.1%} margin (minimum {AUTO_ROUTE_MIN_MARGIN:.1%}); "
            f"routed to {department}.",
        )

    if confidence >= LLM_ASSIST_THRESHOLD and llm_enabled:
        return _decision(
            "llm",
            department,
            f"Request optional AI assistance for the '{intent}' prediction; retain human review.",
            "yellow",
            "pending_review",
            f"Top intent confidence {confidence:.1%} is between "
            f"{LLM_ASSIST_THRESHOLD:.1%} and {AUTO_ROUTE_THRESHOLD:.1%}; AI assistance is enabled, "
            "but the prediction remains pending human review.",
            needs_llm=True,
        )

    if confidence >= LLM_ASSIST_THRESHOLD:
        reason = (
            f"Top intent confidence {confidence:.1%} is between {LLM_ASSIST_THRESHOLD:.1%} "
            f"and {AUTO_ROUTE_THRESHOLD:.1%}; external AI assistance is disabled, so the case "
            "is sent for human review."
        )
    else:
        reason = (
            f"Top intent confidence {confidence:.1%} is below the "
            f"{LLM_ASSIST_THRESHOLD:.1%} review threshold; the case is sent for human review."
        )
    return _decision(
        "human",
        "Human Agent Queue",
        "Route to a human specialist for triage and intent verification.",
        "red",
        "pending_review",
        reason,
    )


def _generate_auto_action(intent: str, entities: dict) -> str:
    """Generate an operational handoff action without inventing missing entities."""
    order_ids = entities.get("order_ids", [])
    amounts = entities.get("amounts", [])
    cards = entities.get("card_last4", [])
    products = entities.get("products", [])

    if intent == "order tracking":
        if order_ids:
            return f"Fetch status for Order #{order_ids[0]}"
        return "Initiate order lookup by customer account profile"
    if intent == "refund request":
        amount = f" of {amounts[0]}" if amounts else ""
        product = f" for {products[0]}" if products else ""
        return f"Initiate refund review{amount}{product}"
    if intent == "billing issue":
        amount = f" regarding charge {amounts[0]}" if amounts else ""
        return f"Flag account for billing review{amount}"
    if intent == "account problem":
        return "Open account support ticket and send verification link"
    if intent == "product complaint":
        product = f" on {products[0]}" if products else ""
        return f"Create Customer Care ticket{product} with quality logs"
    if intent == "payment failure":
        return "Initiate transaction trace and payment retry diagnostics"
    if intent == "card issue":
        card = f" ending {cards[0]}" if cards else ""
        return f"Route to Cards Support for card status check{card}"
    return "Route to General Support queue for representative assignment"
