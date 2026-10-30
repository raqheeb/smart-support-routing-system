"""
backend/classifier.py
Zero-shot intent classification using facebook/bart-large-mnli.
Falls back gracefully if model isn't downloaded yet.
"""

import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import pipeline
import torch
torch.set_num_threads(2)

# ------------------------------------------------------------------
# These are the top-level routing intents we care about.
# Mapped from BANKING77's 77 fine-grained intents into 8 categories.
# ------------------------------------------------------------------
INTENT_LABELS = [
    "order tracking",
    "refund request",
    "billing issue",
    "account problem",
    "product complaint",
    "payment failure",
    "card issue",
    "general inquiry",
]

INTENT_TO_DEPARTMENT = {
    "order tracking":    "Logistics Team",
    "refund request":    "Billing & Refunds",
    "billing issue":     "Billing & Refunds",
    "account problem":   "Account Support",
    "product complaint": "Customer Care",
    "payment failure":   "Payments Team",
    "card issue":        "Cards Team",
    "general inquiry":   "General Support",
}

_classifier = None


def load_classifier():
    """Lazy-load the zero-shot model (downloads once, cached after)."""
    global _classifier
    if _classifier is None:
        print("[Classifier] Loading facebook/bart-large-mnli ...")
        device = 0 if torch.cuda.is_available() else -1
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
        )
        print("[Classifier] Model ready.")
    return _classifier


def classify_intent(query: str) -> dict:
    """
    Returns:
        {
            "intent": str,
            "confidence": float (0-1),
            "all_scores": dict[label -> score]
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must contain non-whitespace text.")

    clf = load_classifier()
    result = clf(query.strip(), candidate_labels=INTENT_LABELS, multi_label=False)
    if not isinstance(result, dict):
        raise RuntimeError("Classifier returned an invalid response.")
    labels = result.get("labels")
    scores = result.get("scores")
    if not isinstance(labels, list) or not isinstance(scores, list) or len(labels) != len(scores):
        raise RuntimeError("Classifier returned mismatched labels and scores.")
    if not labels or any(label not in INTENT_LABELS for label in labels):
        raise RuntimeError("Classifier returned an unknown or empty intent list.")

    try:
        if any(isinstance(score, bool) for score in scores):
            raise TypeError("Boolean confidence is invalid.")
        numeric_scores = [float(score) for score in scores]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Classifier returned a non-numeric confidence score.") from exc
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in numeric_scores):
        raise RuntimeError("Classifier returned a confidence score outside [0, 1].")

    ranked = sorted(zip(labels, numeric_scores), key=lambda item: item[1], reverse=True)
    top_intent, top_score = ranked[0]
    confidence_margin = top_score - ranked[1][1] if len(ranked) > 1 else 0.0
    all_scores = {label: round(score, 4) for label, score in ranked}

    return {
        "intent":     top_intent,
        "confidence": round(top_score, 4),
        "confidence_margin": round(confidence_margin, 4),
        "all_scores": all_scores,
    }


def get_department(intent: str) -> str:
    return INTENT_TO_DEPARTMENT.get(intent, "General Support")
