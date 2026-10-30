"""
evaluate.py
Evaluates the intent classifier on a held-out test sample from BANKING77.
Computes accuracy, weighted precision, recall, F1 score, per-class metrics,
and a confusion matrix. Results are saved to data/evaluation_results.json.

Usage: python evaluate.py [--samples 50]
"""

import argparse
import json
import os
import random
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import pipeline

from backend.classifier import INTENT_LABELS

# Prevent deadlocks with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------------------
# Map BANKING77's 77 fine-grained labels -> our 8 routing categories
# ------------------------------------------------------------------
BANKING77_TO_ROUTING = {
    # Order Tracking
    "card_delivery_estimate":        "order tracking",
    "track_complaint":               "order tracking",
    "getting_spare_card":            "order tracking",

    # Refund Request
    "refund_not_done":               "refund request",
    "pending_cash_back":             "refund request",
    "cash_withdrawal_charge":        "refund request",
    "cancel_transfer":               "refund request",

    # Billing Issue
    "wrong_amount_of_cash_received": "billing issue",
    "extra_charge_on_statement":     "billing issue",
    "charged_twice":                 "billing issue",
    "direct_debit_payment_not_recognised": "billing issue",
    "balance_not_updated_after_cheque_or_cash_deposit": "billing issue",

    # Account Problem
    "lost_or_stolen_card":           "account problem",
    "verify_my_identity":            "account problem",
    "card_not_working":              "account problem",
    "activate_my_card":              "account problem",
    "age_limit":                     "account problem",
    "change_pin":                    "account problem",
    "compromised_card":              "account problem",
    "reverted_card_payment?":        "account problem",

    # Product Complaint
    "damaged_card":                  "product complaint",
    "card_acceptance":               "product complaint",
    "declined_card_payment":         "product complaint",
    "declined_transfer":             "product complaint",

    # Payment Failure
    "failed_transfer":               "payment failure",
    "transfer_not_received_by_recipient": "payment failure",
    "transfer_timing":               "payment failure",
    "payment_issue":                 "payment failure",

    # Card Issue
    "get_physical_card":             "card issue",
    "lost_or_stolen_phone":          "card issue",
    "apple_pay_or_google_pay":       "card issue",
    "contactless_not_working":       "card issue",
    "card_swallowed":                "card issue",
}


def map_label(banking77_label: str) -> str:
    return BANKING77_TO_ROUTING.get(banking77_label, "general inquiry")


def run_evaluation(sample_size: int = 50, seed: int = 42):
    print("\n" + "=" * 65)
    print("   SMART SUPPORT ROUTER — MODEL EVALUATION")
    print("=" * 65)

    # 1. Load BANKING77 test split
    print("\n[1/4] Loading BANKING77 test split...")
    dataset = load_dataset("legacy-datasets/banking77")
    test_set = dataset["test"]
    label_names = test_set.features["label"].names

    random.seed(seed)
    indices = random.sample(range(len(test_set)), min(sample_size, len(test_set)))
    samples = [test_set[i] for i in indices]
    print(f"      Selected {len(samples)} held-out samples for evaluation.")

    # 2. Initialize classifier
    print("\n[2/4] Initializing zero-shot classifier pipeline...")
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )
    print("      Classifier ready.")

    # 3. Predict on held-out samples
    print(f"\n[3/4] Running inference on {len(samples)} samples...")
    y_true = []
    y_pred = []
    results = []

    for i, sample in enumerate(samples):
        query = sample["text"]
        true_label = label_names[sample["label"]]
        true_routing = map_label(true_label)

        pred_res = clf(query, candidate_labels=INTENT_LABELS, multi_label=False)
        pred_routing = pred_res["labels"][0]
        confidence = round(pred_res["scores"][0], 4)

        y_true.append(true_routing)
        y_pred.append(pred_routing)

        results.append({
            "query": query,
            "true_intent": true_routing,
            "pred_intent": pred_routing,
            "confidence": confidence,
            "correct": true_routing == pred_routing,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"      Processed {i + 1}/{len(samples)} samples...")

    # 4. Metrics & Confusion Matrix
    print("\n[4/4] Computing classification performance metrics...")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=INTENT_LABELS)
    cm_dict = {
        label: {
            INTENT_LABELS[j]: int(cm[i][j])
            for j in range(len(INTENT_LABELS))
        }
        for i, label in enumerate(INTENT_LABELS)
    }

    report_dict = classification_report(y_true, y_pred, labels=INTENT_LABELS, zero_division=0, output_dict=True)
    report_text = classification_report(y_true, y_pred, labels=INTENT_LABELS, zero_division=0)

    print("\n" + "=" * 65)
    print("   EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"   Evaluated Samples : {len(samples)}")
    print(f"   Overall Accuracy  : {acc * 100:.2f}%")
    print(f"   Weighted Precision: {prec * 100:.2f}%")
    print(f"   Weighted Recall   : {rec * 100:.2f}%")
    print(f"   Weighted F1 Score : {f1 * 100:.2f}%")
    print("=" * 65)

    print("\nPer-Class Classification Report:\n")
    print(report_text)

    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    header = f"{'True \\ Pred':<20}" + "".join(f"{lbl[:10]:>12}" for lbl in INTENT_LABELS)
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(INTENT_LABELS):
        row_str = f"{row_label:<20}" + "".join(f"{cm[i][j]:>12}" for j in range(len(INTENT_LABELS)))
        print(row_str)

    output = {
        "samples_evaluated": len(samples),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "classification_report": report_dict,
        "confusion_matrix": cm_dict,
        "samples": results,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results and confusion matrix saved to 'data/evaluation_results.json'\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Intent Classifier on BANKING77")
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples (default: 50)")
    args = parser.parse_args()

    run_evaluation(sample_size=args.samples)
