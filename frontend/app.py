"""
frontend/app.py
Interactive Gradio Dashboard for Smart Support Routing System.
Includes Message Analyzer with Explainability, Human Review Queue,
Real-time Analytics, and Pipeline Architecture Documentation.

Run with: python frontend/app.py
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Tuple, List
import gradio as gr
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    API_PORT,
    UI_PORT,
    AUTO_ROUTE_THRESHOLD,
    AUTO_ROUTE_MIN_MARGIN,
    LLM_ASSIST_THRESHOLD,
)
from backend.classifier import INTENT_LABELS, get_department

API_BASE = f"http://localhost:{API_PORT}"

SAMPLE_QUERIES = [
    "I want a refund for my damaged phone",
    "Where is my order #98765 for iPhone 15?",
    "My card was charged twice this month for $150",
    "I cannot log into my account, forgot my password",
    "This product is absolutely useless, stopped working in 2 days",
    "My payment failed but money was deducted from my account",
    "I need to update my billing address",
    "Urgent: my credit card ending in 4321 was stolen and unauthorized charge of $500 appeared!",
]


# ------------------------------------------------------------------
# Helper: Call /analyze endpoint
# ------------------------------------------------------------------
def analyze_query(query: str):
    if not query or not query.strip():
        empty_html = "<div style='color: #ffa726; padding: 12px;'>⚠️ Please enter a customer query.</div>"
        return empty_html, "", "", "", "", ""

    clean_query = query.strip()
    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{API_BASE}/analyze", json={"query": clean_query})
            if resp.status_code != 200:
                err_msg = resp.json().get("detail", resp.text)
                return f"<div style='color: #ef5350;'>❌ API Error: {err_msg}</div>", "", "", "", "", ""
            data = resp.json()
    except httpx.ConnectError:
        return (
            "<div style='color: #ef5350; padding: 16px; border: 1px solid #ef5350; border-radius: 8px;'>"
            "❌ <b>Cannot connect to backend API server.</b><br/>"
            f"Please ensure the FastAPI service is active on port {API_PORT}:<br/>"
            "<code>uvicorn backend.main:app --port 8000 --reload</code>"
            "</div>",
            "", "", "", "", ""
        )
    except Exception as e:
        return f"<div style='color: #ef5350;'>❌ Error: {str(e)}</div>", "", "", "", "", ""

    # Parse response fields
    intent         = data.get("intent", "Unknown").title()
    confidence     = data.get("confidence", 0.0) * 100
    confidence_margin = data.get("confidence_margin")
    margin_text = f"{confidence_margin * 100:.1f}%" if confidence_margin is not None else "N/A"
    badge          = data.get("badge", "red")
    routing        = data.get("routing", "human").upper()
    department     = data.get("department", "General Support")
    action         = data.get("action", "")
    routing_reason = data.get("routing_reason", "Standard routing rule applied.")
    log_id         = data.get("log_id")
    entities       = data.get("entities", {})
    llm            = data.get("llm_analysis")
    all_scores     = data.get("all_scores", {})

    badge_configs = {
        "green":  ("#00c853", "🟢 AUTO-ROUTED"),
        "yellow": ("#ffd600", "🟡 AI-ASSISTED"),
        "red":    ("#ff1744", "🔴 HUMAN REVIEW REQUIRED"),
    }
    color, badge_label = badge_configs.get(badge, ("#888888", "⚪ UNKNOWN"))

    # 1. Routing Decision Card with Explainability
    routing_html = f"""
    <div style="background: {color}15; border: 2px solid {color}; border-radius: 12px; padding: 20px; margin: 8px 0; font-family: 'Segoe UI', system-ui, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 18px; font-weight: 800; color: {color}; letter-spacing: 1px;">{badge_label}</span>
            <span style="font-size: 12px; background: {color}33; color: {color}; padding: 3px 10px; border-radius: 12px; font-weight: 700;">{routing}</span>
        </div>
        <div style="font-size: 16px; color: #ffffff; margin-top: 10px;">
            Destination Team: <b style="color: #64b5f6;">{department}</b>
        </div>
        <div style="font-size: 14px; color: #e0e0e0; margin-top: 6px;">
            <b>Action Plan:</b> {action}
        </div>
        <div style="margin-top: 12px; padding-top: 10px; border-top: 1px dashed {color}55; font-size: 12px; color: #b0bec5; line-height: 1.5;">
            <b>💡 Decision Rationale:</b> {routing_reason}
        </div>
    </div>
    """

    # 2. Intent Card
    bar_width = max(2, min(100, int(confidence)))
    scores_html = "".join(
        f"<div style='display:flex; justify-content:space-between; font-size:11px; margin:2px 0; color:#aaa;'>"
        f"<span>{lbl.title()}</span><span>{score*100:.1f}%</span></div>"
        for lbl, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    )

    intent_html = f"""
    <div style="background: #1a1a2e; border: 1px solid #33334d; border-radius: 12px; padding: 18px; font-family: 'Segoe UI', system-ui, sans-serif;">
        <div style="font-size: 11px; color: #90caf9; letter-spacing: 2px; text-transform: uppercase; font-weight: 700;">
            Detected Intent
        </div>
        <div style="font-size: 24px; font-weight: 800; color: #ffffff; margin: 6px 0;">
            {intent}
        </div>
        <div style="font-size: 13px; color: #cfd8dc; margin-bottom: 6px;">
            Classifier top score: <b style="color: {color}; font-size: 15px;">{confidence:.1f}%</b>
            <span style="margin-left: 12px;">Top-two margin: <b>{margin_text}</b></span>
        </div>
        <div style="background: #2a2a40; border-radius: 6px; height: 10px; overflow: hidden; margin-bottom: 12px;">
            <div style="background: {color}; width: {bar_width}%; height: 10px; border-radius: 6px; transition: width 0.4s ease;"></div>
        </div>
        <div style="border-top: 1px solid #2a2a40; padding-top: 8px;">
            <div style="font-size: 10px; color: #78909c; text-transform: uppercase; margin-bottom: 4px; font-weight: 600;">Top Candidates:</div>
            {scores_html}
        </div>
    </div>
    """

    # 3. Entities Card
    if entities:
        icons = {
            "order_ids": "📦 Order ID",
            "products": "🛍️ Product",
            "amounts": "💰 Amount",
            "dates": "📅 Date/Time",
            "locations": "📍 Location",
            "card_last4": "💳 Card / Acc",
            "persons": "👤 Contact",
            "emails": "✉️ Email",
            "phones": "📞 Phone",
            "urgency": "🚨 Urgency Flag",
        }
        rows = ""
        for key, values in entities.items():
            if values:
                tag_label = icons.get(key, key.replace("_", " ").title())
                badge_style = "background: #d32f2f33; color: #ff8a80; border: 1px solid #d32f2f;" if key == "urgency" else "background: #2a2a44; color: #81d4fa;"
                badges = "".join(f"<span style='{badge_style} padding: 3px 8px; border-radius: 6px; font-size: 12px; margin-right: 4px; display: inline-block;'>{v}</span>" for v in values)
                rows += f"""
                <div style="margin: 8px 0; font-size: 13px;">
                    <span style="color: #90a4ae; font-weight: 600; min-width: 110px; display: inline-block;">{tag_label}:</span>
                    <span>{badges}</span>
                </div>
                """
        entities_html = f"""
        <div style="background: #1a1a2e; border: 1px solid #33334d; border-radius: 12px; padding: 18px; font-family: 'Segoe UI', system-ui, sans-serif;">
            <div style="font-size: 11px; color: #81d4fa; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">
                Extracted Entities & Metadata
            </div>
            {rows}
        </div>
        """
    else:
        entities_html = """
        <div style="background: #1a1a2e; border: 1px solid #33334d; border-radius: 12px; padding: 18px; color: #78909c; font-size: 13px;">
            ℹ️ No specific structured entities detected in message text.
        </div>
        """

    # 4. LLM Reasoning Card
    if llm:
        tone_colors = {
            "negative": "#ff5252", "frustrated": "#ff6d00",
            "urgent": "#ffd740", "positive": "#69f0ae", "neutral": "#90caf9"
        }
        tone = llm.get("tone", "neutral")
        tone_color = tone_colors.get(tone, "#90caf9")
        priority = llm.get("priority", "medium").upper()
        source = llm.get("source", "claude_api")
        source_note = (
            "<div style='font-size: 11px; color: #ffcc80; margin-top: 8px;'>"
            "Local heuristic suggestion; its confidence is not calibrated."
            "</div>"
            if source != "claude_api" else ""
        )

        llm_html = f"""
        <div style="background: #1a1a2e; border: 1px solid #ffd60088; border-radius: 12px; padding: 18px; font-family: 'Segoe UI', system-ui, sans-serif;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
                <span style="font-size: 11px; color: #ffd600; letter-spacing: 2px; text-transform: uppercase; font-weight: 700;">
                    ✦ AI Reasoning Analysis ({source})
                </span>
                <span style="font-size: 11px; background: #ffd60022; color: #ffd600; padding: 2px 8px; border-radius: 8px; font-weight: bold;">
                    PRIORITY: {priority}
                </span>
            </div>
            <div style="font-size: 13px; color: #ffffff; margin: 6px 0;">
                <b>Core Issue:</b> {llm.get("key_issue", "N/A")}
            </div>
            <div style="display: flex; gap: 12px; margin-top: 10px; flex-wrap: wrap;">
                <div style="background: #0d0d1a; border-radius: 6px; padding: 6px 12px;">
                    <span style="font-size: 10px; color: #78909c;">DETECTED TONE:</span>
                    <span style="color: {tone_color}; font-weight: 700; font-size: 12px; margin-left: 6px;">{tone.upper()}</span>
                </div>
                <div style="background: #0d0d1a; border-radius: 6px; padding: 6px 12px; flex: 1;">
                    <span style="font-size: 10px; color: #78909c;">SUGGESTED AGENT STEP:</span>
                    <span style="color: #e0e0e0; font-size: 12px; margin-left: 6px;">{llm.get("suggested_action", "N/A")}</span>
                </div>
            </div>
            {source_note}
        </div>
        """
    else:
        llm_html = ""

    return routing_html, intent_html, entities_html, llm_html, str(log_id), clean_query


# ------------------------------------------------------------------
# Helper: Feedback Submission
# ------------------------------------------------------------------
def submit_feedback(log_id: str, feedback_type: str, query: str):
    if not log_id or log_id == "None":
        return "⚠️ No active prediction to give feedback on."
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{API_BASE}/feedback",
                json={"log_id": int(log_id), "feedback": feedback_type}
            )
            resp.raise_for_status()
        icon = "✅" if feedback_type == "correct" else "❌"
        return f"{icon} Feedback '{feedback_type}' submitted for Log #{log_id}!"
    except Exception as e:
        return f"❌ Error recording feedback: {str(e)}"


# ------------------------------------------------------------------
# Helper: Load Human Review Queue
# ------------------------------------------------------------------
def load_review_queue(filter_status: str = "pending_review"):
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{API_BASE}/review/queue", params={"status_filter": filter_status, "limit": 40})
            resp.raise_for_status()
            tickets = resp.json()
    except Exception as e:
        return [], gr.update(choices=[], value=None), f"⚠️ Review queue unavailable: {e}"

    if not tickets:
        return [], gr.update(choices=[], value=None), "No tickets match this status filter."

    table_data = []
    ticket_choices = []
    for t in tickets:
        conf_str = f"{t['confidence']*100:.1f}%" if t.get("confidence") is not None else "N/A"
        table_data.append([
            str(t["id"]),
            t.get("query", "")[:50] + ("..." if len(t.get("query", "")) > 50 else ""),
            t.get("predicted_intent", "Unknown"),
            conf_str,
            t.get("routing", "human").upper(),
            t.get("status", "pending_review"),
        ])
        ticket_choices.append((f"#{t['id']} - {t.get('predicted_intent', '')}", str(t["id"])))

    return (
        table_data,
        gr.update(
            choices=ticket_choices,
            value=ticket_choices[0][1] if ticket_choices else None,
        ),
        f"Showing {len(tickets)} ticket(s). Select a ticket to review.",
    )


def load_review_detail(ticket_selection: str):
    """Load complete ticket context and enable editing only for pending cases."""
    if not ticket_selection:
        return (
            "",
            {},
            gr.update(value=None, interactive=False),
            gr.update(value="reviewed", interactive=False),
            "",
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
        )

    try:
        log_id = int(ticket_selection)
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE}/review/{log_id}")
            response.raise_for_status()
            ticket = response.json()
    except (ValueError, httpx.HTTPError) as exc:
        return (
            "",
            {"error": str(exc)},
            gr.update(value=None, interactive=False),
            gr.update(value="reviewed", interactive=False),
            "",
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
        )

    reviewable = ticket.get("reviewable", False)
    predicted_intent = ticket.get("predicted_intent")
    intent_value = predicted_intent if predicted_intent in INTENT_LABELS else None
    details = {
        "ticket_id": ticket["id"],
        "predicted_intent": predicted_intent,
        "confidence": ticket.get("confidence"),
        "routing": ticket.get("routing"),
        "assigned_team": ticket.get("department"),
        "verified_department": ticket.get("reviewed_department"),
        "routing_reason": ticket.get("routing_reason"),
        "entities": ticket.get("entities", {}),
        "status": ticket.get("status"),
        "previous_feedback": ticket.get("feedback"),
        "reviewed_intent": ticket.get("reviewed_intent"),
        "review_notes": ticket.get("review_notes"),
        "submitted_at": ticket.get("timestamp"),
        "reviewed_at": ticket.get("reviewed_at"),
        "reviewable": reviewable,
    }
    return (
        ticket.get("query", ""),
        details,
        gr.update(value=intent_value, interactive=reviewable),
        gr.update(
            value="reviewed" if reviewable else "resolved",
            interactive=reviewable,
        ),
        (
            f"Destination team for verified intent: {get_department(intent_value)}"
            if intent_value else ""
        ),
        gr.update(value=ticket.get("review_notes") or "", interactive=reviewable),
        gr.update(interactive=reviewable),
    )


def preview_review_department(reviewed_intent: str):
    if reviewed_intent not in INTENT_LABELS:
        return ""
    return f"Destination team for verified intent: {get_department(reviewed_intent)}"


# ------------------------------------------------------------------
# Helper: Submit Human Review / Correction
# ------------------------------------------------------------------
def submit_review_action(ticket_selection: str, corrected_intent: str, notes: str, status_decision: str):
    if not ticket_selection:
        return "⚠️ Please select a ticket from the queue first."
    try:
        log_id = int(ticket_selection)
    except (TypeError, ValueError):
        return "⚠️ Invalid ticket ID format."

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{API_BASE}/review/submit",
                json={
                    "log_id": log_id,
                    "reviewed_intent": corrected_intent,
                    "notes": notes,
                    "status": status_decision,
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return (
                f"✅ Ticket #{log_id} reviewed as '{data['reviewed_intent']}' "
                f"for {data['reviewed_department']} "
                f"(Status: {data['new_status']}; feedback: {data['feedback']})."
            )
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        return f"❌ Review was not saved: {detail}"
    except httpx.HTTPError as exc:
        return f"❌ Review service unavailable: {exc}"


# ------------------------------------------------------------------
# Helper: Load Analytics Tab
# ------------------------------------------------------------------
def load_analytics_dashboard():
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{API_BASE}/analytics")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return (
            f"<div style='color:#ef5350;'>❌ Analytics unavailable: {str(e)}</div>",
            "", "", ""
        )

    total        = data.get("total", 0)
    avg_conf     = data.get("avg_confidence", 0.0) * 100
    routing      = data.get("routing_breakdown", {})
    intents      = data.get("intents", {})
    conf_dist    = data.get("confidence_distribution", {})
    correction   = data.get("correction_rate", 0.0) * 100
    corrected_cnt= data.get("corrected_tickets", 0)
    human_cnt    = data.get("human_review_count", 0)

    # 1. KPI Cards
    summary_html = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; font-family: 'Segoe UI', system-ui, sans-serif;">
        <div style="background: #1a1a2e; border: 1px solid #33334d; border-radius: 10px; padding: 18px; text-align: center;">
            <div style="font-size: 32px; font-weight: 800; color: #00e676;">{total}</div>
            <div style="font-size: 11px; color: #90a4ae; letter-spacing: 1px; text-transform: uppercase;">Total Messages</div>
        </div>
        <div style="background: #1a1a2e; border: 1px solid #33334d; border-radius: 10px; padding: 18px; text-align: center;">
            <div style="font-size: 32px; font-weight: 800; color: #ffd600;">{avg_conf:.1f}%</div>
            <div style="font-size: 11px; color: #90a4ae; letter-spacing: 1px; text-transform: uppercase;">Avg Confidence</div>
        </div>
        <div style="background: #1a1a2e; border: 1px solid #33334d; border-radius: 10px; padding: 18px; text-align: center;">
            <div style="font-size: 32px; font-weight: 800; color: #ff5252;">{human_cnt}</div>
            <div style="font-size: 11px; color: #90a4ae; letter-spacing: 1px; text-transform: uppercase;">Review Cases</div>
        </div>
        <div style="background: #1a1a2e; border: 1px solid #33334d; border-radius: 10px; padding: 18px; text-align: center;">
            <div style="font-size: 32px; font-weight: 800; color: #40c4ff;">{correction:.1f}%</div>
            <div style="font-size: 11px; color: #90a4ae; letter-spacing: 1px; text-transform: uppercase;">Correction Rate ({corrected_cnt})</div>
        </div>
    </div>
    """

    # 2. Routing Breakdown Bars
    routing_colors = {"auto": "#00e676", "llm": "#ffd600", "human": "#ff5252"}
    routing_rows = ""
    for r_type in ["auto", "llm", "human"]:
        count = routing.get(r_type, 0)
        pct = int((count / total * 100) if total > 0 else 0)
        rc = routing_colors.get(r_type, "#888")
        routing_rows += f"""
        <div style="margin: 10px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 4px; font-size: 13px;">
                <span style="color: #cfd8dc; text-transform: uppercase; font-weight: 600;">{r_type}</span>
                <span style="color: {rc}; font-weight: bold;">{count} ({pct}%)</span>
            </div>
            <div style="background: #2a2a40; border-radius: 4px; height: 8px;">
                <div style="background: {rc}; width: {pct}%; height: 8px; border-radius: 4px;"></div>
            </div>
        </div>
        """
    routing_html = f"""
    <div style="background: #1a1a2e; border: 1px solid #33334d; border-radius: 12px; padding: 20px; font-family: 'Segoe UI', system-ui, sans-serif;">
        <div style="font-size: 12px; color: #90caf9; letter-spacing: 2px; text-transform: uppercase; font-weight: bold; margin-bottom: 12px;">
            Routing Path Distribution
        </div>
        {routing_rows}
    </div>
    """

    # 3. Confidence Distribution Buckets
    dist_rows = ""
    for b_label, count in conf_dist.items():
        pct = int((count / total * 100) if total > 0 else 0)
        dist_rows += f"""
        <div style="margin: 10px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 4px; font-size: 13px;">
                <span style="color: #cfd8dc; font-weight: 600;">Confidence {b_label}</span>
                <span style="color: #80d8ff; font-weight: bold;">{count} ({pct}%)</span>
            </div>
            <div style="background: #2a2a40; border-radius: 4px; height: 8px;">
                <div style="background: #40c4ff; width: {pct}%; height: 8px; border-radius: 4px;"></div>
            </div>
        </div>
        """
    dist_html = f"""
    <div style="background: #1a1a2e; border: 1px solid #33334d; border-radius: 12px; padding: 20px; font-family: 'Segoe UI', system-ui, sans-serif;">
        <div style="font-size: 12px; color: #80d8ff; letter-spacing: 2px; text-transform: uppercase; font-weight: bold; margin-bottom: 12px;">
            Confidence Score Brackets
        </div>
        {dist_rows}
    </div>
    """

    # 4. Top Intents
    intent_rows = ""
    sorted_intents = sorted(intents.items(), key=lambda x: x[1], reverse=True)
    for iname, icnt in sorted_intents[:6]:
        pct = int((icnt / total * 100) if total > 0 else 0)
        intent_rows += f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding: 8px 0; border-bottom: 1px solid #2a2a40; font-size: 13px;">
            <span style="color: #e0e0e0;">{iname.title()}</span>
            <div style="display:flex; align-items:center; gap: 10px;">
                <div style="background: #2a2a40; border-radius: 3px; height: 6px; width: 80px;">
                    <div style="background: #ab47bc; width: {pct}%; height: 6px; border-radius: 3px;"></div>
                </div>
                <span style="color: #ba68c8; font-weight: 700; min-width: 30px; text-align: right;">{icnt}</span>
            </div>
        </div>
        """
    intents_html = f"""
    <div style="background: #1a1a2e; border: 1px solid #33334d; border-radius: 12px; padding: 20px; font-family: 'Segoe UI', system-ui, sans-serif;">
        <div style="font-size: 12px; color: #ba68c8; letter-spacing: 2px; text-transform: uppercase; font-weight: bold; margin-bottom: 12px;">
            Top Intent Categories
        </div>
        {intent_rows if intent_rows else '<div style="color:#777">No queries recorded yet.</div>'}
    </div>
    """

    return summary_html, routing_html, dist_html, intents_html


# ------------------------------------------------------------------
# Build Gradio UI Application
# ------------------------------------------------------------------
custom_css = """
body, .gradio-container {
    background: #0d0d1a !important;
    color: #e0e0e0 !important;
}
.gr-button-primary {
    background: #00c853 !important;
    border: none !important;
    color: #000 !important;
    font-weight: 800 !important;
}
.gr-button-secondary {
    background: #1e1e30 !important;
    border: 1px solid #444 !important;
    color: #e0e0e0 !important;
}
"""

with gr.Blocks(title="Smart Support Routing System") as demo:

    # State stores
    current_log_id = gr.State(value=None)
    current_query = gr.State(value="")

    # Header
    gr.HTML(f"""
    <div style="text-align: center; padding: 26px 0 16px; border-bottom: 1px solid #1e1e38; margin-bottom: 20px;">
        <div style="font-size: 11px; letter-spacing: 5px; color: #78909c; text-transform: uppercase;">
            Enterprise Multi-Tier AI Routing Platform
        </div>
        <div style="font-size: 34px; font-weight: 800; color: #ffffff; margin: 4px 0;">
            SMART SUPPORT <span style="color: #00e676;">ROUTER</span>
        </div>
        <div style="font-size: 13px; color: #90a4ae;">
            Bart-Large Zero-Shot Classification · spaCy NER · Multi-Tier Routing · Human Review & Feedback Loop
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Real-time Message Analyzer ──────────────────────────
        with gr.Tab("⚡ Support Routing Analyzer"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        placeholder="Enter incoming customer inquiry (e.g., 'I want a refund for my damaged phone')...",
                        label="Customer Message / Query",
                        lines=4,
                    )
                    sample_dropdown = gr.Dropdown(
                        choices=SAMPLE_QUERIES,
                        label="💡 Or load a sample customer message:",
                        value=None,
                    )
                    analyze_btn = gr.Button("⚡ ANALYZE & ROUTE", variant="primary", size="lg")

                with gr.Column(scale=3):
                    routing_output  = gr.HTML(label="Routing Decision")
                    intent_output   = gr.HTML(label="Intent & Confidence")
                    entities_output = gr.HTML(label="Entities")
                    llm_output      = gr.HTML(label="AI Reasoning")

            # Feedback Strip
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("<div style='font-size:12px; color:#90a4ae; font-weight:600; padding-top:8px;'>WAS THIS ROUTING DECISION ACCURATE?</div>")
                with gr.Column(scale=3):
                    with gr.Row():
                        correct_btn = gr.Button("✅ Mark Correct", variant="secondary")
                        incorrect_btn = gr.Button("❌ Mark Incorrect", variant="secondary")
                    feedback_msg = gr.Textbox(label="", interactive=False, show_label=False)

            # Wire up analyze interactions
            sample_dropdown.change(
                fn=lambda s: s or "",
                inputs=sample_dropdown,
                outputs=query_input,
            )

            analyze_btn.click(
                fn=analyze_query,
                inputs=query_input,
                outputs=[
                    routing_output,
                    intent_output,
                    entities_output,
                    llm_output,
                    current_log_id,
                    current_query,
                ],
            )

            correct_btn.click(
                fn=lambda lid, q: submit_feedback(lid, "correct", q),
                inputs=[current_log_id, current_query],
                outputs=feedback_msg,
            )
            incorrect_btn.click(
                fn=lambda lid, q: submit_feedback(lid, "incorrect", q),
                inputs=[current_log_id, current_query],
                outputs=feedback_msg,
            )

        # ── Tab 2: Human Review Queue ────────────────────────────────
        with gr.Tab("🧑‍💼 Human Review Queue"):
            gr.Markdown("### Triage & Validate Low-Confidence or Flagged Support Cases")
            with gr.Row():
                status_filter = gr.Dropdown(
                    choices=["pending_review", "reviewed", "resolved", "auto_routed", "all"],
                    value="pending_review",
                    label="Queue Filter Status",
                )
                refresh_queue_btn = gr.Button("🔄 Refresh Queue", variant="secondary")

            queue_status = gr.Markdown()
            queue_table = gr.Dataframe(
                headers=["ID", "Query Snippet", "Predicted Intent", "Confidence", "Routing", "Status"],
                datatype=["str", "str", "str", "str", "str", "str"],
                label="Tickets Requiring Human Action",
                interactive=False,
            )

            gr.Markdown("#### Review & Correct Ticket")
            with gr.Row():
                ticket_selector = gr.Dropdown(
                    choices=[],
                    label="Select Ticket ID to Review",
                    interactive=True,
                )
            selected_query = gr.Textbox(
                label="Customer Message",
                lines=4,
                interactive=False,
            )
            selected_details = gr.JSON(label="Prediction and Review Context", value={})
            gr.Markdown("#### Record Agent Decision")
            with gr.Row():
                corrected_intent_dropdown = gr.Dropdown(
                    choices=INTENT_LABELS,
                    value=None,
                    label="Verified / Corrected Intent",
                    interactive=False,
                )
                action_status_dropdown = gr.Dropdown(
                    choices=["reviewed", "resolved"],
                    value="reviewed",
                    label="Update Ticket Status To",
                    interactive=False,
                )

            corrected_department_preview = gr.Textbox(
                label="Destination for Verified Intent",
                interactive=False,
            )
            reviewer_notes = gr.Textbox(
                label="Reviewer Notes / Rationale",
                placeholder="Optional notes regarding the correction or triage decision...",
                lines=2,
                interactive=False,
            )
            submit_review_btn = gr.Button("💾 SUBMIT HUMAN REVIEW", variant="primary", interactive=False)
            review_feedback_msg = gr.Textbox(label="", interactive=False, show_label=False)

            ticket_selector.change(
                fn=load_review_detail,
                inputs=[ticket_selector],
                outputs=[
                    selected_query,
                    selected_details,
                    corrected_intent_dropdown,
                    action_status_dropdown,
                    corrected_department_preview,
                    reviewer_notes,
                    submit_review_btn,
                ],
            )

            refresh_queue_btn.click(
                fn=load_review_queue,
                inputs=[status_filter],
                outputs=[queue_table, ticket_selector, queue_status],
            ).then(
                fn=load_review_detail,
                inputs=[ticket_selector],
                outputs=[
                    selected_query,
                    selected_details,
                    corrected_intent_dropdown,
                    action_status_dropdown,
                    corrected_department_preview,
                    reviewer_notes,
                    submit_review_btn,
                ],
            )
            corrected_intent_dropdown.change(
                fn=preview_review_department,
                inputs=[corrected_intent_dropdown],
                outputs=[corrected_department_preview],
            )

            submit_review_btn.click(
                fn=submit_review_action,
                inputs=[ticket_selector, corrected_intent_dropdown, reviewer_notes, action_status_dropdown],
                outputs=review_feedback_msg,
            ).then(
                fn=load_review_queue,
                inputs=[status_filter],
                outputs=[queue_table, ticket_selector, queue_status],
            ).then(
                fn=load_review_detail,
                inputs=[ticket_selector],
                outputs=[
                    selected_query,
                    selected_details,
                    corrected_intent_dropdown,
                    action_status_dropdown,
                    corrected_department_preview,
                    reviewer_notes,
                    submit_review_btn,
                ],
            )

        # ── Tab 3: Analytics & Audit ─────────────────────────────────
        with gr.Tab("📊 Operational Analytics"):
            refresh_analytics_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
            summary_cards_output = gr.HTML()

            with gr.Row():
                routing_dist_output = gr.HTML()
                conf_dist_output    = gr.HTML()

            with gr.Row():
                top_intents_output  = gr.HTML()

            refresh_analytics_btn.click(
                fn=load_analytics_dashboard,
                inputs=[],
                outputs=[summary_cards_output, routing_dist_output, conf_dist_output, top_intents_output],
            )

        # ── Tab 4: System Architecture ───────────────────────────────
        with gr.Tab("ℹ️ Architecture & Thresholds"):
            gr.HTML(f"""
            <div style="font-family: 'Segoe UI', system-ui, sans-serif; max-width: 800px; margin: 20px auto; line-height: 1.7;">
                <h3 style="color: #64b5f6; margin-bottom: 8px;">Multi-Tier Routing Logic</h3>
                <p style="color: #b0bec5; font-size: 14px;">
                    Smart Support Router classifies inquiries using zero-shot inference (<code>facebook/bart-large-mnli</code>)
                    coupled with custom spaCy NER patterns and rule-based urgency detection:
                </p>

                <div style="display: flex; flex-direction: column; gap: 14px; margin: 20px 0;">
                    <div style="background: #1a1a2e; border-left: 4px solid #00e676; padding: 14px 18px; border-radius: 0 8px 8px 0;">
                        <b style="color: #00e676;">🟢 Tier 1: Automated Routing (Confidence ≥ {int(AUTO_ROUTE_THRESHOLD*100)}%, margin ≥ {int(AUTO_ROUTE_MIN_MARGIN*100)}%)</b>
                        <div style="color: #cfd8dc; font-size: 13px; margin-top: 4px;">
                            Automatic routing requires both the top classifier score and top-two score margin to meet their thresholds.
                            Missing entities are not inferred; the assigned team receives a lookup action instead.
                        </div>
                    </div>

                    <div style="background: #1a1a2e; border-left: 4px solid #ffd600; padding: 14px 18px; border-radius: 0 8px 8px 0;">
                        <b style="color: #ffd600;">🟡 Tier 2: AI Reasoning Assist ({int(LLM_ASSIST_THRESHOLD*100)}% - {int(AUTO_ROUTE_THRESHOLD*100)}% Confidence)</b>
                        <div style="color: #cfd8dc; font-size: 13px; margin-top: 4px;">
                            Medium-confidence cases always remain pending human review. Claude suggestions are optional and only sent
                            when explicitly enabled with a configured API key; otherwise no external request is made.
                        </div>
                    </div>

                    <div style="background: #1a1a2e; border-left: 4px solid #ff1744; padding: 14px 18px; border-radius: 0 8px 8px 0;">
                        <b style="color: #ff1744;">🔴 Tier 3: Human Review (&lt; {int(LLM_ASSIST_THRESHOLD*100)}%, ambiguous margin, short query, or security cue)</b>
                        <div style="color: #cfd8dc; font-size: 13px; margin-top: 4px;">
                            Low-confidence messages, ambiguous one-word queries, or messages containing fraud/security alerts
                            route safely into the Human Review Queue for agent verification and continuous model feedback.
                        </div>
                    </div>
                </div>
            </div>
            """)


if __name__ == "__main__":
    print(f"\n[*] Launching Smart Support Routing Dashboard on http://localhost:{UI_PORT}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=UI_PORT,
        share=False,
        css=custom_css,
    )
