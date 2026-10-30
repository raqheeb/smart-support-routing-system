"""
backend/ner.py
Named Entity Recognition using spaCy + custom regex patterns.
Extracts: Order IDs, product names, dates, amounts, locations, card details, contacts, urgency cues.
"""

import re
import spacy

_nlp = None


def load_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[NER] Downloading en_core_web_sm ...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ------------------------------------------------------------------
# Regex patterns for banking/e-commerce specific entities
# ------------------------------------------------------------------
# Fixed: Only match orders with '#' prefix, explicit keywords ('order', 'tracking', 'invoice'),
# or structured alphanumeric codes to prevent plain years/numbers (e.g. 2024, 1500) from false positive matching.
ORDER_ID_PATTERN = re.compile(
    r'(?:#\s*([A-Za-z0-9\-_]{4,16})\b|(?:order|tracking|invoice|receipt|shipment)\s*(?:id|num|number|#)?\s*[:#]?\s*([A-Za-z0-9\-_]{4,16})\b)',
    re.IGNORECASE
)

# Fixed: Non-capturing groups so re.finditer extracts full amount string like '$999', '50.00 USD'
AMOUNT_PATTERN = re.compile(
    r'(?:\$[\d,]+(?:\.\d+)?|\b\d+(?:[\.,]\d{2})?\s*(?:USD|INR|EUR|GBP|CAD|AUD)\b|\b\d+[\.,]\d{2}\b)',
    re.IGNORECASE
)

CARD_LAST4_PATTERN = re.compile(
    r'(?:\b(?:card|ending(?:\s+in)?|last\s*4|account)[\s:#]*(\d{4})\b|(?:\*{3,}|x{3,})(\d{4})\b)',
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)

PHONE_PATTERN = re.compile(
    r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
)

URGENCY_KEYWORDS = [
    "urgent", "asap", "emergency", "immediately", "critical",
    "unauthorized", "fraud", "stolen", "compromised", "hacked"
]


def extract_entities(query: str) -> dict:
    """
    Extracts structured domain entities from customer support query text.
    Returns:
        {
            "order_ids":  list[str],
            "products":   list[str],
            "dates":      list[str],
            "amounts":    list[str],
            "locations":  list[str],
            "card_last4": list[str],
            "persons":    list[str],
            "emails":     list[str],
            "phones":     list[str],
            "urgency":    list[str],
        }
    """
    if not query or not query.strip():
        return {}

    clean_query = query.strip()
    nlp = load_nlp()
    doc = nlp(clean_query)

    # Order IDs
    order_ids = []
    for match in ORDER_ID_PATTERN.finditer(clean_query):
        oid = match.group(1) or match.group(2)
        if oid and not oid.isdigit() or (oid.isdigit() and len(oid) >= 5 and oid not in ["2024", "2025", "2026", "2027"]):
            order_ids.append(oid)

    # Amounts
    amounts = [m.group(0).strip() for m in AMOUNT_PATTERN.finditer(clean_query)]

    # Card Last 4
    cards = []
    for match in CARD_LAST4_PATTERN.finditer(clean_query):
        card_num = match.group(1) or match.group(2)
        if card_num:
            cards.append(card_num)

    # Emails & Phones
    emails = EMAIL_PATTERN.findall(clean_query)
    phones = PHONE_PATTERN.findall(clean_query)

    # Urgency cues
    query_lower = clean_query.lower()
    urgency_cues = [word for word in URGENCY_KEYWORDS if re.search(rf"\b{word}\b", query_lower)]

    entities = {
        "order_ids":  order_ids,
        "products":   [],
        "dates":      [],
        "amounts":    amounts,
        "locations":  [],
        "card_last4": cards,
        "persons":    [],
        "emails":     emails,
        "phones":     phones,
        "urgency":    urgency_cues,
    }

    for ent in doc.ents:
        cleaned_text = ent.text.strip()
        if ent.label_ == "PRODUCT" and cleaned_text not in ["USD", "EUR", "GBP", "INR", "CAD", "AUD"]:
            entities["products"].append(cleaned_text)
        elif ent.label_ in ("DATE", "TIME"):
            # Avoid single numbers matching card last 4 digits
            if cleaned_text not in cards:
                entities["dates"].append(cleaned_text)
        elif ent.label_ in ("GPE", "LOC"):
            entities["locations"].append(cleaned_text)
        elif ent.label_ == "PERSON":
            if not cleaned_text.isdigit() and cleaned_text not in order_ids:
                entities["persons"].append(cleaned_text)
        elif ent.label_ == "MONEY":
            entities["amounts"].append(cleaned_text)

    # De-duplicate while preserving non-empty lists
    result = {}
    for key, values in entities.items():
        unique_vals = sorted(list({str(v).strip() for v in values if str(v).strip()}))
        if unique_vals:
            result[key] = unique_vals

    return result
