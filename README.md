# 🎯 Smart Support Routing System

AI-powered customer intent recognition and routing pipeline.

## Tech Stack
- **Frontend**: Gradio dashboard
- **Backend**: FastAPI + Uvicorn
- **ML/NLP**: facebook/bart-large-mnli (zero-shot), spaCy
- **LLM**: Claude (Anthropic API)
- **Database**: SQLite via SQLAlchemy

---

## ⚙️ Setup

### 1. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. (Optional) Configure Claude
Anthropic requests are disabled by default, even if a key is present. To opt in (may incur API charges), configure both values in `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_ENABLED=true
```
When enabled, Claude is called only for medium-confidence predictions. Its result is a suggestion; the classifier's intent and support-team assignment remain unchanged, and the ticket still requires human review. Calls that fail or return invalid data use the local heuristic suggestion. No confidence from that heuristic is treated as calibrated.

### 4. Test the pipeline
```bash
python test_pipeline.py
```
The pipeline test uses the local heuristic by default. It can call the Anthropic API only when both `ANTHROPIC_ENABLED=true` and a key are configured; that may incur API charges. Do not enable it unless that usage is approved. To force the local fallback in PowerShell:
```powershell
$env:ANTHROPIC_API_KEY = ""
python test_pipeline.py
```

### 5. Start the API server
```bash
uvicorn backend.main:app --reload
```

API docs available at: http://localhost:8000/docs

### 6. Start the frontend
In a separate terminal, run:
```bash
python frontend/app.py
```
The Gradio dashboard is available at: http://localhost:7860

---

## 📁 Project Structure
```
smart_support/
├── backend/
│   ├── main.py         # FastAPI app and API endpoints
│   ├── classifier.py   # Zero-shot intent classifier
│   ├── ner.py          # spaCy entity extraction
│   ├── decision.py     # Routing logic
│   └── llm.py          # Claude API handler
│   ├── config.py       # Environment-based configuration
├── database/
│   └── db.py           # SQLite models & session
├── frontend/
│   └── app.py          # Gradio dashboard
├── evaluate.py         # Held-out BANKING77 evaluation
├── test_pipeline.py    # Full pipeline test
├── test_database_migrations.py
├── .env                # API keys (never commit this)
└── requirements.txt
```

---

## 🔁 Pipeline Flow
```
User Query
    ↓
Intent Classifier (bart-large-mnli)
    ↓
NER (spaCy)
    ↓
Decision Engine
    ├── confidence >= 80% and top-two margin >= 10% → Auto Route (🟢)
    ├── confidence 65-80% → Human Review; optional Claude suggestion (🟡)
    └── confidence < 65%  → Human Handoff (🔴)
    ↓
Log to SQLite
    ↓
Feedback → logged for future retraining
```
Confidence is the zero-shot classifier's top candidate score (not a calibrated probability of correctness); the margin is the difference between the top two candidate scores. The default thresholds are conservative routing rules, not empirically tuned accuracy claims. Automatic routing requires both configured thresholds. A high score with a narrow or unavailable margin, security-related urgency cues, and messages of two words or fewer go directly to human review. Claude is disabled unless both `ANTHROPIC_ENABLED=true` and an API key are configured. Its suggestions never change the predicted intent or destination and never bypass human review. If Claude fails or returns invalid data, the local heuristic provides a clearly marked, uncalibrated suggestion only.

---

## 🌐 API Endpoints
| Method | Endpoint         | Description                                      |
|--------|------------------|--------------------------------------------------|
| GET    | /                | Service status                                   |
| GET    | /health          | Database connectivity and record count           |
| POST   | /analyze         | Classify, extract entities, route, and log query |
| POST   | /feedback        | Submit correct/incorrect feedback                |
| GET    | /review/queue    | List tickets by review status                    |
| GET    | /review/{log_id} | Load the full message and review context         |
| POST   | /review/submit   | Record a human review, correction, and outcome   |
| GET    | /analytics       | Routing, confidence, review, and feedback stats  |
| GET    | /logs            | Recent prediction logs                           |

The review dashboard lets agents inspect the original message, prediction, destination, rationale, and extracted entities before recording a verified intent, optional notes, and `reviewed` or `resolved` status. It previews the support team associated with the verified intent without overwriting the original prediction or destination. Submitting a review persists the corrected intent and timestamp and records `correct`/`incorrect` feedback against the original prediction. Only pending tickets can be reviewed; completed tickets are read-only. No schema migration is needed because the required review fields already exist in `prediction_logs`; the verified team is derived from the verified intent.
