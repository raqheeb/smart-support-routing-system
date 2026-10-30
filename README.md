# 🎯 Smart Support Routing System

An intelligent AI-powered customer support classification and routing platform that automatically identifies customer intent, extracts relevant entities, recommends the appropriate support team, and enables human agents to review uncertain predictions.

The system combines zero-shot NLP classification, Named Entity Recognition (NER), confidence-aware decision-making, and a human-in-the-loop review workflow.

## 🚀 Features

* **Zero-Shot Intent Classification:** Uses Facebook BART Large MNLI to identify customer intent without task-specific model training.
* **Named Entity Recognition:** Uses spaCy to extract relevant entities from customer messages.
* **Confidence-Aware Routing:** Uses classification confidence and top-two prediction margin to determine routing decisions.
* **Human-in-the-Loop Review:** Allows support agents to inspect predictions, correct intents, and record review outcomes.
* **Optional LLM Assistance:** Integrates Anthropic Claude for suggestions in medium-confidence cases. Disabled by default.
* **Persistent Storage:** Stores predictions, review information, and feedback using SQLite and SQLAlchemy.
* **Evaluation Pipeline:** Includes a BANKING77 evaluation script for assessing classification performance.
* **Interactive Dashboard:** Provides a Gradio interface for ticket analysis and human review.

## 🛠️ Technology Stack

| Component               | Technology               |
| ----------------------- | ------------------------ |
| Frontend                | Gradio                   |
| Backend                 | FastAPI, Uvicorn         |
| Intent Classification   | Facebook BART Large MNLI |
| NLP / Entity Extraction | spaCy                    |
| Optional LLM            | Anthropic Claude         |
| Database                | SQLite, SQLAlchemy       |
| Evaluation              | BANKING77                |
| Testing                 | pytest                   |

## 🏗️ System Architecture

The system follows a modular architecture:

1. Customer messages enter through the Gradio interface or FastAPI endpoint.
2. The BART zero-shot classifier predicts the most relevant customer intent.
3. spaCy extracts relevant entities from the message.
4. The decision engine evaluates classification confidence, prediction margin, and review conditions.
5. The system either automatically routes the ticket or sends it for human review.
6. Predictions and review information are persisted in SQLite.
7. Human feedback is collected for future model evaluation and improvement.

## 🔁 Routing Pipeline

```text
             Customer Message
                    |
                    v
          Intent Classification
          (BART Large MNLI)
                    |
                    v
          Named Entity Recognition
                 (spaCy)
                    |
                    v
              Decision Engine
                    |
          +---------+----------+
          |                    |
          v                    v
    High Confidence       Uncertain Prediction
          |                    |
          v                    v
     Auto Routing          Human Review
          |                    |
          |              Opti
```

