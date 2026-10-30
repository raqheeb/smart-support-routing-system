"""
backend/config.py
Centralized configuration management with environment variable overrides.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from dotenv import load_dotenv

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from project root
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

def _read_probability_setting(name: str, default: str) -> float:
    value = float(os.getenv(name, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


# Classifier confidence is the top zero-shot candidate probability.
AUTO_ROUTE_THRESHOLD = _read_probability_setting("AUTO_ROUTE_THRESHOLD", "0.80")
LLM_ASSIST_THRESHOLD = _read_probability_setting("LLM_ASSIST_THRESHOLD", "0.65")
AUTO_ROUTE_MIN_MARGIN = _read_probability_setting("AUTO_ROUTE_MIN_MARGIN", "0.10")
if LLM_ASSIST_THRESHOLD >= AUTO_ROUTE_THRESHOLD:
    raise ValueError("LLM_ASSIST_THRESHOLD must be lower than AUTO_ROUTE_THRESHOLD.")

# LLM / Anthropic Settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_ENABLED = os.getenv("ANTHROPIC_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_TIMEOUT_SECONDS = float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "15.0"))

# Database
DB_PATH = BASE_DIR / "database" / "logs.db"
DEFAULT_DB_URL = f"sqlite:///{DB_PATH.as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "7860"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
