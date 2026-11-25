"""Artifact paths and optional live inference settings."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "live-results"


def load_api_config(model=None):
    base_url = os.environ.get("CBR_API_BASE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("CBR_API_KEY", "").strip()
    model = model or os.environ.get("CBR_MODEL") or "claude-opus-5"
    if not base_url or not api_key:
        raise ValueError("Set CBR_API_BASE_URL and CBR_API_KEY for live inference")
    return {"base_url": base_url, "api_key": api_key, "model": model}

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 8192
LLM_MAX_RETRIES = 8
