"""
agent/config.py — Centralised configuration loaded from environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example to .env and fill in your values."
        )
    return val


# ── LLM / Embeddings ──────────────────────────────────────────────────────────
OPENAI_API_KEY: str = _require("OPENAI_API_KEY")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Paths ─────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_PATH: Path = _ROOT / os.getenv("KNOWLEDGE_BASE_PATH", "knowledge-base")
ORDERS_PATH: Path = _ROOT / os.getenv("ORDERS_PATH", "data/orders.json")
VECTOR_STORE_PATH: Path = _ROOT / os.getenv("VECTOR_STORE_PATH", "vector_store/chroma_db")
LOG_DIR: Path = _ROOT / os.getenv("LOG_DIR", "logs")

# ── Behaviour ─────────────────────────────────────────────────────────────────
DEBUG_LOGGING: bool = os.getenv("DEBUG_LOGGING", "false").lower() == "true"

# ── RAG ───────────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 400          # tokens per chunk
CHUNK_OVERLAP: int = 60        # token overlap between chunks
RETRIEVAL_TOP_K: int = 6       # chunks retrieved per query
HISTORY_WINDOW: int = 8        # last N messages sent to LLM

# ── Collection name inside ChromaDB ──────────────────────────────────────────
CHROMA_COLLECTION: str = "aster_row_kb"
