"""
agent/config.py — Centralised configuration loaded from environment variables.
Configured for top Chinese open-source foundation models (DeepSeek-V3 / Qwen 2.5)
with optional local embedding support.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")


def _get_api_key() -> str:
    key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    return key


# ── LLM / Embeddings ──────────────────────────────────────────────────────────
LLM_API_KEY: str = _get_api_key()
# Default to DeepSeek API endpoint (DeepSeek-V3 / DeepSeek-R1 open-source foundation model)
# Can also be set to Alibaba DashScope for Qwen ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
# or local Ollama ("http://localhost:11434/v1")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")  # deepseek-chat (DeepSeek-V3) or qwen-plus

# Embedding Provider: "local" (runs offline on-device via ONNX) or "openai_compatible"
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local").lower()
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
