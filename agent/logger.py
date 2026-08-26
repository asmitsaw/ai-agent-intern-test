"""
agent/logger.py — Structured JSON logging for debug traces.

When DEBUG_LOGGING=true every agent turn writes one JSONL line per event
to LOG_DIR/<session_id>.jsonl.  Secrets are never written.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from agent.config import DEBUG_LOGGING, LOG_DIR

# Plain stdlib logger for internal warnings
_log = logging.getLogger(__name__)


class AgentLogger:
    """Emits structured JSONL traces for a single agent session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._enabled = DEBUG_LOGGING
        if self._enabled:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self._path = LOG_DIR / f"{session_id}.jsonl"

    # ── public helpers ────────────────────────────────────────────────────────

    def user_message(self, text: str) -> None:
        self._emit("user_message", {"text": text})

    def retrieval(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        self._emit(
            "retrieval",
            {
                "query": query,
                "chunks": [
                    {
                        "source": c.get("source_file"),
                        "heading": c.get("heading"),
                        "score": round(c.get("score", 0), 4),
                        "status": c.get("status"),
                        "policy_authority": c.get("policy_authority"),
                        "preview": c.get("text", "")[:120],
                    }
                    for c in chunks
                ],
            },
        )

    def tool_call(self, tool: str, arguments: dict, result: dict) -> None:
        """Log a tool invocation.  Internal fields are already stripped before
        this is called — we just log what the model received."""
        self._emit("tool_call", {"tool": tool, "arguments": arguments, "result": result})

    def llm_response(
        self,
        text: str,
        sources: list[str],
        handoff: bool,
    ) -> None:
        self._emit(
            "llm_response",
            {"text": text[:500], "sources": sources, "handoff": handoff},
        )

    def error(self, message: str, **extra: Any) -> None:
        self._emit("error", {"message": message, **extra})

    def fallback(self, reason: str) -> None:
        self._emit("fallback", {"reason": reason})

    # ── internals ─────────────────────────────────────────────────────────────

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if not self._enabled:
            return
        record = {
            "ts": time.time(),
            "session_id": self.session_id,
            "event": event,
            **payload,
        }
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            _log.warning("Could not write log: %s", exc)
