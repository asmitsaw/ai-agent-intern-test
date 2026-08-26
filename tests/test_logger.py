"""
tests/test_logger.py — Unit tests for structured logging.
"""
import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.logger import AgentLogger


def test_logger_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.logger.DEBUG_LOGGING", False)
    logger = AgentLogger(session_id="test_disabled")
    logger.user_message("Hello")
    logger.retrieval("query", [])
    logger.tool_call("tool", {}, {})
    logger.llm_response("res", [], False)
    logger.error("err")
    # Should not create any log file
    log_file = Path("logs/test_disabled.jsonl")
    assert not log_file.exists()


def test_logger_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.logger.DEBUG_LOGGING", True)
    monkeypatch.setattr("agent.logger.LOG_DIR", tmp_path)

    logger = AgentLogger(session_id="test_enabled")
    logger.user_message("Where is my order?")
    logger.retrieval("Where is my order?", [{"source_file": "05-shipping.md", "score": 0.9}])
    logger.tool_call("order_lookup", {"order_id": "ORD-1001"}, {"found": True})
    logger.llm_response("Your order shipped.", ["05-shipping.md"], False)

    log_file = tmp_path / "test_enabled.jsonl"
    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    record1 = json.loads(lines[0])
    assert record1["event"] == "user_message"
    assert record1["text"] == "Where is my order?"
