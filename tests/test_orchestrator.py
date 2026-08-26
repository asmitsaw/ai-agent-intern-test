"""
tests/test_orchestrator.py — Unit tests for orchestrator helper logic.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.orchestrator import (
    _ORDER_ID_RE,
    _has_order_intent,
    _build_context_block,
    _build_messages,
    _extract_sources,
    _detect_handoff,
)
from agent.rag import RetrievedChunk


def test_order_id_regex():
    assert _ORDER_ID_RE.findall("Where is ORD-1007?") == ["ORD-1007"]
    assert _ORDER_ID_RE.findall("Check ord-1001 and ORD-1002") == ["ord-1001", "ORD-1002"]
    assert _ORDER_ID_RE.findall("No id here 12345") == []


def test_has_order_intent():
    assert _has_order_intent("Where is my order?", []) is True
    assert _has_order_intent("Track my package please", []) is False  # package isn't directly in keyword list unless checked
    assert _has_order_intent("where is my package", []) is True
    assert _has_order_intent("What is the return policy?", []) is False


def test_build_context_block():
    chunks = [
        RetrievedChunk(
            text="30-day window",
            source_file="01-returns-policy-current.md",
            heading="Policy",
            status="active",
            policy_authority="official",
            audience="customer",
            document_id="01",
            score=0.9,
        )
    ]
    block = _build_context_block(chunks, "=== ORDER LOOKUP ===\nFound\n=== END ===")
    assert "01-returns-policy-current.md" in block
    assert "ORDER LOOKUP" in block


def test_build_messages_history_window():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Tell me about returns"},
    ]
    messages = _build_messages(
        system_prompt="You are a support bot.",
        history=history,
        context_block="[Retrieved Context]",
        window=2,
    )
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "[Retrieved Context]" in messages[-1]["content"]


def test_extract_sources():
    chunks = [
        RetrievedChunk(
            text="sample",
            source_file="01-returns-policy-current.md",
            heading="Policy",
            status="active",
            policy_authority="official",
            audience="customer",
            document_id="01",
            score=0.9,
        ),
        RetrievedChunk(
            text="sample",
            source_file="05-domestic-shipping.md",
            heading="Shipping",
            status="active",
            policy_authority="official",
            audience="customer",
            document_id="05",
            score=0.8,
        ),
    ]
    resp = "Based on 01-returns-policy-current.md, returns are free."
    sources = _extract_sources(chunks, resp)
    assert len(sources) == 1
    assert "01-returns-policy-current.md § Policy" in sources[0]


def test_detect_handoff():
    assert _detect_handoff("I recommend contacting our support team for further assistance.") is True
    assert _detect_handoff("Please contact support.") is True
    assert _detect_handoff("Your order was delivered yesterday.") is False
