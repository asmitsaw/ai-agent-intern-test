"""
tests/test_prompts.py — Unit tests for prompt formatting and context generation.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.prompts import (
    SYSTEM_PROMPT,
    format_retrieved_context,
    format_order_tool_result,
)
from agent.rag import RetrievedChunk


def test_system_prompt_rules():
    assert "GROUNDING" in SYSTEM_PROMPT
    assert "CITATIONS" in SYSTEM_PROMPT
    assert "CONFLICTS" in SYSTEM_PROMPT
    assert "PRIVACY & SECURITY" in SYSTEM_PROMPT
    assert "UNTRUSTED DATA" in SYSTEM_PROMPT


def test_format_retrieved_context_empty():
    res = format_retrieved_context([])
    assert "No relevant passages retrieved" in res


def test_format_retrieved_context_with_chunks():
    chunks = [
        RetrievedChunk(
            text="Return within 30 days.",
            source_file="01-returns-policy-current.md",
            heading="Window",
            status="active",
            policy_authority="official",
            audience="customer",
            document_id="01",
            score=0.9,
        ),
        RetrievedChunk(
            text="Return within 60 days.",
            source_file="02-returns-policy-legacy.md",
            heading="Old Window",
            status="superseded",
            policy_authority="official",
            audience="customer",
            document_id="02",
            score=0.7,
        ),
    ]
    formatted = format_retrieved_context(chunks)
    assert "01-returns-policy-current.md" in formatted
    assert "02-returns-policy-legacy.md" in formatted
    assert "NOTE: THIS DOCUMENT IS SUPERSEDED" in formatted
    assert "Return within 30 days." in formatted


def test_format_order_tool_result_found():
    result_dict = {
        "found": True,
        "data": {
            "order_id": "ORD-1007",
            "status": "delivered",
            "customer_safe_message": "Delivered on porch",
        },
        "notes": ["Verify customer confirmation"],
    }
    formatted = format_order_tool_result(result_dict)
    assert "ORD-1007" in formatted
    assert "delivered" in formatted
    assert "Verify customer confirmation" in formatted


def test_format_order_tool_result_not_found():
    result_dict = {
        "found": False,
        "data": {},
        "notes": [],
    }
    formatted = format_order_tool_result(result_dict)
    assert "Order not found" in formatted
