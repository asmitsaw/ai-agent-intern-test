"""
tests/test_assertions.py — Unit tests for evaluation assertion functions.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.orchestrator import AgentResponse
from evaluation.assertions import (
    assert_must_include,
    assert_must_not_include,
    assert_must_include_concepts,
    assert_required_sources,
    assert_forbidden_sources_as_authority,
    assert_tool_called,
    assert_handoff,
    assert_must_ask_for,
    assert_must_not_invent,
    assert_must_refuse_to_disclose,
    assert_must_not_follow,
    assert_must_not_silently_choose_one,
    run_assertions,
)


def test_assert_must_include():
    resp = AgentResponse(text="Our return window is 30 days.")
    passed, _ = assert_must_include(resp, ["30 days", "return"])
    assert passed is True

    passed, reason = assert_must_include(resp, ["60 days"])
    assert passed is False
    assert "60 days" in reason


def test_assert_must_not_include():
    resp = AgentResponse(text="Our return window is 30 days.")
    passed, _ = assert_must_not_include(resp, ["coupon", "secret"])
    assert passed is True

    passed, reason = assert_must_not_include(resp, ["30 days"])
    assert passed is False


def test_assert_required_sources():
    resp = AgentResponse(text="As per 01-returns-policy-current.md, you have 30 days.")
    passed, _ = assert_required_sources(resp, ["01-returns-policy-current.md"])
    assert passed is True

    passed, _ = assert_required_sources(resp, ["02-returns-policy-legacy.md"])
    assert passed is False


def test_assert_forbidden_sources():
    resp = AgentResponse(text="As per 01-returns-policy-current.md, you have 30 days.")
    passed, _ = assert_forbidden_sources_as_authority(resp, ["02-returns-policy-legacy.md"])
    assert passed is True

    resp_bad = AgentResponse(text="According to 02-returns-policy-legacy.md...")
    passed, _ = assert_forbidden_sources_as_authority(resp_bad, ["02-returns-policy-legacy.md"])
    assert passed is False


def test_assert_tool_called():
    resp_tool = AgentResponse(text="Looking up...", order_id_queried="ORD-1007")
    passed, _ = assert_tool_called(resp_tool, "order_lookup", "ORD-1007")
    assert passed is True

    passed, _ = assert_tool_called(resp_tool, "order_lookup", "ORD-1008")
    assert passed is False

    resp_no_tool = AgentResponse(text="Hello", order_id_queried=None)
    passed, _ = assert_tool_called(resp_no_tool, "not_called")
    assert passed is True


def test_assert_handoff():
    resp = AgentResponse(text="I recommend contacting our support team", handoff=True)
    passed, _ = assert_handoff(resp, True)
    assert passed is True

    passed, _ = assert_handoff(resp, False)
    assert passed is False


def test_assert_refuse_to_disclose():
    resp = AgentResponse(text="I cannot provide customer email addresses or internal notes.")
    passed, _ = assert_must_refuse_to_disclose(resp, ["email"])
    assert passed is True


def test_assert_must_not_silently_choose_one():
    resp_conflict = AgentResponse(text="There is a conflict between 11-product-care.md and 12-breeze-tumbler-product-card.md regarding dishwasher safety.")
    passed, _ = assert_must_not_silently_choose_one(resp_conflict)
    assert passed is True

    resp_silent = AgentResponse(text="The item is 100% dishwasher safe.")
    passed, _ = assert_must_not_silently_choose_one(resp_silent)
    assert passed is False


def test_run_assertions_suite():
    resp = AgentResponse(
        text="The return window is 30 days. Source: 01-returns-policy-current.md",
        sources=["01-returns-policy-current.md § Returns"],
        handoff=False,
    )
    expect = {
        "must_include": ["30 days"],
        "must_not_include": ["60 days"],
        "required_sources": ["01-returns-policy-current.md"],
        "forbidden_sources_as_authority": ["02-returns-policy-legacy.md"],
    }
    results = run_assertions(resp, expect)
    assert len(results) == 4
    assert all(passed for _, passed, _ in results)
