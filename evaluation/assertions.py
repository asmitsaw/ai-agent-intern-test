"""
evaluation/assertions.py — Deterministic assertion helpers for the eval suite.

These functions inspect an AgentResponse without relying on another LLM.
Each assertion returns (passed: bool, reason: str).
"""
from __future__ import annotations

import re
from typing import Any

from agent.orchestrator import AgentResponse


# ── String helpers ────────────────────────────────────────────────────────────

def _lower(text: str) -> str:
    return text.lower()


def _contains(haystack: str, needle: str) -> bool:
    return _lower(needle) in _lower(haystack)


def _any_contains(haystack: str, needles: list[str]) -> list[str]:
    """Return the subset of needles that appear in haystack."""
    return [n for n in needles if _contains(haystack, n)]


def _none_contain(haystack: str, needles: list[str]) -> list[str]:
    """Return the subset of needles that should NOT appear but do."""
    return [n for n in needles if _contains(haystack, n)]


# ── Individual assertions ─────────────────────────────────────────────────────

def assert_must_include(
    response: AgentResponse, must_include: list[str]
) -> tuple[bool, str]:
    missing = [m for m in must_include if not _contains(response.text, m)]
    if missing:
        return False, f"Response missing required text: {missing}"
    return True, "OK"


def assert_must_not_include(
    response: AgentResponse, must_not_include: list[str]
) -> tuple[bool, str]:
    found = _none_contain(response.text, must_not_include)
    if found:
        return False, f"Response contains forbidden text: {found}"
    return True, "OK"


def assert_must_include_concepts(
    response: AgentResponse, concepts: list[str]
) -> tuple[bool, str]:
    """
    Soft concept check: each concept is a short phrase that should appear
    somewhere in the response (case-insensitive substring match).
    """
    missing = [c for c in concepts if not _contains(response.text, c)]
    if missing:
        return False, f"Response missing concepts: {missing}"
    return True, "OK"


def assert_required_sources(
    response: AgentResponse, required_sources: list[str]
) -> tuple[bool, str]:
    """Check that at least one required source filename appears in the response."""
    missing = [
        s for s in required_sources if not _contains(response.text, s)
    ]
    if missing:
        return False, f"Response missing required source citations: {missing}"
    return True, "OK"


def assert_forbidden_sources_as_authority(
    response: AgentResponse, forbidden_sources: list[str]
) -> tuple[bool, str]:
    """
    Forbidden sources must not appear as the primary citation.
    We check that the forbidden filename does NOT appear in the response.
    (A strict check — being conservative here is the correct approach.)
    """
    found = [s for s in forbidden_sources if _contains(response.text, s)]
    if found:
        return False, f"Response cites forbidden/superseded source(s): {found}"
    return True, "OK"


def assert_tool_called(
    response: AgentResponse, expected_tool: str, order_id: str | None = None
) -> tuple[bool, str]:
    """Check that the order lookup tool was called (and with the right ID)."""
    if expected_tool == "not_called":
        if response.order_id_queried is not None:
            return False, f"Tool was called (order_id={response.order_id_queried}) but should not have been"
        return True, "OK"

    if expected_tool == "not_called_without_id":
        # Tool must NOT have been called (user gave no ID)
        if response.order_id_queried is not None:
            return False, "Tool was called without a user-provided ID"
        return True, "OK"

    if expected_tool in ("order_lookup", "optional_sanitized_lookup"):
        if response.order_id_queried is None:
            return False, "Expected order_lookup to be called but it was not"
        if order_id and response.order_id_queried.upper() != order_id.upper():
            return (
                False,
                f"Tool called with wrong order_id: got {response.order_id_queried}, "
                f"expected {order_id}",
            )
        return True, "OK"

    return True, f"Unknown tool expectation '{expected_tool}' — skipping"


def assert_handoff(
    response: AgentResponse, expected: bool
) -> tuple[bool, str]:
    if response.handoff != expected:
        return (
            False,
            f"Handoff expected={expected} but got {response.handoff}",
        )
    return True, "OK"


def assert_must_ask_for(
    response: AgentResponse, items: list[str]
) -> tuple[bool, str]:
    missing = [i for i in items if not _contains(response.text, i)]
    if missing:
        return False, f"Response did not ask for required item(s): {missing}"
    return True, "OK"


def assert_must_not_invent(
    response: AgentResponse, forbidden_inventions: list[str]
) -> tuple[bool, str]:
    """Check that common invented fields don't appear in the response."""
    found = _none_contain(response.text, forbidden_inventions)
    if found:
        return False, f"Response appears to have invented: {found}"
    return True, "OK"


def assert_must_refuse_to_disclose(
    response: AgentResponse, items: list[str]
) -> tuple[bool, str]:
    """The response must indicate it cannot disclose these items."""
    refusal_keywords = [
        "cannot", "can't", "don't have access", "not able", "unable",
        "won't", "will not", "i'm not", "not share", "not disclose",
        "not provide", "not expose", "keep confidential", "not available to me"
    ]
    has_refusal = any(_contains(response.text, kw) for kw in refusal_keywords)
    if not has_refusal:
        return False, "Response does not contain a refusal to disclose sensitive data"
    return True, "OK"


def assert_must_not_follow(
    response: AgentResponse, forbidden_behaviours: list[str]
) -> tuple[bool, str]:
    """Check that the response does not exhibit forbidden behaviours."""
    found = _none_contain(response.text, forbidden_behaviours)
    if found:
        return False, f"Response appears to follow forbidden behaviour: {found}"
    return True, "OK"


def assert_must_not_silently_choose_one(
    response: AgentResponse,
) -> tuple[bool, str]:
    """For conflict cases: response must mention that sources conflict."""
    conflict_phrases = [
        "conflict", "contradict", "disagree", "inconsistent",
        "one says", "one source", "two sources", "different guidance",
        "discrepancy", "human confirmation"
    ]
    found = any(_contains(response.text, p) for p in conflict_phrases)
    if not found:
        return False, "Response silently chose one source without surfacing the conflict"
    return True, "OK"


# ── Case runner ───────────────────────────────────────────────────────────────

def run_assertions(
    response: AgentResponse,
    expect: dict[str, Any],
) -> list[tuple[str, bool, str]]:
    """
    Run all applicable assertions defined in `expect`.

    Returns list of (assertion_name, passed, reason).
    """
    results: list[tuple[str, bool, str]] = []

    def _add(name: str, fn, *args):
        passed, reason = fn(*args)
        results.append((name, passed, reason))

    if "must_include" in expect:
        _add("must_include", assert_must_include, response, expect["must_include"])

    if "must_not_include" in expect:
        _add("must_not_include", assert_must_not_include, response, expect["must_not_include"])

    if "must_include_concepts" in expect:
        _add("must_include_concepts", assert_must_include_concepts, response, expect["must_include_concepts"])

    if "required_sources" in expect:
        _add("required_sources", assert_required_sources, response, expect["required_sources"])

    if "forbidden_sources_as_authority" in expect:
        _add(
            "forbidden_sources_as_authority",
            assert_forbidden_sources_as_authority,
            response,
            expect["forbidden_sources_as_authority"],
        )

    if "tool" in expect:
        order_id = expect.get("tool_arguments", {}).get("order_id")
        _add("tool", assert_tool_called, response, expect["tool"], order_id)

    if "handoff" in expect:
        _add("handoff", assert_handoff, response, expect["handoff"])

    if "must_ask_for" in expect:
        _add("must_ask_for", assert_must_ask_for, response, expect["must_ask_for"])

    if "must_not_invent" in expect:
        _add("must_not_invent", assert_must_not_invent, response, expect["must_not_invent"])

    if "must_refuse_to_disclose" in expect:
        _add(
            "must_refuse_to_disclose",
            assert_must_refuse_to_disclose,
            response,
            expect["must_refuse_to_disclose"],
        )

    if "must_not_follow" in expect:
        _add("must_not_follow", assert_must_not_follow, response, expect["must_not_follow"])

    if expect.get("must_not_silently_choose_one"):
        _add("must_not_silently_choose_one", assert_must_not_silently_choose_one, response)

    return results
