"""
agent/orchestrator.py — Multi-turn agent session.

One AgentSession per conversation.  Call session.chat(user_message) for each turn.

Flow per turn:
  1. Append user message to history.
  2. Detect if order lookup is needed.
  3. Run RAG retrieval.
  4. Run order lookup tool if applicable.
  5. Build prompt (system + rolling history + retrieved context + tool result).
  6. Call LLM.
  7. Extract sources and handoff flag from response.
  8. Log everything (debug mode).
  9. Return AgentResponse.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from agent.config import (
    HISTORY_WINDOW,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    RETRIEVAL_TOP_K,
)
from agent.logger import AgentLogger
from agent.order_tool import lookup_order
from agent.prompts import (
    SYSTEM_PROMPT,
    format_order_tool_result,
    format_retrieved_context,
)
from agent.rag import RetrievedChunk, retrieve

# ── Patterns for detecting order IDs in user text ────────────────────────────
_ORDER_ID_RE = re.compile(r"\bORD-\d+\b", re.IGNORECASE)

# Keywords that suggest the user wants order status without providing an ID
_ORDER_INTENT_KEYWORDS = (
    "where is my order",
    "track my order",
    "order status",
    "where is my package",
    "when will my order",
    "my order arrive",
    "when will it arrive",
    "has my order shipped",
    "shipping update",
)

# Keywords that signal a handoff should be recommended in the response
_HANDOFF_PHRASES = (
    "recommend contacting our support team",
    "contact human support",
    "i recommend contacting",
    "please contact support",
    "human assistance",
)


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    text: str
    sources: list[str] = field(default_factory=list)
    handoff: bool = False
    order_id_queried: str | None = None
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)


# ── Session ───────────────────────────────────────────────────────────────────

class AgentSession:
    """Maintains conversation state for one user session."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self._history: list[dict[str, str]] = []  # OpenAI message dicts
        self._client: OpenAI = OpenAI(
            api_key=LLM_API_KEY or "dummy-key",
            base_url=LLM_BASE_URL,
        )
        self._logger: AgentLogger = AgentLogger(self.session_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> AgentResponse:
        """Process one user turn and return the agent's response."""
        self._logger.user_message(user_message)

        # 1. Append user turn to rolling history
        self._history.append({"role": "user", "content": user_message})

        # 2. Detect order intent
        order_ids_in_message = _ORDER_ID_RE.findall(user_message)
        order_id_queried: str | None = None
        order_tool_content: str = ""
        requires_handoff_from_tool = False

        # 3. RAG retrieval
        chunks = retrieve(user_message, k=RETRIEVAL_TOP_K)
        self._logger.retrieval(user_message, [c.to_dict() for c in chunks])

        # 4. Order lookup
        if order_ids_in_message:
            # Use the first valid-looking order ID found
            raw_id = order_ids_in_message[0]
            result = lookup_order(raw_id)
            order_id_queried = raw_id.upper()
            result_dict = result.to_dict()
            order_tool_content = format_order_tool_result(result_dict)
            requires_handoff_from_tool = result.requires_handoff
            self._logger.tool_call(
                "order_lookup",
                {"order_id": raw_id},
                result_dict,
            )
        elif _has_order_intent(user_message, self._history):
            # User wants order info but hasn't given an ID — don't call the tool
            order_tool_content = (
                "=== ORDER LOOKUP ===\n"
                "The user appears to want order status information but has not provided an order ID.\n"
                "Ask the user for their order ID before performing a lookup.\n"
                "=== END ==="
            )

        # 5. Build the messages list for the LLM
        context_block = _build_context_block(chunks, order_tool_content)
        messages = _build_messages(
            system_prompt=SYSTEM_PROMPT,
            history=self._history,
            context_block=context_block,
            window=HISTORY_WINDOW,
        )

        # 6. Call LLM
        response_text = self._call_llm(messages)

        # 7. Extract metadata from response
        sources = _extract_sources(chunks, response_text)
        handoff = requires_handoff_from_tool or _detect_handoff(response_text)

        # 8. Append assistant turn to history
        self._history.append({"role": "assistant", "content": response_text})

        # 9. Log response
        self._logger.llm_response(response_text, sources, handoff)

        return AgentResponse(
            text=response_text,
            sources=sources,
            handoff=handoff,
            order_id_queried=order_id_queried,
            retrieved_chunks=[c.to_dict() for c in chunks],
        )

    def reset(self) -> None:
        """Clear conversation history (start a new session without new object)."""
        self._history.clear()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict[str, str]], max_retries: int = 3) -> str:
        import time
        for attempt in range(max_retries):
            try:
                completion = self._client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.1,    # low temperature → more deterministic
                    max_tokens=1024,
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = 4 * (attempt + 1)
                    time.sleep(wait_time)
                    continue
                self._logger.error(str(exc))
                return (
                    "I'm sorry, I encountered an error while processing your request. "
                    "I recommend contacting our support team for further assistance."
                )
        return (
            "I'm sorry, I encountered an error while processing your request. "
            "I recommend contacting our support team for further assistance."
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_order_intent(user_message: str, history: list[dict]) -> bool:
    """Return True if the user seems to be asking about an order."""
    text = user_message.lower()
    return any(kw in text for kw in _ORDER_INTENT_KEYWORDS)


def _build_context_block(
    chunks: list[RetrievedChunk],
    order_tool_content: str,
) -> str:
    """Combine retrieval and tool output into a single context string."""
    parts: list[str] = []
    rag_context = format_retrieved_context(chunks)
    parts.append(rag_context)
    if order_tool_content:
        parts.append("\n\n" + order_tool_content)
    return "\n".join(parts)


def _build_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    context_block: str,
    window: int,
) -> list[dict[str, str]]:
    """
    Construct the OpenAI messages list:
      [system] → [rolling history window] → [context injected as user message]

    The context is injected as the last "user" message so the model sees it
    immediately before generating its reply.
    """
    # Take last `window` messages, excluding the most recent user message
    # (we'll inject context with it instead)
    past = history[-(window + 1) : -1] if len(history) > 1 else []
    latest_user = history[-1]["content"]

    # Inject context into the latest user message
    augmented_user = f"{latest_user}\n\n{context_block}"

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(past)
    messages.append({"role": "user", "content": augmented_user})
    return messages


def _extract_sources(chunks: list[RetrievedChunk], response_text: str) -> list[str]:
    """Return source citations for chunks whose source files appear in the response."""
    cited: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.source_file in response_text and chunk.source_file not in seen:
            cited.append(chunk.citation())
            seen.add(chunk.source_file)
    return cited


def _detect_handoff(response_text: str) -> bool:
    """Return True if the response recommends human handoff."""
    lower = response_text.lower()
    return any(phrase in lower for phrase in _HANDOFF_PHRASES)
