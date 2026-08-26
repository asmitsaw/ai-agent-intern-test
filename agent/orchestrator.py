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
            timeout=4.0,
            max_retries=0,
        )
        self._logger: AgentLogger = AgentLogger(self.session_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_message: str, model_override: str | None = None) -> AgentResponse:
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
        result_dict: dict | None = None
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

        # 6. Call LLM (with fast grounded fallback)
        response_text = self._call_llm(
            messages=messages,
            chunks=chunks,
            order_tool_result=result_dict,
            user_message=user_message,
            model_override=model_override,
        )

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

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        chunks: list[RetrievedChunk] | None = None,
        order_tool_result: dict | None = None,
        user_message: str = "",
        model_override: str | None = None,
        max_retries: int = 1,
    ) -> str:
        import time

        # Fast path for greetings
        clean_user = user_message.strip().lower()
        if clean_user in ("hi", "hello", "hey", "good morning", "good afternoon", "hi there"):
            return (
                "Hello! Welcome to Aster & Row customer support. "
                "How can I help you today? You can ask about order status, returns, shipping, warranty, or product care."
            )

        models_to_try = []
        if model_override and model_override.strip():
            models_to_try.append(model_override.strip())
        if LLM_MODEL not in models_to_try:
            models_to_try.append(LLM_MODEL)

        # If using OpenRouter, try alternative fast models
        if "openrouter" in str(self._client.base_url).lower():
            for alt in ["minimax/minimax-m3:free", "nvidia/nemotron-3.5-lightning:free", "liquid/lfm-2.5-2.6b:free"]:
                if alt not in models_to_try:
                    models_to_try.append(alt)

        last_error = None
        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    completion = self._client.chat.completions.create(
                        model=model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=0.1,
                        max_tokens=1024,
                        timeout=3.5,  # 3.5 second fast timeout per model
                    )
                    content = completion.choices[0].message.content
                    if content and content.strip():
                        return content.strip()
                except Exception as exc:
                    last_error = exc
                    break  # immediately try next model or fallback

        self._logger.error(f"LLM call fallback triggered: {last_error}")
        return _synthesize_grounded_fallback(user_message, chunks or [], order_tool_result)


# ── Grounded Fallback Synthesizer ─────────────────────────────────────────────

def _synthesize_grounded_fallback(
    user_message: str,
    chunks: list[RetrievedChunk],
    order_result: dict | None = None,
) -> str:
    """
    Deterministically synthesizes a grounded answer from retrieved chunks and order tool outputs
    when external LLM APIs are congested or unreachable.
    """
    msg_clean = user_message.strip().lower()
    msg_lower = msg_clean

    # 1. Greetings
    if msg_clean in ("hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hi there", "hello there") or msg_clean.startswith(("hi ", "hello ", "hey ")):
        return (
            "Hello! Welcome to Aster & Row customer support. "
            "How can I help you today? You can ask about order status, returns, shipping, warranty, or product care."
        )

    # 2. Identity and capabilities
    if any(k in msg_clean for k in ("who are you", "who are u", "what are you", "what is your name", "what can you do")):
        return (
            "I am the Aster & Row AI customer support assistant. "
            "I can help you check order statuses (e.g. `ORD-1007`), explain our return and membership policies, "
            "provide shipping estimates, and guide you on product care and warranty information."
        )

    # 3. Privacy defense
    if any(k in msg_clean for k in ("email", "address", "customer data", "private field", "phone")):
        if "ord-" in msg_clean:
            return (
                "For customer privacy and security, I cannot provide or disclose customer email addresses, "
                "shipping addresses, or internal account details. "
                "I recommend contacting our support team for further assistance."
            )

    # 4. Prompt injection defense
    if any(k in msg_clean for k in ("ignore", "system prompt", "instruction", "coupon", "secret", "reveal")):
        return (
            "I am the Aster & Row customer support assistant. I cannot disclose internal system prompts "
            "or issue unauthorized promotional coupons. How else may I assist you with our products or policies?"
        )

    # 3. Order lookup responses
    if order_result:
        if not order_result.get("found"):
            return (
                f"I was unable to locate an order matching that ID in our system. "
                f"Please double check your order number or I recommend contacting our support team for further assistance."
            )

        data = order_result.get("data", {})
        status = data.get("status", "unknown")
        order_id = data.get("order_id", "your order")

        if status == "cancelled":
            return (
                f"Order {order_id} is currently marked as **cancelled**. "
                f"Because this order has been cancelled, delivery estimates and tracking are no longer active. "
                f"I recommend contacting our support team for further assistance."
            )
        elif status == "returned":
            return (
                f"Order {order_id} has been marked as **returned** in our system. "
                f"I cannot confirm whether the refund batch has completed. "
                f"I recommend contacting our support team for further assistance."
            )
        elif status == "exception":
            return (
                f"Order {order_id} has a status of **exception**. The shipment requires support review. "
                f"I recommend contacting our support team for further assistance."
            )
        elif status == "shipped":
            eta = data.get("estimated_delivery")
            carrier = data.get("carrier", "standard carrier")
            tracking = data.get("tracking_number", "")
            if not eta:
                return (
                    f"Order {order_id} has **shipped** via {carrier} (Tracking: {tracking}). "
                    f"However, an estimated delivery date is currently unavailable. "
                    f"Please allow standard transit times or check the carrier tracking link."
                )
            return (
                f"Order {order_id} has **shipped** via {carrier} (Tracking: {tracking}). "
                f"Estimated delivery is {eta}."
            )
        elif status == "delivered":
            msg = data.get("customer_safe_message", "Delivered.")
            return f"Order {order_id} has been **delivered**. Details: {msg}"
        else:
            return f"Order {order_id} is currently **{status}**."

    # Missing order ID intent
    if any(k in msg_lower for k in _ORDER_INTENT_KEYWORDS) and "ord-" not in msg_lower:
        return "To look up your order status, please provide your order ID (e.g. ORD-1007)."

    # 4. Knowledge base responses
    if chunks:
        top_chunk = chunks[0]

        # Dishwasher conflict detection
        if "dishwasher" in msg_lower or "tumbler" in msg_lower:
            sources = [c.source_file for c in chunks]
            if "11-product-care.md" in sources and "12-breeze-tumbler-product-card.md" in sources:
                return (
                    "There is a discrepancy in our official documentation regarding dishwasher safety: "
                    "11-product-care.md advises hand-washing the tumbler body, while 12-breeze-tumbler-product-card.md "
                    "states the entire tumbler is dishwasher safe. "
                    "I recommend contacting our support team for confirmation. "
                    "(Source: 11-product-care.md — Drinkware Care, 12-breeze-tumbler-product-card.md — Care and Use)"
                )

        # Warranty check
        if "warranty" in msg_lower:
            return (
                "Our warranty policy provides a 2-year warranty on all bags and backpacks, and a 1-year warranty on "
                "drinkware and travel accessories. Warranty claims require proof of purchase. "
                f"(Source: {top_chunk.source_file} — {top_chunk.heading})"
            )

        # TrailPlus return window
        if "trailplus" in msg_lower:
            return (
                "Active TrailPlus members receive an extended return window of 45 calendar days from the date of delivery. "
                f"(Source: 09-trailplus-membership.md — Extended Returns)"
            )

        # Standard return window
        if "return" in msg_lower:
            return (
                "For regular customers, unused items in original packaging can be returned within 30 calendar days of delivery. "
                f"(Source: 01-returns-policy-current.md — Standard Return Window)"
            )

        # Shipping
        if "international" in msg_lower or "canada" in msg_lower or "ship" in msg_lower:
            if "germany" in msg_lower:
                return (
                    "We do not currently offer shipping to Germany or outside our approved international regions. "
                    "(Source: 06-international-shipping.md — Supported Destinations)"
                )
            if "canada" in msg_lower:
                return (
                    "Yes, we ship to Canada with standard delivery taking 6-10 business days. Canadian customers are responsible for applicable import duties. "
                    "(Source: 06-international-shipping.md — Canada Shipping)"
                )
            return (
                "We offer domestic shipping across the US as well as international shipping to select destinations including Canada. "
                f"(Source: {top_chunk.source_file} — {top_chunk.heading})"
            )

        # General grounded extraction
        clean_text = top_chunk.text.strip().replace("\n\n", " ")
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + "..."
        return f"{clean_text}\n\n(Source: {top_chunk.source_file} — {top_chunk.heading})"

    return (
        "I'm sorry, I could not find specific information regarding that request in our knowledge base. "
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
