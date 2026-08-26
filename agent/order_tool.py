"""
agent/order_tool.py — Safe order lookup from data/orders.json.

Rules enforced here (from orders-data-dictionary.md):
  • Only SAFE_FIELDS are ever returned to the model.
  • customer.* and internal.* are stripped unconditionally.
  • When status is cancelled/returned, stale ETA/carrier fields are suppressed.
  • When status is shipped but estimated_delivery is null, we flag it explicitly.
  • When status is exception, we flag that a human handoff is required.
  • Internal warehouse notes must NEVER reach the model (prompt-injection risk).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.config import ORDERS_PATH

# ── Fields that are allowed to leave this module ──────────────────────────────
SAFE_FIELDS: frozenset[str] = frozenset(
    {
        "order_id",
        "membership_tier",
        "items",          # filtered sub-fields below
        "placed_at",
        "status",
        "status_updated_at",
        "shipped_at",
        "delivered_at",
        "carrier",
        "tracking_number",
        "estimated_delivery",
        "customer_safe_message",
    }
)

# Safe sub-fields within each item
SAFE_ITEM_FIELDS: frozenset[str] = frozenset({"name", "quantity", "final_sale"})

# Statuses where stale ETA / carrier data should be suppressed
STALE_ETA_STATUSES: frozenset[str] = frozenset({"cancelled", "returned"})


# ── Dataset loader (cached so we only hit disk once) ──────────────────────────

@lru_cache(maxsize=1)
def _load_dataset() -> dict[str, Any]:
    path = Path(ORDERS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Orders file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _get_orders() -> list[dict[str, Any]]:
    return _load_dataset().get("orders", [])


def snapshot_timestamp() -> str:
    """Return the dataset snapshot timestamp for use in time calculations."""
    return _load_dataset().get("snapshot_at", "")


# ── Input normalisation ───────────────────────────────────────────────────────

def _normalise_order_id(raw: str) -> str:
    """
    Normalise harmless differences: strip whitespace, uppercase.
    We do NOT guess substantially different IDs.
    """
    return raw.strip().upper()


def _looks_like_order_id(value: str) -> bool:
    """Return True if value matches the ORD-NNNN pattern (after normalisation)."""
    return bool(re.fullmatch(r"ORD-\d+", _normalise_order_id(value)))


# ── Field sanitisation ───────────────────────────────────────────────────────

def _sanitise_items(items: list[dict]) -> list[dict]:
    """Keep only customer-safe sub-fields from each item."""
    return [
        {k: v for k, v in item.items() if k in SAFE_ITEM_FIELDS}
        for item in items
    ]


def _sanitise_order(order: dict[str, Any], status: str) -> dict[str, Any]:
    """
    Return a sanitised copy containing only SAFE_FIELDS.
    Suppresses stale ETA/carrier when order is cancelled or returned.
    """
    safe: dict[str, Any] = {}

    for field in SAFE_FIELDS:
        if field not in order:
            continue
        value = order[field]

        # Suppress stale ETA / carrier for cancelled / returned orders
        if status in STALE_ETA_STATUSES and field in (
            "carrier",
            "tracking_number",
            "estimated_delivery",
            "shipped_at",
        ):
            safe[field] = None
            continue

        # Sanitise items sub-document
        if field == "items" and isinstance(value, list):
            safe[field] = _sanitise_items(value)
            continue

        safe[field] = value

    return safe


# ── Public API ────────────────────────────────────────────────────────────────

class OrderLookupResult:
    """Value object returned by lookup_order()."""

    def __init__(
        self,
        found: bool,
        data: dict[str, Any] | None = None,
        requires_handoff: bool = False,
        notes: list[str] | None = None,
    ) -> None:
        self.found = found
        self.data = data or {}
        self.requires_handoff = requires_handoff
        # Agent-facing notes (e.g. "ETA unavailable", "exception status")
        self.notes: list[str] = notes or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "data": self.data,
            "requires_handoff": self.requires_handoff,
            "notes": self.notes,
        }


def lookup_order(order_id_raw: str) -> OrderLookupResult:
    """
    Look up a single order by ID.

    Args:
        order_id_raw: Raw order ID from the user (may be lowercase / padded).

    Returns:
        OrderLookupResult with only customer-safe fields.
    """
    order_id = _normalise_order_id(order_id_raw)

    # Reject clearly malformed IDs early (do not guess)
    if not _looks_like_order_id(order_id):
        return OrderLookupResult(
            found=False,
            notes=[
                f"'{order_id_raw}' does not look like a valid order ID. "
                "Order IDs follow the format ORD-XXXX."
            ],
            requires_handoff=True,
        )

    orders = _get_orders()
    match = next((o for o in orders if o.get("order_id") == order_id), None)

    if match is None:
        return OrderLookupResult(
            found=False,
            notes=[
                f"Order {order_id} was not found in our system. "
                "Please double-check the order ID or contact support."
            ],
            requires_handoff=True,
        )

    status: str = match.get("status", "unknown")
    safe_data = _sanitise_order(match, status)

    notes: list[str] = []
    requires_handoff = False

    # ── Status-specific agent notes ───────────────────────────────────────────

    if status in STALE_ETA_STATUSES:
        notes.append(
            f"This order is {status}. "
            "Stale carrier/ETA fields have been suppressed."
        )

    elif status == "shipped" and not match.get("estimated_delivery"):
        notes.append(
            "The order has shipped but no delivery estimate is available. "
            "Do not invent or calculate an arrival date."
        )

    elif status == "exception":
        notes.append(
            "This shipment has an exception that requires support review. "
            "Recommend a human handoff."
        )
        requires_handoff = True

    elif status == "delayed":
        notes.append(
            "The carrier has reported a delay. "
            "Use customer_safe_message for the delay explanation."
        )

    return OrderLookupResult(
        found=True,
        data=safe_data,
        requires_handoff=requires_handoff,
        notes=notes,
    )
