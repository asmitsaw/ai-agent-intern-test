"""
tests/test_order_tool.py — Unit tests for the order lookup tool.

These tests run entirely offline (no API calls).
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.order_tool import lookup_order, _normalise_order_id, _looks_like_order_id


class TestNormalisation:
    def test_uppercase(self):
        assert _normalise_order_id("ord-1007") == "ORD-1007"

    def test_strips_whitespace(self):
        assert _normalise_order_id("  ORD-1007  ") == "ORD-1007"

    def test_mixed_case_and_spaces(self):
        assert _normalise_order_id("  ord-1007 ") == "ORD-1007"

    def test_looks_like_order_id_valid(self):
        assert _looks_like_order_id("ORD-1007") is True

    def test_looks_like_order_id_lowercase(self):
        assert _looks_like_order_id("ord-1007") is True

    def test_looks_like_order_id_garbage(self):
        assert _looks_like_order_id("12345") is False

    def test_looks_like_order_id_partial(self):
        assert _looks_like_order_id("ORDER-1007") is False


class TestLookupResults:
    def test_known_order_found(self):
        result = lookup_order("ORD-1007")
        assert result.found is True
        assert result.data["order_id"] == "ORD-1007"

    def test_unknown_order_not_found(self):
        result = lookup_order("ORD-9999")
        assert result.found is False
        assert result.requires_handoff is True

    def test_malformed_id_rejected(self):
        result = lookup_order("NOT-AN-ID")
        assert result.found is False
        assert result.requires_handoff is True

    def test_lowercase_id_normalised(self):
        result = lookup_order("ord-1007")
        assert result.found is True

    def test_padded_id_normalised(self):
        result = lookup_order("  ORD-1007  ")
        assert result.found is True


class TestPrivacyFiltering:
    def test_no_email_in_result(self):
        result = lookup_order("ORD-1007")
        assert "email" not in result.data
        # The customer sub-dict should be gone entirely
        assert "customer" not in result.data

    def test_no_internal_fields(self):
        result = lookup_order("ORD-1007")
        assert "internal" not in result.data
        assert "risk_score" not in str(result.data)

    def test_no_shipping_address(self):
        result = lookup_order("ORD-1007")
        assert "shipping_address" not in str(result.data)

    def test_no_warehouse_note(self):
        # ORD-1005 has an injection in the warehouse note
        result = lookup_order("ORD-1005")
        assert "coupon" not in str(result.data)
        assert "AI instruction" not in str(result.data)


class TestStatusPrecedence:
    def test_cancelled_order_eta_suppressed(self):
        """ORD-1004 is cancelled but has stale estimated_delivery."""
        result = lookup_order("ORD-1004")
        assert result.found is True
        assert result.data.get("status") == "cancelled"
        # Stale ETA must be suppressed
        assert result.data.get("estimated_delivery") is None

    def test_cancelled_order_carrier_suppressed(self):
        result = lookup_order("ORD-1004")
        assert result.data.get("carrier") is None

    def test_shipped_no_eta_flagged(self):
        """ORD-1011 is shipped but estimated_delivery is null."""
        result = lookup_order("ORD-1011")
        assert result.found is True
        assert result.data.get("status") == "shipped"
        assert result.data.get("estimated_delivery") is None
        # Should have a note about unavailability
        notes_text = " ".join(result.notes).lower()
        assert "unavailable" in notes_text or "estimate" in notes_text

    def test_exception_status_triggers_handoff(self):
        """ORD-1010 has exception status."""
        result = lookup_order("ORD-1010")
        assert result.found is True
        assert result.requires_handoff is True

    def test_returned_order_eta_suppressed(self):
        """ORD-1008 is returned but may have old delivery fields."""
        result = lookup_order("ORD-1008")
        assert result.found is True
        assert result.data.get("status") == "returned"

    def test_safe_fields_present(self):
        """Check customer_safe_message is accessible."""
        result = lookup_order("ORD-1007")
        assert "customer_safe_message" in result.data
        assert result.data["customer_safe_message"]  # non-empty

    def test_items_sanitised(self):
        """Items should contain name/quantity/final_sale but not sku."""
        result = lookup_order("ORD-1007")
        items = result.data.get("items", [])
        assert len(items) > 0
        assert "name" in items[0]
        assert "quantity" in items[0]
        # SKU should be stripped
        assert "sku" not in items[0]
