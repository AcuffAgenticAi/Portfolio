"""
================================================================================
billing/usage_meter.py  —  Per-Tenant Usage Metering & Billing Simulation
================================================================================

Purpose:
    Records prediction API calls per tenant and generates usage-based invoices.
    This module simulates the billing engine of a SaaS AI platform where
    customers pay per inference call.

Billing model:
    $0.002 per prediction (configurable via PRICE_PER_PREDICTION).

    Tiers can be added by extending the _apply_tier_pricing() method:
        • Starter tier:     $0.003 / prediction  (0 – 10 k calls/month)
        • Growth tier:      $0.002 / prediction  (10 k – 100 k calls/month)
        • Enterprise tier:  $0.001 / prediction  (100 k+ calls/month)

Thread safety:
    UsageMeter is NOT thread-safe in its current dict-based implementation.
    For concurrent access wrap record_usage() with a threading.Lock or
    replace usage_counter with a Redis INCR command in production.

Usage:
    meter = UsageMeter()
    meter.record_usage("bank_a")
    meter.record_usage("bank_a")
    meter.record_usage("bank_b")
    print(meter.generate_invoice())
    # {'bank_a': 0.004, 'bank_b': 0.002}
================================================================================
"""

import logging
from collections import defaultdict
from typing import Dict, Optional

logger = logging.getLogger("platform.usage_meter")

# ── Pricing constants (override via subclass or env var in production) ────────
PRICE_PER_PREDICTION: float = 0.002   # USD per prediction call


class UsageMeter:
    """
    In-memory usage counter and invoice generator for multi-tenant billing.

    In a production SaaS platform this would:
        • Persist counts to Redis (fast INCR) or a time-series DB
        • Integrate with Stripe / Chargebee for actual invoicing
        • Support monthly billing cycles and overage alerts

    Attributes:
        _usage_counter:        Dict mapping tenant_id → call count this period.
        _price_per_prediction: USD charged per prediction call.
    """

    def __init__(self, price_per_prediction: float = PRICE_PER_PREDICTION) -> None:
        self._usage_counter:        Dict[str, int]   = defaultdict(int)
        self._price_per_prediction: float            = price_per_prediction

    # ─────────────────────────────────────────────────────────────────────────
    # Recording usage
    # ─────────────────────────────────────────────────────────────────────────

    def record_usage(self, tenant_id: str, count: int = 1) -> None:
        """
        Increment the prediction call counter for a tenant.

        Args:
            tenant_id:  Unique identifier for the tenant making the call.
            count:      Number of predictions to record (default 1).
                        Use count > 1 for batch prediction endpoints.

        Raises:
            ValueError:  If count is not a positive integer.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1; got {count}")

        self._usage_counter[tenant_id] += count
        logger.debug(
            "Usage recorded  tenant='%s'  +%d  total=%d",
            tenant_id, count, self._usage_counter[tenant_id],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Billing
    # ─────────────────────────────────────────────────────────────────────────

    def generate_invoice(
        self,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute the invoice amounts for all tenants (or one specific tenant).

        Args:
            tenant_id:  If provided, return the invoice for this tenant only.
                        Returns all tenants when None (default).

        Returns:
            Dict mapping tenant_id → USD amount owed for the current period.
        """
        if tenant_id is not None:
            calls  = self._usage_counter.get(tenant_id, 0)
            amount = self._apply_tier_pricing(calls)
            return {tenant_id: round(amount, 4)}

        invoices = {}
        for tid, calls in self._usage_counter.items():
            invoices[tid] = round(self._apply_tier_pricing(calls), 4)

        logger.info(
            "Invoice generated  tenants=%d  total_calls=%d  total_revenue=$%.4f",
            len(invoices),
            sum(self._usage_counter.values()),
            sum(invoices.values()),
        )
        return invoices

    def get_usage(self, tenant_id: str) -> int:
        """Return the raw prediction call count for a tenant this period."""
        return self._usage_counter.get(tenant_id, 0)

    def reset_usage(self, tenant_id: Optional[str] = None) -> None:
        """
        Reset usage counters at the start of a new billing period.

        Args:
            tenant_id:  Reset a single tenant when provided; reset all when None.
        """
        if tenant_id is not None:
            self._usage_counter[tenant_id] = 0
            logger.info("Usage counter reset for tenant '%s'", tenant_id)
        else:
            self._usage_counter.clear()
            logger.info("All usage counters reset (new billing period)")

    # ─────────────────────────────────────────────────────────────────────────
    # Pricing logic
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_tier_pricing(self, call_count: int) -> float:
        """
        Calculate the USD cost for a given number of prediction calls.

        Currently flat-rate.  Replace with tiered logic as needed:

            if call_count <= 10_000:
                return call_count * 0.003
            elif call_count <= 100_000:
                return 10_000 * 0.003 + (call_count - 10_000) * 0.002
            else:
                return (10_000 * 0.003
                        + 90_000 * 0.002
                        + (call_count - 100_000) * 0.001)

        Args:
            call_count:  Number of prediction calls made this period.

        Returns:
            USD amount due (unrounded; caller rounds as needed).
        """
        return call_count * self._price_per_prediction
