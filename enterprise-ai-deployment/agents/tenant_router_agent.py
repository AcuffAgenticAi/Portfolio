"""
================================================================================
agents/tenant_router_agent.py  —  Multi-Tenant Model Router
================================================================================

Purpose:
    Routes inference requests to the correct model artefact for each tenant
    (customer organisation) served by the platform.

    Multi-tenancy is fundamental to enterprise SaaS AI platforms.  Each tenant:
        • May have a domain-specific model trained on their own data
        • Must be isolated from other tenants (no cross-contamination)
        • Should be billed independently for their prediction volume
        • Can be onboarded or offboarded without affecting other tenants

Architecture:
    TENANT_REGISTRY maps tenant_id → model_path.  In a production deployment
    this would be backed by a config database (Postgres, DynamoDB, etc.) and
    hot-reloaded without server restarts.  The dict-based implementation here
    is a transparent stand-in that can be swapped in a single method.

Canary routing:
    The router supports an optional canary weight per tenant so that a new
    model version can be shadow-tested on a percentage of real traffic before
    a full rollout.  When a canary is configured, CANARY_WEIGHT % of requests
    are routed to the candidate model; the remainder go to the stable model.

Usage:
    router    = TenantRouterAgent()
    model_path = router.get_model_path("bank_a")
================================================================================
"""

import logging
import random
from typing import Dict, Optional

logger = logging.getLogger("platform.tenant_router")


# ── Tenant registry ──────────────────────────────────────────────────────────
# Maps tenant_id → configuration dict.
#
# Keys per tenant:
#   stable_model  — path to the primary (fully-released) model artefact
#   canary_model  — path to the candidate model artefact (optional)
#   canary_weight — fraction of traffic routed to canary  (0.0 – 1.0)
#
# In production: load this from a config DB; hot-reload on change events.
# ────────────────────────────────────────────────────────────────────────────

TENANT_REGISTRY: Dict[str, Dict] = {
    "bank_a": {
        "stable_model":  "models/bank_a_fraud_model.pkl",
        "canary_model":  None,
        "canary_weight": 0.0,
    },
    "bank_b": {
        "stable_model":  "models/bank_b_fraud_model.pkl",
        "canary_model":  "models/bank_b_fraud_model_v2.pkl",
        "canary_weight": 0.1,   # 10 % of bank_b traffic goes to v2 (canary)
    },
    "bank_c": {
        "stable_model":  "models/bank_c_fraud_model.pkl",
        "canary_model":  None,
        "canary_weight": 0.0,
    },
}


class TenantRouterAgent:
    """
    Routes inference requests to per-tenant model artefacts.

    Supports canary deployments where a candidate model receives a configurable
    slice of production traffic for shadow-testing before full rollout.

    Attributes:
        _registry:  Dict of tenant configurations (stable model path, canary
                    model path, canary traffic weight).  Defaults to the
                    module-level TENANT_REGISTRY constant.
    """

    def __init__(self, registry: Optional[Dict] = None) -> None:
        # Allow injection of a custom registry for testing.
        self._registry = registry if registry is not None else TENANT_REGISTRY

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def get_model_path(self, tenant_id: str) -> str:
        """
        Return the model artefact path that should serve this request.

        When a canary model is configured, traffic is split according to
        canary_weight using a simple uniform random draw.

        Args:
            tenant_id:  The unique identifier for the requesting tenant.

        Returns:
            Filesystem path to the serialised model artefact.

        Raises:
            ValueError:  If tenant_id is not registered in the registry.
        """
        config = self._lookup_tenant(tenant_id)

        # ── Canary routing ────────────────────────────────────────────────────
        if config.get("canary_model") and config.get("canary_weight", 0) > 0:
            if random.random() < config["canary_weight"]:
                logger.debug(
                    "Routing tenant '%s' to canary model: %s",
                    tenant_id, config["canary_model"],
                )
                return config["canary_model"]

        # ── Stable routing ────────────────────────────────────────────────────
        logger.debug(
            "Routing tenant '%s' to stable model: %s",
            tenant_id, config["stable_model"],
        )
        return config["stable_model"]

    def register_tenant(
        self,
        tenant_id:     str,
        stable_model:  str,
        canary_model:  Optional[str]  = None,
        canary_weight: float          = 0.0,
    ) -> None:
        """
        Register a new tenant or update an existing one at runtime.

        Args:
            tenant_id:     Unique tenant identifier.
            stable_model:  Path to the stable model artefact.
            canary_model:  Path to the canary model artefact (optional).
            canary_weight: Fraction of traffic to send to canary (0.0 – 1.0).
        """
        if not 0.0 <= canary_weight <= 1.0:
            raise ValueError(f"canary_weight must be in [0, 1]; got {canary_weight}")

        self._registry[tenant_id] = {
            "stable_model":  stable_model,
            "canary_model":  canary_model,
            "canary_weight": canary_weight,
        }
        logger.info("Tenant '%s' registered/updated in routing table", tenant_id)

    def deregister_tenant(self, tenant_id: str) -> None:
        """Remove a tenant from the routing table (graceful offboarding)."""
        if tenant_id not in self._registry:
            raise ValueError(f"Tenant '{tenant_id}' not found in routing table")
        del self._registry[tenant_id]
        logger.info("Tenant '%s' deregistered from routing table", tenant_id)

    def list_tenants(self) -> Dict[str, Dict]:
        """Return a copy of the current routing table (safe for external use)."""
        return dict(self._registry)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _lookup_tenant(self, tenant_id: str) -> Dict:
        """
        Look up tenant configuration and raise a clear error if not found.

        Args:
            tenant_id:  Tenant identifier to look up.

        Returns:
            Tenant configuration dict.

        Raises:
            ValueError:  If the tenant is not registered.
        """
        config = self._registry.get(tenant_id)
        if config is None:
            registered = list(self._registry.keys())
            raise ValueError(
                f"Unknown tenant '{tenant_id}'.  "
                f"Registered tenants: {registered}"
            )
        return config
