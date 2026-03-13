"""
================================================================================
config/settings.py  —  Centralised platform configuration
================================================================================

Purpose:
    Single source of truth for every tuneable parameter in the platform.
    Values are read from environment variables so that the same codebase can
    run in local dev, staging, and production without code changes — only the
    environment differs.

    All settings have safe local-dev defaults so the platform runs out of the
    box with `python main.py` on a developer laptop.

Usage:
    from config.settings import Settings
    cfg = Settings()
    print(cfg.MODEL_PATH)

Environment variables:
    ENV                 deployment environment: local | staging | production
    MODEL_PATH          path to the primary serialised model artefact
    BASELINE_DATA_PATH  path to baseline feature CSV (used by drift detection)
    CURRENT_DATA_PATH   path to current-window feature CSV
    LABEL_COLUMN        name of the target/label column in data CSVs
    MONITOR_INTERVAL    seconds between monitoring loop iterations
    DRIFT_P_THRESHOLD   p-value below which drift is flagged (KS test)
    ACCURACY_THRESHOLD  minimum acceptable accuracy before retraining alert
    LOG_LEVEL           Python logging level string (DEBUG | INFO | WARNING …)
    ACTIVE_TENANTS      comma-separated list of active tenant IDs
================================================================================
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    """
    Immutable platform configuration loaded from environment variables.

    Attributes are read once at construction time so the object is safe to
    share across threads and subprocesses.
    """

    # ── Environment ──────────────────────────────────────────────────────────
    ENV: str = field(default_factory=lambda: os.getenv("ENV", "local"))

    # ── Model artefact paths ─────────────────────────────────────────────────
    MODEL_PATH: str = field(
        default_factory=lambda: os.getenv("MODEL_PATH", "models/fraud_model.pkl")
    )

    # ── Data paths (feature store stand-ins) ─────────────────────────────────
    BASELINE_DATA_PATH: str = field(
        default_factory=lambda: os.getenv(
            "BASELINE_DATA_PATH", "data/baseline_transactions.csv"
        )
    )
    CURRENT_DATA_PATH: str = field(
        default_factory=lambda: os.getenv(
            "CURRENT_DATA_PATH", "data/current_transactions.csv"
        )
    )
    LABEL_COLUMN: str = field(
        default_factory=lambda: os.getenv("LABEL_COLUMN", "label")
    )

    # ── Monitoring behaviour ──────────────────────────────────────────────────
    MONITOR_INTERVAL: int = field(
        default_factory=lambda: int(os.getenv("MONITOR_INTERVAL", "60"))
    )

    # ── Drift detection thresholds ────────────────────────────────────────────
    # KS-test p-value: if p < DRIFT_P_THRESHOLD the distribution has shifted.
    DRIFT_P_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_P_THRESHOLD", "0.05"))
    )

    # ── Performance thresholds ────────────────────────────────────────────────
    # Alert and optionally trigger retrain if accuracy falls below this floor.
    ACCURACY_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("ACCURACY_THRESHOLD", "0.85"))
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # ── Multi-tenancy ─────────────────────────────────────────────────────────
    # Comma-separated list of tenant IDs that the platform currently serves.
    ACTIVE_TENANTS: List[str] = field(
        default_factory=lambda: os.getenv("ACTIVE_TENANTS", "bank_a,bank_b").split(",")
    )

    # ── API server ────────────────────────────────────────────────────────────
    API_HOST: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    API_PORT: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    def __post_init__(self) -> None:
        """Validate critical settings immediately after construction."""
        if self.MONITOR_INTERVAL < 10:
            raise ValueError(
                f"MONITOR_INTERVAL must be >= 10 seconds; got {self.MONITOR_INTERVAL}"
            )
        if not (0 < self.DRIFT_P_THRESHOLD < 1):
            raise ValueError(
                f"DRIFT_P_THRESHOLD must be in (0, 1); got {self.DRIFT_P_THRESHOLD}"
            )
        if not (0 < self.ACCURACY_THRESHOLD <= 1):
            raise ValueError(
                f"ACCURACY_THRESHOLD must be in (0, 1]; got {self.ACCURACY_THRESHOLD}"
            )
