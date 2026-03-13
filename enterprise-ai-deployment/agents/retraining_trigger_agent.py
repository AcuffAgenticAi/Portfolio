"""
================================================================================
agents/retraining_trigger_agent.py  —  Retraining Trigger Agent
================================================================================

Purpose:
    Examines the drift report produced by DriftDetectionAgent and decides
    whether to initiate a retraining cycle.  Acts as the decision layer
    between detection and action — keeping that logic out of both the drift
    agent (which only measures) and the retraining pipeline (which only trains).

Trigger logic:
    Retraining is initiated when ANY feature column shows drift_detected=True.
    In a production system this can be extended to require:
        • N features drifting simultaneously
        • Accuracy dropping below Settings.ACCURACY_THRESHOLD
        • A minimum cool-down period between retrain runs
        • Manual approval gate (human-in-the-loop)

Retraining is idempotent from the caller's perspective: if no drift is
detected the function returns early without touching the model on disk.

Usage:
    trigger = RetrainingTriggerAgent()
    result  = trigger.check_and_trigger(drift_report)
    # result → "retraining_triggered" | "model_stable" | "error:<detail>"
================================================================================
"""

import logging
import time
from typing import Any, Dict

from pipelines.retraining_pipeline import retrain_model

logger = logging.getLogger("platform.retraining_trigger")


class RetrainingTriggerAgent:
    """
    Decision agent that triggers model retraining when drift is detected.

    Attributes:
        _last_retrain_ts:  Unix timestamp of the most recent retraining run.
                           Used to enforce a minimum cool-down between runs
                           and prevent thrashing in high-drift environments.
        _cool_down_secs:   Minimum seconds between consecutive retrain runs
                           (default: 3600 = 1 hour).
    """

    _COOL_DOWN_DEFAULT = 3_600  # 1 hour

    def __init__(self, cool_down_secs: int = _COOL_DOWN_DEFAULT) -> None:
        self._last_retrain_ts: float = 0.0
        self._cool_down_secs:  int   = cool_down_secs

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def check_and_trigger(self, drift_report: Dict[str, Any]) -> str:
        """
        Evaluate the drift report and trigger retraining if warranted.

        Args:
            drift_report:  Output of DriftDetectionAgent.detect_drift().
                           Must contain a ``_summary`` key with ``any_drift``.

        Returns:
            "retraining_triggered"  — retrain pipeline was invoked.
            "model_stable"          — no drift detected; no action taken.
            "cool_down_active"      — drift detected but cool-down not expired.
            "error:<detail>"        — retraining raised an exception.
        """
        # Use the pre-computed summary block when available; fall back to
        # scanning individual feature entries for backward compatibility.
        any_drift = self._any_drift_detected(drift_report)

        if not any_drift:
            logger.info("No drift detected — model is stable; no retraining required")
            return "model_stable"

        # ── Cool-down guard ───────────────────────────────────────────────────
        secs_since_last = time.time() - self._last_retrain_ts
        if secs_since_last < self._cool_down_secs:
            remaining = int(self._cool_down_secs - secs_since_last)
            logger.warning(
                "Drift detected but cool-down active — next retrain in %ds", remaining
            )
            return "cool_down_active"

        # ── Initiate retraining ───────────────────────────────────────────────
        drifted_features = self._drifted_features(drift_report)
        logger.warning(
            "Drift detected in feature(s): %s — initiating retraining",
            drifted_features,
        )

        try:
            retrain_model()
            self._last_retrain_ts = time.time()
            logger.info("Retraining pipeline completed successfully")
            return "retraining_triggered"
        except Exception as exc:  # noqa: BLE001
            logger.error("Retraining pipeline failed: %s", exc, exc_info=True)
            return f"error:{exc}"

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _any_drift_detected(drift_report: Dict[str, Any]) -> bool:
        """
        Return True if any feature in the report shows drift_detected=True.

        Prefers the pre-computed ``_summary.any_drift`` field for efficiency.
        Falls back to iterating feature entries if the summary is absent.
        """
        if "_summary" in drift_report:
            return drift_report["_summary"].get("any_drift", False)

        # Fallback: scan individual feature entries.
        return any(
            v.get("drift_detected", False)
            for k, v in drift_report.items()
            if isinstance(v, dict) and k != "_summary"
        )

    @staticmethod
    def _drifted_features(drift_report: Dict[str, Any]) -> list:
        """Return a list of feature names where drift was detected."""
        return [
            k
            for k, v in drift_report.items()
            if isinstance(v, dict) and v.get("drift_detected", False) and k != "_summary"
        ]
