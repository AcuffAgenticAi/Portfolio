"""
================================================================================
agents/performance_monitor_agent.py  —  Performance Monitoring Agent
================================================================================

Purpose:
    Evaluates a deployed model's predictive quality against recent ground-truth
    labels.  Computes a standard classification scorecard and persists every
    run to the metrics log for trend analysis and alerting.

Metrics computed:
    • Accuracy   — fraction of correct predictions
    • Precision  — TP / (TP + FP)  per class, then macro-averaged
    • Recall     — TP / (TP + FN)  per class, then macro-averaged
    • F1 Score   — harmonic mean of precision and recall (macro)
    • Confusion matrix (serialised as nested list for JSON compatibility)

Threshold alerting:
    If accuracy falls below Settings.ACCURACY_THRESHOLD an alert dict is
    appended to the returned report and a WARNING is emitted to the log.
    In production this hook can be extended to publish to PagerDuty / Slack.

Usage:
    agent  = PerformanceMonitorAgent()
    report = agent.evaluate(predictions=[0, 1, 1, 0], actuals=[0, 1, 0, 0])
================================================================================
"""

import logging
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings   import Settings
from monitoring.metrics_tracker import MetricsTracker

logger = logging.getLogger("platform.performance_monitor")


class PerformanceMonitorAgent:
    """
    Evaluates model predictions against ground-truth labels and logs results.

    Attributes:
        metrics_tracker:  Persistence layer for the metrics log (JSON-lines).
        accuracy_threshold:  Minimum acceptable accuracy before an alert fires.
    """

    def __init__(self) -> None:
        settings              = Settings()
        self.metrics_tracker  = MetricsTracker()
        self.accuracy_threshold = settings.ACCURACY_THRESHOLD

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        predictions: List[Any],
        actuals:     List[Any],
    ) -> Dict[str, Any]:
        """
        Compute a classification scorecard and persist it to the metrics log.

        Args:
            predictions:  Sequence of values predicted by the deployed model.
            actuals:      Corresponding ground-truth labels.

        Returns:
            dict with keys: accuracy, precision, recall, f1, confusion_matrix,
            total_predictions, alerts.

        Raises:
            ValueError:  If predictions and actuals have different lengths.
        """
        if len(predictions) != len(actuals):
            raise ValueError(
                f"Length mismatch: predictions={len(predictions)}, actuals={len(actuals)}"
            )

        # ── Compute metrics ───────────────────────────────────────────────────
        accuracy  = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, average="macro", zero_division=0)
        recall    = recall_score(actuals, predictions,    average="macro", zero_division=0)
        f1        = f1_score(actuals, predictions,        average="macro", zero_division=0)
        cm        = confusion_matrix(actuals, predictions).tolist()

        report: Dict[str, Any] = {
            "accuracy":           round(accuracy,  4),
            "precision":          round(precision, 4),
            "recall":             round(recall,    4),
            "f1":                 round(f1,        4),
            "confusion_matrix":   cm,
            "total_predictions":  len(predictions),
            "alerts":             [],
        }

        # ── Threshold alerting ────────────────────────────────────────────────
        report["alerts"] = self._check_thresholds(report)

        # ── Persist to metrics log ────────────────────────────────────────────
        self.metrics_tracker.log_metrics(report)

        logger.info(
            "Performance evaluation complete  accuracy=%.4f  precision=%.4f  "
            "recall=%.4f  f1=%.4f  n=%d",
            accuracy, precision, recall, f1, len(predictions),
        )

        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _check_thresholds(self, report: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Compare metric values against configured thresholds.

        Returns a (possibly empty) list of alert dicts.  Each alert has:
            metric, value, threshold, severity, message
        """
        alerts = []

        if report["accuracy"] < self.accuracy_threshold:
            msg = (
                f"Accuracy {report['accuracy']:.4f} is below threshold "
                f"{self.accuracy_threshold:.4f} — consider retraining"
            )
            logger.warning("ALERT: %s", msg)
            alerts.append({
                "metric":    "accuracy",
                "value":     report["accuracy"],
                "threshold": self.accuracy_threshold,
                "severity":  "warning",
                "message":   msg,
            })

        return alerts
