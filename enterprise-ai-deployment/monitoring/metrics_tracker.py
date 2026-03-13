"""
================================================================================
monitoring/metrics_tracker.py  —  Metrics Persistence Layer
================================================================================

Purpose:
    Provides a simple, append-only JSON-lines log for model performance metrics.
    Every monitoring cycle writes one record so the log becomes a time-series
    of model health that can be queried, graphed, or exported to Prometheus /
    Grafana in a production deployment.

Log format (one JSON object per line):
    {
        "timestamp": "2025-06-01T14:32:01.123456",
        "metrics": {
            "accuracy":          0.9234,
            "precision":         0.8812,
            "recall":            0.9011,
            "f1":                0.8910,
            "total_predictions": 1024,
            "alerts":            []
        }
    }

Why JSON-lines?
    • Human-readable and easily parsed by grep / jq / pandas
    • Supports streaming ingestion (no need to read the whole file)
    • Append-only writes are safe under concurrent access

Production integration points:
    • Export to Prometheus by exposing a /metrics endpoint that reads the log
    • Forward records to Elasticsearch / Splunk for alerting
    • Stream to a data warehouse via a Fluentd / Logstash pipeline

Usage:
    tracker = MetricsTracker()
    tracker.log_metrics({"accuracy": 0.92, "total_predictions": 500})
    history = tracker.load_history()
================================================================================
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("platform.metrics_tracker")

# Default log file path.  Created automatically if it does not exist.
DEFAULT_LOG_PATH = "monitoring/metrics_log.jsonl"


class MetricsTracker:
    """
    Append-only JSON-lines store for model performance metrics.

    Attributes:
        log_path:  Filesystem path to the metrics log file.
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        # Ensure the parent directory exists so the first write never fails.
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Write
    # ─────────────────────────────────────────────────────────────────────────

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Append a metrics record to the log file.

        Each record is wrapped with a UTC timestamp and serialised as a single
        JSON line.  The file is opened in append mode so concurrent writers
        are safe as long as each write is a single os.write() call (which
        Python's 'a' mode guarantees on POSIX systems).

        Args:
            metrics:  Dict of metric name → value to persist.
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics":   metrics,
        }

        line = json.dumps(record, default=str) + "\n"

        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
            logger.debug("Metrics logged to '%s'", self.log_path)
        except OSError as exc:
            logger.error("Failed to write metrics log: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Read
    # ─────────────────────────────────────────────────────────────────────────

    def load_history(
        self,
        last_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load metrics records from the log file.

        Args:
            last_n:  If provided, return only the most recent N records.
                     Returns all records when None (default).

        Returns:
            List of record dicts in chronological order.
        """
        if not self.log_path.exists():
            logger.warning("Metrics log '%s' does not exist yet", self.log_path)
            return []

        records: List[Dict[str, Any]] = []

        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON on line %d: %s", line_num, exc
                    )

        if last_n is not None:
            records = records[-last_n:]

        return records

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recent metrics record, or None if the log is empty.

        Useful for the health dashboard which only needs the latest snapshot.
        """
        history = self.load_history(last_n=1)
        return history[0] if history else None

    # ─────────────────────────────────────────────────────────────────────────
    # Maintenance
    # ─────────────────────────────────────────────────────────────────────────

    def rotate_log(self, archive_suffix: Optional[str] = None) -> None:
        """
        Rename the current log file to an archive and start a fresh log.

        Args:
            archive_suffix:  String appended to the archived filename.
                             Defaults to the current UTC timestamp.
        """
        if not self.log_path.exists():
            return

        suffix   = archive_suffix or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        archive  = self.log_path.with_suffix(f".{suffix}.jsonl")
        os.rename(self.log_path, archive)
        logger.info("Log rotated: '%s' → '%s'", self.log_path, archive)
