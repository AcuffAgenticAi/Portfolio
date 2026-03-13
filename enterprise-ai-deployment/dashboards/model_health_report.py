"""
================================================================================
dashboards/model_health_report.py  —  Model Health Report Generator
================================================================================

Purpose:
    Reads the metrics log and renders a human-readable health report to stdout
    (and optionally to a file).  Acts as the CLI companion to the Streamlit
    dashboard for environments where a GUI is unavailable (CI pipelines,
    server SSH sessions, automated alert emails).

Report sections:
    1. Summary line  — timestamp, model version, pass/fail status
    2. Metrics table — accuracy, precision, recall, F1
    3. Active alerts — any threshold violations flagged by PerformanceMonitor
    4. Trend line    — accuracy over the last N monitoring cycles

Usage:
    # Programmatic (called from main.py monitoring loop):
    from dashboards.model_health_report import generate_report
    generate_report()

    # CLI (run standalone):
    python dashboards/model_health_report.py
    python dashboards/model_health_report.py --last-n 10
================================================================================
"""

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional

from monitoring.metrics_tracker import MetricsTracker

logger = logging.getLogger("platform.health_report")

# ── Constants ─────────────────────────────────────────────────────────────────
REPORT_SEPARATOR = "─" * 60
PASS_ICON = "✓"
FAIL_ICON = "✗"
WARN_ICON = "⚠"


def generate_report(
    last_n:   int           = 1,
    out_file: Optional[str] = None,
) -> Optional[str]:
    """
    Generate and print a model health report from the metrics log.

    Args:
        last_n:    Number of most-recent monitoring cycles to include in the
                   trend section.  Pass 1 for a single-snapshot report.
        out_file:  If provided, also write the report to this file path in
                   addition to printing to stdout.

    Returns:
        The rendered report string, or None if the log is empty.
    """
    tracker = MetricsTracker()
    history = tracker.load_history(last_n=last_n)

    if not history:
        msg = "No metrics data available yet.  Run a monitoring cycle first."
        logger.warning(msg)
        print(msg)
        return None

    latest  = history[-1]
    report  = _render_report(latest, trend_records=history)

    # ── Output ────────────────────────────────────────────────────────────────
    print(report)

    if out_file:
        try:
            with open(out_file, "w", encoding="utf-8") as fh:
                fh.write(report)
            logger.info("Health report written to '%s'", out_file)
        except OSError as exc:
            logger.error("Failed to write report to '%s': %s", out_file, exc)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_report(
    latest:        Dict[str, Any],
    trend_records: List[Dict[str, Any]],
) -> str:
    """Build the full report string from the latest metrics record."""
    metrics = latest.get("metrics", {})
    alerts  = metrics.get("alerts", [])
    status  = FAIL_ICON if alerts else PASS_ICON

    lines = [
        "",
        REPORT_SEPARATOR,
        "  ENTERPRISE AI PLATFORM  —  MODEL HEALTH REPORT",
        REPORT_SEPARATOR,
        f"  Timestamp : {latest.get('timestamp', 'N/A')}",
        f"  Status    : {status}  {'ALERTS ACTIVE' if alerts else 'ALL SYSTEMS NOMINAL'}",
        REPORT_SEPARATOR,
        "",
        "  PERFORMANCE METRICS",
        f"  {'Metric':<22} {'Value':>10}",
        f"  {'-'*22} {'-'*10}",
        f"  {'Accuracy':<22} {_fmt_pct(metrics.get('accuracy'))}",
        f"  {'Precision (macro)':<22} {_fmt_pct(metrics.get('precision'))}",
        f"  {'Recall (macro)':<22} {_fmt_pct(metrics.get('recall'))}",
        f"  {'F1 Score (macro)':<22} {_fmt_pct(metrics.get('f1'))}",
        f"  {'Total Predictions':<22} {metrics.get('total_predictions', 'N/A'):>10}",
        "",
    ]

    # ── Alerts section ────────────────────────────────────────────────────────
    if alerts:
        lines += [
            "  ACTIVE ALERTS",
            f"  {'-'*56}",
        ]
        for alert in alerts:
            lines.append(
                f"  {WARN_ICON}  [{alert.get('severity', '?').upper()}] "
                f"{alert.get('message', '')}"
            )
        lines.append("")

    # ── Accuracy trend ────────────────────────────────────────────────────────
    if len(trend_records) > 1:
        lines += [
            "  ACCURACY TREND  (most-recent last)",
            f"  {'-'*56}",
        ]
        for record in trend_records:
            ts  = record.get("timestamp", "?")[:19]   # trim to seconds
            acc = record["metrics"].get("accuracy")
            bar = _spark_bar(acc)
            lines.append(f"  {ts}  {bar}  {_fmt_pct(acc)}")
        lines.append("")

    lines += [REPORT_SEPARATOR, ""]
    return "\n".join(lines)


def _fmt_pct(value: Optional[float]) -> str:
    """Format a float as a right-aligned percentage string."""
    if value is None:
        return f"{'N/A':>10}"
    return f"{value * 100:>9.2f}%"


def _spark_bar(value: Optional[float], width: int = 20) -> str:
    """Render a simple ASCII progress bar for a value in [0, 1]."""
    if value is None:
        return "?" * width
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="model_health_report.py",
        description="Print a model health report from the metrics log.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=5,
        help="Number of most-recent cycles to include in trend (default: 5)",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Optional file path to also write the report to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    result = generate_report(last_n=args.last_n, out_file=args.out_file)
    sys.exit(0 if result else 1)
