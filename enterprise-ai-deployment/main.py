"""
================================================================================
Enterprise AI Model Deployment & Monitoring Agent  —  main.py
================================================================================

Purpose:
    Top-level orchestrator for the MLOps platform.  Bootstraps every subsystem,
    starts the FastAPI inference server in a background process, then enters a
    continuous monitoring loop that detects drift and fires retraining when
    necessary.

Architecture overview:
    main.py
      ├── deployment_pipeline     →  FastAPI inference server (port 8000)
      ├── DriftDetectionAgent     →  KS-test per feature column
      ├── PerformanceMonitorAgent →  accuracy / precision / recall metrics
      ├── RetrainingTriggerAgent  →  kicks off retraining_pipeline on drift
      ├── TenantRouterAgent       →  routes requests to per-tenant models
      └── UsageMeter              →  per-tenant billing simulation

Environment variables (see config/settings.py for full list):
    MODEL_PATH          Serialised model artefact    (default: models/fraud_model.pkl)
    MONITOR_INTERVAL    Seconds between monitor runs (default: 60)
    LOG_LEVEL           Python logging level          (default: INFO)

Usage:
    python main.py                   # full platform startup
    python main.py --monitor-only    # skip deployment; run monitoring loop only
    python main.py --retrain-only    # force one immediate retrain cycle then exit
================================================================================
"""

import argparse
import logging
import multiprocessing
import sys
import time

import pandas as pd

# ── Internal subsystems ──────────────────────────────────────────────────────
from agents.drift_detection_agent      import DriftDetectionAgent
from agents.performance_monitor_agent  import PerformanceMonitorAgent
from agents.retraining_trigger_agent   import RetrainingTriggerAgent
from agents.tenant_router_agent        import TenantRouterAgent
from billing.usage_meter               import UsageMeter
from config.settings                   import Settings
from dashboards.model_health_report    import generate_report
from pipelines.deployment_pipeline     import deploy_model
from pipelines.retraining_pipeline     import retrain_model

# ── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("monitoring/platform.log", mode="a"),
    ],
)
logger = logging.getLogger("platform.main")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: launch the FastAPI server in a separate OS process so it does not
# block the monitoring loop that runs in the main process.
# ─────────────────────────────────────────────────────────────────────────────
def _start_api_server() -> multiprocessing.Process:
    """Spawn the FastAPI inference server as a non-blocking background process."""
    proc = multiprocessing.Process(target=deploy_model, daemon=True)
    proc.start()
    logger.info("API server process started (PID %d)", proc.pid)
    return proc


# ─────────────────────────────────────────────────────────────────────────────
# Core monitoring loop
# ─────────────────────────────────────────────────────────────────────────────
def run_monitoring_loop(settings: "Settings", one_shot: bool = False) -> None:
    """
    Continuous monitoring loop.

    Each iteration:
      1. Loads baseline and current feature data from configured paths.
      2. Runs KS-based drift detection across every feature column.
      3. Evaluates recent prediction accuracy against ground-truth labels.
      4. Checks whether retraining should be triggered.
      5. Records per-tenant usage for billing.
      6. Writes a health report snapshot to the monitoring log.

    Args:
        settings:  Platform configuration object (see config/settings.py).
        one_shot:  When True, execute exactly one iteration and return.
                   Used internally by the --retrain-only CLI flag.
    """
    # Instantiate agents once; they are stateless across iterations.
    drift_agent   = DriftDetectionAgent()
    perf_agent    = PerformanceMonitorAgent()
    retrain_agent = RetrainingTriggerAgent()
    tenant_router = TenantRouterAgent()
    billing       = UsageMeter()

    logger.info("Monitoring loop started  (interval=%ds)", settings.MONITOR_INTERVAL)

    while True:
        logger.info("── Monitor cycle begin ──────────────────────────────────")

        # ------------------------------------------------------------------
        # 1. Load feature data
        #    In production these would come from a feature store or data lake.
        #    CSV reads serve as a portable local stand-in.
        # ------------------------------------------------------------------
        try:
            baseline_df = pd.read_csv(settings.BASELINE_DATA_PATH)
            current_df  = pd.read_csv(settings.CURRENT_DATA_PATH)
        except FileNotFoundError as exc:
            logger.error("Data file not found: %s — skipping cycle", exc)
            _sleep_or_exit(one_shot, settings.MONITOR_INTERVAL)
            continue

        # ------------------------------------------------------------------
        # 2. Drift detection  (Kolmogorov–Smirnov test, per feature column)
        # ------------------------------------------------------------------
        feature_cols  = [c for c in baseline_df.columns if c != settings.LABEL_COLUMN]
        drift_report  = drift_agent.detect_drift(
            baseline_data=baseline_df[feature_cols],
            current_data=current_df[feature_cols],
        )
        logger.info("Drift report: %s", drift_report)

        # ------------------------------------------------------------------
        # 3. Performance evaluation
        #    Requires ground-truth labels present in the current dataset.
        # ------------------------------------------------------------------
        if settings.LABEL_COLUMN in current_df.columns:
            perf_report = perf_agent.evaluate(
                predictions=current_df.get("prediction", current_df[settings.LABEL_COLUMN]),
                actuals=current_df[settings.LABEL_COLUMN],
            )
            logger.info("Performance report: %s", perf_report)

        # ------------------------------------------------------------------
        # 4. Retraining trigger
        # ------------------------------------------------------------------
        trigger_result = retrain_agent.check_and_trigger(drift_report)
        logger.info("Retraining check: %s", trigger_result)

        # ------------------------------------------------------------------
        # 5. Per-tenant usage recording + billing snapshot
        # ------------------------------------------------------------------
        for tenant_id in settings.ACTIVE_TENANTS:
            try:
                tenant_router.get_model_path(tenant_id)   # validates route exists
                billing.record_usage(tenant_id)
            except ValueError as exc:
                logger.warning("Tenant routing error: %s", exc)

        logger.info("Billing snapshot: %s", billing.generate_invoice())

        # ------------------------------------------------------------------
        # 6. Write health report to monitoring log
        # ------------------------------------------------------------------
        generate_report()

        logger.info("── Monitor cycle complete ───────────────────────────────")
        _sleep_or_exit(one_shot, settings.MONITOR_INTERVAL)


def _sleep_or_exit(one_shot: bool, interval: int) -> None:
    """Sleep between monitoring cycles, or exit immediately when one_shot=True."""
    if one_shot:
        logger.info("one_shot mode — exiting after single monitoring cycle")
        sys.exit(0)
    time.sleep(interval)


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Enterprise AI Model Deployment & Monitoring Agent",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--monitor-only",
        action="store_true",
        help="Skip model deployment; start the monitoring loop only.",
    )
    group.add_argument(
        "--retrain-only",
        action="store_true",
        help="Force one retrain cycle then exit (ignores drift thresholds).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Platform entry point — bootstraps and runs the MLOps platform."""
    args     = parse_args()
    settings = Settings()

    logger.info("=" * 72)
    logger.info("Enterprise AI MLOps Platform  —  starting up")
    logger.info("Model path : %s", settings.MODEL_PATH)
    logger.info("Environment: %s", settings.ENV)
    logger.info("=" * 72)

    if args.retrain_only:
        # Useful for scheduled CI/CD jobs that need a forced retrain.
        logger.info("--retrain-only: forcing immediate retraining cycle")
        retrain_model()
        logger.info("Retrain complete — exiting")
        return

    if not args.monitor_only:
        # Launch inference server in background; wait briefly for port bind.
        _start_api_server()
        time.sleep(2)

    run_monitoring_loop(settings, one_shot=args.retrain_only)


if __name__ == "__main__":
    main()
