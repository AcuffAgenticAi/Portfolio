"""
================================================================================
pipelines/deployment_pipeline.py  —  Model Deployment Pipeline
================================================================================

Purpose:
    Orchestrates the end-to-end steps required to deploy a trained model
    artefact as a live REST API endpoint.

    Deployment steps:
        1. Validate the model artefact exists and is loadable.
        2. Run a smoke-test prediction to verify the model is functional.
        3. Build the FastAPI application via ModelDeploymentAgent.
        4. Start the uvicorn ASGI server on the configured host/port.

    The pipeline is called either:
        • Directly from main.py in a subprocess (normal platform startup)
        • By a CI/CD job after a successful retraining run

Pre-deployment validation:
    A configurable smoke-test payload is run against the freshly loaded model
    before the server socket is opened.  If the smoke test fails the pipeline
    raises immediately so the bad model never receives production traffic.

Usage:
    from pipelines.deployment_pipeline import deploy_model
    deploy_model()                          # uses Settings defaults
    deploy_model(model_path="models/v2.pkl")  # explicit path override
================================================================================
"""

import logging
import os
from typing import Optional

import joblib
import uvicorn

from agents.model_deployment_agent import create_api
from config.settings import Settings

logger = logging.getLogger("platform.deployment_pipeline")

# Smoke-test payload: a representative feature dict used to verify the model
# can produce a prediction without raising an exception.
_SMOKE_TEST_FEATURES = {
    "transaction_amount": 250.0,
    "account_age":        365,
    "num_transactions":   5,
    "merchant_risk":      2,
}


def deploy_model(model_path: Optional[str] = None) -> None:
    """
    Full deployment pipeline: validate → smoke-test → serve.

    Blocks indefinitely once uvicorn starts (designed to run in a subprocess
    or as the main process in a container).

    Args:
        model_path:  Override for the model artefact path.  Defaults to
                     Settings.MODEL_PATH when not supplied.

    Raises:
        FileNotFoundError:  If the model artefact does not exist on disk.
        RuntimeError:       If the smoke-test prediction fails.
    """
    settings   = Settings()
    model_path = model_path or settings.MODEL_PATH

    logger.info("=" * 60)
    logger.info("Deployment pipeline starting  model='%s'", model_path)
    logger.info("=" * 60)

    # ── Step 1: Validate artefact exists on disk ──────────────────────────────
    _validate_artefact(model_path)

    # ── Step 2: Smoke-test the model before opening the server socket ─────────
    _smoke_test_model(model_path)

    # ── Step 3: Build the FastAPI application ─────────────────────────────────
    logger.info("Building FastAPI application…")
    app = create_api(model_path)

    # ── Step 4: Start the ASGI server (blocks until process is killed) ────────
    logger.info(
        "Starting uvicorn server  host=%s  port=%d",
        settings.API_HOST, settings.API_PORT,
    )
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        # access_log=False keeps uvicorn's own request logs quiet because
        # the FastAPI middleware in model_deployment_agent.py already logs them.
        access_log=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_artefact(model_path: str) -> None:
    """
    Confirm the model artefact exists and is a non-empty file.

    Args:
        model_path:  Filesystem path to check.

    Raises:
        FileNotFoundError:  If the file is missing or empty.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model artefact not found at '{model_path}'.  "
            "Run the retraining pipeline first: python main.py --retrain-only"
        )

    if os.path.getsize(model_path) == 0:
        raise FileNotFoundError(
            f"Model artefact at '{model_path}' is empty — it may be corrupt."
        )

    logger.info("Artefact validation passed: '%s'", model_path)


def _smoke_test_model(model_path: str) -> None:
    """
    Load the model and run a single test prediction.

    This catches common failure modes (schema mismatch, corrupt weights,
    missing feature columns) before any traffic hits the endpoint.

    Args:
        model_path:  Path to the serialised model artefact.

    Raises:
        RuntimeError:  If the smoke-test prediction raises any exception.
    """
    logger.info("Running pre-deployment smoke test…")
    try:
        import pandas as pd
        model  = joblib.load(model_path)
        input_df = pd.DataFrame([_SMOKE_TEST_FEATURES])
        result = model.predict(input_df)
        logger.info("Smoke test PASSED  result=%s", result)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Pre-deployment smoke test FAILED for '{model_path}': {exc}"
        ) from exc
