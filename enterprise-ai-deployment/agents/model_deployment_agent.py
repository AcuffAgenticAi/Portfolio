"""
================================================================================
agents/model_deployment_agent.py  —  Model Deployment Agent
================================================================================

Purpose:
    Loads a serialised ML model from disk and exposes it as a production-grade
    REST API via FastAPI.  The agent handles:

        • Model loading with error-safe fallback logging
        • Health / readiness / liveness endpoints (Kubernetes-compatible)
        • Single-sample and batch prediction endpoints
        • Per-request structured logging for observability
        • Request validation via Pydantic schemas
        • Graceful error handling with RFC 7807 problem-detail responses

Endpoints:
    GET  /healthz          → liveness probe   (always returns 200 if process alive)
    GET  /readyz           → readiness probe  (returns 503 if model not loaded)
    GET  /                 → human-readable status summary
    POST /predict          → single-record prediction
    POST /predict/batch    → batch prediction (list of records)

Integration:
    Used by pipelines/deployment_pipeline.py which calls create_api() and
    passes the resulting FastAPI app instance to uvicorn.
================================================================================
"""

import logging
import time
import uuid
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

logger = logging.getLogger("platform.deployment_agent")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request / response schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Schema for a single inference request.

    All feature values are passed as a flat dict so the schema stays
    model-agnostic.  The model deployment agent converts this to a
    single-row DataFrame before calling model.predict().

    Example payload:
        {
            "features": {
                "transaction_amount": 450.0,
                "account_age":        730,
                "num_transactions":   8,
                "merchant_risk":      3
            }
        }
    """
    features: Dict[str, Any]

    @validator("features")
    def features_must_not_be_empty(cls, v: Dict) -> Dict:  # noqa: N805
        if not v:
            raise ValueError("features dict must contain at least one key")
        return v


class BatchPredictRequest(BaseModel):
    """Schema for a batch inference request (list of feature dicts)."""
    records: List[Dict[str, Any]]

    @validator("records")
    def records_must_not_be_empty(cls, v: List) -> List:  # noqa: N805
        if not v:
            raise ValueError("records list must contain at least one item")
        return v


class PredictResponse(BaseModel):
    """Schema for a single prediction response."""
    request_id:  str
    prediction:  List[Any]
    model_version: str
    latency_ms:  float


class BatchPredictResponse(BaseModel):
    """Schema for a batch prediction response."""
    request_id:   str
    predictions:  List[Any]
    record_count: int
    model_version: str
    latency_ms:   float


# ─────────────────────────────────────────────────────────────────────────────
# Core agent class
# ─────────────────────────────────────────────────────────────────────────────

class ModelDeploymentAgent:
    """
    Wraps a serialised scikit-learn-compatible model and exposes a predict()
    interface used by the FastAPI route handlers.

    The agent is intentionally stateless with respect to HTTP — it holds only
    the loaded model object and a version label derived from the file path.

    Attributes:
        model:          The deserialised model object (sklearn Pipeline or estimator).
        model_path:     Filesystem path the model was loaded from.
        model_version:  Human-readable version string derived from the path.
        _ready:         Boolean flag; True once model is successfully loaded.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path    = model_path
        self.model_version = model_path.replace("/", "_").replace(".pkl", "")
        self._ready        = False
        self.model         = None
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Load the serialised model from disk.

        Uses joblib.load() which handles numpy arrays and sklearn objects
        efficiently via memory-mapped files when available.

        Sets self._ready = True on success, logs the error and leaves
        _ready = False on failure so the readiness probe can surface the
        problem to Kubernetes without crashing the pod.
        """
        try:
            self.model   = joblib.load(self.model_path)
            self._ready  = True
            logger.info("Model loaded successfully from '%s'", self.model_path)
        except FileNotFoundError:
            logger.error("Model file not found: '%s'", self.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load model '%s': %s", self.model_path, exc)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, features: Dict[str, Any]) -> List[Any]:
        """
        Run inference on a single feature record.

        Args:
            features:  Dict mapping feature name → value.

        Returns:
            List containing the model's prediction(s).

        Raises:
            RuntimeError:  If the model has not been loaded successfully.
            ValueError:    If the feature dict is incompatible with the model.
        """
        if not self._ready:
            raise RuntimeError("Model is not ready — cannot serve predictions")

        # Wrap the flat dict in a single-row DataFrame so sklearn Pipelines
        # receive the expected column structure.
        input_df = pd.DataFrame([features])
        predictions = self.model.predict(input_df)
        return predictions.tolist()

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Any]:
        """
        Run inference on a batch of feature records.

        Args:
            records:  List of feature dicts.

        Returns:
            List of predictions in the same order as the input records.
        """
        if not self._ready:
            raise RuntimeError("Model is not ready — cannot serve predictions")

        input_df    = pd.DataFrame(records)
        predictions = self.model.predict(input_df)
        return predictions.tolist()

    @property
    def is_ready(self) -> bool:
        """Return True if the model is loaded and the agent can serve traffic."""
        return self._ready


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_api(model_path: str) -> FastAPI:
    """
    Build and return a configured FastAPI application instance.

    The function creates one ModelDeploymentAgent (model load happens here)
    and registers all HTTP route handlers as closures over that agent instance.

    Args:
        model_path:  Filesystem path to the serialised model (.pkl file).

    Returns:
        Configured FastAPI app ready to be passed to uvicorn.run().
    """
    agent = ModelDeploymentAgent(model_path)

    app = FastAPI(
        title="Enterprise Model Inference API",
        description="Production-grade ML model serving endpoint with monitoring hooks.",
        version="1.0.0",
    )

    # ── Middleware: per-request logging ──────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log every inbound request and its response status + latency."""
        start    = time.perf_counter()
        response = await call_next(request)
        elapsed  = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s  →  %d  (%.1fms)",
            request.method, request.url.path, response.status_code, elapsed,
        )
        return response

    # ── Health / readiness endpoints ─────────────────────────────────────────

    @app.get("/healthz", tags=["Health"], summary="Liveness probe")
    def liveness() -> Dict[str, str]:
        """
        Kubernetes liveness probe.
        Returns 200 as long as the process is running.
        """
        return {"status": "alive"}

    @app.get("/readyz", tags=["Health"], summary="Readiness probe")
    def readiness() -> Dict[str, str]:
        """
        Kubernetes readiness probe.
        Returns 503 if the model has not been loaded; 200 otherwise.
        """
        if not agent.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded — pod is not ready to serve traffic",
            )
        return {"status": "ready", "model_version": agent.model_version}

    @app.get("/", tags=["Health"], summary="Human-readable status")
    def root() -> Dict[str, Any]:
        """Return a summary of the running model and server status."""
        return {
            "service":       "Enterprise Model Inference API",
            "model_version": agent.model_version,
            "model_ready":   agent.is_ready,
        }

    # ── Inference endpoints ───────────────────────────────────────────────────

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["Inference"],
        summary="Single-record prediction",
    )
    def predict(body: PredictRequest) -> PredictResponse:
        """
        Run inference on a single feature record.

        Accepts a JSON body with a ``features`` dict and returns the model
        prediction along with request metadata for tracing.
        """
        request_id = str(uuid.uuid4())
        t0         = time.perf_counter()

        try:
            prediction = agent.predict(body.features)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction error (request_id=%s): %s", request_id, exc)
            raise HTTPException(status_code=500, detail="Internal inference error") from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("Prediction served  request_id=%s  latency=%.1fms", request_id, latency_ms)

        return PredictResponse(
            request_id    = request_id,
            prediction    = prediction,
            model_version = agent.model_version,
            latency_ms    = round(latency_ms, 2),
        )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictResponse,
        tags=["Inference"],
        summary="Batch prediction",
    )
    def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
        """
        Run inference on a batch of feature records in a single request.

        More efficient than calling /predict repeatedly for large workloads
        because it avoids per-request DataFrame construction overhead.
        """
        request_id = str(uuid.uuid4())
        t0         = time.perf_counter()

        try:
            predictions = agent.predict_batch(body.records)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Batch prediction error (request_id=%s): %s", request_id, exc)
            raise HTTPException(status_code=500, detail="Internal inference error") from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Batch prediction served  request_id=%s  records=%d  latency=%.1fms",
            request_id, len(predictions), latency_ms,
        )

        return BatchPredictResponse(
            request_id    = request_id,
            predictions   = predictions,
            record_count  = len(predictions),
            model_version = agent.model_version,
            latency_ms    = round(latency_ms, 2),
        )

    return app
