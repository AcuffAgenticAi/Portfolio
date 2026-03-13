# Enterprise AI Model Deployment & Monitoring Agent

> A fully automated, production-grade MLOps platform for deploying, monitoring,
> and maintaining machine learning models at scale.

---

## Overview

This platform automates the complete ML model lifecycle:

- **Deployment** — serves any scikit-learn model as a FastAPI REST endpoint with health probes, batch inference, and request tracing
- **Drift Detection** — Kolmogorov–Smirnov test detects distribution shifts in production data before they degrade model performance
- **Performance Monitoring** — accuracy, precision, recall, and F1 computed on every monitoring cycle with threshold alerting
- **Automatic Retraining** — when drift is detected, a quality-gated retraining pipeline fires automatically and atomically replaces the deployed artefact
- **Multi-tenancy** — per-tenant model routing with canary deployment support
- **Billing Simulation** — usage-based SaaS invoicing per tenant
- **Adversarial Security** — red-team test suite validates model robustness before every deployment
- **Kubernetes-ready** — HPA, rolling updates, liveness/readiness probes, ConfigMap configuration

---

## Architecture

```
                     ┌─────────────────────────────┐
                     │     Customer Dashboard       │
                     │   (Streamlit  port 8501)     │
                     └──────────────┬──────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │   FastAPI Inference Server   │
                     │        port 8000             │
                     │  /predict  /predict/batch    │
                     │  /healthz  /readyz           │
                     └──────────────┬──────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │             Tenant Router                  │
              │   bank_a → model_a   bank_b → model_b     │
              │          Canary split supported            │
              └───────────┬────────────────┬──────────────┘
                          │                │
              ┌───────────▼──┐    ┌────────▼───────────┐
              │  Stable Model│    │  Canary Model (10%) │
              └───────────┬──┘    └────────────────────-┘
                          │
          ┌───────────────▼───────────────┐
          │       Monitoring Loop         │
          │  DriftDetectionAgent  (KS)    │
          │  PerformanceMonitorAgent      │
          │  RetrainingTriggerAgent       │
          │  MetricsTracker  (JSON-lines) │
          └───────────────┬───────────────┘
                          │  drift detected
          ┌───────────────▼───────────────┐
          │     Retraining Pipeline       │
          │  load data → engineer feats   │
          │  train RF → quality gate      │
          │  atomic save → MLflow log     │
          └───────────────────────────────┘
```

---

## Project Structure

```
ai-model-deployment-agent/
│
├── agents/
│   ├── model_deployment_agent.py     # FastAPI app factory + ModelDeploymentAgent
│   ├── performance_monitor_agent.py  # Accuracy / precision / recall / F1
│   ├── drift_detection_agent.py      # KS two-sample drift detection
│   ├── retraining_trigger_agent.py   # Drift → retrain decision with cool-down
│   └── tenant_router_agent.py        # Per-tenant routing + canary splits
│
├── billing/
│   └── usage_meter.py                # Per-tenant call counting + invoicing
│
├── config/
│   └── settings.py                   # Env-var-backed configuration dataclass
│
├── dashboards/
│   ├── model_health_report.py        # CLI health report with ASCII trend chart
│   └── platform_dashboard.py         # Streamlit interactive dashboard
│
├── data/
│   └── generate_sample_data.py       # Synthetic fraud transaction generator
│
├── models/
│   └── fraud_model.pkl               # Serialised model artefact (generated)
│
├── monitoring/
│   ├── metrics_tracker.py            # JSON-lines metrics log
│   └── metrics_log.jsonl             # Append-only metrics log (generated)
│
├── pipelines/
│   ├── deployment_pipeline.py        # Validate → smoke-test → serve
│   └── retraining_pipeline.py        # Load → engineer → train → gate → save
│
├── security/
│   └── adversarial_validator.py      # Red-team test suite (5 attack patterns)
│
├── tests/
│   └── test_platform.py              # pytest suite (10 test classes, 30+ tests)
│
├── k8s/
│   └── deployment.yaml               # Namespace, Deployments, Service, Ingress, HPA
│
├── scripts/
│   └── start_platform.sh             # One-command local startup script
│
├── Dockerfile                        # Multi-stage production image
├── requirements.txt                  # Pinned Python dependencies
└── main.py                           # Platform orchestrator entry point
```

---

## Quick Start

### Local (bare Python)

```bash
# 1. Clone and enter the project
git clone https://github.com/yourname/ai-model-deployment-agent
cd ai-model-deployment-agent

# 2. One-command startup (installs deps, generates data, trains, starts platform)
chmod +x scripts/start_platform.sh
./scripts/start_platform.sh

# API docs: http://localhost:8000/docs
# Dashboard: streamlit run dashboards/platform_dashboard.py
```

### Manual steps

```bash
pip install -r requirements.txt

# Generate sample data
python data/generate_sample_data.py

# Train initial model
python main.py --retrain-only

# Run tests
pytest tests/ -v

# Start the full platform
python main.py

# Start monitoring loop only (no API server)
python main.py --monitor-only
```

### Docker

```bash
docker build -t ai-platform:latest .
docker run -p 8000:8000 ai-platform:latest

# Force retrain inside container
docker run ai-platform:latest python main.py --retrain-only
```

### Kubernetes

```bash
# Update image path in k8s/deployment.yaml first
kubectl apply -f k8s/deployment.yaml
kubectl -n ai-platform get pods
kubectl -n ai-platform logs -f deployment/inference-api
```

---

## API Reference

| Method | Endpoint         | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/healthz`      | Liveness probe (always 200)          |
| GET    | `/readyz`       | Readiness probe (503 if not loaded)  |
| GET    | `/`             | Status summary                       |
| POST   | `/predict`      | Single-record prediction             |
| POST   | `/predict/batch`| Batch prediction                     |

**Single prediction request:**
```json
POST /predict
{
  "features": {
    "transaction_amount": 450.0,
    "account_age": 730,
    "num_transactions": 8,
    "merchant_risk": 3
  }
}
```

**Response:**
```json
{
  "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "prediction": [1],
  "model_version": "models_fraud_model",
  "latency_ms": 12.4
}
```

---

## Configuration

All settings are read from environment variables with safe local defaults.

| Variable             | Default                          | Description                          |
|----------------------|----------------------------------|--------------------------------------|
| `ENV`                | `local`                          | Deployment environment               |
| `MODEL_PATH`         | `models/fraud_model.pkl`         | Model artefact path                  |
| `BASELINE_DATA_PATH` | `data/baseline_transactions.csv` | Reference feature distribution       |
| `CURRENT_DATA_PATH`  | `data/current_transactions.csv`  | Current production window            |
| `LABEL_COLUMN`       | `label`                          | Target column name                   |
| `MONITOR_INTERVAL`   | `60`                             | Seconds between monitor cycles       |
| `DRIFT_P_THRESHOLD`  | `0.05`                           | KS test p-value threshold            |
| `ACCURACY_THRESHOLD` | `0.85`                           | Minimum acceptable accuracy          |
| `ACTIVE_TENANTS`     | `bank_a,bank_b`                  | Comma-separated active tenant IDs    |
| `API_HOST`           | `0.0.0.0`                        | Inference server bind address        |
| `API_PORT`           | `8000`                           | Inference server port                |

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Single test class
pytest tests/test_platform.py::TestDriftDetectionAgent -v
```

---

## Technology Stack

| Layer              | Technology                                  |
|--------------------|---------------------------------------------|
| API serving        | FastAPI + uvicorn                           |
| ML framework       | scikit-learn (RandomForestClassifier)       |
| Drift detection    | scipy KS two-sample test                   |
| Data processing    | pandas + numpy                              |
| Model serialisation| joblib                                      |
| Dashboard          | Streamlit + Altair                          |
| Experiment tracking| MLflow (optional)                           |
| Containerisation   | Docker (multi-stage)                        |
| Orchestration      | Kubernetes (Deployment, HPA, Ingress)       |
| Testing            | pytest                                      |

---

## Skills Demonstrated

- MLOps architecture and ML lifecycle management
- Production model deployment pipelines
- Automated drift detection and retraining
- Multi-tenant SaaS platform design
- Adversarial ML security testing
- Kubernetes-native deployment patterns
- Observability and metrics tracking
- Enterprise-grade Python (type hints, logging, error handling, atomic I/O)
