# ================================================================================
# Dockerfile  —  Enterprise AI Platform container image
# ================================================================================
#
# Purpose:
#   Multi-stage Docker build that produces a lean, production-ready container
#   image for the AI Model Deployment & Monitoring Agent.
#
# Stages:
#   builder  —  installs Python dependencies into a virtual environment
#   runtime  —  copies only the venv and application code (no build tools)
#
# Build:
#   docker build -t your-registry/ai-platform:latest .
#   docker build -t your-registry/ai-platform:1.0.0 .
#
# Run locally:
#   docker run -p 8000:8000 your-registry/ai-platform:latest
#   docker run your-registry/ai-platform:latest python main.py --monitor-only
#
# Security notes:
#   - Runs as a non-root user (uid=1000) to limit blast radius of any exploit
#   - No secrets in the image; inject via environment variables or k8s Secrets
#   - Base image pinned to a specific digest in production (use Dependabot)
# ================================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Set working directory inside the container.
WORKDIR /build

# Install system packages needed to compile Python dependencies
# (numpy, scipy, scikit-learn have C extensions).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment so we can copy it cleanly to runtime.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first — leverages Docker layer caching so a code-only
# change does not re-install all packages.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Copy the pre-built virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only the minimal runtime system library (libgomp for OpenMP / sklearn).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group.
RUN groupadd --gid 1000 appgroup \
 && useradd  --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy application source code.
COPY --chown=appuser:appgroup . .

# Create directories the app writes to at runtime.
RUN mkdir -p monitoring data models \
 && chown -R appuser:appgroup monitoring data models

# Switch to non-root user.
USER appuser

# ── Runtime configuration ─────────────────────────────────────────────────────
# These are safe defaults; override via environment variables at run time.
ENV ENV=production \
    LOG_LEVEL=INFO \
    MONITOR_INTERVAL=60 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Expose the FastAPI inference port.
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
# Docker engine polls this to determine container health.
# Kubernetes uses its own probes (/healthz, /readyz) — this is a belt-and-
# suspenders check for bare Docker deployments.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Default: start the full platform (API + monitoring loop).
# Override with `docker run ... python main.py --monitor-only` as needed.
CMD ["python", "main.py"]
