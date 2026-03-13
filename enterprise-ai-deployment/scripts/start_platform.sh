#!/usr/bin/env bash
# ================================================================================
# scripts/start_platform.sh  —  One-command local platform startup
# ================================================================================
#
# Purpose:
#   Bootstraps the full Enterprise AI MLOps Platform locally in a single
#   terminal session.  Performs all prerequisite checks then starts each
#   platform subsystem in the correct order.
#
# What it does:
#   1. Verifies Python version (3.10+) and pip
#   2. Installs Python dependencies from requirements.txt
#   3. Generates synthetic sample data (baseline + current CSVs)
#   4. Trains an initial model artefact
#   5. Runs the adversarial security validation suite
#   6. Runs the pytest test suite
#   7. Starts the full platform (API + monitoring loop)
#
# Usage:
#   chmod +x scripts/start_platform.sh
#   ./scripts/start_platform.sh
#   ./scripts/start_platform.sh --skip-tests    # skip test suite (faster)
#   ./scripts/start_platform.sh --monitor-only  # monitoring loop only
# ================================================================================

set -euo pipefail   # exit on error, undefined vars, and pipe failures

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"   # no colour

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
section() { echo -e "\n${GREEN}══════════════════════════════════════════════${NC}"; \
            echo -e "${GREEN}  $*${NC}"; \
            echo -e "${GREEN}══════════════════════════════════════════════${NC}"; }

# ── CLI flags ─────────────────────────────────────────────────────────────────
SKIP_TESTS=false
MONITOR_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-tests)   SKIP_TESTS=true   ;;
        --monitor-only) MONITOR_ONLY=true ;;
        *)              warn "Unknown flag: $arg" ;;
    esac
done

# ── Script directory (so we can run from any cwd) ─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

section "Enterprise AI MLOps Platform  —  Local Startup"
info "Project directory: $PROJECT_DIR"

# ── Step 1: Python version check ──────────────────────────────────────────────
section "Step 1/7: Checking Python version"
python3 --version || error "python3 not found — install Python 3.10+"

PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
    error "Python 3.10+ required; found 3.$PYTHON_MINOR"
fi
info "Python check passed"

# ── Step 2: Install dependencies ──────────────────────────────────────────────
section "Step 2/7: Installing dependencies"

if [[ ! -d ".venv" ]]; then
    info "Creating virtual environment..."
    python3 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "Dependencies installed"

# ── Step 3: Generate sample data ──────────────────────────────────────────────
section "Step 3/7: Generating sample data"
python3 data/generate_sample_data.py
info "Sample data ready"

# ── Step 4: Train initial model ───────────────────────────────────────────────
section "Step 4/7: Training initial model"
python3 main.py --retrain-only
info "Initial model artefact ready"

# ── Step 5: Adversarial security validation ───────────────────────────────────
section "Step 5/7: Adversarial security validation"
python3 -c "
from security.adversarial_validator import AdversarialValidator
v = AdversarialValidator('models/fraud_model.pkl')
r = v.run_all_tests()
s = r['summary']
print(f'  Passed: {s[\"passed\"]}/{s[\"total\"]}  |  Risk: {s[\"overall_risk\"]}')
if s['failed'] > 0:
    print('  Failed tests:', s['failed_tests'])
"

# ── Step 6: Run test suite ────────────────────────────────────────────────────
if [[ "$SKIP_TESTS" == "false" ]]; then
    section "Step 6/7: Running test suite"
    python3 -m pytest tests/ -v --tb=short
    info "All tests passed"
else
    section "Step 6/7: Skipping test suite  (--skip-tests)"
fi

# ── Step 7: Start the platform ────────────────────────────────────────────────
section "Step 7/7: Starting platform"
info "API server will be available at  http://localhost:8000"
info "Swagger docs available at        http://localhost:8000/docs"
info "Press Ctrl+C to stop the platform"

if [[ "$MONITOR_ONLY" == "true" ]]; then
    python3 main.py --monitor-only
else
    python3 main.py
fi
