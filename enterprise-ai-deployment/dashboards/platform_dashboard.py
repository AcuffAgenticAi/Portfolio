"""
================================================================================
dashboards/platform_dashboard.py  —  Streamlit Customer Dashboard
================================================================================

Purpose:
    Interactive web dashboard for the Enterprise AI Platform.  Provides:

        • Live fraud detection demo — sliders to configure a transaction and
          submit it to the running inference API
        • Platform KPI metrics  — predictions today, active models, latency
        • Billing snapshot      — per-tenant invoice simulation
        • Model accuracy trend  — line chart of historical accuracy readings

    Designed to be served alongside the FastAPI inference server, giving
    customers a visual portal into the platform's capabilities.

Run:
    streamlit run dashboards/platform_dashboard.py

Requirements:
    streamlit, requests, pandas, altair (all in requirements.txt)

Environment variables:
    API_URL   Base URL of the inference API  (default: http://localhost:8000)
================================================================================
"""

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL    = os.getenv("API_URL", "http://localhost:8000")
METRICS_LOG = Path("monitoring/metrics_log.jsonl")

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Enterprise AI Platform",
    page_icon  = "🤖",
    layout     = "wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 Enterprise AI Platform")
st.caption("Real-time fraud detection · Automated monitoring · Multi-tenant SaaS")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Fraud Detection Demo
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("🔍 Fraud Detection Demo")

col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.slider("Transaction Amount ($)", 1,   1_000, 250)
    account_age        = st.slider("Account Age (days)",     1,   2_000, 365)

with col2:
    num_transactions   = st.slider("Transactions Today",     1,      20,   5)
    merchant_risk      = st.slider("Merchant Risk Score",    0,       5,   2)

if st.button("▶  Run Fraud Check", type="primary"):
    payload: Dict[str, Any] = {
        "features": {
            "transaction_amount": transaction_amount,
            "account_age":        account_age,
            "num_transactions":   num_transactions,
            "merchant_risk":      merchant_risk,
        }
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        prediction = result.get("prediction", [None])[0]
        label      = "🚨 FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE"
        color      = "red"               if prediction == 1 else "green"

        st.markdown(
            f"**Prediction:** <span style='color:{color}; font-size:1.2em'>"
            f"{label}</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Model: {result.get('model_version', 'N/A')}  ·  "
            f"Latency: {result.get('latency_ms', 'N/A')} ms  ·  "
            f"Request ID: {result.get('request_id', 'N/A')}"
        )

    except requests.exceptions.ConnectionError:
        st.warning(
            "⚠️  Cannot connect to the inference API at "
            f"`{API_URL}`.  Start the platform with `python main.py` first."
        )
    except requests.HTTPError as exc:
        st.error(f"API error: {exc}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Platform KPIs
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📊 Platform Metrics")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# In production these would be pulled from Prometheus / a metrics API.
# We simulate plausible values here.
kpi1.metric("Predictions Today",  f"{random.randint(3_000, 6_000):,}")
kpi2.metric("Active Models",      "3")
kpi3.metric("Avg Latency",        f"{random.randint(30, 60)} ms")
kpi4.metric("Uptime",             "99.97 %")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Billing Snapshot
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("💳 Billing Snapshot  (current period)")

billing_data = {
    "Tenant":      ["Bank A",  "Bank B",  "Bank C"],
    "Predictions": [7_100,      4_900,     2_300],
    "Invoice ($)": [14.20,       9.80,      4.60],
}

billing_df = pd.DataFrame(billing_data)
st.dataframe(billing_df, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Accuracy Trend
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📈 Model Accuracy Trend")

def _load_accuracy_trend() -> Optional[pd.DataFrame]:
    """Load historical accuracy data from the metrics log."""
    if not METRICS_LOG.exists():
        return None

    records = []
    with open(METRICS_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append({
                    "Timestamp": rec.get("timestamp", "")[:19],
                    "Accuracy":  rec.get("metrics", {}).get("accuracy"),
                })
            except json.JSONDecodeError:
                continue

    if not records:
        return None

    df = pd.DataFrame(records).dropna()
    return df


trend_df = _load_accuracy_trend()

if trend_df is not None and not trend_df.empty:
    st.line_chart(trend_df.set_index("Timestamp")["Accuracy"])
else:
    # Generate synthetic demo data so the dashboard looks populated even
    # before the first monitoring cycle has run.
    demo_accuracy = [0.90 + random.uniform(-0.03, 0.03) for _ in range(12)]
    st.line_chart({"Accuracy": demo_accuracy})
    st.caption("_(Demo data — run a monitoring cycle to see real metrics)_")
