"""
================================================================================
data/generate_sample_data.py  —  Sample Transaction Data Generator
================================================================================

Purpose:
    Generates synthetic fraud-detection datasets for local development and
    testing.  Produces two CSV files:

        baseline_transactions.csv  — training-time distribution (reference)
        current_transactions.csv   — current-window distribution
                                     (optionally shifted to simulate drift)

    Run this script once before starting the platform locally.

Usage:
    python data/generate_sample_data.py                    # no drift
    python data/generate_sample_data.py --inject-drift     # simulate drift
    python data/generate_sample_data.py --rows 5000        # larger dataset
================================================================================
"""

import argparse
import os

import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ── Output paths ─────────────────────────────────────────────────────────────
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASELINE  = os.path.join(OUT_DIR, "baseline_transactions.csv")
CURRENT   = os.path.join(OUT_DIR, "current_transactions.csv")


def generate_transactions(
    n_rows:       int   = 2_000,
    inject_drift: bool  = False,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset.

    Features:
        transaction_amount  — dollar amount of the transaction
        account_age         — account age in days
        num_transactions    — number of transactions the account made today
        merchant_risk       — risk score for the merchant category (0–5)
        label               — 0 = legitimate, 1 = fraud

    When inject_drift=True the transaction_amount distribution is shifted
    upward to simulate the kind of input drift the KS test will detect.

    Args:
        n_rows:        Number of rows to generate.
        inject_drift:  If True, shift feature distributions to simulate drift.

    Returns:
        DataFrame with the synthetic transaction data.
    """
    # Fraudulent transactions are ~5 % of the dataset.
    fraud_mask = RNG.random(n_rows) < 0.05

    # ── Feature generation ────────────────────────────────────────────────────
    transaction_amount = RNG.lognormal(mean=5.5, sigma=1.2, size=n_rows)
    account_age        = RNG.integers(1, 2_000, size=n_rows).astype(float)
    num_transactions   = RNG.integers(1, 21, size=n_rows).astype(float)
    merchant_risk      = RNG.integers(0, 6, size=n_rows).astype(float)

    if inject_drift:
        # Shift transaction_amount up by 50 % to simulate a distribution change
        # that the KS test should flag.
        transaction_amount *= 1.5
        merchant_risk       = np.clip(merchant_risk + 1, 0, 5)

    # ── Fraud boost: fraudulent records skew toward high-risk features ─────────
    transaction_amount[fraud_mask] *= 2.0
    merchant_risk[fraud_mask]       = np.clip(merchant_risk[fraud_mask] + 2, 0, 5)

    df = pd.DataFrame({
        "transaction_amount": np.round(transaction_amount, 2),
        "account_age":        account_age,
        "num_transactions":   num_transactions,
        "merchant_risk":      merchant_risk,
        "label":              fraud_mask.astype(int),
    })

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample transaction data")
    parser.add_argument("--rows",         type=int,  default=2_000)
    parser.add_argument("--inject-drift", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Baseline — no drift
    baseline_df = generate_transactions(n_rows=args.rows, inject_drift=False)
    baseline_df.to_csv(BASELINE, index=False)
    print(f"Baseline data written to '{BASELINE}'  ({len(baseline_df):,} rows)")

    # Current — optionally drifted
    current_df = generate_transactions(n_rows=args.rows, inject_drift=args.inject_drift)
    current_df.to_csv(CURRENT, index=False)
    drift_tag = "WITH drift" if args.inject_drift else "no drift"
    print(f"Current data  written to '{CURRENT}'  ({len(current_df):,} rows, {drift_tag})")


if __name__ == "__main__":
    main()
