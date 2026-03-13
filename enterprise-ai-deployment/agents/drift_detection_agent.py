"""
================================================================================
agents/drift_detection_agent.py  —  Data Drift Detection Agent
================================================================================

Purpose:
    Detects statistical shifts between a baseline feature distribution and the
    current production window.  Drift indicates that the real-world data the
    model sees today differs from the data it was trained on — a leading signal
    that model performance will degrade if left unaddressed.

Statistical test:
    Kolmogorov–Smirnov (KS) two-sample test.
    • Null hypothesis: the two samples are drawn from the same distribution.
    • If p-value < DRIFT_P_THRESHOLD (default 0.05) we reject H₀ → drift flagged.
    • KS is non-parametric (no assumption of normality) and works on any
      continuous or ordinal feature.

Output (drift_report dict):
    {
        "transaction_amount": {
            "ks_statistic":   0.312,
            "p_value":        0.003,
            "drift_detected": True
        },
        "account_age": {
            "ks_statistic":   0.045,
            "p_value":        0.821,
            "drift_detected": False
        },
        ...
        "_summary": {
            "features_tested":   4,
            "features_drifted":  1,
            "any_drift":         True
        }
    }

Usage:
    agent  = DriftDetectionAgent()
    report = agent.detect_drift(baseline_df, current_df)
================================================================================
"""

import logging
from typing import Any, Dict

import pandas as pd
from scipy.stats import ks_2samp

from config.settings import Settings

logger = logging.getLogger("platform.drift_detection")


class DriftDetectionAgent:
    """
    Detects input-feature distribution drift using the KS two-sample test.

    Attributes:
        p_threshold:  p-value below which drift is considered detected.
                      Loaded from Settings.DRIFT_P_THRESHOLD (env-configurable).
    """

    def __init__(self) -> None:
        self.p_threshold = Settings().DRIFT_P_THRESHOLD

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def detect_drift(
        self,
        baseline_data: pd.DataFrame,
        current_data:  pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run the KS test for every numeric column shared between both DataFrames.

        Non-numeric columns are silently skipped (they require a different test
        such as chi-squared; extend this method if categorical drift matters).

        Args:
            baseline_data:  Historical feature data the model was trained on
                            (or a representative reference window).
            current_data:   Features from the current production window.

        Returns:
            drift_report dict keyed by feature name, plus a ``_summary`` entry.

        Raises:
            ValueError:  If the DataFrames share no common numeric columns.
        """
        common_cols  = self._get_common_numeric_columns(baseline_data, current_data)

        if not common_cols:
            raise ValueError(
                "baseline_data and current_data share no common numeric columns — "
                "cannot run drift detection"
            )

        drift_report: Dict[str, Any] = {}
        drifted_count = 0

        for col in common_cols:
            result = self._test_column(
                baseline_series=baseline_data[col],
                current_series=current_data[col],
                col_name=col,
            )
            drift_report[col] = result
            if result["drift_detected"]:
                drifted_count += 1

        # ── Summary block ─────────────────────────────────────────────────────
        drift_report["_summary"] = {
            "features_tested":  len(common_cols),
            "features_drifted": drifted_count,
            "any_drift":        drifted_count > 0,
        }

        logger.info(
            "Drift detection complete: %d/%d features drifted  (p_threshold=%.3f)",
            drifted_count, len(common_cols), self.p_threshold,
        )

        return drift_report

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _test_column(
        self,
        baseline_series: pd.Series,
        current_series:  pd.Series,
        col_name:        str,
    ) -> Dict[str, Any]:
        """
        Apply the KS two-sample test to a single feature column.

        NaN values are dropped before the test so missing data does not skew
        the distribution comparison.

        Args:
            baseline_series:  Baseline values for one feature.
            current_series:   Current-window values for the same feature.
            col_name:         Column name (used only for logging).

        Returns:
            dict with keys: ks_statistic, p_value, drift_detected.
        """
        clean_baseline = baseline_series.dropna()
        clean_current  = current_series.dropna()

        ks_stat, p_value = ks_2samp(clean_baseline, clean_current)
        drift_detected   = bool(p_value < self.p_threshold)

        if drift_detected:
            logger.warning(
                "Drift detected in feature '%s'  KS=%.4f  p=%.4f  (threshold=%.3f)",
                col_name, ks_stat, p_value, self.p_threshold,
            )
        else:
            logger.debug(
                "No drift in '%s'  KS=%.4f  p=%.4f", col_name, ks_stat, p_value
            )

        return {
            "ks_statistic":   round(float(ks_stat), 6),
            "p_value":        round(float(p_value),  6),
            "drift_detected": drift_detected,
        }

    @staticmethod
    def _get_common_numeric_columns(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
    ) -> list:
        """
        Return a sorted list of numeric column names present in both DataFrames.

        Only numeric dtypes (int, float) are included; KS requires ordinal data.
        """
        numeric1 = set(df1.select_dtypes(include="number").columns)
        numeric2 = set(df2.select_dtypes(include="number").columns)
        return sorted(numeric1 & numeric2)
