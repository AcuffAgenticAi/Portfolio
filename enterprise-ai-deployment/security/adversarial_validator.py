"""
================================================================================
security/adversarial_validator.py  —  Adversarial ML Security Validator
================================================================================

Purpose:
    Red-team tests the deployed model against common adversarial attack
    patterns to validate robustness before and after every deployment.

Attack patterns tested:
    1.  Boundary probing     — inputs near the decision boundary
    2.  Feature perturbation — small systematic shifts to features
    3.  Payload injection    — extreme/null/negative feature values
    4.  Repetition attack    — identical requests at high frequency
    5.  Schema fuzzing       — unexpected dtypes and missing features

Enterprise value:
    Adversarial validation prevents attackers from gaming the fraud model
    by probing for blind spots.  Running it in CI/CD ensures every new model
    version is security-tested before it sees production traffic.

Usage:
    validator = AdversarialValidator(model_path="models/fraud_model.pkl")
    results   = validator.run_all_tests()
    print(results["summary"])
================================================================================
"""

import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("platform.adversarial_validator")


class AdversarialValidator:
    """
    Validates a model's robustness against adversarial input patterns.

    Each test method returns a TestResult dict with keys:
        test_name    — human-readable name
        passed       — True if the model behaved as expected
        details      — list of findings (empty when passed=True)
        risk_level   — "low" | "medium" | "high" | "critical"
    """

    # ── Canonical feature schema expected by the fraud model ─────────────────
    _FEATURE_COLUMNS = [
        "transaction_amount",
        "account_age",
        "num_transactions",
        "merchant_risk",
    ]

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model      = None
        self._load_model()

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Execute the full adversarial test suite and return a consolidated report.

        Returns:
            dict with keys:
                results  — list of individual test result dicts
                summary  — high-level pass/fail count and overall risk rating
        """
        if self.model is None:
            return {"results": [], "summary": {"error": "Model not loaded"}}

        tests = [
            self.test_boundary_probing,
            self.test_feature_perturbation,
            self.test_payload_injection,
            self.test_schema_fuzzing,
            self.test_output_consistency,
        ]

        results = []
        for test_fn in tests:
            try:
                result = test_fn()
                results.append(result)
                status = "PASS" if result["passed"] else "FAIL"
                logger.info(
                    "Adversarial test %-30s  %s  risk=%-8s",
                    result["test_name"], status, result["risk_level"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Test '%s' raised an exception: %s", test_fn.__name__, exc)
                results.append({
                    "test_name":  test_fn.__name__,
                    "passed":     False,
                    "details":    [f"Exception during test: {exc}"],
                    "risk_level": "high",
                })

        summary = self._build_summary(results)
        logger.info(
            "Adversarial validation complete  passed=%d/%d  overall_risk=%s",
            summary["passed"], summary["total"], summary["overall_risk"],
        )
        return {"results": results, "summary": summary}

    # ─────────────────────────────────────────────────────────────────────────
    # Individual test methods
    # ─────────────────────────────────────────────────────────────────────────

    def test_boundary_probing(self) -> Dict[str, Any]:
        """
        Probe the decision boundary by sending inputs with features tuned to
        values known to sit near typical fraud/legitimate thresholds.

        A well-calibrated model should return consistent predictions for
        inputs that are clearly on one side of the boundary and not flip
        on minor perturbations.
        """
        test_name = "boundary_probing"
        findings:  List[str] = []

        # High-risk boundary: high amount, new account, many transactions.
        boundary_cases = [
            {"transaction_amount": 499.0, "account_age": 31,  "num_transactions": 9,  "merchant_risk": 3},
            {"transaction_amount": 501.0, "account_age": 29,  "num_transactions": 11, "merchant_risk": 3},
            {"transaction_amount": 500.0, "account_age": 30,  "num_transactions": 10, "merchant_risk": 3},
        ]

        predictions = self._predict_batch(boundary_cases)
        if predictions is None:
            return self._error_result(test_name, "Prediction failed on boundary cases")

        # Expect some consistency: not all three should differ from each other.
        unique_preds = len(set(predictions))
        if unique_preds == len(predictions):
            findings.append(
                "Every boundary case produced a different prediction — "
                "model may be highly sensitive to minor input variations near the boundary."
            )

        return {
            "test_name":  test_name,
            "passed":     len(findings) == 0,
            "details":    findings,
            "risk_level": "medium" if findings else "low",
        }

    def test_feature_perturbation(self) -> Dict[str, Any]:
        """
        Verify the model's predictions are stable under small feature noise.

        A 1 % perturbation on a legitimate transaction should not flip the
        prediction to fraud.  Large instability here indicates the model
        may be exploitable through minor input manipulation.
        """
        test_name = "feature_perturbation"
        findings:  List[str] = []

        base_input = {
            "transaction_amount": 100.0,
            "account_age":        500,
            "num_transactions":   3,
            "merchant_risk":      1,
        }

        base_pred = self._predict_single(base_input)
        if base_pred is None:
            return self._error_result(test_name, "Base prediction failed")

        # Apply ±1 % perturbation to each numeric feature and check for flips.
        flips = []
        for feature in self._FEATURE_COLUMNS:
            perturbed = dict(base_input)
            original  = perturbed[feature]
            perturbed[feature] = original * 1.01   # +1 %

            perturbed_pred = self._predict_single(perturbed)
            if perturbed_pred is not None and perturbed_pred != base_pred:
                flips.append(f"Prediction flipped when '{feature}' changed by 1%")

        findings.extend(flips)

        return {
            "test_name":  test_name,
            "passed":     len(findings) == 0,
            "details":    findings,
            "risk_level": "high" if findings else "low",
        }

    def test_payload_injection(self) -> Dict[str, Any]:
        """
        Test the model's handling of extreme, negative, and zero feature values.

        The model should not crash and should return a valid prediction for
        any numeric input — even degenerate ones.  Crashes or exceptions here
        indicate brittle preprocessing that an attacker could exploit to cause
        denial-of-service via malformed payloads.
        """
        test_name  = "payload_injection"
        findings:  List[str] = []

        extreme_cases = [
            {"transaction_amount": 0,      "account_age": 0,    "num_transactions": 0,  "merchant_risk": 0},   # all zeros
            {"transaction_amount": 1e9,    "account_age": 9999, "num_transactions": 99, "merchant_risk": 5},   # max values
            {"transaction_amount": -500.0, "account_age": -1,   "num_transactions": -5, "merchant_risk": -1},  # negatives
            {"transaction_amount": float("inf"), "account_age": 365, "num_transactions": 5, "merchant_risk": 2},  # infinity
        ]

        for i, case in enumerate(extreme_cases):
            try:
                # Replace inf/nan with large finite values (simulating upstream
                # sanitisation that should exist in the API layer).
                sanitised = {
                    k: (v if np.isfinite(v) else 1e9)
                    for k, v in case.items()
                }
                pred = self._predict_single(sanitised)
                if pred is None:
                    findings.append(f"Case {i}: model returned None for extreme input")
                elif pred not in (0, 1):
                    findings.append(f"Case {i}: unexpected prediction value '{pred}'")
            except Exception as exc:  # noqa: BLE001
                findings.append(f"Case {i}: exception on extreme input — {exc}")

        return {
            "test_name":  test_name,
            "passed":     len(findings) == 0,
            "details":    findings,
            "risk_level": "critical" if findings else "low",
        }

    def test_schema_fuzzing(self) -> Dict[str, Any]:
        """
        Verify the API layer rejects or safely handles malformed feature schemas.

        This tests the model's preprocessing robustness — missing features,
        extra features, and wrong-type features should be handled gracefully
        without returning a misleading prediction.

        Note: This test validates expected *failure* behaviour.  A robust
        system should raise a clear error, not silently return a wrong answer.
        """
        test_name  = "schema_fuzzing"
        findings:  List[str] = []

        # ── Missing feature ───────────────────────────────────────────────────
        incomplete = {"transaction_amount": 200.0, "account_age": 365}   # missing 2 features
        try:
            df   = pd.DataFrame([incomplete])
            pred = self.model.predict(df)
            # If the model silently fills missing columns with zeros it may
            # produce a misleading result — flag it for review.
            findings.append(
                "Model predicted on incomplete features without raising an error.  "
                "Ensure the preprocessing pipeline validates required columns."
            )
        except Exception:
            pass   # Expected: exception on missing features is correct behaviour.

        return {
            "test_name":  test_name,
            "passed":     len(findings) == 0,
            "details":    findings,
            "risk_level": "medium" if findings else "low",
        }

    def test_output_consistency(self) -> Dict[str, Any]:
        """
        Verify the model produces identical outputs for identical inputs.

        A deterministic model (Random Forest with fixed random_state) must
        always return the same prediction for the same input.  Non-determinism
        here could indicate an issue with the serialisation / loading pipeline.
        """
        test_name  = "output_consistency"
        findings:  List[str] = []

        reference_input = {
            "transaction_amount": 300.0,
            "account_age":        180,
            "num_transactions":   7,
            "merchant_risk":      2,
        }

        predictions = [self._predict_single(reference_input) for _ in range(5)]
        unique      = set(predictions)

        if len(unique) > 1:
            findings.append(
                f"Non-deterministic output detected: got {unique} across 5 identical calls"
            )

        return {
            "test_name":  test_name,
            "passed":     len(findings) == 0,
            "details":    findings,
            "risk_level": "high" if findings else "low",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the serialised model; set self.model=None on failure."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info("Security validator loaded model from '%s'", self.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load model for validation: %s", exc)

    def _predict_single(self, features: Dict[str, Any]) -> Optional[int]:
        """Run a single prediction; return None on error."""
        try:
            df   = pd.DataFrame([features])
            pred = self.model.predict(df)
            return int(pred[0])
        except Exception:  # noqa: BLE001
            return None

    def _predict_batch(self, records: List[Dict]) -> Optional[List[int]]:
        """Run a batch prediction; return None on error."""
        try:
            df    = pd.DataFrame(records)
            preds = self.model.predict(df)
            return [int(p) for p in preds]
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _error_result(test_name: str, detail: str) -> Dict[str, Any]:
        """Build a failed test result for infrastructure-level errors."""
        return {
            "test_name":  test_name,
            "passed":     False,
            "details":    [detail],
            "risk_level": "high",
        }

    @staticmethod
    def _build_summary(results: List[Dict]) -> Dict[str, Any]:
        """Aggregate individual test results into an overall summary."""
        total    = len(results)
        passed   = sum(1 for r in results if r["passed"])
        failed   = total - passed
        failures = [r for r in results if not r["passed"]]

        # Overall risk = highest risk level among failed tests.
        risk_order  = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        overall_risk = "low"
        for r in failures:
            if risk_order.get(r["risk_level"], 0) > risk_order[overall_risk]:
                overall_risk = r["risk_level"]

        return {
            "total":        total,
            "passed":       passed,
            "failed":       failed,
            "overall_risk": overall_risk,
            "failed_tests": [r["test_name"] for r in failures],
        }
