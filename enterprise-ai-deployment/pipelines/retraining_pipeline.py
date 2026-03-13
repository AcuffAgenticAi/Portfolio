"""
================================================================================
pipelines/retraining_pipeline.py  —  Automated Retraining Pipeline
================================================================================

Purpose:
    Retrains the fraud-detection model from scratch using the most recent
    production data and saves the new artefact, atomically replacing the
    previously deployed model.

Pipeline steps:
    1.  Load labelled training data from the configured CSV path.
    2.  Run feature engineering (same transforms as training time).
    3.  Split into train / validation sets.
    4.  Train a RandomForestClassifier (can be swapped for any sklearn estimator).
    5.  Evaluate on the hold-out validation set.
    6.  If validation accuracy meets the quality gate, persist the new model.
    7.  Log the run to MLflow (optional; skipped gracefully if not installed).

Atomic save:
    The new model is first written to a temp file, then renamed over the
    existing artefact.  This prevents a partially-written file from being
    loaded by a concurrently running inference server.

Quality gate:
    If the retrained model's validation accuracy is below
    Settings.ACCURACY_THRESHOLD the pipeline raises and the existing artefact
    is left untouched.  This prevents a regression from going live.

Usage:
    from pipelines.retraining_pipeline import retrain_model
    retrain_model()                              # uses Settings defaults
    retrain_model(data_path="data/custom.csv")   # explicit data path override
================================================================================
"""

import logging
import os
import tempfile
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import Settings

logger = logging.getLogger("platform.retraining_pipeline")


def retrain_model(
    data_path:  Optional[str] = None,
    model_path: Optional[str] = None,
) -> dict:
    """
    Execute the full retraining pipeline.

    Args:
        data_path:   Path to the labelled training CSV.  Defaults to
                     Settings.CURRENT_DATA_PATH.
        model_path:  Destination path for the saved model artefact.  Defaults
                     to Settings.MODEL_PATH.

    Returns:
        dict with keys: accuracy, report, model_path — the results of the
        validation evaluation.

    Raises:
        FileNotFoundError:  If the training data CSV does not exist.
        ValueError:         If the quality gate check fails.
    """
    settings   = Settings()
    data_path  = data_path  or settings.CURRENT_DATA_PATH
    model_path = model_path or settings.MODEL_PATH

    logger.info("=" * 60)
    logger.info("Retraining pipeline starting  data='%s'", data_path)
    logger.info("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    df = _load_data(data_path, settings.LABEL_COLUMN)

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    X, y = _engineer_features(df, settings.LABEL_COLUMN)

    # ── Step 3: Train / validation split (stratified to preserve class balance)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(
        "Data split  train=%d  val=%d  features=%d",
        len(X_train), len(X_val), X.shape[1],
    )

    # ── Step 4: Train model ───────────────────────────────────────────────────
    model = _build_and_train(X_train, y_train)

    # ── Step 5: Evaluate on hold-out validation set ───────────────────────────
    accuracy, report = _evaluate(model, X_val, y_val)
    logger.info("Validation accuracy: %.4f", accuracy)
    logger.info("Classification report:\n%s", report)

    # ── Step 6: Quality gate — only save if model meets the minimum bar ────────
    if accuracy < settings.ACCURACY_THRESHOLD:
        raise ValueError(
            f"Retrained model accuracy {accuracy:.4f} is below quality gate "
            f"{settings.ACCURACY_THRESHOLD:.4f} — artefact NOT saved"
        )

    # ── Step 7: Atomic artefact save ──────────────────────────────────────────
    _atomic_save(model, model_path)
    logger.info("New model artefact saved to '%s'", model_path)

    # ── Step 8: Optional MLflow logging ───────────────────────────────────────
    _log_to_mlflow(accuracy, model_path)

    logger.info("Retraining pipeline complete")
    return {"accuracy": accuracy, "report": report, "model_path": model_path}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(data_path: str, label_column: str) -> pd.DataFrame:
    """Load and minimally validate the training CSV."""
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Training data not found at '{data_path}'.  "
            "Ensure the data export job has run before triggering retraining."
        )

    df = pd.read_csv(data_path)

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in '{data_path}'.  "
            f"Available columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError(f"Training data at '{data_path}' is empty")

    logger.info("Loaded %d rows, %d columns from '%s'", len(df), df.shape[1], data_path)
    return df


def _engineer_features(
    df: pd.DataFrame,
    label_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features from the label and apply any domain-specific transforms.

    Extend this function with:
        • Log-transforming skewed monetary amounts
        • One-hot encoding categorical merchant categories
        • Time-of-day cyclical features (sin/cos encoding)
    """
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Drop any non-numeric columns that slipped through (e.g. transaction IDs).
    X = X.select_dtypes(include="number")

    # Fill remaining NaNs with column medians (simple, robust imputation).
    X = X.fillna(X.median())

    return X, y


def _build_and_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Build an sklearn Pipeline and fit it on the training set.

    The Pipeline wraps:
        1. StandardScaler  — zero-mean, unit-variance normalisation
        2. RandomForestClassifier — ensemble of decision trees

    Using a Pipeline guarantees that the same transforms applied at training
    time are automatically applied at inference time (no train/serve skew).

    Returns:
        Fitted sklearn Pipeline object.
    """
    pipeline = Pipeline([
        ("scaler",      StandardScaler()),
        ("classifier",  RandomForestClassifier(
            n_estimators=200,       # 200 trees — good bias/variance balance
            max_depth=None,         # trees grow to pure leaves by default
            min_samples_leaf=5,     # prevents overfitting on small leaves
            class_weight="balanced",# handles class imbalance in fraud datasets
            random_state=42,        # reproducible runs
            n_jobs=-1,              # use all available CPU cores
        )),
    ])

    logger.info("Fitting RandomForestClassifier  n_estimators=200…")
    pipeline.fit(X_train, y_train)
    return pipeline


def _evaluate(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[float, str]:
    """Run predictions on the validation set and return accuracy + full report."""
    y_pred   = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report   = classification_report(y_val, y_pred, zero_division=0)
    return accuracy, report


def _atomic_save(model: Pipeline, model_path: str) -> None:
    """
    Save the model artefact atomically using a write-then-rename pattern.

    This prevents a concurrent inference server from loading a partially-written
    (corrupt) artefact.

    Args:
        model:       Fitted sklearn Pipeline to serialise.
        model_path:  Final destination path for the artefact.
    """
    dir_name = os.path.dirname(model_path) or "."
    os.makedirs(dir_name, exist_ok=True)

    # Write to a temp file in the same directory so os.rename() is atomic
    # (same filesystem, single inode rename — no partial reads possible).
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".pkl.tmp")
    os.close(fd)

    try:
        joblib.dump(model, tmp_path, compress=3)    # compress=3 reduces artefact size
        os.rename(tmp_path, model_path)
    except Exception:
        # Clean up the temp file if anything goes wrong before the rename.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _log_to_mlflow(accuracy: float, model_path: str) -> None:
    """
    Log the retraining run to MLflow Tracking (optional dependency).

    Silently skips if mlflow is not installed so the pipeline works in
    environments without the full MLflow server stack.
    """
    try:
        import mlflow  # noqa: PLC0415 — optional import

        with mlflow.start_run(run_name="automated_retrain"):
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 200)
            mlflow.log_metric("val_accuracy", accuracy)
            mlflow.log_artifact(model_path)

        logger.info("MLflow run logged successfully")
    except ImportError:
        logger.debug("mlflow not installed — skipping MLflow logging")
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging failed (non-fatal): %s", exc)
