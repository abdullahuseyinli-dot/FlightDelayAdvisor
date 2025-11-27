"""
Evaluate the trained delay and cancellation models on the processed dataset.

Outputs:
- ROC-AUC, PR-AUC, Brier score for both models.
- Simple text summary printed to stdout (ASCII-only, safe for Windows/CP1252).

Run from the project root:

    (.project1venv) PS> python .\src\evaluate_models.py
    (.project1venv) PS> python .\src\evaluate_models.py > reports\evaluate_models_log.txt
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

# --------------------------------------------------------------------
# Make sure we can import app.py from the project root
# --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import (  # type: ignore  # noqa: E402
    DATA_PATH,
    DELAY_MODEL_PATH,
    CANCEL_MODEL_PATH,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

# --------------------------------------------------------------------
# Optional: suppress LightGBM "feature names" warning in this script
# --------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


def load_data_and_models():
    df = pd.read_parquet(DATA_PATH)
    delay_model = joblib.load(DELAY_MODEL_PATH)
    cancel_model = joblib.load(CANCEL_MODEL_PATH)
    return df, delay_model, cancel_model


def evaluate_delay_model(df: pd.DataFrame, model) -> dict:
    """
    Evaluate the delay>=15min model on all non-cancelled flights
    with a defined ArrDel15 label.
    """
    df_delay = df[(df["Cancelled"] == 0) & df["ArrDel15"].notna()].copy()
    X = df_delay[FEATURE_COLS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    y = df_delay["ArrDel15"].astype(int).to_numpy()

    proba = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    return {
        "n": len(y),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
    }


def evaluate_cancel_model(df: pd.DataFrame, model) -> dict:
    """
    Evaluate the cancellation model on all flights with Cancelled in {0,1}.
    """
    df_c = df[df["Cancelled"].isin([0, 1])].copy()
    X = df_c[FEATURE_COLS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    y = df_c["Cancelled"].astype(int).to_numpy()

    proba = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    return {
        "n": len(y),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
    }


def main() -> None:
    df, delay_model, cancel_model = load_data_and_models()

    delay_metrics = evaluate_delay_model(df, delay_model)
    cancel_metrics = evaluate_cancel_model(df, cancel_model)

    print("=" * 72)
    print("MODEL EVALUATION SUMMARY (full dataset)")
    print("=" * 72)

    # NOTE: ASCII only (>= instead of ≥) to avoid Unicode issues on Windows.
    print("\nDelay >= 15 min model:")
    print(f"  Samples       : {delay_metrics['n']:,}")
    print(f"  ROC-AUC       : {delay_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC        : {delay_metrics['pr_auc']:.4f}")
    print(f"  Brier score   : {delay_metrics['brier']:.4f}")

    print("\nCancellation model:")
    print(f"  Samples       : {cancel_metrics['n']:,}")
    print(f"  ROC-AUC       : {cancel_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC        : {cancel_metrics['pr_auc']:.4f}")
    print(f"  Brier score   : {cancel_metrics['brier']:.4f}")


if __name__ == "__main__":
    main()
