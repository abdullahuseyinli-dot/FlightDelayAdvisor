# src/generate_plots_only.py
from __future__ import annotations

from pathlib import Path
import sys
import warnings

import joblib
import pandas as pd

# --------------------------------------------------------------------
# Ensure we can import both train_models (from src) and app (from root)
# --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_models import (  # type: ignore  # noqa: E402
    DATA_PATH,
    MODELS_DIR,
    make_time_splits,
    evaluate_probs,
    find_best_threshold,
    create_diagnostic_plots_for_catboost,
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


if __name__ == "__main__":
    print(f"[INFO] Loading dataset from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    # ---------- Task 2: Cancellation (Cancelled) ----------
    print("\n[INFO] Preparing cancellation task splits for diagnostics only...")
    df_cancel = df[df["Cancelled"].isin([0, 1])].copy()
    df_cancel["Cancelled"] = df_cancel["Cancelled"].astype("int16")

    X_train_c, y_train_c, X_val_c, y_val_c, X_test_c, y_test_c = make_time_splits(
        df_cancel, "Cancelled"
    )

    # 1) Load calibrated CatBoost cancellation model from disk
    model_path = MODELS_DIR / "catboost_cancel_calibrated.joblib"
    print(f"[INFO] Loading calibrated CatBoost cancellation model from {model_path}")
    calibrator = joblib.load(model_path)

    # 2) Compute calibrated probabilities on val and test
    print("[INFO] Computing probabilities for validation and test splits...")
    y_proba_val_cal = calibrator.predict_proba(X_val_c)[:, 1]
    y_proba_test_cal = calibrator.predict_proba(X_test_c)[:, 1]

    # 3) Recompute best threshold on validation (F1-optimal)
    best_t, best_f1 = find_best_threshold(y_val_c, y_proba_val_cal, metric="f1")
    print(
        f"[INFO] Best validation threshold for cancel (CatBoost, calibrated): "
        f"{best_t:.3f} (F1={best_f1:.3f})"
    )

    # Optional: print metrics on test again for sanity
    evaluate_probs(
        y_test_c,
        y_proba_test_cal,
        context="cancel – CatBoost (calibrated, t* from val) [plots only rerun]",
        split_name="test",
        threshold=best_t,
        print_top_bucket=True,
        top_quantile=0.9,
    )

    # 4) Extract underlying CatBoost model (if available)
    base_model = None
    if hasattr(calibrator, "estimator"):
        base_model = calibrator.estimator
    elif hasattr(calibrator, "base_estimator"):
        base_model = calibrator.base_estimator

    if base_model is None:
        print(
            "[WARN] Could not find underlying CatBoost estimator on calibrated model. "
            "Feature-importance plot will be skipped; other plots will still be generated."
        )

    # 5) Generate diagnostic plots only
    create_diagnostic_plots_for_catboost(
        task_name="cancel",
        X_test=X_test_c,              # <-- NEW: pass test features
        y_test=y_test_c,
        y_proba_test_cal=y_proba_test_cal,
        best_threshold=best_t,
        base_model=base_model,
        is_cancellation=True,
    )

    print("\n[INFO] Finished regenerating CatBoost cancellation diagnostics.")
