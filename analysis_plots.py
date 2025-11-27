# analysis_plots.py
"""
Generate research-style diagnostic plots for delay and cancellation models.

- Compares LogReg, LGBM, CatBoost, TorchNN for both tasks
- Highlights the *best* model for each task
- Saves ~12 PNG figures under reports/figures_analysis/

Assumes:
    - Data: data/processed/bts_delay_2010_2024_balanced_research_weather.parquet
    - Models:
        models/logreg_delay15_calibrated.joblib
        models/lgbm_delay15_calibrated.joblib
        models/catboost_delay15_calibrated.joblib
        models/torchnn_delay15_calibrated.joblib

        models/logreg_cancel_calibrated.joblib
        models/lgbm_cancel_calibrated.joblib
        models/catboost_cancel_calibrated.joblib
        models/torchnn_cancel_calibrated.joblib

You can comment out any models you don’t have.
"""

from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

from catboost import CatBoostClassifier  # for feature importance

# -------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------
DATA_PATH = Path("data/processed/bts_delay_2010_2024_balanced_research_weather.parquet")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures_analysis")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# -------------------------------------------------------------------
# Features (must match training script)
# -------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Year",
    "Month",
    "DayOfWeek",
    "DayOfMonth",
    "DayOfYear",
    "DepHour",
    "IsWeekend",
    "IsHolidaySeason",
    "Distance",
    "RouteDelayRate",
    "RouteCancelRate",
    "RouteFlights",
    "AirlineDelayRate",
    "AirlineCancelRate",
    "AirlineFlights",
    "OriginSlotFlights",
    "DepHour_sin",
    "DepHour_cos",
    "Origin_tavg",
    "Origin_prcp",
    "Origin_snow",
    "Origin_wspd",
    "Origin_BadWeather",
    "Dest_tavg",
    "Dest_prcp",
    "Dest_snow",
    "Dest_wspd",
    "Dest_BadWeather",
]

CATEGORICAL_FEATURES = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Route",
    "Season",
    "DistanceBand",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# -------------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------------
def make_time_splits(df: pd.DataFrame, target_col: str):
    """
    Recreate the time-based train/val/test split used in training.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    X = df[FEATURE_COLS].copy()
    y = df[target_col].astype("int16")

    train_mask = df["Year"].between(2010, 2018)
    val_mask = df["Year"].between(2019, 2021)
    test_mask = df["Year"].between(2022, 2024)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(
        f"[SPLIT] {target_col}: "
        f"train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def find_best_threshold(y_true, y_proba, metric="f1"):
    """
    Scan thresholds to find the one that maximises F1 on validation.
    Used just for confusion matrices / qualitative interpretation.
    """
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


# -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------
def plot_roc_comparison(y_test, proba_dict, title, out_path):
    plt.figure(figsize=(6, 5))
    for name, p in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = roc_auc_score(y_test, p)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] Saved {out_path}")


def plot_pr_comparison(y_test, proba_dict, title, out_path):
    plt.figure(figsize=(6, 5))
    baseline = np.mean(y_test)
    for name, p in proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, p)
        ap = average_precision_score(y_test, p)
        plt.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.3f})")
    plt.hlines(baseline, 0, 1, colors="grey", linestyles="--", label="Baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] Saved {out_path}")


def plot_calibration_comparison(y_test, proba_dict, title, out_path, n_bins=20):
    plt.figure(figsize=(6, 5))
    for name, p in proba_dict.items():
        prob_true, prob_pred = calibration_curve(
            y_test, p, n_bins=n_bins, strategy="quantile"
        )
        plt.plot(prob_pred, prob_true, marker="o", lw=2, label=name)
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] Saved {out_path}")


def plot_probability_histograms(y_true, y_proba, title, out_path, bins=30):
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    plt.figure(figsize=(6, 5))
    plt.hist(
        y_proba_arr[y_true_arr == 0],
        bins=bins,
        alpha=0.6,
        label="Class 0",
        density=True,
    )
    plt.hist(
        y_proba_arr[y_true_arr == 1],
        bins=bins,
        alpha=0.6,
        label="Class 1",
        density=True,
    )
    plt.xlabel("Predicted probability of class 1")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] Saved {out_path}")


def plot_confusion(y_true, y_proba, threshold, title, out_path):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    classes = ["0", "1"]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] Saved {out_path}")


def plot_top_decile_bar(y_true, y_proba, title, out_path, quantile=0.9):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    cutoff = np.quantile(y_proba, quantile)
    mask_top = y_proba >= cutoff
    rate_top = float(y_true[mask_top].mean()) if mask_top.any() else np.nan
    mask_rest = ~mask_top
    rate_rest = float(y_true[mask_rest].mean()) if mask_rest.any() else np.nan

    plt.figure(figsize=(5, 4))
    plt.bar(["Top bucket", "Rest"], [rate_top, rate_rest])
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] Saved {out_path}")


def plot_catboost_feature_importance_from_calibrator(
    calibrator_path: Path, feature_names, title: str, out_path: Path, top_k: int = 25
):
    if not calibrator_path.exists():
        print(f"[WARN] {calibrator_path} not found; skipping CatBoost feature importance.")
        return

    calib = joblib.load(calibrator_path)
    base = getattr(calib, "base_estimator", None)

    if not isinstance(base, CatBoostClassifier):
        print(
            f"[WARN] base_estimator of {calibrator_path} is not CatBoostClassifier; "
            "cannot plot feature importance."
        )
        return

    importances = np.asarray(base.get_feature_importance(type="FeatureImportance"))
    indices = np.argsort(importances)[::-1][:top_k]
    feat_names = np.array(feature_names)[indices]
    feat_imp = importances[indices]

    plt.figure(figsize=(8, max(4, 0.35 * len(indices))))
    y_pos = np.arange(len(indices))
    plt.barh(y_pos, feat_imp[::-1])
    plt.yticks(y_pos, feat_names[::-1])
    plt.xlabel("Importance score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[FIG] Saved {out_path}")


# -------------------------------------------------------------------
# Model evaluation wrapper
# -------------------------------------------------------------------
def evaluate_models_for_task(
    task_name: str,
    model_files: dict,
    X_val,
    y_val,
    X_test,
    y_test,
):
    """
    Load calibrated models, get val/test probabilities and best thresholds.

    Returns:
        proba_val_dict, proba_test_dict, best_thresholds, metrics_df
    """
    from sklearn.metrics import (
        f1_score,
    )

    proba_val_dict = {}
    proba_test_dict = {}
    best_thresholds = {}
    rows = []

    for name, path in model_files.items():
        if not path.exists():
            print(f"[WARN] Model for {task_name} – {name} not found at {path}, skipping.")
            continue

        print(f"[LOAD] {task_name} – {name} from {path}")
        model = joblib.load(path)

        proba_val = model.predict_proba(X_val)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]

        t_best, f1_best = find_best_threshold(y_val, proba_val)

        # Basic metrics on test for table + legends
        auc = roc_auc_score(y_test, proba_test)
        ap = average_precision_score(y_test, proba_test)
        brier = brier_score_loss(y_test, proba_test)
        y_pred_best = (proba_test >= t_best).astype(int)
        f1_test = f1_score(y_test, y_pred_best)

        rows.append(
            {
                "task": task_name,
                "model": name,
                "auc_test": auc,
                "pr_auc_test": ap,
                "brier_test": brier,
                "best_t": t_best,
                "best_f1_val": f1_best,
                "f1_test_at_best_t": f1_test,
            }
        )

        proba_val_dict[name] = proba_val
        proba_test_dict[name] = proba_test
        best_thresholds[name] = t_best

    metrics_df = pd.DataFrame(rows).sort_values("pr_auc_test", ascending=False)
    return proba_val_dict, proba_test_dict, best_thresholds, metrics_df


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[INFO] Loading processed dataset from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    # ---------- Task 1: Delay >= 15 minutes ----------
    print("\n================ DELAY ≥ 15 MINUTES (ArrDel15) ================")
    df_delay = df[(df["Cancelled"] == 0) & df["ArrDel15"].notna()].copy()
    df_delay["ArrDel15"] = df_delay["ArrDel15"].astype("int16")

    X_train_d, y_train_d, X_val_d, y_val_d, X_test_d, y_test_d = make_time_splits(
        df_delay, "ArrDel15"
    )

    delay_models = {
        "LogReg": MODELS_DIR / "logreg_delay15_calibrated.joblib",
        "LGBM": MODELS_DIR / "lgbm_delay15_calibrated.joblib",
        "CatBoost": MODELS_DIR / "catboost_delay15_calibrated.joblib",
        "TorchNN": MODELS_DIR / "torchnn_delay15_calibrated.joblib",
    }

    (
        proba_val_delay,
        proba_test_delay,
        thresholds_delay,
        metrics_delay,
    ) = evaluate_models_for_task(
        "delay15",
        delay_models,
        X_val_d,
        y_val_d,
        X_test_d,
        y_test_d,
    )

    print("\n[METRICS] Delay models (sorted by PR-AUC on test):")
    print(metrics_delay.to_string(index=False))

    # Pick best delay model by PR-AUC on test (should be CatBoost)
    if not metrics_delay.empty:
        best_delay_name = metrics_delay.iloc[0]["model"]
        print(f"[BEST] Delay model: {best_delay_name}")
    else:
        best_delay_name = None

    # ----- Plots for delay task -----
    if proba_test_delay:
        # 1) ROC comparison
        plot_roc_comparison(
            y_test_d,
            proba_test_delay,
            "Delay ≥ 15 min – ROC comparison (test)",
            FIGURES_DIR / "delay15_roc_comparison.png",
        )

        # 2) PR comparison
        plot_pr_comparison(
            y_test_d,
            proba_test_delay,
            "Delay ≥ 15 min – Precision–Recall comparison (test)",
            FIGURES_DIR / "delay15_pr_comparison.png",
        )

        # 3) Calibration comparison (best 2–3 models)
        top_delay_models = (
            metrics_delay["model"].tolist()[:3] if len(metrics_delay) >= 3 else metrics_delay["model"].tolist()
        )
        proba_calib_delay = {
            m: proba_test_delay[m] for m in top_delay_models if m in proba_test_delay
        }
        if proba_calib_delay:
            plot_calibration_comparison(
                y_test_d,
                proba_calib_delay,
                "Delay ≥ 15 min – Calibration (test)",
                FIGURES_DIR / "delay15_calibration_comparison.png",
            )

        # 4) Probability histograms for best delay model
        if best_delay_name is not None:
            plot_probability_histograms(
                y_test_d,
                proba_test_delay[best_delay_name],
                f"Delay ≥ 15 min – {best_delay_name} probability distribution (test)",
                FIGURES_DIR / f"delay15_{best_delay_name}_prob_hist.png",
            )

            # 5) Confusion matrix at best threshold
            plot_confusion(
                y_test_d,
                proba_test_delay[best_delay_name],
                thresholds_delay[best_delay_name],
                f"Delay ≥ 15 min – {best_delay_name} confusion (t* from val)",
                FIGURES_DIR / f"delay15_{best_delay_name}_confusion.png",
            )

    # 6) Feature importance for CatBoost delay model
    plot_catboost_feature_importance_from_calibrator(
        MODELS_DIR / "catboost_delay15_calibrated.joblib",
        FEATURE_COLS,
        "Delay ≥ 15 min – CatBoost feature importance",
        FIGURES_DIR / "delay15_catboost_feature_importance.png",
        top_k=25,
    )

    # ---------- Task 2: Cancellation ----------
    print("\n================ CANCELLATION (Cancelled) ================")
    df_cancel = df[df["Cancelled"].isin([0, 1])].copy()
    df_cancel["Cancelled"] = df_cancel["Cancelled"].astype("int16")

    X_train_c, y_train_c, X_val_c, y_val_c, X_test_c, y_test_c = make_time_splits(
        df_cancel, "Cancelled"
    )

    cancel_models = {
        "LogReg": MODELS_DIR / "logreg_cancel_calibrated.joblib",
        "LGBM": MODELS_DIR / "lgbm_cancel_calibrated.joblib",
        "CatBoost": MODELS_DIR / "catboost_cancel_calibrated.joblib",
        "TorchNN": MODELS_DIR / "torchnn_cancel_calibrated.joblib",
    }

    (
        proba_val_cancel,
        proba_test_cancel,
        thresholds_cancel,
        metrics_cancel,
    ) = evaluate_models_for_task(
        "cancel",
        cancel_models,
        X_val_c,
        y_val_c,
        X_test_c,
        y_test_c,
    )

    print("\n[METRICS] Cancellation models (sorted by PR-AUC on test):")
    print(metrics_cancel.to_string(index=False))

    if not metrics_cancel.empty:
        best_cancel_name = metrics_cancel.iloc[0]["model"]  # expected: LGBM
        print(f"[BEST] Cancellation model: {best_cancel_name}")
    else:
        best_cancel_name = None

    # ----- Plots for cancellation task -----
    if proba_test_cancel:
        # 7) ROC comparison
        plot_roc_comparison(
            y_test_c,
            proba_test_cancel,
            "Cancellation – ROC comparison (test)",
            FIGURES_DIR / "cancel_roc_comparison.png",
        )

        # 8) PR comparison
        plot_pr_comparison(
            y_test_c,
            proba_test_cancel,
            "Cancellation – Precision–Recall comparison (test)",
            FIGURES_DIR / "cancel_pr_comparison.png",
        )

        # 9) Calibration comparison (best 2–3 models)
        top_cancel_models = (
            metrics_cancel["model"].tolist()[:3] if len(metrics_cancel) >= 3 else metrics_cancel["model"].tolist()
        )
        proba_calib_cancel = {
            m: proba_test_cancel[m] for m in top_cancel_models if m in proba_test_cancel
        }
        if proba_calib_cancel:
            plot_calibration_comparison(
                y_test_c,
                proba_calib_cancel,
                "Cancellation – Calibration (test)",
                FIGURES_DIR / "cancel_calibration_comparison.png",
            )

        # 10) Probability histograms for best cancellation model
        if best_cancel_name is not None:
            plot_probability_histograms(
                y_test_c,
                proba_test_cancel[best_cancel_name],
                f"Cancellation – {best_cancel_name} probability distribution (test)",
                FIGURES_DIR / f"cancel_{best_cancel_name}_prob_hist.png",
            )

            # 11) Confusion matrix at best threshold
            plot_confusion(
                y_test_c,
                proba_test_cancel[best_cancel_name],
                thresholds_cancel[best_cancel_name],
                f"Cancellation – {best_cancel_name} confusion (t* from val)",
                FIGURES_DIR / f"cancel_{best_cancel_name}_confusion.png",
            )

            # 12) Top-decile vs rest positive rate (best model)
            plot_top_decile_bar(
                y_test_c,
                proba_test_cancel[best_cancel_name],
                f"Cancellation – {best_cancel_name} top-decile vs rest positive rate",
                FIGURES_DIR / f"cancel_{best_cancel_name}_top_decile.png",
            )

    print("\n[DONE] All plots written to:", FIGURES_DIR.resolve())
