# train_models.py
"""
End-to-end training script for FlightDelayAdvisor.

Trains and evaluates multiple probabilistic classifiers for:
    1. Arrival delay ≥ 15 minutes (ArrDel15)
    2. Cancellations (Cancelled)

On top of the "baseline" models (Logistic Regression, LightGBM,
CatBoost, Tabular Neural Net), this script adds:

    - Rigorous time-based train/val/test splitting
    - Class-weighting and negative subsampling for cancellations
    - Probability calibration via isotonic regression
    - Rich scalar metrics + ASCII summary
    - Research-grade diagnostic plots (ROC, PR, calibration, etc.)
    - Fairness-style group diagnostics (per-airline performance)
    - Temporal robustness / drift analysis (metrics by year)
    - Optional bootstrap significance testing between models

All paths & feature definitions are consistent with:

    - prepare_dataset.py
    - add_weather_to_dataset.py  (adds *_weather features)
    - src/app.py                 (Streamlit application)

Outputs
-------
Models:
    models/logreg_delay15_calibrated.joblib
    models/lgbm_delay15_calibrated.joblib
    models/catboost_delay15_calibrated.joblib
    models/torchnn_delay15.pt + calibrator

    models/logreg_cancel_calibrated.joblib
    models/lgbm_cancel_calibrated.joblib
    models/catboost_cancel_calibrated.joblib
    models/torchnn_cancel.pt + calibrator

Reports:
    reports/metrics_summary.txt
    reports/figures/*.png
    reports/fairness/*.csv
    reports/drift/*.csv

This file is deliberately verbose and heavily commented so that
another student can reproduce and extend the experiments easily.
"""

from pathlib import Path
from typing import Callable, Dict, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# NEW: optional SHAP for global explainability (guarded import)
try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# --- PyTorch tabular NN imports ---
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------
DATA_PATH = Path("data/processed/bts_delay_2010_2024_balanced_research_weather.parquet")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
FAIRNESS_DIR = REPORTS_DIR / "fairness"
DRIFT_DIR = REPORTS_DIR / "drift"

for d in [REPORTS_DIR, FIGURES_DIR, FAIRNESS_DIR, DRIFT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

USE_GPU = True  # Set to False if you don't have CUDA
RANDOM_STATE = 42

# For cancellation, keep all positives but only some negatives in TRAIN
NEGATIVE_SUBSAMPLE_FRACTION_CANCEL = 0.3  # 30% of negatives

# Limit sample size for hyper‑parameter tuning (GridSearch)
MAX_GRIDSEARCH_SAMPLES = 300_000

# Limit sample size for Torch NN training (to keep runtime reasonable)
MAX_TORCH_TRAIN_SAMPLES = 500_000

# Limit sample size for bootstrap significance / SHAP / some heavy diagnostics
MAX_DIAG_SAMPLES = 100_000

# Global container for metrics we will also write to an ASCII report
METRIC_ROWS: List[Dict] = []

# -------------------------------------------------------------------
# Features (MUST match prepare_dataset + add_weather_to_dataset + app)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Features (MUST match prepare_dataset + add_weather_to_dataset + app)
# -------------------------------------------------------------------
NUMERIC_FEATURES = [
    # Calendar / time
    "Year",
    "Month",
    "DayOfWeek",
    "DayOfMonth",
    "DayOfYear",
    "DepHour",
    "IsWeekend",
    "IsHolidaySeason",
    "IsHoliday",
    "IsDayBeforeHoliday",
    "IsDayAfterHoliday",
    # Distance / basic route
    "Distance",
    # Historical reliability aggregates (2010–2018)
    "RouteDelayRate",
    "RouteCancelRate",
    "RouteFlights",
    "AirlineDelayRate",
    "AirlineCancelRate",
    "AirlineFlights",
    # Congestion features
    "OriginSlotFlights",
    "OriginDailyFlights",
    "OriginDailyFlightsAirline",
    # Hub flags (stored as 0/1 but treated as numeric)
    "IsAirlineHubAtOrigin",
    "IsAirlineHubAtDest",
    # Cyclical encodings
    "DepHour_sin",
    "DepHour_cos",
    # Weather (added by add_weather_to_dataset.py)
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
    "OriginState",
    "DestState",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# A coarse grouping of features for ablation / interpretation
FEATURE_GROUPS: Dict[str, List[str]] = {
    "calendar": [
        "Year",
        "Month",
        "DayOfWeek",
        "DayOfMonth",
        "DayOfYear",
        "IsWeekend",
        "IsHolidaySeason",
        "IsHoliday",
        "IsDayBeforeHoliday",
        "IsDayAfterHoliday",
        "DepHour",
        "DepHour_sin",
        "DepHour_cos",
        "Season",
    ],
    "route_airline_congestion": [
        "RouteDelayRate",
        "RouteCancelRate",
        "RouteFlights",
        "AirlineDelayRate",
        "AirlineCancelRate",
        "AirlineFlights",
        "OriginSlotFlights",
        "OriginDailyFlights",
        "OriginDailyFlightsAirline",
        "IsAirlineHubAtOrigin",
        "IsAirlineHubAtDest",
        "Origin",
        "Dest",
        "Route",
        "Distance",
        "DistanceBand",
    ],
    "weather": [
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
    ],
    # full list is implicit in FEATURE_COLS
}



# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def make_time_splits(df: pd.DataFrame, target_col: str):
    """
    Create time-based train/val/test splits for a given target.

    IMPORTANT: y_* are returned as pandas Series with indices aligned
    to X_* so that any subsampling by index remains consistent.
    """
    X = df[FEATURE_COLS].copy()
    y = df[target_col].astype("int16")  # Series, not .values

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


def top_decile_positive_rate(y_true, y_proba, quantile: float = 0.9) -> float:
    """
    Fraction of positives among the top (1-quantile) risk bucket.
    Useful for imbalanced tasks like cancellations.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if len(y_true) == 0:
        return np.nan

    cutoff = np.quantile(y_proba, quantile)
    mask = y_proba >= cutoff
    if not mask.any():
        return np.nan

    return float(y_true[mask].mean())


def evaluate_probs(
    y_true,
    y_proba,
    context: str,
    split_name: str,
    threshold: float = 0.5,
    print_top_bucket: bool = False,
    top_quantile: float = 0.9,
):
    """
    Print classification + probability metrics for a given threshold.

    For the report you care about ROC-AUC, PR-AUC, Brier; for imbalanced
    tasks (cancellation) we also report the positive rate in the top bucket.
    """
    global METRIC_ROWS

    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n[METRICS] {context} – {split_name} (threshold={threshold:.3f})")
    print(classification_report(y_true, y_pred, digits=3))

    metrics_row = {
        "context": context,
        "split": split_name,
        "threshold": float(threshold),
        "roc_auc": None,
        "pr_auc": None,
        "brier": None,
        "top_bucket_rate": None,
        "top_bucket_quantile": top_quantile if print_top_bucket else None,
        "accuracy": None,
        "precision_pos": None,
        "recall_pos": None,
        "f1_pos": None,
    }

    try:
        auc = roc_auc_score(y_true, y_proba)
        print(f"ROC-AUC: {auc:.4f}")
        metrics_row["roc_auc"] = float(auc)
    except ValueError:
        print("ROC-AUC not defined (single class in this split).")

    try:
        ap = average_precision_score(y_true, y_proba)
        print(f"PR-AUC: {ap:.4f}")
        metrics_row["pr_auc"] = float(ap)
    except ValueError:
        print("PR-AUC not defined (single class in this split).")

    try:
        brier = brier_score_loss(y_true, y_proba)
        print(f"Brier score: {brier:.4f}")
        metrics_row["brier"] = float(brier)
    except ValueError:
        print("Brier score not defined.")

    # Extra scalar metrics for ASCII report
    try:
        metrics_row["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics_row["precision_pos"] = float(precision_score(y_true, y_pred))
        metrics_row["recall_pos"] = float(recall_score(y_true, y_pred))
        metrics_row["f1_pos"] = float(f1_score(y_true, y_pred))
    except Exception:
        pass

    if print_top_bucket:
        rate = top_decile_positive_rate(y_true, y_proba, quantile=top_quantile)
        if not np.isnan(rate):
            top_pct = int((1.0 - top_quantile) * 100)
            print(f"Top {top_pct}% bucket positive rate: {rate:.4f}")
            metrics_row["top_bucket_rate"] = float(rate)

    METRIC_ROWS.append(metrics_row)
    return metrics_row


def find_best_threshold(y_true, y_proba, metric="f1"):
    """
    Scan thresholds on the validation set to find the one that maximises F1.
    This is only for reporting; in the app we use probabilities directly.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred)
        else:
            raise ValueError("Unsupported metric")
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


def check_two_classes(y, name: str) -> bool:
    classes = np.unique(y)
    if len(classes) < 2:
        print(
            f"[WARN] Target '{name}' has only one class in this split: {classes}. "
            f"Skipping this model."
        )
        return False
    return True


def subsample_negatives(X_train, y_train, fraction_neg: float, random_state: int):
    """
    For highly imbalanced tasks (like cancellation), keep all positives
    but only a fraction of negatives in the TRAINING set to give the model
    a more balanced signal. Validation and test remain untouched.

    Returns X_sub (DataFrame) and y_sub (Series) with aligned indices.
    """
    # Ensure y is a Series aligned to X_train
    if isinstance(y_train, pd.Series):
        y_series = y_train
    else:
        y_series = pd.Series(np.asarray(y_train), index=X_train.index)

    y_arr = y_series.values
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X_train, y_series  # nothing to do

    n_neg_keep = max(1, int(len(neg_idx) * fraction_neg))
    rng = np.random.RandomState(random_state)
    keep_neg_idx = rng.choice(neg_idx, size=n_neg_keep, replace=False)

    keep_idx = np.concatenate([pos_idx, keep_neg_idx])
    rng.shuffle(keep_idx)

    X_sub = X_train.iloc[keep_idx]
    y_sub = y_series.iloc[keep_idx]

    print(
        f"[SUBSAMPLE] Kept {len(y_sub):,} training samples "
        f"({len(pos_idx)} positives, {n_neg_keep} negatives)"
    )
    return X_sub, y_sub


def compute_class_weights(y_train):
    """Return class weights [w0, w1] based on inverse frequency."""
    y_arr = np.asarray(y_train)
    pos = int(np.sum(y_arr))
    neg = len(y_arr) - pos
    if pos == 0:
        return [1.0, 1.0]
    ratio = neg / pos
    return [1.0, float(ratio)]


# -------------------------------------------------------------------
# Plotting helpers (researcher-level diagnostics)
# -------------------------------------------------------------------
def plot_roc_curve(y_true, y_proba, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1, color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_proba, title, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    baseline = np.mean(y_true)
    plt.hlines(baseline, 0, 1, colors="grey", linestyles="--", label="Baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_calibration(y_true, y_proba, title, out_path, n_bins: int = 20):
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="quantile"
    )
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", lw=2, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_probability_histograms(y_true, y_proba, title, out_path, bins: int = 30):
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

    # annotate cells
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


def plot_top_decile_bar(y_true, y_proba, title, out_path, quantile: float = 0.9):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    cutoff = np.quantile(y_proba, quantile)
    mask_top = y_proba >= cutoff
    rate_top = float(y_true[mask_top].mean()) if mask_top.any() else np.nan
    mask_rest = ~mask_top
    rate_rest = float(y_true[mask_rest].mean()) if mask_rest.any() else np.nan

    plt.figure(figsize=(5, 4))
    plt.bar(["Top decile", "Rest"], [rate_top, rate_rest])
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_catboost_feature_importance(
    cat_model, feature_names, title, out_path, top_k: int = 25
):
    importances = cat_model.get_feature_importance(type="FeatureImportance")
    importances = np.asarray(importances)
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


def create_diagnostic_plots_for_catboost(
    task_name: str,
    y_test,
    y_proba_test_cal,
    best_threshold: float,
    X_test: pd.DataFrame,
    base_model=None,
    is_cancellation: bool = False,
):
    """
    Generate a suite of researcher-level plots for the calibrated CatBoost model.
    If base_model is not provided or does not support get_feature_importance,
    the feature-importance / SHAP plots are skipped gracefully.
    """
    prefix = f"{task_name}_catboost_calibrated"

    # ROC and PR curves
    plot_roc_curve(
        y_test,
        y_proba_test_cal,
        title=f"{task_name} – CatBoost ROC (test)",
        out_path=FIGURES_DIR / f"{prefix}_roc.png",
    )
    plot_pr_curve(
        y_test,
        y_proba_test_cal,
        title=f"{task_name} – CatBoost Precision–Recall (test)",
        out_path=FIGURES_DIR / f"{prefix}_pr.png",
    )

    # Calibration curve
    plot_calibration(
        y_test,
        y_proba_test_cal,
        title=f"{task_name} – CatBoost calibration (test)",
        out_path=FIGURES_DIR / f"{prefix}_calibration.png",
    )

    # Probability histograms
    plot_probability_histograms(
        y_test,
        y_proba_test_cal,
        title=f"{task_name} – CatBoost probability distribution (test)",
        out_path=FIGURES_DIR / f"{prefix}_prob_hist.png",
    )

    # Confusion matrix at best threshold
    plot_confusion(
        y_test,
        y_proba_test_cal,
        threshold=best_threshold,
        title=f"{task_name} – CatBoost confusion (t*={best_threshold:.2f})",
        out_path=FIGURES_DIR / f"{prefix}_confusion.png",
    )

    # Top decile plot for imbalanced cancellation task
    if is_cancellation:
        plot_top_decile_bar(
            y_test,
            y_proba_test_cal,
            title=f"{task_name} – positive rate in top decile vs rest (test)",
            out_path=FIGURES_DIR / f"{prefix}_top_decile.png",
        )

    # Feature importance plot (optional, robust)
    if base_model is not None:
        try:
            plot_catboost_feature_importance(
                base_model,
                FEATURE_COLS,
                title=f"{task_name} – CatBoost feature importance",
                out_path=FIGURES_DIR / f"{prefix}_feature_importance.png",
            )
        except Exception as e:
            print(f"[WARN] Skipping feature importance plot for {task_name}: {e}")

        # Optional SHAP summary (can be heavy; run on subsample)
        if SHAP_AVAILABLE:
            try:
                n_sample = min(MAX_DIAG_SAMPLES, len(X_test))
                if n_sample >= 1000:
                    X_sample = X_test.sample(n=n_sample, random_state=RANDOM_STATE)
                else:
                    X_sample = X_test.copy()

                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(X_sample)

                # SHAP summary plot (bar + beeswarm style)
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    show=False,
                    plot_type="bar",
                    max_display=25,
                )
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / f"{prefix}_shap_summary_bar.png", dpi=150)
                plt.close()

                shap.summary_plot(
                    shap_values,
                    X_sample,
                    show=False,
                    max_display=25,
                )
                plt.tight_layout()
                plt.savefig(
                    FIGURES_DIR / f"{prefix}_shap_summary_beeswarm.png", dpi=150
                )
                plt.close()
            except Exception as e:
                print(f"[WARN] SHAP plots failed for {task_name}: {e}")


def create_diagnostic_plots_generic(
    task_name: str,
    model_name: str,
    y_test,
    y_proba_test_cal,
    best_threshold: float,
    is_cancellation: bool = False,
):
    """
    Generic diagnostic plot suite: ROC, PR, calibration, histogram, confusion,
    and optionally top-decile enrichment (for imbalanced tasks).
    """
    prefix = f"{task_name}_{model_name}_calibrated"

    plot_roc_curve(
        y_test,
        y_proba_test_cal,
        title=f"{task_name} – {model_name} ROC (test)",
        out_path=FIGURES_DIR / f"{prefix}_roc.png",
    )
    plot_pr_curve(
        y_test,
        y_proba_test_cal,
        title=f"{task_name} – {model_name} Precision–Recall (test)",
        out_path=FIGURES_DIR / f"{prefix}_pr.png",
    )
    plot_calibration(
        y_test,
        y_proba_test_cal,
        title=f"{task_name} – {model_name} calibration (test)",
        out_path=FIGURES_DIR / f"{prefix}_calibration.png",
    )
    plot_probability_histograms(
        y_test,
        y_proba_test_cal,
        title=f"{task_name} – {model_name} probability distribution (test)",
        out_path=FIGURES_DIR / f"{prefix}_prob_hist.png",
    )
    plot_confusion(
        y_test,
        y_proba_test_cal,
        threshold=best_threshold,
        title=f"{task_name} – {model_name} confusion (t*={best_threshold:.2f})",
        out_path=FIGURES_DIR / f"{prefix}_confusion.png",
    )

    if is_cancellation:
        plot_top_decile_bar(
            y_test,
            y_proba_test_cal,
            title=f"{task_name} – {model_name} top-decile vs rest (test)",
            out_path=FIGURES_DIR / f"{prefix}_top_decile.png",
        )


# -------------------------------------------------------------------
# Fairness-style group diagnostics & temporal drift
# -------------------------------------------------------------------
def evaluate_group_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    group: pd.Series,
    group_name: str,
    context: str,
    min_samples: int = 1000,
):
    """
    Compute simple group-wise metrics (AUC, Brier, prevalence) for
    high-level "fairness" diagnostics, e.g. per airline.

    Results are saved as CSV and a bar plot of Brier score.
    """
    df_g = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=int),
            "y_proba": np.asarray(y_proba, dtype=float),
            "group": group.values,
        }
    )

    rows = []
    for g, df_sub in df_g.groupby("group"):
        n = len(df_sub)
        if n < min_samples:
            continue
        y_t = df_sub["y_true"].values
        y_p = df_sub["y_proba"].values
        try:
            auc = roc_auc_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan
        except Exception:
            auc = np.nan
        try:
            brier = brier_score_loss(y_t, y_p)
        except Exception:
            brier = np.nan
        prevalence = float(y_t.mean())
        rows.append(
            {
                group_name: g,
                "n_samples": int(n),
                "auc": float(auc) if auc == auc else np.nan,
                "brier": float(brier) if brier == brier else np.nan,
                "prevalence": prevalence,
            }
        )

    if not rows:
        print(f"[FAIRNESS] No groups with ≥{min_samples} samples for {context}.")
        return

    df_metrics = pd.DataFrame(rows).sort_values("brier")
    out_csv = FAIRNESS_DIR / f"{context}_by_{group_name}.csv"
    df_metrics.to_csv(out_csv, index=False)
    print(f"[FAIRNESS] Wrote group metrics for {context} to {out_csv}")

    # Bar plot of Brier score (lower is better)
    plt.figure(figsize=(max(6, 0.4 * len(df_metrics)), 4))
    plt.bar(df_metrics[group_name].astype(str), df_metrics["brier"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Brier score (lower is better)")
    plt.title(f"{context}: Brier score by {group_name}")
    plt.tight_layout()
    out_png = FIGURES_DIR / f"{context}_brier_by_{group_name}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()


def evaluate_drift_by_year(
    df: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    model,
    task_name: str,
    split_label: str,
):
    """
    Evaluate AUC / Brier by calendar year for a given model to get a handle
    on temporal robustness / dataset shift.
    """
    years = sorted(df["Year"].unique())
    rows = []
    for year in years:
        mask = df["Year"] == year
        if not mask.any():
            continue
        y_true = y_all[mask]
        X_year = X_all[mask]
        if len(np.unique(y_true)) < 2:
            continue
        y_proba = model.predict_proba(X_year)[:, 1]
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = np.nan
        try:
            brier = brier_score_loss(y_true, y_proba)
        except Exception:
            brier = np.nan
        rows.append({"Year": int(year), "AUC": float(auc), "Brier": float(brier)})

    if not rows:
        print(f"[DRIFT] No per-year metrics for {task_name} ({split_label}).")
        return

    df_year = pd.DataFrame(rows).sort_values("Year")
    out_csv = DRIFT_DIR / f"{task_name}_{split_label}_metrics_by_year.csv"
    df_year.to_csv(out_csv, index=False)
    print(f"[DRIFT] Wrote per-year metrics for {task_name} to {out_csv}")

    # Line plots
    plt.figure(figsize=(7, 4))
    plt.plot(df_year["Year"], df_year["AUC"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("ROC-AUC")
    plt.title(f"{task_name} – ROC-AUC by year ({split_label})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{task_name}_{split_label}_auc_by_year.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(df_year["Year"], df_year["Brier"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Brier score")
    plt.title(f"{task_name} – Brier by year ({split_label})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{task_name}_{split_label}_brier_by_year.png", dpi=150)
    plt.close()


# -------------------------------------------------------------------
# Bootstrap significance testing between models
# -------------------------------------------------------------------
def bootstrap_metric_difference(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 500,
    max_samples: int = MAX_DIAG_SAMPLES,
    higher_is_better: bool = True,
    context: str = "",
):
    """
    Simple non-parametric bootstrap to estimate the distribution of the
    difference in a scalar metric between two models (A - B).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels on a test set.
    proba_a, proba_b : np.ndarray
        Predicted probabilities from model A and model B.
    metric_fn : callable
        Function mapping (y_true, y_proba) -> scalar metric.
    higher_is_better : bool
        If False, diffs are negated so that positive values indicate that
        model A is better.

    Saves a histogram of bootstrap differences and prints a 95% CI.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba_a = np.asarray(proba_a, dtype=float)
    proba_b = np.asarray(proba_b, dtype=float)
    assert y_true.shape == proba_a.shape == proba_b.shape

    n = len(y_true)
    if n == 0:
        print(f"[BOOT] No data for bootstrap comparison: {context}")
        return

    # Optional downsampling for speed
    if n > max_samples:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, size=max_samples, replace=False)
        y_true = y_true[idx]
        proba_a = proba_a[idx]
        proba_b = proba_b[idx]
        n = len(y_true)

    rng = np.random.RandomState(RANDOM_STATE + 123)
    diffs = []

    for _ in range(n_bootstrap):
        b_idx = rng.randint(0, n, size=n)
        y_b = y_true[b_idx]
        pa_b = proba_a[b_idx]
        pb_b = proba_b[b_idx]
        try:
            m_a = metric_fn(y_b, pa_b)
            m_b = metric_fn(y_b, pb_b)
            diff = m_a - m_b
            if not higher_is_better:
                diff = -diff
            diffs.append(diff)
        except Exception:
            continue

    if not diffs:
        print(f"[BOOT] Could not compute bootstrap diffs for {context}")
        return

    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    print(
        f"[BOOT] {context}: mean diff={mean_diff:.4f}, "
        f"95% CI=({ci_low:.4f}, {ci_high:.4f})"
    )

    plt.figure(figsize=(6, 4))
    plt.hist(diffs, bins=30, alpha=0.8)
    plt.axvline(0.0, color="black", linestyle="--", label="No difference")
    plt.axvline(mean_diff, color="red", linestyle="-", label="Mean diff")
    plt.xlabel("Metric difference (A − B)")
    plt.ylabel("Bootstrap frequency")
    plt.title(context)
    plt.legend()
    plt.tight_layout()
    out_png = FIGURES_DIR / (context.replace(" ", "_") + "_bootstrap_diff.png")
    plt.savefig(out_png, dpi=150)
    plt.close()


# -------------------------------------------------------------------
# Hyper-parameter tuning helpers (GridSearch etc.)
# -------------------------------------------------------------------
def tune_logistic_hyperparams(X_train, y_train, task_name: str):
    """
    Small GridSearchCV over C for LogisticRegression
    on a subsample of the training set.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    clf = LogisticRegression(
        solver="saga",
        max_iter=300,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    # Subsample for grid search (to keep runtime reasonable)
    if len(y_train) > MAX_GRIDSEARCH_SAMPLES:
        X_gs, _, y_gs, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_GRIDSEARCH_SAMPLES,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
    else:
        X_gs, y_gs = X_train, y_train

    param_grid = {
        "clf__C": [0.1, 0.3, 1.0, 3.0],
    }

    print(
        f"[GRID] Searching Logistic Regression hyper‑params for {task_name} "
        f"on {len(y_gs):,} samples..."
    )
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
        refit=False,
    )
    grid.fit(X_gs, y_gs)

    print(
        f"[GRID] Best LogReg params for {task_name}: {grid.best_params_}, "
        f"best CV ROC‑AUC={grid.best_score_:.4f}"
    )

    # Return a fresh pipeline with the best hyperparameters
    pipe.set_params(**grid.best_params_)
    return pipe


def tune_lgbm_hyperparams(X_train, y_train, task_name: str):
    """
    Small GridSearchCV for LightGBM on a subsample of the training set.
    We use a sklearn-style pipeline with one-hot encoding.
    """
    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    clf = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
        # If you build GPU LightGBM, you can set device_type="gpu" here
        # device_type="gpu",
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    # Subsample for grid search (to keep runtime reasonable)
    if len(y_train) > MAX_GRIDSEARCH_SAMPLES:
        X_gs, _, y_gs, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_GRIDSEARCH_SAMPLES,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
    else:
        X_gs, y_gs = X_train, y_train

    param_grid = {
        "clf__num_leaves": [31, 63],
        "clf__n_estimators": [200, 400],
        "clf__learning_rate": [0.05, 0.1],
        "clf__min_child_samples": [20, 100],
    }

    print(
        f"[GRID] Searching LightGBM hyper‑params for {task_name} "
        f"on {len(y_gs):,} samples..."
    )
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
        refit=False,
    )
    grid.fit(X_gs, y_gs)

    print(
        f"[GRID] Best LightGBM params for {task_name}: {grid.best_params_}, "
        f"best CV ROC‑AUC={grid.best_score_:.4f}"
    )

    pipe.set_params(**grid.best_params_)
    return pipe


def tune_catboost_hyperparams(
    X_train,
    y_train,
    X_val,
    y_val,
    cat_feature_indices,
    class_weights,
    task_name: str,
    max_train_samples: int = 300_000,
    max_val_samples: int = 150_000,
):
    """
    Lightweight manual grid search for CatBoost hyper‑parameters,
    using AUC on the validation set as objective.
    Assumes y_train and y_val are pandas Series with aligned indices.
    """
    if isinstance(X_train, pd.DataFrame) and len(X_train) > max_train_samples:
        X_tr_sub = X_train.sample(n=max_train_samples, random_state=RANDOM_STATE)
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_tr_sub = y_train.loc[X_tr_sub.index]
        else:
            y_tr_sub = pd.Series(np.asarray(y_train), index=X_train.index).loc[
                X_tr_sub.index
            ]
    else:
        X_tr_sub, y_tr_sub = X_train, y_train

    if isinstance(X_val, pd.DataFrame) and len(X_val) > max_val_samples:
        X_val_sub = X_val.sample(n=max_val_samples, random_state=RANDOM_STATE)
        if isinstance(y_val, (pd.Series, pd.DataFrame)):
            y_val_sub = y_val.loc[X_val_sub.index]
        else:
            y_val_sub = pd.Series(np.asarray(y_val), index=X_val.index).loc[
                X_val_sub.index
            ]
    else:
        X_val_sub, y_val_sub = X_val, y_val

    train_pool = Pool(X_tr_sub, y_tr_sub, cat_features=cat_feature_indices)
    val_pool = Pool(X_val_sub, y_val_sub, cat_features=cat_feature_indices)

    # Small grid of reasonable CatBoost configurations
    param_grid = [
        {"depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3.0, "iterations": 400},
        {"depth": 8, "learning_rate": 0.1, "l2_leaf_reg": 3.0, "iterations": 600},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0, "iterations": 600},
        {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 5.0, "iterations": 800},
    ]

    best_params = None
    best_auc = -np.inf

    print(f"[GRID] Tuning CatBoost hyper‑params for {task_name}...")
    for i, params in enumerate(param_grid, start=1):
        print(f"[GRID]  Trial {i}/{len(param_grid)}: {params}")
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            class_weights=class_weights,
            task_type="GPU" if USE_GPU and torch.cuda.is_available() else "CPU",
            devices="0" if USE_GPU and torch.cuda.is_available() else None,
            verbose=False,
            allow_writing_files=False,
            border_count=128,
            **params,
        )
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        y_val_proba = model.predict_proba(val_pool)[:, 1]
        auc = roc_auc_score(y_val_sub, y_val_proba)
        print(f"[GRID]    Validation ROC‑AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_params = params

    print(
        f"[GRID] Best CatBoost params for {task_name}: {best_params}, "
        f"val ROC‑AUC={best_auc:.4f}"
    )
    return best_params


# -------------------------------------------------------------------
# Training functions: Logistic, LightGBM, CatBoost
# -------------------------------------------------------------------
def train_logistic_baseline(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    task_name: str,
    print_top_bucket=False,
):
    """
    Baseline model: Logistic Regression with one-hot encoding.
    Includes a GridSearchCV over C before final training.
    """
    if not check_two_classes(y_train, f"{task_name} (train)"):
        return None

    # First: small grid search to choose C
    pipe = tune_logistic_hyperparams(X_train, y_train, task_name)

    # Then: refit tuned pipeline on the full training set
    print(f"\n[TRAIN] Logistic Regression baseline for {task_name} (refit with best C)")
    pipe.fit(X_train, y_train)

    # Evaluate on test before calibration
    y_proba_test = pipe.predict_proba(X_test)[:, 1]
    evaluate_probs(
        y_test,
        y_proba_test,
        f"{task_name} – LogReg (uncalibrated)",
        "test",
        threshold=0.5,
        print_top_bucket=print_top_bucket,
    )

    # Calibrate on validation set
    print(
        f"[CALIB] Calibrating Logistic Regression for {task_name} using isotonic regression..."
    )
    calibrator = CalibratedClassifierCV(pipe, cv="prefit", method="isotonic")
    calibrator.fit(X_val, y_val)

    # Evaluate calibrated model on test (threshold 0.5)
    y_proba_test_cal = calibrator.predict_proba(X_test)[:, 1]
    evaluate_probs(
        y_test,
        y_proba_test_cal,
        f"{task_name} – LogReg (calibrated, t=0.5)",
        "test",
        threshold=0.5,
        print_top_bucket=print_top_bucket,
    )

    # Find best threshold on VALIDATION set (using calibrated model)
    y_proba_val_cal = calibrator.predict_proba(X_val)[:, 1]
    best_t, best_f1 = find_best_threshold(y_val, y_proba_val_cal, metric="f1")
    print(
        f"[THRESH] Best validation threshold for {task_name} (LogReg, calibrated): "
        f"{best_t:.3f} (F1={best_f1:.3f})"
    )

    # Evaluate calibrated model on test with best threshold
    evaluate_probs(
        y_test,
        y_proba_test_cal,
        f"{task_name} – LogReg (calibrated, t* from val)",
        "test",
        threshold=best_t,
        print_top_bucket=print_top_bucket,
    )

    # Diagnostics for report
    create_diagnostic_plots_generic(
        task_name=task_name,
        model_name="LogReg",
        y_test=y_test,
        y_proba_test_cal=y_proba_test_cal,
        best_threshold=best_t,
        is_cancellation=(task_name == "cancel"),
    )

    model_path = MODELS_DIR / f"logreg_{task_name}_calibrated.joblib"
    joblib.dump(calibrator, model_path)
    print(f"[OK] Saved calibrated Logistic Regression for {task_name} to {model_path}")

    return calibrator


def train_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    task_name: str,
    print_top_bucket=False,
):
    """
    Additional model: LightGBM with one-hot encoding and small GridSearch.
    """
    if not check_two_classes(y_train, f"{task_name} (train)"):
        return None

    # Grid search for LightGBM hyperparams
    pipe = tune_lgbm_hyperparams(X_train, y_train, task_name)

    print(f"\n[TRAIN] LightGBM for {task_name} (refit with best params)")
    pipe.fit(X_train, y_train)

    # Evaluate uncalibrated
    y_proba_test = pipe.predict_proba(X_test)[:, 1]
    evaluate_probs(
        y_test,
        y_proba_test,
        f"{task_name} – LGBM (uncalibrated)",
        "test",
        threshold=0.5,
        print_top_bucket=print_top_bucket,
    )

    # Calibrate on validation set
    print(f"[CALIB] Calibrating LightGBM for {task_name} using isotonic regression...")
    calibrator = CalibratedClassifierCV(pipe, cv="prefit", method="isotonic")
    calibrator.fit(X_val, y_val)

    # Evaluate calibrated model (threshold 0.5)
    y_proba_test_cal = calibrator.predict_proba(X_test)[:, 1]
    evaluate_probs(
        y_test,
        y_proba_test_cal,
        f"{task_name} – LGBM (calibrated, t=0.5)",
        "test",
        threshold=0.5,
        print_top_bucket=print_top_bucket,
    )

    # Best threshold
    y_proba_val_cal = calibrator.predict_proba(X_val)[:, 1]
    best_t, best_f1 = find_best_threshold(y_val, y_proba_val_cal, metric="f1")
    print(
        f"[THRESH] Best validation threshold for {task_name} (LGBM, calibrated): "
        f"{best_t:.3f} (F1={best_f1:.3f})"
    )

    evaluate_probs(
        y_test,
        y_proba_test_cal,
        f"{task_name} – LGBM (calibrated, t* from val)",
        "test",
        threshold=best_t,
        print_top_bucket=print_top_bucket,
    )

    # Diagnostics for report
    create_diagnostic_plots_generic(
        task_name=task_name,
        model_name="LGBM",
        y_test=y_test,
        y_proba_test_cal=y_proba_test_cal,
        best_threshold=best_t,
        is_cancellation=(task_name == "cancel"),
    )

    model_path = MODELS_DIR / f"lgbm_{task_name}_calibrated.joblib"
    joblib.dump(calibrator, model_path)
    print(f"[OK] Saved calibrated LightGBM for {task_name} to {model_path}")

    return calibrator


def train_catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    task_name: str,
    print_top_bucket=False,
):
    """
    Main tree model: CatBoostClassifier (GPU-ready) with hyper‑parameter tuning
    and probability calibration.
    """
    if not check_two_classes(y_train, f"{task_name} (train)"):
        return None

    print(f"\n[TRAIN] CatBoost for {task_name}")

    cat_feature_indices = [X_train.columns.get_loc(col) for col in CATEGORICAL_FEATURES]

    class_weights = compute_class_weights(y_train)
    print(f"[INFO]  Class weights for {task_name}: {class_weights}")

    # Hyper‑parameter tuning on a subsample
    best_params = tune_catboost_hyperparams(
        X_train,
        y_train,
        X_val,
        y_val,
        cat_feature_indices,
        class_weights,
        task_name,
    )

    # Final model with tuned hyper‑parameters
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)

    base_params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        class_weights=class_weights,
        task_type="GPU" if USE_GPU and torch.cuda.is_available() else "CPU",
        devices="0" if USE_GPU and torch.cuda.is_available() else None,
        verbose=100,
        use_best_model=True,
        allow_writing_files=False,
        border_count=128,
        od_type="Iter",
        od_wait=100,
    )
    if best_params is not None:
        base_params.update(best_params)

    model = CatBoostClassifier(**base_params)
    model.fit(train_pool, eval_set=val_pool)

    # Evaluate on test set (uncalibrated) at threshold 0.5
    test_pool = Pool(X_test, cat_features=cat_feature_indices)
    y_proba_test = model.predict_proba(test_pool)[:, 1]
    evaluate_probs(
        y_test,
        y_proba_test,
        f"{task_name} – CatBoost (uncalibrated)",
        "test",
        threshold=0.5,
        print_top_bucket=print_top_bucket,
    )

    # Calibrate using validation set
    print(f"[CALIB] Calibrating CatBoost for {task_name} using isotonic regression...")
    calibrator = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    calibrator.fit(X_val, y_val)

    # Evaluate calibrated model on test (threshold 0.5)
    y_proba_test_cal = calibrator.predict_proba(X_test)[:, 1]
    evaluate_probs(
        y_test,
        y_proba_test_cal,
        f"{task_name} – CatBoost (calibrated, t=0.5)",
        "test",
        threshold=0.5,
        print_top_bucket=print_top_bucket,
    )

    # Best threshold on validation set
    y_proba_val_cal = calibrator.predict_proba(X_val)[:, 1]
    best_t, best_f1 = find_best_threshold(y_val, y_proba_val_cal, metric="f1")
    print(
        f"[THRESH] Best validation threshold for {task_name} (CatBoost, calibrated): "
        f"{best_t:.3f} (F1={best_f1:.3f})"
    )

    # Evaluate calibrated model on test with best threshold
    evaluate_probs(
        y_test,
        y_proba_test_cal,
        f"{task_name} – CatBoost (calibrated, t* from val)",
        "test",
        threshold=best_t,
        print_top_bucket=print_top_bucket,
    )

    # Research plots for calibrated CatBoost on test data
    create_diagnostic_plots_for_catboost(
        task_name=task_name,
        y_test=y_test,
        y_proba_test_cal=y_proba_test_cal,
        best_threshold=best_t,
        X_test=X_test,
        base_model=model,
        is_cancellation=(task_name == "cancel"),
    )

    model_path = MODELS_DIR / f"catboost_{task_name}_calibrated.joblib"
    joblib.dump(calibrator, model_path)
    print(f"[OK] Saved calibrated CatBoost for {task_name} to {model_path}")

    return calibrator


# -------------------------------------------------------------------
# Simple two-model ensemble (CatBoost + Logistic Regression)
# -------------------------------------------------------------------
def evaluate_two_model_ensemble(
    model_a,
    model_b,
    X_val,
    y_val,
    X_test,
    y_test,
    task_name: str,
    name_a: str,
    name_b: str,
    print_top_bucket: bool = False,
):
    """
    Simple ensemble of two calibrated models by weighted averaging of probabilities.
    Weight w is tuned on validation to minimize Brier score.
    """
    w_candidates = [0.3, 0.5, 0.7]

    proba_val_a = model_a.predict_proba(X_val)[:, 1]
    proba_val_b = model_b.predict_proba(X_val)[:, 1]

    best_w = 0.5
    best_brier = float("inf")

    for w in w_candidates:
        p_val = w * proba_val_a + (1.0 - w) * proba_val_b
        try:
            brier = brier_score_loss(y_val, p_val)
        except ValueError:
            continue
        if brier < best_brier:
            best_brier = brier
            best_w = w

    print(
        f"[ENSEMBLE] Best weight for {task_name} ensemble ({name_a},{name_b}) "
        f"on validation: w={best_w:.2f}, Brier={best_brier:.4f}"
    )

    proba_test_a = model_a.predict_proba(X_test)[:, 1]
    proba_test_b = model_b.predict_proba(X_test)[:, 1]
    p_test = best_w * proba_test_a + (1.0 - best_w) * proba_test_b

    context = (
        f"{task_name} – Ensemble ({name_a}*{best_w:.2f} + "
        f"{name_b}*{1.0 - best_w:.2f})"
    )
    evaluate_probs(
        y_test,
        p_test,
        context,
        "test",
        threshold=0.5,
        print_top_bucket=print_top_bucket,
    )


# -------------------------------------------------------------------
# NEW: PyTorch tabular NN (embeddings + MLP) for delay / cancellation
# -------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, numeric_cols, cat_cols):
        self.X_num = X[numeric_cols].astype(np.float32).values

        cat_arrays = []
        for col in cat_cols:
            # X[col] is categorical for the whole df; codes are consistent
            codes = X[col].cat.codes.to_numpy()
            # Reserve 0 for "unknown"/NaN; shift valid codes by +1
            codes = np.where(codes < 0, 0, codes + 1).astype(np.int64)
            cat_arrays.append(codes)

        if cat_arrays:
            self.X_cat = np.stack(cat_arrays, axis=1)
        else:
            self.X_cat = np.zeros((len(X), 0), dtype=np.int64)

        self.y = y.to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


class TabularNN(nn.Module):
    def __init__(
        self, num_numeric: int, embedding_sizes, hidden_dims=(256, 128), dropout=0.2
    ):
        """
        embedding_sizes: list of (num_embeddings, emb_dim) for each categorical feature.
        """
        super().__init__()
        self.embeds = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, emb_dim)
                for num_embeddings, emb_dim in embedding_sizes
            ]
        )
        emb_total_dim = sum(emb_dim for _, emb_dim in embedding_sizes)
        input_dim = num_numeric + emb_total_dim

        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        if self.embeds:
            emb_list = []
            for i, emb in enumerate(self.embeds):
                emb_list.append(emb(x_cat[:, i]))
            x = torch.cat([x_num] + emb_list, dim=1)
        else:
            x = x_num
        logits = self.net(x).squeeze(-1)
        return logits


def build_embedding_sizes_for_training_df(df: pd.DataFrame):
    """Compute (num_embeddings, emb_dim) for each categorical feature."""
    sizes = []
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
        n_categories = df[col].cat.categories.size
        num_embeddings = n_categories + 1  # +1 for "unknown" / NaN
        # Simple embedding dimension heuristic
        emb_dim = min(50, max(4, num_embeddings // 2))
        sizes.append((num_embeddings, emb_dim))
    return sizes


def train_torch_tabular_nn(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    task_name: str,
    embedding_sizes,
    print_top_bucket: bool = False,
    num_epochs: int = 8,
    batch_size: int = 2048,
):
    """
    GPU-ready tabular NN with categorical embeddings + MLP.
    Uses class-weighted BCEWithLogitsLoss and isotonic calibration.
    """
    if not check_two_classes(y_train, f"{task_name} (train)"):
        return None, None

    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"\n[TRAIN] Torch TabularNN for {task_name} on device: {device}")

    # Optional subsampling for speed
    if len(y_train) > MAX_TORCH_TRAIN_SAMPLES:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(y_train), size=MAX_TORCH_TRAIN_SAMPLES, replace=False
        )
        X_train_sub = X_train.iloc[idx]
        y_train_sub = y_train.iloc[idx]
        print(f"[TORCH] Subsampled train to {len(y_train_sub):,} rows for NN.")
    else:
        X_train_sub, y_train_sub = X_train, y_train

    train_ds = TabularDataset(
        X_train_sub, y_train_sub, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    val_ds = TabularDataset(X_val, y_val, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    test_ds = TabularDataset(X_test, y_test, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)

    model = TabularNN(
        num_numeric=len(NUMERIC_FEATURES), embedding_sizes=embedding_sizes
    ).to(device)

    # Class imbalance handling
    y_train_np = y_train_sub.to_numpy()
    pos = int(y_train_np.sum())
    neg = len(y_train_np) - pos
    if pos > 0:
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_auc = -np.inf
    best_state = None

    def predict_proba_loader(model_, loader_):
        model_.eval()
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for x_num_b, x_cat_b, y_b in loader_:
                x_num_b = x_num_b.to(device)
                x_cat_b = x_cat_b.to(device)
                logits = model_(x_num_b, x_cat_b)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y_b.numpy())
        return np.concatenate(all_probs), np.concatenate(all_targets)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for x_num_b, x_cat_b, y_b in train_loader:
            x_num_b = x_num_b.to(device)
            x_cat_b = x_cat_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_num_b, x_cat_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_b)

        epoch_loss = running_loss / len(train_ds)

        # Validation AUC
        val_probs_raw, val_targets = predict_proba_loader(model, val_loader)
        try:
            val_auc = roc_auc_score(val_targets, val_probs_raw)
        except ValueError:
            val_auc = np.nan

        print(
            f"[TORCH] {task_name} epoch {epoch}: loss={epoch_loss:.4f}, val AUC={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"[TORCH] Loaded best model for {task_name} (val AUC={best_val_auc:.4f})."
        )

    # Predictions for calibration and test
    val_probs_raw, val_targets = predict_proba_loader(model, val_loader)
    test_probs_raw, test_targets = predict_proba_loader(model, test_loader)

    print(f"[CALIB] Calibrating TorchNN for {task_name} using isotonic regression...")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_probs_raw, val_targets)

    val_probs_cal = iso.predict(val_probs_raw)
    test_probs_cal = iso.predict(test_probs_raw)

    # Evaluate calibrated NN on test
    evaluate_probs(
        y_test,
        test_probs_cal,
        f"{task_name} – TorchNN (calibrated, t=0.5)",
        "test",
        threshold=0.5,
        print_top_bucket=print_top_bucket,
    )

    best_t, best_f1 = find_best_threshold(y_val, val_probs_cal, metric="f1")
    print(
        f"[THRESH] Best validation threshold for {task_name} (TorchNN, calibrated): "
        f"{best_t:.3f} (F1={best_f1:.3f})"
    )

    evaluate_probs(
        y_test,
        test_probs_cal,
        f"{task_name} – TorchNN (calibrated, t* from val)",
        "test",
        threshold=best_t,
        print_top_bucket=print_top_bucket,
    )

    # Diagnostics for report
    create_diagnostic_plots_generic(
        task_name=task_name,
        model_name="TorchNN",
        y_test=y_test,
        y_proba_test_cal=test_probs_cal,
        best_threshold=best_t,
        is_cancellation=(task_name == "cancel"),
    )

    # Save model + calibrator for later experiments (not used by app yet)
    model_path = MODELS_DIR / f"torchnn_{task_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "embedding_sizes": embedding_sizes,
        },
        model_path,
    )
    print(f"[OK] Saved TorchNN for {task_name} to {model_path}")

    calib_path = MODELS_DIR / f"torchnn_{task_name}_iso_calibrator.joblib"
    joblib.dump(iso, calib_path)
    print(f"[OK] Saved TorchNN isotonic calibrator for {task_name} to {calib_path}")

    return model, iso


# -------------------------------------------------------------------
# Feature ablation experiments (logistic regression, light-weight)
# -------------------------------------------------------------------
def run_feature_group_ablation(
    df_task: pd.DataFrame,
    target_col: str,
    task_name: str,
    is_cancellation: bool = False,
):
    """
    Very lightweight ablation: train a simple Logistic Regression model
    on different feature groups to quantify their standalone predictive
    power. This is NOT used in the app, purely for the report.
    """
    print(f"\n[ABLATION] Running feature-group ablation for {task_name}...")

    X = df_task[FEATURE_COLS].copy()
    y = df_task[target_col].astype("int16")

    # Same temporal split as the main models
    train_mask = df_task["Year"].between(2010, 2018)
    df_task["Year"].between(2019, 2021)
    test_mask = df_task["Year"].between(2022, 2024)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Optional downsampling for speed
    if len(y_train) > MAX_GRIDSEARCH_SAMPLES:
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_GRIDSEARCH_SAMPLES,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

    configs = {
        "calendar_only": FEATURE_GROUPS["calendar"],
        "route_airline_congestion_only": FEATURE_GROUPS["route_airline_congestion"],
        "weather_only": FEATURE_GROUPS["weather"],
        "all_features": FEATURE_COLS,
        "no_weather": [c for c in FEATURE_COLS if c not in FEATURE_GROUPS["weather"]],
    }

    ablation_rows = []

    for cfg_name, feature_list in configs.items():
        num_cols = [c for c in feature_list if c in NUMERIC_FEATURES]
        cat_cols = [c for c in feature_list if c in CATEGORICAL_FEATURES]

        print(
            f"[ABLATION]  Config={cfg_name}, num={len(num_cols)}, cat={len(cat_cols)}"
        )

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

        clf = LogisticRegression(
            solver="saga",
            max_iter=300,
            n_jobs=-1,
            class_weight="balanced",
            C=1.0,
            random_state=RANDOM_STATE,
        )

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", clf),
            ]
        )

        pipe.fit(X_train[feature_list], y_train)

        y_proba_test = pipe.predict_proba(X_test[feature_list])[:, 1]
        ctx = f"{task_name} – Ablation ({cfg_name})"
        metrics = evaluate_probs(
            y_test,
            y_proba_test,
            context=ctx,
            split_name="test",
            threshold=0.5,
            print_top_bucket=is_cancellation,
        )
        ablation_rows.append({"config": cfg_name, **metrics})

    # Summarise ablation results for later inspection
    df_ablate = pd.DataFrame(ablation_rows)
    out_csv = REPORTS_DIR / f"{task_name}_feature_ablation_logreg.csv"
    df_ablate.to_csv(out_csv, index=False)
    print(f"[ABLATION] Wrote ablation summary for {task_name} to {out_csv}")


# -------------------------------------------------------------------
# ASCII metrics report
# -------------------------------------------------------------------
def write_ascii_metrics_report(rows, out_path: Path):
    if not rows:
        return

    lines = []
    lines.append("=" * 80)
    lines.append("MODEL EVALUATION SUMMARY")
    lines.append("=" * 80)
    for row in rows:
        lines.append("-" * 80)
        lines.append(f"Context   : {row.get('context')}")
        lines.append(f"Split     : {row.get('split')}")
        lines.append(f"Threshold : {row.get('threshold'):.3f}")
        roc_auc = row.get("roc_auc")
        pr_auc = row.get("pr_auc")
        brier = row.get("brier")
        acc = row.get("accuracy")
        prec = row.get("precision_pos")
        rec = row.get("recall_pos")
        f1 = row.get("f1_pos")

        lines.append(
            f"ROC-AUC   : {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC   : N/A"
        )
        lines.append(
            f"PR-AUC    : {pr_auc:.4f}" if pr_auc is not None else "PR-AUC    : N/A"
        )
        lines.append(
            f"Brier     : {brier:.4f}" if brier is not None else "Brier     : N/A"
        )
        lines.append(f"Accuracy  : {acc:.4f}" if acc is not None else "Accuracy  : N/A")
        lines.append(
            f"Precision : {prec:.4f}" if prec is not None else "Precision : N/A"
        )
        lines.append(f"Recall    : {rec:.4f}" if rec is not None else "Recall    : N/A")
        lines.append(f"F1 (pos)  : {f1:.4f}" if f1 is not None else "F1 (pos)  : N/A")

        if row.get("top_bucket_quantile") is not None:
            top_rate = row.get("top_bucket_rate")
            top_q = row.get("top_bucket_quantile")
            top_pct = int((1.0 - top_q) * 100)
            if top_rate is not None:
                lines.append(f"Top {top_pct}% bucket positive rate : {top_rate:.4f}")
            else:
                lines.append(f"Top {top_pct}% bucket positive rate : N/A")

    lines.append("-" * 80)
    text = "\n".join(lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[OK] Wrote ASCII metrics report to {out_path}")


# -------------------------------------------------------------------
# Main entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[INFO] Loading processed dataset from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    # Ensure categorical dtypes are consistent (important for CatBoost & Torch)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    # Precompute embedding sizes for NN on the full df (shared across tasks)
    embedding_sizes = build_embedding_sizes_for_training_df(df)

    # ---------- Task 1: Delay ≥ 15 minutes (ArrDel15) ----------
    print("\n================ DELAY ≥ 15 MINUTES (ArrDel15) ================")

    df_delay = df[(df["Cancelled"] == 0) & df["ArrDel15"].notna()].copy()
    df_delay["ArrDel15"] = df_delay["ArrDel15"].astype("int16")

    X_train, y_train, X_val, y_val, X_test, y_test = make_time_splits(
        df_delay, "ArrDel15"
    )

    logreg_delay = train_logistic_baseline(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        task_name="delay15",
        print_top_bucket=False,
    )

    lgbm_delay = train_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        task_name="delay15",
        print_top_bucket=False,
    )

    catboost_delay = train_catboost(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        task_name="delay15",
        print_top_bucket=False,
    )

    torch_delay, torch_delay_calib = train_torch_tabular_nn(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        task_name="delay15",
        embedding_sizes=embedding_sizes,
        print_top_bucket=False,
    )

    # Simple ensemble of CatBoost + LogReg for delay (tree-only ensemble)
    if (logreg_delay is not None) and (catboost_delay is not None):
        evaluate_two_model_ensemble(
            model_a=catboost_delay,
            model_b=logreg_delay,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            task_name="delay15",
            name_a="CatBoost",
            name_b="LogReg",
            print_top_bucket=False,
        )

    # Feature ablation (delay)
    run_feature_group_ablation(
        df_task=df_delay,
        target_col="ArrDel15",
        task_name="delay15",
        is_cancellation=False,
    )

    # ---------- Task 2: Cancellation (Cancelled) ----------
    print("\n================ CANCELLATION (Cancelled) ================")

    df_cancel = df[df["Cancelled"].isin([0, 1])].copy()
    df_cancel["Cancelled"] = df_cancel["Cancelled"].astype("int16")

    X_train_c, y_train_c, X_val_c, y_val_c, X_test_c, y_test_c = make_time_splits(
        df_cancel, "Cancelled"
    )

    # Subsample negatives ONLY on the training set for cancellation
    X_train_c_bal, y_train_c_bal = subsample_negatives(
        X_train_c, y_train_c, NEGATIVE_SUBSAMPLE_FRACTION_CANCEL, RANDOM_STATE
    )

    logreg_cancel = train_logistic_baseline(
        X_train_c_bal,
        y_train_c_bal,
        X_val_c,
        y_val_c,
        X_test_c,
        y_test_c,
        task_name="cancel",
        print_top_bucket=True,
    )

    lgbm_cancel = train_lightgbm(
        X_train_c_bal,
        y_train_c_bal,
        X_val_c,
        y_val_c,
        X_test_c,
        y_test_c,
        task_name="cancel",
        print_top_bucket=True,
    )

    catboost_cancel = train_catboost(
        X_train_c_bal,
        y_train_c_bal,
        X_val_c,
        y_val_c,
        X_test_c,
        y_test_c,
        task_name="cancel",
        print_top_bucket=True,
    )

    torch_cancel, torch_cancel_calib = train_torch_tabular_nn(
        X_train_c_bal,
        y_train_c_bal,
        X_val_c,
        y_val_c,
        X_test_c,
        y_test_c,
        task_name="cancel",
        embedding_sizes=embedding_sizes,
        print_top_bucket=True,
    )

    # Ensemble of CatBoost + LogReg for cancellation
    if (logreg_cancel is not None) and (catboost_cancel is not None):
        evaluate_two_model_ensemble(
            model_a=catboost_cancel,
            model_b=logreg_cancel,
            X_val=X_val_c,
            y_val=y_val_c,
            X_test=X_test_c,
            y_test=y_test_c,
            task_name="cancel",
            name_a="CatBoost",
            name_b="LogReg",
            print_top_bucket=True,
        )

    # Feature ablation (cancellation)
    run_feature_group_ablation(
        df_task=df_cancel,
        target_col="Cancelled",
        task_name="cancel",
        is_cancellation=True,
    )

    # ---------- Fairness & drift diagnostics for "best" models ----------
    # For delay we treat CatBoost as the primary model (used in the app).
    # For cancellation, the app currently uses LightGBM.
    print("\n================ ADDITIONAL DIAGNOSTICS ================")

    best_delay_model = catboost_delay or lgbm_delay or logreg_delay
    best_cancel_model = lgbm_cancel or catboost_cancel or logreg_cancel

    if best_delay_model is not None:
        # Delay: fairness by airline on test period
        df_delay_test = df_delay[df_delay["Year"].between(2022, 2024)].copy()
        X_delay_test = df_delay_test[FEATURE_COLS]
        y_delay_test = df_delay_test["ArrDel15"].astype("int16")
        y_delay_proba = best_delay_model.predict_proba(X_delay_test)[:, 1]
        evaluate_group_metrics(
            y_true=y_delay_test,
            y_proba=y_delay_proba,
            group=df_delay_test["Reporting_Airline"],
            group_name="Reporting_Airline",
            context="delay15_bestmodel_airline",
            min_samples=2000,
        )
        # Delay: drift by year across the full dataset used for that task
        evaluate_drift_by_year(
            df=df_delay,
            X_all=df_delay[FEATURE_COLS],
            y_all=df_delay["ArrDel15"].astype("int16"),
            model=best_delay_model,
            task_name="delay15",
            split_label="full",
        )

    if best_cancel_model is not None:
        # Cancellation: fairness by airline on test period
        df_cancel_test = df_cancel[df_cancel["Year"].between(2022, 2024)].copy()
        X_cancel_test = df_cancel_test[FEATURE_COLS]
        y_cancel_test = df_cancel_test["Cancelled"].astype("int16")
        y_cancel_proba = best_cancel_model.predict_proba(X_cancel_test)[:, 1]
        evaluate_group_metrics(
            y_true=y_cancel_test,
            y_proba=y_cancel_proba,
            group=df_cancel_test["Reporting_Airline"],
            group_name="Reporting_Airline",
            context="cancel_bestmodel_airline",
            min_samples=2000,
        )
        # Cancellation: drift by year across the full dataset
        evaluate_drift_by_year(
            df=df_cancel,
            X_all=df_cancel[FEATURE_COLS],
            y_all=df_cancel["Cancelled"].astype("int16"),
            model=best_cancel_model,
            task_name="cancel",
            split_label="full",
        )

    # ---------- Bootstrap comparisons (CatBoost vs LGBM) ----------
    if (catboost_delay is not None) and (lgbm_delay is not None):
        y_proba_cat = catboost_delay.predict_proba(X_test)[:, 1]
        y_proba_lgbm = lgbm_delay.predict_proba(X_test)[:, 1]
        bootstrap_metric_difference(
            y_true=y_test.values,
            proba_a=y_proba_cat,
            proba_b=y_proba_lgbm,
            metric_fn=roc_auc_score,
            n_bootstrap=400,
            higher_is_better=True,
            context="delay15 CatBoost vs LGBM (AUC)",
        )
        bootstrap_metric_difference(
            y_true=y_test.values,
            proba_a=y_proba_cat,
            proba_b=y_proba_lgbm,
            metric_fn=brier_score_loss,
            n_bootstrap=400,
            higher_is_better=False,  # lower Brier is better
            context="delay15 CatBoost vs LGBM (Brier)",
        )

    if (lgbm_cancel is not None) and (catboost_cancel is not None):
        y_proba_lgbm = lgbm_cancel.predict_proba(X_test_c)[:, 1]
        y_proba_cat = catboost_cancel.predict_proba(X_test_c)[:, 1]
        bootstrap_metric_difference(
            y_true=y_test_c.values,
            proba_a=y_proba_lgbm,
            proba_b=y_proba_cat,
            metric_fn=roc_auc_score,
            n_bootstrap=400,
            higher_is_better=True,
            context="cancel LGBM vs CatBoost (AUC)",
        )
        bootstrap_metric_difference(
            y_true=y_test_c.values,
            proba_a=y_proba_lgbm,
            proba_b=y_proba_cat,
            metric_fn=brier_score_loss,
            n_bootstrap=400,
            higher_is_better=False,
            context="cancel LGBM vs CatBoost (Brier)",
        )

    # Write ASCII metrics summary including all models + ensembles + ablations
    report_path = REPORTS_DIR / "metrics_summary.txt"
    write_ascii_metrics_report(METRIC_ROWS, report_path)

    print("\n[SUMMARY] Training complete. Models, figures and reports saved.")
