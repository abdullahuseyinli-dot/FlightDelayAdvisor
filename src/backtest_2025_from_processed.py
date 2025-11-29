# src/backtest_2025_from_processed.py
r"""
Backtest FlightDelayAdvisor models on processed 2025 BTS data.

Pipeline:
- Load processed 2025 dataset from data/processed/bts_2025_backtest_base.parquet
  (created by prepare_bts_2025_for_backtest.py).
- Use the SAME metadata and feature builder as the app (app.py) so that
  all route/airline/slot/weather logic is identical to production.
- Evaluate the deployed models (CatBoost delay, LGBM cancel) on 2025 flights.
- Print metrics to stdout and write a summary to reports/backtest_2025_metrics.txt.

Run from repo root (example):

    (.project1venv) PS> python .\src\backtest_2025_from_processed.py --max-rows 500000 ^
                                   > reports\backtest_2025_log.txt
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
from textwrap import dedent

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# --------------------------------------------------------------------
# Import app + train_models helpers safely
# --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import (  # type: ignore  # noqa: E402
    CATEGORICAL_FEATURES,
    build_feature_row,
    load_metadata,
    load_models,
)
from train_models import evaluate_probs  # type: ignore  # noqa: E402

PROCESSED_2025_PATH = REPO_ROOT / "data" / "processed" / "bts_2025_backtest_base.parquet"
REPORTS_DIR = REPO_ROOT / "reports"


# --------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------
def compute_core_metrics(
    y: np.ndarray,
    proba: np.ndarray,
    top_quantile: float = 0.09,
) -> dict[str, float]:
    """
    Compute core metrics (ROC-AUC, PR-AUC, Brier) plus:
      - base_rate: overall event rate
      - top_bucket_positive_rate: event rate in top quantile of risk
      - top_bucket_event_share: fraction of all events in the top bucket
    """
    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    base_rate = float(y.mean())

    if len(proba) == 0:
        top_rate = np.nan
        top_share = np.nan
    else:
        cutoff = np.quantile(proba, 1.0 - top_quantile)
        mask_top = proba >= cutoff
        if mask_top.any():
            top_rate = float(y[mask_top].mean())
            if y.sum() > 0:
                top_share = float(y[mask_top].sum() / y.sum())
            else:
                top_share = np.nan
        else:
            top_rate = np.nan
            top_share = np.nan

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "base_rate": float(base_rate),
        "top_quantile": float(top_quantile),
        "top_bucket_positive_rate": float(top_rate),
        "top_bucket_event_share": float(top_share),
    }


def print_risk_stratification(
    y: np.ndarray,
    proba: np.ndarray,
    name: str,
    top_fracs: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
) -> None:
    """
    Print a small table showing, for several top-X% highest-risk segments:
      - event rate in that segment
      - uplift vs global base rate
      - fraction of all events captured

    This directly answers: "If I focus on the riskiest X% of flights,
    how concentrated are the delays/cancellations?"
    """
    if len(y) == 0:
        print(f"[RISK] No data for {name}; cannot stratify.")
        return

    base_rate = float(y.mean())
    if base_rate == 0.0:
        print(f"[RISK] No positive events for {name}; cannot stratify.")
        return

    df_rows: list[dict[str, float]] = []
    y = y.astype(float)
    proba = proba.astype(float)

    for frac in top_fracs:
        if frac <= 0 or frac >= 1:
            continue
        cutoff = np.quantile(proba, 1.0 - frac)
        mask_top = proba >= cutoff
        if not mask_top.any():
            continue

        n_top = int(mask_top.sum())
        rate_top = float(y[mask_top].mean())
        event_share = float(y[mask_top].sum() / y.sum())
        uplift = rate_top / base_rate if base_rate > 0 else np.nan

        df_rows.append(
            {
                "top_fraction": frac,
                "num_flights": n_top,
                "segment_event_rate": rate_top,
                "uplift_vs_base": uplift,
                "share_of_all_events": event_share,
            }
        )

    if not df_rows:
        print(f"[RISK] Could not compute risk stratification for {name}.")
        return

    df_strat = pd.DataFrame(df_rows)
    print(f"[RISK] Risk stratification for {name}")
    print(f"       Global base rate: {base_rate:.4f}")
    print(
        df_strat.assign(
            top_percent=lambda d: d["top_fraction"] * 100.0,
            segment_event_rate_pct=lambda d: d["segment_event_rate"] * 100.0,
            share_of_all_events_pct=lambda d: d["share_of_all_events"] * 100.0,
        )[
            [
                "top_percent",
                "num_flights",
                "segment_event_rate_pct",
                "uplift_vs_base",
                "share_of_all_events_pct",
            ]
        ]
        .rename(
            columns={
                "top_percent": "top_%_highest_risk",
                "num_flights": "n_flights",
                "segment_event_rate_pct": "event_rate_%_in_segment",
                "uplift_vs_base": "uplift_vs_base",
                "share_of_all_events_pct": "share_%_of_all_events",
            }
        )
        .to_string(index=False, float_format=lambda x: f"{x:6.2f}")
    )


# --------------------------------------------------------------------
# Feature construction for 2025 flights
# --------------------------------------------------------------------
def build_feature_matrix_for_2025(
    df_2025: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use app.py's build_feature_row + metadata (route stats, airline stats,
    weather climatology) to create the feature matrix for 2025 flights.

    Returns:
        X_all  : DataFrame of features for all 2025 flights where Cancelled in {0,1}
        df_eval: Subset of df_2025 matching X_all (aligned index)
    """
    (
        df_hist,
        airlines,
        origins,
        dests,
        route_meta,
        airline_meta,
        slot_meta,
        route_pair_meta,
        route_distance_meta,
        origin_weather_meta,
        dest_weather_meta,
        defaults,
    ) = load_metadata()

    df_eval = df_2025[df_2025["Cancelled"].isin([0, 1])].copy()
    print(f"[INFO] Using {len(df_eval):,} 2025 rows with Cancelled in {{0,1}}.")

    feature_rows: list[pd.DataFrame] = []
    for idx, row in df_eval.iterrows():
        travel_date = date(
            int(row["Year"]),
            int(row["Month"]),
            int(row["DayOfMonth"]),
        )
        dep_hour = int(row["DepHour"])
        airline = str(row["Reporting_Airline"])
        origin = str(row["Origin"])
        dest = str(row["Dest"])
        distance = float(row["Distance"])

        X_row = build_feature_row(
            travel_date=travel_date,
            dep_hour=dep_hour,
            airline=airline,
            origin=origin,
            dest=dest,
            distance=distance,
            route_meta=route_meta,
            airline_meta=airline_meta,
            slot_meta=slot_meta,
            origin_weather_meta=origin_weather_meta,
            dest_weather_meta=dest_weather_meta,
            defaults=defaults,
        )
        X_row.index = [idx]
        feature_rows.append(X_row)

    X_all = pd.concat(feature_rows, axis=0)
    for col in CATEGORICAL_FEATURES:
        X_all[col] = X_all[col].astype("category")

    print(f"[INFO] Built feature matrix for {len(X_all):,} 2025 flights.")
    return X_all, df_eval


# --------------------------------------------------------------------
# Backtest core
# --------------------------------------------------------------------
def run_backtest(max_rows: int | None = None) -> None:
    if not PROCESSED_2025_PATH.exists():
        raise FileNotFoundError(
            f"Processed 2025 file not found: {PROCESSED_2025_PATH} "
            "Run src/prepare_bts_2025_for_backtest.py first."
        )

    print("=" * 70)
    print("BACKTEST: FlightDelayAdvisor models on real-world 2025 BTS data")
    print("=" * 70)
    print(f"[PATH] Repo root:       {REPO_ROOT}")
    print(f"[PATH] Processed 2025:  {PROCESSED_2025_PATH}")
    print()

    df_2025 = pd.read_parquet(PROCESSED_2025_PATH)
    print(f"[INFO] Loaded processed 2025 dataset with {len(df_2025):,} rows.")

    if max_rows is not None and len(df_2025) > max_rows:
        df_2025 = df_2025.sample(max_rows, random_state=42).reset_index(drop=True)
        print(f"[INFO] Subsampled 2025 dataset to {len(df_2025):,} rows for backtest.")

    X_all, df_eval = build_feature_matrix_for_2025(df_2025)
    delay_model, cancel_model = load_models()

    # ---------------- Delay backtest ----------------
    df_delay = df_eval[(df_eval["Cancelled"] == 0) & df_eval["ArrDel15"].notna()].copy()
    X_delay = X_all.loc[df_delay.index]
    y_delay = df_delay["ArrDel15"].astype(int).to_numpy()

    print(
        f"\n[DELAY] Evaluating on {len(df_delay):,} non-cancelled 2025 flights "
        "with ArrDel15 label..."
    )
    delay_probs = delay_model.predict_proba(X_delay)[:, 1]
    delay_metrics = compute_core_metrics(y_delay, delay_probs, top_quantile=0.09)

    evaluate_probs(
        y_delay,
        delay_probs,
        context="delay15 - CatBoost (2025 backtest)",
        split_name="2025",
        threshold=0.5,  # neutral threshold; ranking is what matters most
        print_top_bucket=True,
        top_quantile=0.09,
    )
    print()
    print_risk_stratification(y_delay, delay_probs, "delay >= 15 minutes")

    # ---------------- Cancellation backtest ----------------
    df_cancel = df_eval[df_eval["Cancelled"].isin([0, 1])].copy()
    X_cancel = X_all.loc[df_cancel.index]
    y_cancel = df_cancel["Cancelled"].astype(int).to_numpy()

    print(f"\n[CANCEL] Evaluating on {len(df_cancel):,} 2025 flights with Cancelled label...")
    cancel_probs = cancel_model.predict_proba(X_cancel)[:, 1]
    cancel_metrics = compute_core_metrics(y_cancel, cancel_probs, top_quantile=0.09)

    evaluate_probs(
        y_cancel,
        cancel_probs,
        context="cancel - LGBM (2025 backtest)",
        split_name="2025",
        threshold=0.5,
        print_top_bucket=True,
        top_quantile=0.09,
    )
    print()
    print_risk_stratification(y_cancel, cancel_probs, "cancellation")

    # -------------- Write summary --------------
    REPORTS_DIR.mkdir(exist_ok=True)
    summary_path = REPORTS_DIR / "backtest_2025_metrics.txt"

    summary = dedent(
        f"""
        ========================================================================
        REAL-WORLD BACKTEST SUMMARY - FlightDelayAdvisor on 2025 BTS data
        ========================================================================
        Source data:
          - Processed BTS 2025 dataset from data/processed/bts_2025_backtest_base.parquet
          - Features built using 2010-2024 metadata (route/airline stats, weather)

        Delay >= 15 min model (CatBoost, calibrated)
          Samples (non-cancelled 2025 flights with ArrDel15) : {len(df_delay):,}
          Base delay rate                                   : {delay_metrics["base_rate"]:.4f}
          ROC-AUC                                           : {delay_metrics["roc_auc"]:.4f}
          PR-AUC                                            : {delay_metrics["pr_auc"]:.4f}
          Brier score                                       : {delay_metrics["brier"]:.4f}
          Top {delay_metrics["top_quantile"]*100:.0f}% bucket positive rate        : {delay_metrics["top_bucket_positive_rate"]:.4f}
          Share of all delays in top bucket                 : {delay_metrics["top_bucket_event_share"]:.4f}

        Cancellation model (LGBM, calibrated)
          Samples (2025 flights with Cancelled in {{0,1}})   : {len(df_cancel):,}
          Base cancellation rate                            : {cancel_metrics["base_rate"]:.4f}
          ROC-AUC                                           : {cancel_metrics["roc_auc"]:.4f}
          PR-AUC                                            : {cancel_metrics["pr_auc"]:.4f}
          Brier score                                       : {cancel_metrics["brier"]:.4f}
          Top {cancel_metrics["top_quantile"]*100:.0f}% bucket positive rate       : {cancel_metrics["top_bucket_positive_rate"]:.4f}
          Share of all cancellations in top bucket          : {cancel_metrics["top_bucket_event_share"]:.4f}

        Notes:
          - This backtest reflects genuinely out-of-time generalisation.
          - Features for 2025 flights are constructed using only 2010-2024
            historical patterns and climatology (no 2025 leakage).
          - For detailed classification reports and risk stratification tables,
            see backtest_2025_log.txt.
        """
    ).strip()

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"\n[INFO] Backtest summary written to: {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest on processed 2025 BTS data.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of 2025 rows to use (for speed). Default: use all.",
    )
    args = parser.parse_args()
    run_backtest(max_rows=args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


