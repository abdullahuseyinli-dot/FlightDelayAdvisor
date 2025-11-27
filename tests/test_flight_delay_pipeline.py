"""
End‑to‑end regression tests for the FlightDelayAdvisor project.

These tests are designed to be:

- **Research‑grade**: they check not only that the code runs, but that the
  models have non‑trivial predictive signal on the historical data.
- **Robust**: they skip gracefully if the large parquet file is not present.
- **Aligned with the app**: they use the same feature builder and model loader
  as the Streamlit UI (src/app.py).

Run with:

    (.project1venv) PS> pip install pytest
    (.project1venv) PS> python -m pytest -q

"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------
# Make sure we can import from src/
# --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Now import from your app
from app import (  # type: ignore
    DATA_PATH as REL_DATA_PATH,
    FEATURE_COLS,
    build_feature_row,
    load_metadata,
    load_models,
)

# Resolve data path relative to the repo root
DATA_PATH = (REPO_ROOT / REL_DATA_PATH).resolve()

# --------------------------------------------------------------------
# Basic existence / integrity tests
# --------------------------------------------------------------------


def test_processed_data_file_exists():
    """Check that the main processed parquet file exists."""
    assert DATA_PATH.exists(), f"Expected data file not found: {DATA_PATH}"


@pytest.mark.skipif(
    not DATA_PATH.exists(), reason="Processed parquet file is not available locally."
)
def test_data_has_minimum_required_columns():
    """
    Ensure that the processed dataset has the columns needed by the app/model.
    This guards against accidentally changing the preprocessing pipeline.
    """
    df = pd.read_parquet(DATA_PATH)

    required_cols = {
        "Year",
        "Month",
        "DayOfMonth",
        "DayOfWeek",
        "DepHour",
        "Distance",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "ArrDel15",
        "Cancelled",
    }

    missing = required_cols - set(df.columns)
    assert not missing, f"Data is missing required columns: {missing}"


# --------------------------------------------------------------------
# Feature builder & model sanity checks
# --------------------------------------------------------------------


@pytest.mark.skipif(
    not DATA_PATH.exists(), reason="Processed parquet file is not available locally."
)
def test_feature_row_matches_training_schema_and_is_clean():
    """
    Build a single feature row from a real historical flight and verify that:

    - the columns exactly match FEATURE_COLS (training schema)
    - there are no NaNs or infs in the features
    """
    (
        df,
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

    # Use a random non‑cancelled flight with known delay label
    row = (
        df[(df["Cancelled"].isin([0, 1])) & df["ArrDel15"].notna()]
        .sample(1, random_state=42)
        .iloc[0]
    )

    travel_date = date(int(row["Year"]), int(row["Month"]), int(row["DayOfMonth"]))
    dep_hour = int(row["DepHour"])
    airline = str(row["Reporting_Airline"])
    origin = str(row["Origin"])
    dest = str(row["Dest"])
    distance = float(row["Distance"])

    X = build_feature_row(
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

    # 1) Exact same columns as in training
    assert (
        list(X.columns) == FEATURE_COLS
    ), "Feature columns do not match training schema."

    # 2) No missing or infinite values
    assert not X.isnull().any().any(), "Feature row contains NaNs."
    assert np.isfinite(
        X.select_dtypes(include=[np.number]).values
    ).all(), "Feature row contains non‑finite numeric values."


@pytest.mark.skipif(
    not DATA_PATH.exists(), reason="Processed parquet file is not available locally."
)
def test_models_produce_probabilities_in_0_1_range():
    """
    Load both delay and cancellation models and ensure they return
    sensible probability outputs for a realistic flight.
    """
    (
        df,
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
    delay_model, cancel_model = load_models()

    row = (
        df[(df["Cancelled"].isin([0, 1])) & df["ArrDel15"].notna()]
        .sample(1, random_state=7)
        .iloc[0]
    )

    travel_date = date(int(row["Year"]), int(row["Month"]), int(row["DayOfMonth"]))
    dep_hour = int(row["DepHour"])
    airline = str(row["Reporting_Airline"])
    origin = str(row["Origin"])
    dest = str(row["Dest"])
    distance = float(row["Distance"])

    X = build_feature_row(
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

    p_delay = float(delay_model.predict_proba(X)[0, 1])
    p_cancel = float(cancel_model.predict_proba(X)[0, 1])

    assert 0.0 <= p_delay <= 1.0, f"Delay prob out of range: {p_delay}"
    assert 0.0 <= p_cancel <= 1.0, f"Cancel prob out of range: {p_cancel}"


# --------------------------------------------------------------------
# Research‑style performance checks
# --------------------------------------------------------------------


@pytest.mark.skipif(
    not DATA_PATH.exists(), reason="Processed parquet file is not available locally."
)
def test_delay_model_has_predictive_signal():
    """
    Check that the delay model has real predictive signal:

    - Compute ROC‑AUC for ArrDel15 using the saved model.
    - Assert that AUC is comfortably above 0.5 (random), e.g. > 0.60.

    This is a regression‑style test: if you accidentally break the
    feature pipeline or load the wrong model, this will likely fail.
    """
    (
        df,
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
    delay_model, _ = load_models()

    df_eval = df[(df["Cancelled"].isin([0, 1])) & df["ArrDel15"].notna()].copy()

    # Build features for a manageable subset to keep the test reasonably fast
    df_eval = df_eval.sample(min(10000, len(df_eval)), random_state=123)

    X_rows = []
    for _, row in df_eval.iterrows():
        travel_date = date(int(row["Year"]), int(row["Month"]), int(row["DayOfMonth"]))
        dep_hour = int(row["DepHour"])
        airline = str(row["Reporting_Airline"])
        origin = str(row["Origin"])
        dest = str(row["Dest"])
        distance = float(row["Distance"])

        X_rows.append(
            build_feature_row(
                travel_date,
                dep_hour,
                airline,
                origin,
                dest,
                distance,
                route_meta,
                airline_meta,
                slot_meta,
                origin_weather_meta,
                dest_weather_meta,
                defaults,
            )
        )

    X = pd.concat(X_rows, ignore_index=True)
    y = df_eval["ArrDel15"].astype(int).values

    proba = delay_model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)

    # Your metrics_summary.txt shows ROC‑AUC around 0.68 on test, so 0.60
    # is a safe lower bound while still catching serious regressions.
    assert auc > 0.60, f"Delay model AUC too low: {auc:.3f}"


@pytest.mark.skipif(
    not DATA_PATH.exists(), reason="Processed parquet file is not available locally."
)
def test_cancel_model_has_predictive_signal():
    """
    Similar to the delay test, but for cancellations.

    We check that the ROC‑AUC is clearly better than random (> 0.60).
    """
    (
        df,
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
    _, cancel_model = load_models()

    df_eval = df[(df["Cancelled"].isin([0, 1]))].copy()
    df_eval = df_eval.sample(min(15000, len(df_eval)), random_state=456)

    X_rows = []
    for _, row in df_eval.iterrows():
        travel_date = date(int(row["Year"]), int(row["Month"]), int(row["DayOfMonth"]))
        dep_hour = int(row["DepHour"])
        airline = str(row["Reporting_Airline"])
        origin = str(row["Origin"])
        dest = str(row["Dest"])
        distance = float(row["Distance"])

        X_rows.append(
            build_feature_row(
                travel_date,
                dep_hour,
                airline,
                origin,
                dest,
                distance,
                route_meta,
                airline_meta,
                slot_meta,
                origin_weather_meta,
                dest_weather_meta,
                defaults,
            )
        )

    X = pd.concat(X_rows, ignore_index=True)
    y = df_eval["Cancelled"].astype(int).values

    proba = cancel_model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)

    # From your metrics summary, AUC ~0.69 – we use a conservative lower bound.
    assert auc > 0.60, f"Cancellation model AUC too low: {auc:.3f}"


if __name__ == "__main__":
    # Allow running this file directly for quick manual checks
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
