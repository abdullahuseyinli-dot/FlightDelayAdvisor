# src/prepare_bts_2025_for_backtest.py
"""
prepare_bts_2025_for_backtest.py

Clean and normalise raw BTS On-Time Performance data for 2025
into a schema compatible with the 2010–2024 pipeline.

- Reads monthly BTS files from data/raw_2025/
  (supports both .zip and .csv files).
- Keeps both cancelled and non-cancelled flights.
- Adds the same time-based and route features as prepare_dataset.py.
- Standardises optional state information into OriginState / DestState.

Output:
    data/processed/bts_2025_backtest_base.parquet

This processed file is then used by a separate backtest script
to evaluate the deployed models on real 2025 data.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------
RAW_2025_DIR = Path("data/raw_2025")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# We only process 2025
YEAR_2025 = 2025

# If None, use all rows; otherwise random sample for speed.
MAX_ROWS_2025: int | None = None

# Columns we need from BTS On-Time Performance data
USE_COLS = [
    "Year",
    "Month",
    "DayOfWeek",
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "Distance",
    "ArrDel15",
    "Cancelled",
    "Diverted",
]

# -------------------------------------------------------------------
# Helpers copied/adapted from prepare_dataset.py
# -------------------------------------------------------------------


def extract_csv_from_zip(zip_path: Path) -> BytesIO:
    """Return a file-like CSV handle from a BTS zip file."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found inside {zip_path}")
        csv_name = csv_names[0]
        raw_bytes = zf.read(csv_name)
    return BytesIO(raw_bytes)


def clean_2025_file(path: Path) -> pd.DataFrame:
    """
    Load and clean data for one 2025 file (zip or csv).

    - Keeps both cancelled and non-cancelled flights.
    - ArrDel15 may be NaN for cancelled flights.
    - Adds time-based and route features but NO aggregates.
    - Detects state columns (if present) and standardises them to
      OriginState and DestState for downstream use.
    """
    suffix = path.suffix.lower()

    if suffix == ".zip":
        # Read CSV from inside the zip
        csv_file = extract_csv_from_zip(path)
        preview = pd.read_csv(csv_file, nrows=0, low_memory=False)
        csv_file.seek(0)
    else:
        # Plain CSV
        preview = pd.read_csv(path, nrows=0, low_memory=False)
        csv_file = path  # type: ignore[assignment]

    available_cols = set(preview.columns)

    # Optional state columns
    state_candidates = [
        "OriginState",
        "DestState",
        "OriginStateName",
        "DestStateName",
        "ORIGIN_STATE_ABR",
        "DEST_STATE_ABR",
        "ORIGIN_STATE_NM",
        "DEST_STATE_NM",
    ]
    state_extra_cols = [c for c in state_candidates if c in available_cols]

    usecols = USE_COLS + state_extra_cols

    df = pd.read_csv(csv_file, usecols=usecols, low_memory=False)

    # Basic cleaning: drop missing in core predictors (but NOT ArrDel15)
    df = df.dropna(
        subset=[
            "Cancelled",
            "CRSDepTime",
            "Origin",
            "Dest",
            "Reporting_Airline",
            "Distance",
            "Year",
            "Month",
            "DayOfWeek",
            "FlightDate",
        ]
    )

    # Remove diverted flights (we do not model them)
    if "Diverted" in df.columns:
        df = df[df["Diverted"] != 1]
    df = df.drop(columns=["Diverted"], errors="ignore")

    # Cast types
    df["Cancelled"] = df["Cancelled"].astype("int8")
    df["ArrDel15"] = df["ArrDel15"].astype("float32")

    df["Year"] = df["Year"].astype("int16")
    df["Month"] = df["Month"].astype("int8")
    df["DayOfWeek"] = df["DayOfWeek"].astype("int8")
    df["Distance"] = df["Distance"].astype("float32")

    # Parse date
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df = df.dropna(subset=["FlightDate"])

    # --- Time-based features ---

    # Departure hour from CRSDepTime (hhmm -> 0..23)
    def dep_hour(x: float | int | str) -> float:
        try:
            x_int = int(x)
        except (TypeError, ValueError):
            return np.nan
        if x_int == 2400:
            return 0
        return x_int // 100

    df["DepHour"] = (
        df["CRSDepTime"].astype("float64").astype("Int64").apply(dep_hour).astype("float32")
    )
    df = df.dropna(subset=["DepHour"])
    df["DepHour"] = df["DepHour"].astype("int8")

    df = df.drop(columns=["CRSDepTime"])

    df["DayOfMonth"] = df["FlightDate"].dt.day.astype("int8")
    df["DayOfYear"] = df["FlightDate"].dt.dayofyear.astype("int16")

    # Weekend flag (BTS: 1=Mon,...,7=Sun)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype("int8")

    # Season
    def month_to_season(m: int) -> str:
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "spring"
        if m in (6, 7, 8):
            return "summer"
        return "autumn"

    df["Season"] = df["Month"].apply(month_to_season).astype("category")

    # Holiday season (rough proxy: Nov–Jan)
    df["IsHolidaySeason"] = df["Month"].isin([11, 12, 1]).astype("int8")

    # Cyclical encoding for hour-of-day
    df["DepHour_sin"] = np.sin(2 * np.pi * df["DepHour"] / 24).astype("float32")
    df["DepHour_cos"] = np.cos(2 * np.pi * df["DepHour"] / 24).astype("float32")

    # --- Route & geography ---
    df["Route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).astype(
        "category"
    )

    def distance_band(d: float) -> str:
        if d < 500:
            return "short"
        if d < 1500:
            return "medium"
        if d < 3000:
            return "long"
        return "ultra_long"

    df["DistanceBand"] = df["Distance"].apply(distance_band).astype("category")

    # Core categoricals
    df["Reporting_Airline"] = df["Reporting_Airline"].astype("category")
    df["Origin"] = df["Origin"].astype("category")
    df["Dest"] = df["Dest"].astype("category")

    # --- Optional state columns (standardised) --------------------
    origin_state_series = None
    for cand in ["OriginState", "ORIGIN_STATE_ABR", "OriginStateName", "ORIGIN_STATE_NM"]:
        if cand in df.columns:
            origin_state_series = df[cand]
            break

    dest_state_series = None
    for cand in ["DestState", "DEST_STATE_ABR", "DestStateName", "DEST_STATE_NM"]:
        if cand in df.columns:
            dest_state_series = df[cand]
            break

    if origin_state_series is not None:
        df["OriginState"] = origin_state_series.astype("category")
    else:
        df["OriginState"] = pd.Categorical([np.nan] * len(df))

    if dest_state_series is not None:
        df["DestState"] = dest_state_series.astype("category")
    else:
        df["DestState"] = pd.Categorical([np.nan] * len(df))

    return df


def build_2025_dataset(raw_dir: Path = RAW_2025_DIR, max_rows: int | None = MAX_ROWS_2025) -> pd.DataFrame:
    """
    Clean and concatenate all 2025 BTS files in data/raw_2025.

    If max_rows is not None, randomly subsample to that many rows
    (for speed / memory reasons).
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"2025 raw folder not found: {raw_dir}")

    files = sorted(list(raw_dir.glob("*.zip")) + list(raw_dir.glob("*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No .zip or .csv files found in {raw_dir}. "
            "Expected monthly 2025 BTS files."
        )

    all_months: list[pd.DataFrame] = []
    for path in files:
        print(f"[INFO] Cleaning 2025 file: {path.name}")
        try:
            df_month = clean_2025_file(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to clean {path.name}: {exc}. Skipping this file.")
            continue

        # Sanity filter on Year to avoid mixing older years
        df_month = df_month[df_month["Year"] == YEAR_2025]
        if df_month.empty:
            print(f"[WARN] {path.name} had no Year == {YEAR_2025}, skipping.")
            continue

        all_months.append(df_month)

    if not all_months:
        raise RuntimeError(
            f"Found files in {raw_dir} but could not clean any as valid 2025 BTS data."
        )

    df_all = pd.concat(all_months, ignore_index=True)
    print(f"[INFO] Combined 2025 cleaned dataset size: {len(df_all):,} rows")

    if max_rows is not None and len(df_all) > max_rows:
        df_all = df_all.sample(max_rows, random_state=42).reset_index(drop=True)
        print(f"[INFO] Subsampled 2025 dataset to {len(df_all):,} rows")

    return df_all


if __name__ == "__main__":
    df_2025 = build_2025_dataset()
    out_path = PROCESSED_DIR / "bts_2025_backtest_base.parquet"
    df_2025.to_parquet(out_path, index=False)

    print(f"[OK] Saved 2025 backtest dataset to {out_path}")
    print("\n[INFO] 2025 dataset info:")
    print(df_2025.info(memory_usage="deep"))
