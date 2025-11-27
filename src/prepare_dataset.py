"""
prepare_dataset.py

Builds a sampled, feature-engineered BTS On‑Time Performance dataset
for 2010–2024, with ~ROWS_PER_YEAR rows per year, plus route/airline/
congestion aggregates computed from the training period (2010–2018).

This script deliberately:
    - Keeps both cancelled and non‑cancelled flights (we model both).
    - Uses only past years (2010–2018) to compute aggregate statistics
      such as RouteDelayRate, AirlineDelayRate, OriginSlotFlights, to
      avoid target leakage into later evaluation years.
    - Standardises any available state information into OriginState and
      DestState (used by the Streamlit app for state‑level guidance).
    - Produces a compact, balanced dataset suitable for downstream
      heavy modelling & diagnostics in train_models.py.

Output
------
    data/processed/bts_delay_2010_2024_balanced_research.parquet

A follow‑up script (add_weather_to_dataset.py, not shown here) enriches
this dataset with historical weather features and writes:

    data/processed/bts_delay_2010_2024_balanced_research_weather.parquet
"""

import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Years to include
START_YEAR = 2010
END_YEAR = 2024

# Target sample size per year (balanced across years)
ROWS_PER_YEAR = 500_000  # 500k per year ~ 7.5M rows total

# Training years for aggregate statistics (avoid leakage)
AGG_STATS_START_YEAR = 2010
AGG_STATS_END_YEAR = 2018

# Global random seed base (for reproducible sampling)
RANDOM_SEED_BASE = 42

# Columns we need from BTS On-Time Performance data
# (state columns will be added dynamically if present in the raw files)
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
# Helpers
# -------------------------------------------------------------------
def extract_csv_from_zip(zip_path: Path) -> BytesIO:
    """Return a file-like CSV handle from a BTS zip file."""
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found inside {zip_path}")
        csv_name = csv_names[0]
        raw_bytes = z.read(csv_name)
    return BytesIO(raw_bytes)


def clean_month(zip_path: Path) -> pd.DataFrame:
    """
    Load and clean data for one month from a BTS zip file.

    - Keeps both cancelled and non-cancelled flights.
    - ArrDel15 may be NaN for cancelled flights.
    - Adds time-based and route features but NO aggregates yet.
    - Detects state columns (if present) and standardises them to
      OriginState and DestState for downstream use.
    """
    csv_file = extract_csv_from_zip(zip_path)

    # --- Detect which state columns are available in this file ---
    preview = pd.read_csv(csv_file, nrows=0, low_memory=False)
    available_cols = set(preview.columns)
    csv_file.seek(0)

    # Common BTS / processed variants for state columns
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

    df = pd.read_csv(
        csv_file,
        usecols=usecols,
        low_memory=False,
    )

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

    # Remove diverted flights (we don't model them here)
    if "Diverted" in df.columns:
        df = df[df["Diverted"] != 1]
    df = df.drop(columns=["Diverted"], errors="ignore")

    # Cast types
    df["Cancelled"] = df["Cancelled"].astype("int8")
    # ArrDel15 can be NaN for cancelled flights; keep as float for now
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
    def dep_hour(x):
        try:
            x_int = int(x)
        except (TypeError, ValueError):
            return np.nan
        if x_int == 2400:
            return 0
        return x_int // 100

    df["DepHour"] = (
        df["CRSDepTime"]
        .astype("float64")
        .astype("Int64")
        .apply(dep_hour)
        .astype("float32")
    )
    # Drop rows with invalid dep hour (very rare)
    df = df.dropna(subset=["DepHour"])
    df["DepHour"] = df["DepHour"].astype("int8")

    df = df.drop(columns=["CRSDepTime"])

    df["DayOfMonth"] = df["FlightDate"].dt.day.astype("int8")
    df["DayOfYear"] = df["FlightDate"].dt.dayofyear.astype("int16")

    # Weekend flag (BTS: 1=Mon,...,7=Sun)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype("int8")

    # Season
    def month_to_season(m):
        if m in (12, 1, 2):
            return "winter"
        elif m in (3, 4, 5):
            return "spring"
        elif m in (6, 7, 8):
            return "summer"
        else:
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

    def distance_band(d):
        if d < 500:
            return "short"
        elif d < 1500:
            return "medium"
        elif d < 3000:
            return "long"
        else:
            return "ultra_long"

    df["DistanceBand"] = df["Distance"].apply(distance_band).astype("category")

    # Set core categoricals
    df["Reporting_Airline"] = df["Reporting_Airline"].astype("category")
    df["Origin"] = df["Origin"].astype("category")
    df["Dest"] = df["Dest"].astype("category")

    # --- Optional state columns (standardised) --------------------
    # Try several common raw column names and map them into
    # OriginState / DestState for the processed dataset.
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
        # Create the column so downstream code can rely on its existence
        df["OriginState"] = pd.Categorical([np.nan] * len(df))

    if dest_state_series is not None:
        df["DestState"] = dest_state_series.astype("category")
    else:
        df["DestState"] = pd.Categorical([np.nan] * len(df))

    return df


def build_yearly_sample(start_year: int, end_year: int, rows_per_year: int) -> pd.DataFrame:
    """
    Clean and sample approximately rows_per_year flights per year.

    Sampling is uniform across months within each year:
      rows_per_month ≈ rows_per_year / 12
    """
    all_years = []
    rows_per_month = max(rows_per_year // 12, 1)

    for year in range(start_year, end_year + 1):
        print(f"\n[INFO] Processing year {year}")
        year_frames = []

        for month in range(1, 13):
            zip_path = RAW_DIR / f"ontime_{year}_{month:02d}.zip"
            if not zip_path.exists():
                print(f"[WARN]  {zip_path.name} not found, skipping.")
                continue

            print(f"[INFO]  Cleaning {zip_path.name}")
            try:
                df_month = clean_month(zip_path)
            except Exception as e:
                print(f"[ERROR] Failed {zip_path.name}: {e}")
                continue

            if len(df_month) == 0:
                continue

            n_sample = min(rows_per_month, len(df_month))
            if n_sample <= 0:
                continue

            # Year- and month-dependent seed for deterministic but varied sampling
            rs = RANDOM_SEED_BASE + year * 100 + month
            df_sample = df_month.sample(n=n_sample, random_state=rs)
            year_frames.append(df_sample)

        if not year_frames:
            print(f"[WARN] No data collected for year {year}, skipping.")
            continue

        df_year = pd.concat(year_frames, ignore_index=True)
        print(f"[INFO]  Year {year} sampled rows: {len(df_year):,}")
        all_years.append(df_year)

    if not all_years:
        raise RuntimeError("No data loaded. Did you download the raw BTS files?")

    df_all = pd.concat(all_years, ignore_index=True)
    print(f"\n[INFO] Combined sampled dataset size: {len(df_all):,} rows")
    return df_all


def add_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add route/airline reliability and congestion features using ONLY
    the training years (AGG_STATS_START_YEAR–AGG_STATS_END_YEAR)
    to avoid using future information in aggregates.
    """
    stats_train_mask = df["Year"].between(AGG_STATS_START_YEAR, AGG_STATS_END_YEAR)
    df_stats = df[stats_train_mask].copy()

    if df_stats.empty:
        raise RuntimeError(
            "No rows in the specified aggregate stats training period "
            f"{AGG_STATS_START_YEAR}–{AGG_STATS_END_YEAR}."
        )

    # --- Route-level reliability ---
    route_stats = (
        df_stats.groupby(["Reporting_Airline", "Origin", "Dest"])
        .agg(
            RouteDelayRate=("ArrDel15", "mean"),  # NaNs ignored
            RouteCancelRate=("Cancelled", "mean"),
            RouteFlights=("Cancelled", "size"),
        )
        .reset_index()
    )

    # --- Airline-level reliability ---
    airline_stats = (
        df_stats.groupby("Reporting_Airline")
        .agg(
            AirlineDelayRate=("ArrDel15", "mean"),
            AirlineCancelRate=("Cancelled", "mean"),
            AirlineFlights=("Cancelled", "size"),
        )
        .reset_index()
    )

    # --- Congestion proxy: avg flights per (Origin, Month, DOW, DepHour) ---
    slot_stats = (
        df_stats.groupby(["Origin", "Month", "DayOfWeek", "DepHour"])
        .size()
        .reset_index(name="OriginSlotFlights")
    )

    # Merge back into full dataset
    df = df.merge(
        route_stats,
        on=["Reporting_Airline", "Origin", "Dest"],
        how="left",
    )
    df = df.merge(
        airline_stats,
        on="Reporting_Airline",
        how="left",
    )
    df = df.merge(
        slot_stats,
        on=["Origin", "Month", "DayOfWeek", "DepHour"],
        how="left",
    )

    # Fill missing aggregates with global defaults
    global_delay_rate = df_stats["ArrDel15"].mean()
    global_cancel_rate = df_stats["Cancelled"].mean()
    global_slot_mean = slot_stats["OriginSlotFlights"].mean()

    df["RouteDelayRate"] = df["RouteDelayRate"].fillna(global_delay_rate).astype(
        "float32"
    )
    df["RouteCancelRate"] = df["RouteCancelRate"].fillna(global_cancel_rate).astype(
        "float32"
    )
    df["RouteFlights"] = df["RouteFlights"].fillna(0).astype("int32")

    df["AirlineDelayRate"] = df["AirlineDelayRate"].fillna(global_delay_rate).astype(
        "float32"
    )
    df["AirlineCancelRate"] = df["AirlineCancelRate"].fillna(global_cancel_rate).astype(
        "float32"
    )
    df["AirlineFlights"] = df["AirlineFlights"].fillna(0).astype("int32")

    df["OriginSlotFlights"] = df["OriginSlotFlights"].fillna(global_slot_mean).astype(
        "float32"
    )

    return df


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    df = build_yearly_sample(START_YEAR, END_YEAR, ROWS_PER_YEAR)
    df = add_aggregates(df)

    out_path = PROCESSED_DIR / "bts_delay_2010_2024_balanced_research.parquet"
    df.to_parquet(out_path, index=False)

    print(f"[OK] Saved processed dataset to {out_path}")
    print("\n[INFO] Dataset info:")
    print(df.info(memory_usage="deep"))
