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

New in this version:
    A) Airline‑hub flags:
       - IsAirlineHubAtOrigin, IsAirlineHubAtDest based on 2010–2018 data.
    B) Fine‑grained holiday features:
       - IsHoliday, IsDayBeforeHoliday, IsDayAfterHoliday.
    C) Daily congestion features:
       - OriginDailyFlights, OriginDailyFlightsAirline.

Output
------
    data/processed/bts_delay_2010_2024_balanced_research.parquet

A follow‑up script (add_weather_to_dataset.py) enriches this dataset with
historical weather features and writes:

    data/processed/bts_delay_2010_2024_balanced_research_weather.parquet
"""

from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Years to include
START_YEAR = 2010
END_YEAR = 2024

# Target sample size per year (balanced across years).
# You are currently using ~1M/year → ~15M before weather join.
# You can reduce this if runtime/memory becomes an issue.
ROWS_PER_YEAR = 1_000_000

# Training years for aggregate statistics (avoid leakage)
AGG_STATS_START_YEAR = 2010
AGG_STATS_END_YEAR = 2018

# Global random seed base (for reproducible sampling)
RANDOM_SEED_BASE = 42

# Airline‑hub thresholds
HUB_SHARE_THRESHOLD = 0.25  # 25% of airport departures
HUB_MIN_FLIGHTS = 10_000    # minimum flights at that airport for that airline

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


def add_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fine-grained US holiday features based on FlightDate:

        - IsHoliday
        - IsDayBeforeHoliday
        - IsDayAfterHoliday

    Uses pandas' USFederalHolidayCalendar.
    """
    if df["FlightDate"].isna().all():
        df["IsHoliday"] = 0
        df["IsDayBeforeHoliday"] = 0
        df["IsDayAfterHoliday"] = 0
        return df

    cal = USFederalHolidayCalendar()
    start = df["FlightDate"].min() - pd.Timedelta(days=2)
    end = df["FlightDate"].max() + pd.Timedelta(days=2)
    holidays = cal.holidays(start=start, end=end)

    holidays_minus_1 = holidays - pd.Timedelta(days=1)
    holidays_plus_1 = holidays + pd.Timedelta(days=1)

    df["IsHoliday"] = df["FlightDate"].isin(holidays).astype("int8")
    df["IsDayBeforeHoliday"] = df["FlightDate"].isin(holidays_minus_1).astype("int8")
    df["IsDayAfterHoliday"] = df["FlightDate"].isin(holidays_plus_1).astype("int8")

    return df


def clean_month(zip_path: Path) -> pd.DataFrame:
    """
    Load and clean data for one month from a BTS zip file.

    - Keeps both cancelled and non-cancelled flights.
    - ArrDel15 may be NaN for cancelled flights.
    - Adds time-based and route features but NO aggregates yet.
    - Detects state columns (if present) and standardises them to
      OriginState and DestState for downstream use.
    - Adds fine-grained holiday features.
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

    # --- Fine-grained holiday features ---
    df = add_holiday_flags(df)

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

    # Holiday season (rough proxy: Nov–Jan) – keep for continuity
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
    for cand in [
        "OriginState",
        "ORIGIN_STATE_ABR",
        "OriginStateName",
        "ORIGIN_STATE_NM",
    ]:
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


def build_yearly_sample(
    start_year: int, end_year: int, rows_per_year: int
) -> pd.DataFrame:
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
            except Exception as e:  # noqa: BLE001
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

    New in this version:
      - Airline hub flags: IsAirlineHubAtOrigin / IsAirlineHubAtDest
      - Daily congestion: OriginDailyFlights / OriginDailyFlightsAirline
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

    # --- Daily congestion (Origin) ---
    # Total departures by (Origin, FlightDate)
    origin_daily = (
        df_stats.groupby(["Origin", "FlightDate"])
        .size()
        .reset_index(name="OriginDailyFlights")
    )

    # Total departures by (Origin, Reporting_Airline, FlightDate)
    origin_daily_airline = (
        df_stats.groupby(["Origin", "Reporting_Airline", "FlightDate"])
        .size()
        .reset_index(name="OriginDailyFlightsAirline")
    )

    # --- Airline hub features (Origin) ---
    airport_totals = (
        df_stats.groupby("Origin")
        .size()
        .reset_index(name="OriginTotalFlights")
    )

    airline_origin = (
        df_stats.groupby(["Origin", "Reporting_Airline"])
        .size()
        .reset_index(name="OriginAirlineFlights")
    )
    airline_origin = airline_origin.merge(airport_totals, on="Origin", how="left")
    airline_origin["AirlineOriginShare"] = (
        airline_origin["OriginAirlineFlights"]
        / airline_origin["OriginTotalFlights"].clip(lower=1)
    )
    airline_origin["IsAirlineHubAtOrigin"] = (
        (airline_origin["AirlineOriginShare"] >= HUB_SHARE_THRESHOLD)
        & (airline_origin["OriginAirlineFlights"] >= HUB_MIN_FLIGHTS)
    ).astype("int8")

    # --- Airline hub features (Dest) ---
    dest_totals = (
        df_stats.groupby("Dest")
        .size()
        .reset_index(name="DestTotalFlights")
    )
    airline_dest = (
        df_stats.groupby(["Dest", "Reporting_Airline"])
        .size()
        .reset_index(name="DestAirlineFlights")
    )
    airline_dest = airline_dest.merge(dest_totals, on="Dest", how="left")
    airline_dest["AirlineDestShare"] = (
        airline_dest["DestAirlineFlights"]
        / airline_dest["DestTotalFlights"].clip(lower=1)
    )
    airline_dest["IsAirlineHubAtDest"] = (
        (airline_dest["AirlineDestShare"] >= HUB_SHARE_THRESHOLD)
        & (airline_dest["DestAirlineFlights"] >= HUB_MIN_FLIGHTS)
    ).astype("int8")

    # ------------------------------------------------------------------
    # Merge back into full dataset
    # ------------------------------------------------------------------
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
    df = df.merge(
        origin_daily,
        on=["Origin", "FlightDate"],
        how="left",
    )
    df = df.merge(
        origin_daily_airline,
        on=["Origin", "Reporting_Airline", "FlightDate"],
        how="left",
    )
    df = df.merge(
        airline_origin[["Origin", "Reporting_Airline", "IsAirlineHubAtOrigin"]],
        on=["Origin", "Reporting_Airline"],
        how="left",
    )
    df = df.merge(
        airline_dest[["Dest", "Reporting_Airline", "IsAirlineHubAtDest"]],
        on=["Dest", "Reporting_Airline"],
        how="left",
    )

    # ------------------------------------------------------------------
    # Fill missing aggregates with global defaults
    # ------------------------------------------------------------------
    global_delay_rate = df_stats["ArrDel15"].mean()
    global_cancel_rate = df_stats["Cancelled"].mean()
    global_slot_mean = slot_stats["OriginSlotFlights"].mean()

    df["RouteDelayRate"] = (
        df["RouteDelayRate"].fillna(global_delay_rate).astype("float32")
    )
    df["RouteCancelRate"] = (
        df["RouteCancelRate"].fillna(global_cancel_rate).astype("float32")
    )
    df["RouteFlights"] = df["RouteFlights"].fillna(0).astype("int32")

    df["AirlineDelayRate"] = (
        df["AirlineDelayRate"].fillna(global_delay_rate).astype("float32")
    )
    df["AirlineCancelRate"] = (
        df["AirlineCancelRate"].fillna(global_cancel_rate).astype("float32")
    )
    df["AirlineFlights"] = df["AirlineFlights"].fillna(0).astype("int32")

    df["OriginSlotFlights"] = (
        df["OriginSlotFlights"].fillna(global_slot_mean).astype("float32")
    )

    # Daily congestion defaults
    global_origin_daily = origin_daily["OriginDailyFlights"].mean()
    global_origin_daily_airline = origin_daily_airline[
        "OriginDailyFlightsAirline"
    ].mean()

    df["OriginDailyFlights"] = (
        df["OriginDailyFlights"].fillna(global_origin_daily).astype("float32")
    )
    df["OriginDailyFlightsAirline"] = (
        df["OriginDailyFlightsAirline"]
        .fillna(global_origin_daily_airline)
        .astype("float32")
    )

    # Hub flags: treat missing as non-hub
    df["IsAirlineHubAtOrigin"] = df["IsAirlineHubAtOrigin"].fillna(0).astype("int8")
    df["IsAirlineHubAtDest"] = df["IsAirlineHubAtDest"].fillna(0).astype("int8")

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

