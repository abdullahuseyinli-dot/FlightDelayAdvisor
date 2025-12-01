# src/app.py
import math
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import requests


# -------------------------------------------------------------------
# Paths (consistent with training)
# -------------------------------------------------------------------
from pathlib import Path

DATA_PATH = Path(
    "data/processed/bts_delay_2010_2024_balanced_research_weather_sample.parquet"
)


# Google Drive file IDs for the parquet datasets.
# Google Drive file IDs for the parquet datasets.
GDRIVE_FILE_ID_FULL = "1PFxYfpn2pT-kg_JvCVjgy5Q1qSNZdxbm"      # full (optional, not used)
GDRIVE_FILE_ID_WEATHER = "1zYCfrenfIMVyUZ8sFK89M68NOHNfUGe5"    # <-- NEW 1M SAMPLE


MODELS_DIR = Path("models")

MODELS_DIR = Path("models")
DELAY_MODEL_PATH = MODELS_DIR / "catboost_delay15_calibrated.joblib"
# use the new best cancellation model from training (LGBM, calibrated)
CANCEL_MODEL_PATH = MODELS_DIR / "lgbm_cancel_calibrated.joblib"

# Aggregation period for stats (match training script)
AGG_STATS_START_YEAR = 2010
AGG_STATS_END_YEAR = 2018

# -------------------------------------------------------------------
# Features (MUST match train_models.py)
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
    # Hub flags
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

# thresholds for route guidance
MIN_ROUTE_FLIGHTS_FOR_WARNING = 50
MIN_FLIGHTS_AIRLINE_COMPARE = 100
MIN_FLIGHTS_AIRPORT_COMPARE = 500

# extra thresholds / weights used in guidance logic
MIN_LEG_FLIGHTS_FOR_ONE_STOP = 150  # for one‑stop suggestions
ALPHA_CANCEL_WEIGHT = 3.0           # cancellation is rarer but more serious


# -------------------------------------------------------------------
# Helpers for feature engineering (same logic as prepare_dataset.py)
# -------------------------------------------------------------------
def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "winter"
    elif m in (3, 4, 5):
        return "spring"
    elif m in (6, 7, 8):
        return "summer"
    else:
        return "autumn"


def distance_band(d: float) -> str:
    if d < 500:
        return "short"
    elif d < 1500:
        return "medium"
    elif d < 3000:
        return "long"
    else:
        return "ultra_long"


def risk_label(p: float) -> str:
    """Soft risk label for UI; thresholds are arbitrary but intuitive."""
    if p < 0.2:
        return "Low"
    elif p < 0.5:
        return "Medium"
    else:
        return "High"


def risk_emoji(label: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(label, "")


def pick_first_existing(columns, candidates):
    """Return the first column name from candidates that exists in df.columns."""
    for c in candidates:
        if c in columns:
            return c
    return None


# ---- Holiday helpers ------------------------------------------------
def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """
    weekday: Monday=0 .. Sunday=6
    n: 1=first, 2=second, ...
    """
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    for _ in range(n - 1):
        d += timedelta(days=7)
    return d


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Last given weekday (0=Mon..6=Sun) in a month."""
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def get_us_federal_holidays(year: int):
    """
    Simple US federal holiday calendar (not perfect but good enough
    for 'IsHoliday / before / after' features).
    """
    holidays = set()

    # Fixed-date holidays
    holidays.add(date(year, 1, 1))   # New Year
    holidays.add(date(year, 7, 4))   # Independence Day
    holidays.add(date(year, 11, 11)) # Veterans Day
    holidays.add(date(year, 12, 25)) # Christmas

    # MLK Day: 3rd Monday of Jan
    holidays.add(_nth_weekday_of_month(year, 1, 0, 3))
    # Presidents' Day: 3rd Monday of Feb
    holidays.add(_nth_weekday_of_month(year, 2, 0, 3))
    # Memorial Day: last Monday of May
    holidays.add(_last_weekday_of_month(year, 5, 0))
    # Labor Day: 1st Monday of September
    holidays.add(_nth_weekday_of_month(year, 9, 0, 1))
    # Columbus Day: 2nd Monday of October
    holidays.add(_nth_weekday_of_month(year, 10, 0, 2))
    # Thanksgiving: 4th Thursday of November
    holidays.add(_nth_weekday_of_month(year, 11, 3, 4))

    return holidays


# -------------------------------------------------------------------
# Cached loading of data & models
# -------------------------------------------------------------------

def _download_file_from_google_drive(file_id: str, destination: Path) -> None:
    """
    Download a (possibly large) file from Google Drive into ``destination``.

    The file IDs are defined at the top of this module. This allows the app
    to run on Streamlit Cloud without checking the large parquet file into
    the Git repository.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    # Initial request
    response = session.get(url, params={"id": file_id}, stream=True)

    # For large files Google Drive adds a confirmation token that we need
    # to send again; see e.g. https://stackoverflow.com/a/39225272
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token is not None:
        response = session.get(
            url,
            params={"id": file_id, "confirm": token},
            stream=True,
        )

    # Raise a helpful error if something went wrong (404, permissions, ...)
    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

@st.cache_data(show_spinner=True)
def load_metadata():
    """
    Load processed dataset and derive:
    - lists of airlines & airports
    - route-level and airline-level stats
    - congestion stats
    - daily congestion stats (OriginDailyFlights, OriginDailyFlightsAirline)
    - airline hub flags (IsAirlineHubAtOrigin/Dest)
    - typical distance per route
    - monthly climatological weather per airport (Origin/Dest)
    - global defaults for fallback
    """
    if not DATA_PATH.exists():
        _download_file_from_google_drive(GDRIVE_FILE_ID_WEATHER, DATA_PATH)

    df = pd.read_parquet(DATA_PATH)

    # Use full data for choices
    airlines = sorted(df["Reporting_Airline"].dropna().unique().tolist())
    origins = sorted(df["Origin"].dropna().unique().tolist())
    dests = sorted(df["Dest"].dropna().unique().tolist())

    # Subset for aggregate stats (match training script: 2010–2018)
    stats_mask = (df["Year"] >= AGG_STATS_START_YEAR) & (
        df["Year"] <= AGG_STATS_END_YEAR
    )
    df_stats = df[stats_mask].copy()
    if df_stats.empty:
        df_stats = df.copy()

    # Route-level reliability (per airline + route)
    route_meta = (
        df_stats.groupby(["Reporting_Airline", "Origin", "Dest"])
        .agg(
            RouteDelayRate=("RouteDelayRate", "mean"),
            RouteCancelRate=("RouteCancelRate", "mean"),
            RouteFlights=("RouteFlights", "mean"),
        )
        .reset_index()
    )

    # Airline-level reliability
    airline_meta = (
        df_stats.groupby("Reporting_Airline")
        .agg(
            AirlineDelayRate=("AirlineDelayRate", "mean"),
            AirlineCancelRate=("AirlineCancelRate", "mean"),
            AirlineFlights=("AirlineFlights", "mean"),
        )
        .reset_index()
    )

    # Congestion: avg flights per (Origin, Month, DOW, DepHour)
    slot_meta = (
        df_stats.groupby(["Origin", "Month", "DayOfWeek", "DepHour"])
        .agg(OriginSlotFlights=("OriginSlotFlights", "mean"))
        .reset_index()
    )

    # --- New congestion features: daily flights --------------------
    # Daily total departures at each origin
    daily_origin = (
        df_stats.groupby(["Origin", "FlightDate"])["Cancelled"]
        .size()
        .reset_index(name="DailyFlightsOrigin")
    )
    # Attach calendar info
    calendar_cols = (
        df_stats[["FlightDate", "Month", "DayOfWeek"]].drop_duplicates()
    )
    daily_origin = daily_origin.merge(calendar_cols, on="FlightDate", how="left")
    daily_origin_meta = (
        daily_origin.groupby(["Origin", "Month", "DayOfWeek"])
        .agg(OriginDailyFlights=("DailyFlightsOrigin", "mean"))
        .reset_index()
    )

    # Daily airline-specific departures at each origin
    daily_origin_airline = (
        df_stats.groupby(["Origin", "Reporting_Airline", "FlightDate"])["Cancelled"]
        .size()
        .reset_index(name="DailyFlightsOriginAirline")
    )
    daily_origin_airline = daily_origin_airline.merge(
        calendar_cols, on="FlightDate", how="left"
    )
    daily_origin_airline_meta = (
        daily_origin_airline.groupby(
            ["Origin", "Reporting_Airline", "Month", "DayOfWeek"]
        )
        .agg(OriginDailyFlightsAirline=("DailyFlightsOriginAirline", "mean"))
        .reset_index()
    )

    # --- Airline-hub flags (A) ------------------------------------
    # Origin hubs
    origin_totals = (
        df_stats.groupby("Origin")["Cancelled"]
        .size()
        .reset_index(name="TotalOriginFlights")
    )
    origin_airline_totals = (
        df_stats.groupby(["Origin", "Reporting_Airline"])["Cancelled"]
        .size()
        .reset_index(name="OriginAirlineFlights")
    )
    hub_origin = origin_airline_totals.merge(
        origin_totals, on="Origin", how="left"
    )
    hub_origin["OriginAirlineShare"] = (
        hub_origin["OriginAirlineFlights"] / hub_origin["TotalOriginFlights"]
    )
    hub_origin["IsAirlineHubAtOrigin"] = (
        (hub_origin["OriginAirlineShare"] >= 0.25)
        & (hub_origin["OriginAirlineFlights"] >= 10_000)
    ).astype("int8")
    hub_origin_meta = hub_origin[
        ["Origin", "Reporting_Airline", "IsAirlineHubAtOrigin"]
    ]

    # Destination hubs
    dest_totals = (
        df_stats.groupby("Dest")["Cancelled"]
        .size()
        .reset_index(name="TotalDestFlights")
    )
    dest_airline_totals = (
        df_stats.groupby(["Dest", "Reporting_Airline"])["Cancelled"]
        .size()
        .reset_index(name="DestAirlineFlights")
    )
    hub_dest = dest_airline_totals.merge(
        dest_totals, on="Dest", how="left"
    )
    hub_dest["DestAirlineShare"] = (
        hub_dest["DestAirlineFlights"] / hub_dest["TotalDestFlights"]
    )
    hub_dest["IsAirlineHubAtDest"] = (
        (hub_dest["DestAirlineShare"] >= 0.25)
        & (hub_dest["DestAirlineFlights"] >= 10_000)
    ).astype("int8")
    hub_dest_meta = hub_dest[
        ["Dest", "Reporting_Airline", "IsAirlineHubAtDest"]
    ]

    # Route‑pair meta across all airlines (all years, for suggestions)
    route_pair_meta = (
        df.groupby(["Origin", "Dest"])
        .agg(
            Distance=("Distance", "mean"),
            Flights=("Cancelled", "size"),
            DelayRate=("ArrDel15", "mean"),
            CancelRate=("Cancelled", "mean"),
        )
        .reset_index()
    )

    # Distance per (Origin, Dest) for quick lookup
    route_distance_meta = route_pair_meta[["Origin", "Dest", "Distance"]].copy()

    # Monthly climatological weather: Origin side
    origin_weather_meta = (
        df.groupby(["Origin", "Month"])
        .agg(
            Origin_tavg=("Origin_tavg", "mean"),
            Origin_prcp=("Origin_prcp", "mean"),
            Origin_snow=("Origin_snow", "mean"),
            Origin_wspd=("Origin_wspd", "mean"),
            Origin_BadWeather=("Origin_BadWeather", "mean"),
        )
        .reset_index()
    )

    # Monthly climatological weather: Destination side
    dest_weather_meta = (
        df.groupby(["Dest", "Month"])
        .agg(
            Dest_tavg=("Dest_tavg", "mean"),
            Dest_prcp=("Dest_prcp", "mean"),
            Dest_snow=("Dest_snow", "mean"),
            Dest_wspd=("Dest_wspd", "mean"),
            Dest_BadWeather=("Dest_BadWeather", "mean"),
        )
        .reset_index()
    )

    # Global defaults (used when combination is unseen)
    non_cancel = df[(df["Cancelled"] == 0) & df["ArrDel15"].notna()]
    global_delay_rate = float(non_cancel["ArrDel15"].mean()) if len(non_cancel) > 0 else 0.2
    global_cancel_rate = float(df["Cancelled"].mean())
    global_slot_mean = float(df["OriginSlotFlights"].mean())
    global_distance_mean = float(df["Distance"].mean())

    global_daily_flights_mean = (
        float(daily_origin_meta["OriginDailyFlights"].mean())
        if not daily_origin_meta.empty
        else 0.0
    )
    global_daily_flights_airline_mean = (
        float(daily_origin_airline_meta["OriginDailyFlightsAirline"].mean())
        if not daily_origin_airline_meta.empty
        else 0.0
    )

    defaults = {
        "global_delay_rate": global_delay_rate,
        "global_cancel_rate": global_cancel_rate,
        "global_slot_mean": global_slot_mean,
        "global_distance_mean": global_distance_mean,
        "OriginDailyFlights_mean": global_daily_flights_mean,
        "OriginDailyFlightsAirline_mean": global_daily_flights_airline_mean,
        # weather defaults
        "Origin_tavg_mean": float(df["Origin_tavg"].mean()),
        "Origin_prcp_mean": float(df["Origin_prcp"].mean()),
        "Origin_snow_mean": float(df["Origin_snow"].mean()),
        "Origin_wspd_mean": float(df["Origin_wspd"].mean()),
        "Origin_BadWeather_mean": float(df["Origin_BadWeather"].mean()),
        "Dest_tavg_mean": float(df["Dest_tavg"].mean()),
        "Dest_prcp_mean": float(df["Dest_prcp"].mean()),
        "Dest_snow_mean": float(df["Dest_snow"].mean()),
        "Dest_wspd_mean": float(df["Dest_wspd"].mean()),
        "Dest_BadWeather_mean": float(df["Dest_BadWeather"].mean()),
    }

    return (
        df,
        airlines,
        origins,
        dests,
        route_meta,
        airline_meta,
        slot_meta,
        daily_origin_meta,
        daily_origin_airline_meta,
        hub_origin_meta,
        hub_dest_meta,
        route_pair_meta,
        route_distance_meta,
        origin_weather_meta,
        dest_weather_meta,
        defaults,
    )


@st.cache_resource(show_spinner=True)
def load_models():
    delay_model = joblib.load(DELAY_MODEL_PATH)
    cancel_model = joblib.load(CANCEL_MODEL_PATH)
    return delay_model, cancel_model


# -------------------------------------------------------------------
# Lookup aggregated stats & weather
# -------------------------------------------------------------------
def get_route_stats(airline, origin, dest, route_meta, defaults):
    mask = (
        (route_meta["Reporting_Airline"] == airline)
        & (route_meta["Origin"] == origin)
        & (route_meta["Dest"] == dest)
    )
    if mask.any():
        row = route_meta.loc[mask].iloc[0]
        return (
            float(row["RouteDelayRate"]),
            float(row["RouteCancelRate"]),
            float(row["RouteFlights"]),
        )
    # Fallback
    return (
        defaults["global_delay_rate"],
        defaults["global_cancel_rate"],
        0.0,
    )


def get_airline_stats(airline, airline_meta, defaults):
    mask = airline_meta["Reporting_Airline"] == airline
    if mask.any():
        row = airline_meta.loc[mask].iloc[0]
        return (
            float(row["AirlineDelayRate"]),
            float(row["AirlineCancelRate"]),
            float(row["AirlineFlights"]),
        )
    return (
        defaults["global_delay_rate"],
        defaults["global_cancel_rate"],
        0.0,
    )


def get_slot_stats(origin, month, dow, dep_hour, slot_meta, defaults):
    mask = (
        (slot_meta["Origin"] == origin)
        & (slot_meta["Month"] == month)
        & (slot_meta["DayOfWeek"] == dow)
        & (slot_meta["DepHour"] == dep_hour)
    )
    if mask.any():
        row = slot_meta.loc[mask].iloc[0]
        return float(row["OriginSlotFlights"])
    return defaults["global_slot_mean"]


def get_daily_origin_flights(
    origin, month, dow, daily_origin_meta, defaults
) -> float:
    mask = (
        (daily_origin_meta["Origin"] == origin)
        & (daily_origin_meta["Month"] == month)
        & (daily_origin_meta["DayOfWeek"] == dow)
    )
    if mask.any():
        return float(
            daily_origin_meta.loc[mask, "OriginDailyFlights"].iloc[0]
        )
    return defaults.get("OriginDailyFlights_mean", 0.0)


def get_daily_origin_airline_flights(
    origin, airline, month, dow, daily_origin_airline_meta, defaults
) -> float:
    mask = (
        (daily_origin_airline_meta["Origin"] == origin)
        & (daily_origin_airline_meta["Reporting_Airline"] == airline)
        & (daily_origin_airline_meta["Month"] == month)
        & (daily_origin_airline_meta["DayOfWeek"] == dow)
    )
    if mask.any():
        return float(
            daily_origin_airline_meta.loc[
                mask, "OriginDailyFlightsAirline"
            ].iloc[0]
        )
    return defaults.get("OriginDailyFlightsAirline_mean", 0.0)


def get_hub_flags(
    airline,
    origin,
    dest,
    hub_origin_meta,
    hub_dest_meta,
) -> tuple[float, float]:
    is_origin_hub = 0.0
    is_dest_hub = 0.0

    mask_o = (
        (hub_origin_meta["Origin"] == origin)
        & (hub_origin_meta["Reporting_Airline"] == airline)
    )
    if mask_o.any():
        is_origin_hub = float(
            hub_origin_meta.loc[mask_o, "IsAirlineHubAtOrigin"].iloc[0]
        )

    mask_d = (
        (hub_dest_meta["Dest"] == dest)
        & (hub_dest_meta["Reporting_Airline"] == airline)
    )
    if mask_d.any():
        is_dest_hub = float(
            hub_dest_meta.loc[mask_d, "IsAirlineHubAtDest"].iloc[0]
        )

    return is_origin_hub, is_dest_hub


def get_route_distance_default(origin, dest, route_distance_meta, defaults):
    mask = (route_distance_meta["Origin"] == origin) & (
        route_distance_meta["Dest"] == dest
    )
    if mask.any():
        return float(route_distance_meta.loc[mask].iloc[0]["Distance"])
    return defaults["global_distance_mean"]


def get_origin_weather(origin, month, origin_weather_meta, defaults):
    mask = (origin_weather_meta["Origin"] == origin) & (
        origin_weather_meta["Month"] == month
    )
    if mask.any():
        row = origin_weather_meta.loc[mask].iloc[0]
        return (
            float(row["Origin_tavg"]),
            float(row["Origin_prcp"]),
            float(row["Origin_snow"]),
            float(row["Origin_wspd"]),
            float(row["Origin_BadWeather"]),
        )
    return (
        defaults["Origin_tavg_mean"],
        defaults["Origin_prcp_mean"],
        defaults["Origin_snow_mean"],
        defaults["Origin_wspd_mean"],
        defaults["Origin_BadWeather_mean"],
    )


def get_dest_weather(dest, month, dest_weather_meta, defaults):
    mask = (dest_weather_meta["Dest"] == dest) & (
        dest_weather_meta["Month"] == month
    )
    if mask.any():
        row = dest_weather_meta.loc[mask].iloc[0]
        return (
            float(row["Dest_tavg"]),
            float(row["Dest_prcp"]),
            float(row["Dest_snow"]),
            float(row["Dest_wspd"]),
            float(row["Dest_BadWeather"]),
        )
    return (
        defaults["Dest_tavg_mean"],
        defaults["Dest_prcp_mean"],
        defaults["Dest_snow_mean"],
        defaults["Dest_wspd_mean"],
        defaults["Dest_BadWeather_mean"],
    )


# -------------------------------------------------------------------
# Build feature row for the models
# -------------------------------------------------------------------
def build_feature_row(
    travel_date: date,
    dep_hour: int,
    airline: str,
    origin: str,
    dest: str,
    distance: float,
    route_meta: pd.DataFrame,
    airline_meta: pd.DataFrame,
    slot_meta: pd.DataFrame,
    daily_origin_meta: pd.DataFrame,
    daily_origin_airline_meta: pd.DataFrame,
    hub_origin_meta: pd.DataFrame,
    hub_dest_meta: pd.DataFrame,
    origin_weather_meta: pd.DataFrame,
    dest_weather_meta: pd.DataFrame,
    defaults: dict,
    airport_to_state: dict,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame with exactly the same columns
    as used during training (including new hub, holiday and
    daily congestion features).
    """
    year = travel_date.year
    month = travel_date.month
    day_of_month = travel_date.day
    day_of_year = int(travel_date.timetuple().tm_yday)

    # Python weekday: 0=Mon,...,6=Sun  -> BTS DayOfWeek: 1=Mon,...,7=Sun
    dow_py = travel_date.weekday()
    day_of_week = dow_py + 1

    is_weekend = 1 if day_of_week in (6, 7) else 0
    is_holiday_season = 1 if month in (11, 12, 1) else 0

    # Holiday flags (B)
    holidays = get_us_federal_holidays(year)
    is_holiday = 1 if travel_date in holidays else 0
    is_day_before_holiday = 1 if (travel_date + timedelta(days=1)) in holidays else 0
    is_day_after_holiday = 1 if (travel_date - timedelta(days=1)) in holidays else 0

    season = month_to_season(month)
    route = f"{origin}_{dest}"

    dep_sin = math.sin(2 * math.pi * dep_hour / 24.0)
    dep_cos = math.cos(2 * math.pi * dep_hour / 24.0)

    dist_band = distance_band(distance)

    # Aggregated stats
    route_delay_rate, route_cancel_rate, route_flights = get_route_stats(
        airline, origin, dest, route_meta, defaults
    )
    airline_delay_rate, airline_cancel_rate, airline_flights = get_airline_stats(
        airline, airline_meta, defaults
    )
    origin_slot_flights = get_slot_stats(
        origin, month, day_of_week, dep_hour, slot_meta, defaults
    )

    # New daily congestion features (C)
    origin_daily_flights = get_daily_origin_flights(
        origin, month, day_of_week, daily_origin_meta, defaults
    )
    origin_daily_flights_airline = get_daily_origin_airline_flights(
        origin,
        airline,
        month,
        day_of_week,
        daily_origin_airline_meta,
        defaults,
    )

    # Hub flags (A)
    is_hub_origin, is_hub_dest = get_hub_flags(
        airline, origin, dest, hub_origin_meta, hub_dest_meta
    )

    # Climatological weather for this airport/month
    (
        origin_tavg,
        origin_prcp,
        origin_snow,
        origin_wspd,
        origin_bad,
    ) = get_origin_weather(origin, month, origin_weather_meta, defaults)
    (
        dest_tavg,
        dest_prcp,
        dest_snow,
        dest_wspd,
        dest_bad,
    ) = get_dest_weather(dest, month, dest_weather_meta, defaults)

    # State information (for OriginState / DestState categoricals)
    origin_state = airport_to_state.get(origin, "Unknown")
    dest_state = airport_to_state.get(dest, "Unknown")

    data = {
        # Numeric
        "Year": [year],
        "Month": [month],
        "DayOfWeek": [day_of_week],
        "DayOfMonth": [day_of_month],
        "DayOfYear": [day_of_year],
        "DepHour": [dep_hour],
        "IsWeekend": [is_weekend],
        "IsHolidaySeason": [is_holiday_season],
        "IsHoliday": [is_holiday],
        "IsDayBeforeHoliday": [is_day_before_holiday],
        "IsDayAfterHoliday": [is_day_after_holiday],
        "Distance": [float(distance)],
        "RouteDelayRate": [route_delay_rate],
        "RouteCancelRate": [route_cancel_rate],
        "RouteFlights": [route_flights],
        "AirlineDelayRate": [airline_delay_rate],
        "AirlineCancelRate": [airline_cancel_rate],
        "AirlineFlights": [airline_flights],
        "OriginSlotFlights": [origin_slot_flights],
        "OriginDailyFlights": [origin_daily_flights],
        "OriginDailyFlightsAirline": [origin_daily_flights_airline],
        "IsAirlineHubAtOrigin": [is_hub_origin],
        "IsAirlineHubAtDest": [is_hub_dest],
        "DepHour_sin": [dep_sin],
        "DepHour_cos": [dep_cos],
        "Origin_tavg": [origin_tavg],
        "Origin_prcp": [origin_prcp],
        "Origin_snow": [origin_snow],
        "Origin_wspd": [origin_wspd],
        "Origin_BadWeather": [origin_bad],
        "Dest_tavg": [dest_tavg],
        "Dest_prcp": [dest_prcp],
        "Dest_snow": [dest_snow],
        "Dest_wspd": [dest_wspd],
        "Dest_BadWeather": [dest_bad],
        # Categoricals
        "Reporting_Airline": [airline],
        "Origin": [origin],
        "Dest": [dest],
        "Route": [route],
        "Season": [season],
        "DistanceBand": [dist_band],
        "OriginState": [origin_state],
        "DestState": [dest_state],
    }

    X = pd.DataFrame(data, columns=FEATURE_COLS)

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    return X


# -------------------------------------------------------------------
# Simple explanation of why a flight is risky/safe
# -------------------------------------------------------------------
def explain_risk_factors(
    travel_date: date,
    dep_hour: int,
    airline: str,
    origin: str,
    dest: str,
    distance: float,
    route_meta: pd.DataFrame,
    airline_meta: pd.DataFrame,
    slot_meta: pd.DataFrame,
    origin_weather_meta: pd.DataFrame,
    dest_weather_meta: pd.DataFrame,
    defaults: dict,
):
    """Return a list of short, human-readable reasons for the risk level."""
    reasons = []

    month = travel_date.month
    dow_py = travel_date.weekday()
    day_of_week = dow_py + 1

    # Global baselines
    global_delay = defaults["global_delay_rate"]
    global_cancel = defaults["global_cancel_rate"]

    # Stats for this configuration
    route_delay_rate, route_cancel_rate, route_flights = get_route_stats(
        airline, origin, dest, route_meta, defaults
    )
    airline_delay_rate, airline_cancel_rate, airline_flights = get_airline_stats(
        airline, airline_meta, defaults
    )
    origin_slot_flights = get_slot_stats(
        origin, month, day_of_week, dep_hour, slot_meta, defaults
    )
    (
        origin_tavg,
        origin_prcp,
        origin_snow,
        origin_wspd,
        origin_bad,
    ) = get_origin_weather(origin, month, origin_weather_meta, defaults)
    (
        dest_tavg,
        dest_prcp,
        dest_snow,
        dest_wspd,
        dest_bad,
    ) = get_dest_weather(dest, month, dest_weather_meta, defaults)

    # Seasonality / calendar
    if month in (11, 12, 1):
        reasons.append(
            "The flight is in the late‑autumn / winter holiday period, when congestion "
            "and weather‑related disruptions are historically higher."
        )
    elif month in (6, 7, 8):
        reasons.append(
            "The flight is in the summer peak travel season, which often shows elevated "
            "traffic and delay risk."
        )

    if day_of_week in (6, 7):
        reasons.append(
            "Departure is on a weekend, when traffic volumes at many airports are higher."
        )

    # Route‑level behaviour
    if route_flights > 0:
        if route_delay_rate > global_delay + 0.03:
            reasons.append(
                "This specific route has a **higher‑than‑average** historical delay rate "
                "across all airlines."
            )
        elif route_delay_rate < global_delay - 0.03:
            reasons.append(
                "This route has a **lower‑than‑average** historical delay rate compared "
                "to the network overall."
            )

        if route_cancel_rate > global_cancel + 0.002:
            reasons.append(
                "This route has a slightly elevated historical cancellation rate."
            )

    # Airline‑level behaviour
    if airline_flights > 0:
        if airline_delay_rate > global_delay + 0.03:
            reasons.append(
                "The chosen airline is historically more delay‑prone than the network "
                "average."
            )
        elif airline_delay_rate < global_delay - 0.03:
            reasons.append(
                "The chosen airline is historically more punctual than the network "
                "average."
            )

        if airline_cancel_rate > global_cancel + 0.002:
            reasons.append(
                "The airline has a somewhat higher cancellation rate than average."
            )

    # Congestion around chosen hour
    if origin_slot_flights > defaults["global_slot_mean"] * 1.3:
        reasons.append(
            "The chosen departure hour falls in a **busy period** at the origin airport, "
            "which tends to increase delay risk."
        )
    elif origin_slot_flights < defaults["global_slot_mean"] * 0.7:
        reasons.append(
            "The departure hour is relatively **quiet** at the origin airport, which "
            "helps reliability."
        )

    # Weather climatology
    if origin_bad > 0.35 or dest_bad > 0.35:
        reasons.append(
            "For this month, the origin and/or destination airport experiences bad "
            "weather on a relatively large share of days, nudging disruption risk upward."
        )

    return reasons


# -------------------------------------------------------------------
# Model‑based suggestions for safer options
# -------------------------------------------------------------------
def suggest_safer_hours(
    travel_date: date,
    airline: str,
    origin: str,
    dest: str,
    distance: float,
    route_meta: pd.DataFrame,
    airline_meta: pd.DataFrame,
    slot_meta: pd.DataFrame,
    daily_origin_meta: pd.DataFrame,
    daily_origin_airline_meta: pd.DataFrame,
    hub_origin_meta: pd.DataFrame,
    hub_dest_meta: pd.DataFrame,
    origin_weather_meta: pd.DataFrame,
    dest_weather_meta: pd.DataFrame,
    defaults: dict,
    delay_model,
    current_hour: int,
    airport_to_state: dict,
    max_suggestions: int = 3,
):
    """Return a list of (hour, delay_proba) that look safer than the current one."""
    records = []
    for h in range(24):
        X_row = build_feature_row(
            travel_date,
            h,
            airline,
            origin,
            dest,
            distance,
            route_meta,
            airline_meta,
            slot_meta,
            daily_origin_meta,
            daily_origin_airline_meta,
            hub_origin_meta,
            hub_dest_meta,
            origin_weather_meta,
            dest_weather_meta,
            defaults,
            airport_to_state,
        )
        proba_delay = float(delay_model.predict_proba(X_row)[0, 1])
        records.append((h, proba_delay))

    # sort by delay risk
    records.sort(key=lambda t: t[1])

    safer = [(h, p) for (h, p) in records if h != current_hour]
    return safer[:max_suggestions]


def suggest_safer_airlines(
    travel_date: date,
    dep_hour: int,
    origin: str,
    dest: str,
    distance: float,
    route_meta: pd.DataFrame,
    airline_meta: pd.DataFrame,
    slot_meta: pd.DataFrame,
    daily_origin_meta: pd.DataFrame,
    daily_origin_airline_meta: pd.DataFrame,
    hub_origin_meta: pd.DataFrame,
    hub_dest_meta: pd.DataFrame,
    origin_weather_meta: pd.DataFrame,
    dest_weather_meta: pd.DataFrame,
    defaults: dict,
    delay_model,
    current_airline: str,
    airport_to_state: dict,
    max_suggestions: int = 3,
):
    """Suggest alternative airlines on the same route with lower predicted delay risk."""
    mask_route_all = (route_meta["Origin"] == origin) & (route_meta["Dest"] == dest)
    candidates = route_meta.loc[mask_route_all].copy()

    if candidates.empty:
        return []

    # Only airlines with real route history and different from the current one
    candidates = candidates[candidates["RouteFlights"] > 0]
    candidates = candidates[candidates["Reporting_Airline"] != current_airline]

    if candidates.empty:
        return []

    suggestions = []
    for _, row in candidates.iterrows():
        code = row["Reporting_Airline"]
        route_flights = float(row["RouteFlights"])
        X_row = build_feature_row(
            travel_date,
            dep_hour,
            code,
            origin,
            dest,
            distance,
            route_meta,
            airline_meta,
            slot_meta,
            daily_origin_meta,
            daily_origin_airline_meta,
            hub_origin_meta,
            hub_dest_meta,
            origin_weather_meta,
            dest_weather_meta,
            defaults,
            airport_to_state,
        )
        proba_delay = float(delay_model.predict_proba(X_row)[0, 1])
        suggestions.append((code, proba_delay, route_flights))

    suggestions.sort(key=lambda t: t[1])
    return suggestions[:max_suggestions]


def suggest_alternative_airports(
    origin: str,
    dest: str,
    airport_to_state: dict,
    route_pair_meta: pd.DataFrame,
    max_suggestions_per_side: int = 3,
    min_flights_per_route: int = 200,
    alpha: float = ALPHA_CANCEL_WEIGHT,
):
    """
    Suggest nearby origin/destination airports (same state) that have
    stronger direct history between origin/dest states.

    Returns (alt_origins, alt_dests) as lists of dicts with keys:
        airport, flights, delay, cancel
    """
    alt_origins = []
    alt_dests = []

    orig_state = airport_to_state.get(origin)
    dest_state = airport_to_state.get(dest)

    # Alternative origins in the same state, flying to the same destination
    if orig_state is not None:
        state_origins = [
            a for a, s in airport_to_state.items() if s == orig_state and a != origin
        ]
        if state_origins:
            rows = route_pair_meta[
                (route_pair_meta["Origin"].isin(state_origins))
                & (route_pair_meta["Dest"] == dest)
            ].copy()
            rows = rows[rows["Flights"] >= min_flights_per_route]
            if not rows.empty:
                rows["CombinedRisk"] = rows["DelayRate"] + alpha * rows["CancelRate"]
                rows = rows.sort_values(
                    ["CombinedRisk", "Flights"], ascending=[True, False]
                ).head(max_suggestions_per_side)
                for _, r in rows.iterrows():
                    alt_origins.append(
                        {
                            "airport": r["Origin"],
                            "flights": int(r["Flights"]),
                            "delay": float(r["DelayRate"]),
                            "cancel": float(r["CancelRate"]),
                        }
                    )

    # Alternative destinations in the same state, from the same origin
    if dest_state is not None:
        state_dests = [
            a for a, s in airport_to_state.items() if s == dest_state and a != dest
        ]
        if state_dests:
            rows = route_pair_meta[
                (route_pair_meta["Origin"] == origin)
                & (route_pair_meta["Dest"].isin(state_dests))
            ].copy()
            rows = rows[rows["Flights"] >= min_flights_per_route]
            if not rows.empty:
                rows["CombinedRisk"] = rows["DelayRate"] + alpha * rows["CancelRate"]
                rows = rows.sort_values(
                    ["CombinedRisk", "Flights"], ascending=[True, False]
                ).head(max_suggestions_per_side)
                for _, r in rows.iterrows():
                    alt_dests.append(
                        {
                            "airport": r["Dest"],
                            "flights": int(r["Flights"]),
                            "delay": float(r["DelayRate"]),
                            "cancel": float(r["CancelRate"]),
                        }
                    )

    return alt_origins, alt_dests


def suggest_one_stop_paths(
    origin: str,
    dest: str,
    route_pair_meta: pd.DataFrame,
    max_paths: int = 3,
    min_leg_flights: int = MIN_LEG_FLIGHTS_FOR_ONE_STOP,
    alpha: float = ALPHA_CANCEL_WEIGHT,
):
    """
    Suggest one‑stop connections origin -> hub -> dest based on volume + reliability.

    Returns a list of dicts:
        origin, hub, dest, flights_out, flights_in, combined_delay, combined_cancel
    """
    from_routes = route_pair_meta[route_pair_meta["Origin"] == origin].copy()
    to_routes = route_pair_meta[route_pair_meta["Dest"] == dest].copy()

    if from_routes.empty or to_routes.empty:
        return []

    from_routes = from_routes.rename(
        columns={
            "Dest": "Hub",
            "Flights": "Flights_out",
            "DelayRate": "DelayRate_out",
            "CancelRate": "CancelRate_out",
            "Distance": "Distance_out",
        }
    )

    to_routes = to_routes.rename(
        columns={
            "Origin": "Hub",
            "Flights": "Flights_in",
            "DelayRate": "DelayRate_in",
            "CancelRate": "CancelRate_in",
            "Distance": "Distance_in",
        }
    )

    transit = from_routes.merge(to_routes, on="Hub", how="inner")
    if transit.empty:
        return []

    transit["MinFlights"] = transit[["Flights_out", "Flights_in"]].min(axis=1)
    transit = transit[transit["MinFlights"] >= min_leg_flights]
    if transit.empty:
        return []

    transit["CombinedDelay"] = transit["DelayRate_out"] + transit["DelayRate_in"]
    transit["CombinedCancel"] = transit["CancelRate_out"] + transit["CancelRate_in"]
    transit["TotalDistance"] = transit["Distance_out"] + transit["Distance_in"]
    transit["RiskScore"] = transit["CombinedDelay"] + alpha * transit["CombinedCancel"]

    transit = transit.sort_values(
        ["RiskScore", "TotalDistance", "MinFlights"],
        ascending=[True, True, False],
    ).head(max_paths)

    suggestions = []
    for _, r in transit.iterrows():
        suggestions.append(
            {
                "origin": origin,
                "hub": r["Hub"],
                "dest": dest,
                "flights_out": int(r["Flights_out"]),
                "flights_in": int(r["Flights_in"]),
                "combined_delay": float(r["CombinedDelay"]),
                "combined_cancel": float(r["CombinedCancel"]),
            }
        )
    return suggestions


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="FlightDelayAdvisor",
        page_icon="✈️",
        layout="wide",
    )

    st.title("✈️ FlightDelayAdvisor")
    st.write(
        "Estimate the historical **delay** and **cancellation** risk of US flights "
        "using 2010–2024 performance and weather‑aware models. "
        "Predictions apply those patterns to future dates."
    )

    # Onboarding / how-to panel
    with st.expander("ℹ️ How to use FlightDelayAdvisor", expanded=False):
        st.markdown(
            """
            - **Single flight** – enter a specific airline, route, date and departure time to get
              delay and cancellation risk, plus tailored guidance and alternatives.
            - **High‑risk hours** – explore which departure hours are historically riskier or
              safer for a given airport and/or airline.
            - **Explore & compare** – compare airlines on a route, rank airports by
              reliability, and (new) get **state‑level recommendations** for airports.
            - Models are trained on **2010–2024** US BTS data with historical airport weather;
              outputs are **probabilities**, not guarantees or live forecasts.
            """
        )

    with st.spinner("Loading data and models..."):
        (
            df,
            airlines,
            origins,
            dests,
            route_meta,
            airline_meta,
            slot_meta,
            daily_origin_meta,
            daily_origin_airline_meta,
            hub_origin_meta,
            hub_dest_meta,
            route_pair_meta,
            route_distance_meta,
            origin_weather_meta,
            dest_weather_meta,
            defaults,
        ) = load_metadata()
        delay_model, cancel_model = load_models()

    # --- derive state information for the state-based tab & features ---
    origin_state_col = pick_first_existing(
        df.columns,
        [
            "OriginState",
            "ORIGIN_STATE_ABR",
            "Origin_State",
            "OriginStateAbbr",
            "OriginStateName",
            "ORIGIN_STATE_NM",
        ],
    )
    dest_state_col = pick_first_existing(
        df.columns,
        [
            "DestState",
            "DEST_STATE_ABR",
            "Dest_State",
            "DestStateAbbr",
            "DestStateName",
            "DEST_STATE_NM",
        ],
    )

    airport_to_state = {}
    states = []
    if origin_state_col is not None:
        origin_state_map = (
            df.groupby("Origin")[origin_state_col]
            .agg(lambda x: x.mode().iat[0])
            .dropna()
        )
        airport_to_state.update(origin_state_map.to_dict())
        states.extend(origin_state_map.unique().tolist())
    if dest_state_col is not None:
        dest_state_map = (
            df.groupby("Dest")[dest_state_col]
            .agg(lambda x: x.mode().iat[0])
            .dropna()
        )
        for k, v in dest_state_map.to_dict().items():
            airport_to_state.setdefault(k, v)
        states.extend(dest_state_map.unique().tolist())
    states = sorted(set(states))

    tab_single, tab_whatif, tab_explore = st.tabs(
        ["🔮 Single flight", "🕒 High‑risk hours", "📊 Explore & compare"]
    )

    # --------------------------------------------------------------
    # TAB 1: Single flight prediction
    # --------------------------------------------------------------
    with tab_single:
        st.subheader("Single flight risk estimate")
        st.caption(
            "Start here with a concrete flight idea. Enter airline, route, date and time "
            "to see predicted risk and model‑based suggestions for safer options."
        )

        col_left, col_right = st.columns([3, 2])

        # ----------------- left: compact input form ----------------
        with col_left:
            st.markdown("#### Flight details")

            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                airline = st.selectbox("Airline", airlines)
            with row1_col2:
                # Only allow dates from 2025 onwards
                today = date(2025, 6, 1)
                travel_date = st.date_input(
                    "Flight date",
                    value=today,
                    min_value=date(2025, 1, 1),
                    max_value=date(2025, 12, 31),
                )

            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                origin = st.selectbox("Origin airport", origins)
            with row2_col2:
                dest = st.selectbox("Destination airport", dests)

            # Departure hour and round-trip setup
            round_trip_enabled = False
            return_date = None
            return_dep_hour = None

            row3_col1, _ = st.columns(2)
            with row3_col1:
                dep_hour = st.slider(
                    "Scheduled departure hour (local time)", 0, 23, 9
                )

            st.markdown("#### Optional: round‑trip details")
            round_trip_enabled = st.checkbox(
                "Estimate round‑trip risk for this itinerary", value=False
            )
            if round_trip_enabled:
                rt_col1, rt_col2 = st.columns(2)
                default_return_date = min(
                    date(2025, 12, 31), travel_date + timedelta(days=7)
                )
                with rt_col1:
                    return_date = st.date_input(
                        "Return flight date",
                        value=default_return_date,
                        min_value=travel_date,
                        max_value=date(2025, 12, 31),
                        key="return_date_single",
                    )
                with rt_col2:
                    return_dep_hour = st.slider(
                        "Return departure hour (local time)",
                        0,
                        23,
                        min((dep_hour + 12) % 24, 23),
                        key="return_dep_hour_single",
                    )

        # ----------------- right: predictions + guidance ------------
        with col_right:
            st.markdown("#### Risk estimates")

            predict_clicked = st.button("Predict risk", type="primary")

            if predict_clicked:
                # Guard against impossible routes
                if origin == dest:
                    st.error(
                        "Origin and destination airport can’t be the same. "
                        "Please choose a different origin airport or destination airport."
                    )
                else:
                    # Remember last searched configuration for cross‑tab linking
                    st.session_state["last_origin"] = origin
                    st.session_state["last_dest"] = dest
                    st.session_state["last_airline"] = airline
                    st.session_state["cmp_origin"] = origin
                    st.session_state["cmp_dest"] = dest

                    # Infer typical distance for this route (not shown to user here)
                    distance = get_route_distance_default(
                        origin, dest, route_distance_meta, defaults
                    )

                    X_input = build_feature_row(
                        travel_date,
                        dep_hour,
                        airline,
                        origin,
                        dest,
                        distance,
                        route_meta,
                        airline_meta,
                        slot_meta,
                        daily_origin_meta,
                        daily_origin_airline_meta,
                        hub_origin_meta,
                        hub_dest_meta,
                        origin_weather_meta,
                        dest_weather_meta,
                        defaults,
                        airport_to_state,
                    )

                    proba_delay = float(delay_model.predict_proba(X_input)[0, 1])
                    proba_cancel = float(cancel_model.predict_proba(X_input)[0, 1])

                    delay_label = risk_label(proba_delay)
                    cancel_label = risk_label(proba_cancel)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Delay ≥ 15 min probability",
                            f"{proba_delay * 100:.1f}%",
                        )
                        st.write(
                            f"Risk level: {risk_emoji(delay_label)} **{delay_label}**"
                        )
                    with col_b:
                        st.metric(
                            "Cancellation probability",
                            f"{proba_cancel * 100:.4f}%",
                        )
                        st.write(
                            f"Risk level: {risk_emoji(cancel_label)} **{cancel_label}**"
                        )

                    # Extra context vs network average cancellation rate
                    cancel_delta = proba_cancel - defaults["global_cancel_rate"]
                    if abs(cancel_delta) < 0.001:
                        cancel_delta_text = "very close to the network average"
                    elif cancel_delta > 0:
                        cancel_delta_text = (
                            f"about {cancel_delta * 100:.2f} percentage points above "
                            "the network average"
                        )
                    else:
                        cancel_delta_text = (
                            f"about {abs(cancel_delta) * 100:.2f} percentage points below "
                            "the network average"
                        )

                    st.caption(
                        f"Across all flights in 2010–2024, the average cancellation probability is "
                        f"{defaults['global_cancel_rate'] * 100:.2f}%. "
                        f"This itinerary is {cancel_delta_text}."
                    )

                    # Round‑trip metrics, if requested
                    if (
                        round_trip_enabled
                        and return_date is not None
                        and return_dep_hour is not None
                    ):
                        distance_rt = get_route_distance_default(
                            dest, origin, route_distance_meta, defaults
                        )
                        X_return = build_feature_row(
                            return_date,
                            return_dep_hour,
                            airline,
                            dest,
                            origin,
                            distance_rt,
                            route_meta,
                            airline_meta,
                            slot_meta,
                            daily_origin_meta,
                            daily_origin_airline_meta,
                            hub_origin_meta,
                            hub_dest_meta,
                            origin_weather_meta,
                            dest_weather_meta,
                            defaults,
                            airport_to_state,
                        )
                        proba_delay_rt = float(
                            delay_model.predict_proba(X_return)[0, 1]
                        )
                        proba_cancel_rt = float(
                            cancel_model.predict_proba(X_return)[0, 1]
                        )

                        round_delay = 1 - (1 - proba_delay) * (1 - proba_delay_rt)
                        round_cancel = 1 - (1 - proba_cancel) * (1 - proba_cancel_rt)

                        st.markdown("##### Round‑trip disruption risk")
                        rt_c1, rt_c2 = st.columns(2)
                        with rt_c1:
                            st.metric(
                                "At least one delayed flight ≥ 15 min",
                                f"{round_delay * 100:.1f}%",
                            )
                        with rt_c2:
                            st.metric(
                                "At least one cancellation",
                                f"{round_cancel * 100:.2f}%",
                            )
                        st.caption(
                            "Assumes the two legs are independent and use the same airline in both directions."
                        )

                    # Explanation of risk factors
                    reasons = explain_risk_factors(
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
                    if reasons:
                        st.markdown("##### Why this risk level (historical factors)")
                        for r in reasons:
                            st.write(f"- {r}")
                    else:
                        st.markdown("##### Why this risk level")
                        st.write(
                            "This itinerary looks fairly typical relative to similar flights "
                            "in 2010–2024; no strong risk factors stand out."
                        )

                    # Additional context about how well we know this carrier+route
                    route_delay_rate, route_cancel_rate, route_flights = get_route_stats(
                        airline, origin, dest, route_meta, defaults
                    )

                    # Other airlines that operate this origin–dest route
                    mask_route_all = (
                        (route_meta["Origin"] == origin)
                        & (route_meta["Dest"] == dest)
                    )
                    other_carriers = route_meta.loc[mask_route_all].copy()
                    if not other_carriers.empty:
                        other_carriers = other_carriers[
                            other_carriers["Reporting_Airline"] != airline
                        ]
                    rich_other_carriers = other_carriers[
                        other_carriers["RouteFlights"] >= MIN_ROUTE_FLIGHTS_FOR_WARNING
                    ]

                    if route_flights < MIN_ROUTE_FLIGHTS_FOR_WARNING:
                        # We have little or no direct history for this airline on this route.
                        if route_flights == 0:
                            freq_text = "has not operated this non‑stop route regularly"
                        else:
                            freq_text = (
                                f"operates this route only occasionally "
                                f"(≈{int(route_flights)} flights in 2010–2024)"
                            )

                        st.info(
                            f"{airline} airline {freq_text}. "
                            "The estimate leans more on the airline’s broader network, "
                            "route peers and typical weather than on rich, route‑specific data."
                        )

                        if not rich_other_carriers.empty:
                            rich_other_carriers["CombinedRisk"] = (
                                rich_other_carriers["RouteDelayRate"]
                                + ALPHA_CANCEL_WEIGHT
                                * rich_other_carriers["RouteCancelRate"]
                            )
                            rich_other_carriers = rich_other_carriers.sort_values(
                                "CombinedRisk", ascending=True
                            )
                            airlines_for_route = rich_other_carriers[
                                "Reporting_Airline"
                            ].tolist()
                            better_airlines = ", ".join(
                                f"{code} airline" for code in airlines_for_route[:5]
                            )

                            st.info(
                                f"For **{origin} airport → {dest} airport**, the model has much richer "
                                "history with these airlines (ordered by lower combined delay "
                                f"and cancellation risk): {better_airlines}. "
                                "If you can switch carrier while keeping the same airports, "
                                "these are usually the most data‑backed options. "
                                "You can explore them in detail in the **“Explore & compare → Airlines”** tab."
                            )
                        else:
                            # No strong alternative carriers – look for alternative airports / hubs
                            alt_origins, alt_dests = suggest_alternative_airports(
                                origin,
                                dest,
                                airport_to_state,
                                route_pair_meta,
                            )
                            one_stop_paths = suggest_one_stop_paths(
                                origin,
                                dest,
                                route_pair_meta,
                            )

                            msg_parts = [
                                f"There haven’t been many non‑stop flights between "
                                f"**{origin} airport** and **{dest} airport** in the 2010–2024 data, "
                                "so the model has limited route‑specific history here."
                            ]

                            if alt_origins:
                                msg_parts.append(
                                    "Within your departure state, travellers more often depart from "
                                    + ", ".join(
                                        f"**{item['airport']} airport**"
                                        for item in alt_origins
                                    )
                                    + f" when flying to **{dest} airport**."
                                )

                            if alt_dests:
                                msg_parts.append(
                                    "Within the destination state, popular arrival airports from "
                                    f"**{origin} airport** include "
                                    + ", ".join(
                                        f"**{item['airport']} airport**"
                                        for item in alt_dests
                                    )
                                    + "."
                                )

                            if one_stop_paths:
                                msg_parts.append(
                                    "Most real‑world itineraries between these cities are flown as a "
                                    "one‑stop via a hub airport."
                                )

                            st.info(" ".join(msg_parts))

                            if alt_origins:
                                st.markdown(
                                    "**Nearby departure airports with stronger direct history to your destination:**"
                                )
                                for item in alt_origins:
                                    st.write(
                                        f"- **{item['airport']} airport → {dest} airport** – delay ≈ "
                                        f"{item['delay'] * 100:.1f}%, cancellation ≈ "
                                        f"{item['cancel'] * 100:.2f}% "
                                        f"(n≈{item['flights']:,} flights)"
                                    )

                            if alt_dests:
                                st.markdown(
                                    "**Nearby arrival airports that are well‑served from your origin:**"
                                )
                                for item in alt_dests:
                                    st.write(
                                        f"- **{origin} airport → {item['airport']} airport** – delay ≈ "
                                        f"{item['delay'] * 100:.1f}%, cancellation ≈ "
                                        f"{item['cancel'] * 100:.2f}% "
                                        f"(n≈{item['flights']:,} flights)"
                                    )

                            if one_stop_paths:
                                st.markdown(
                                    "**One‑stop patterns that historically work well between these airports:**"
                                )
                                for p in one_stop_paths:
                                    st.write(
                                        f"- **{origin} airport → {p['hub']} airport → {dest} airport** – "
                                        f"combined delay ≈ {p['combined_delay'] * 100:.1f}%, "
                                        f"combined cancellation ≈ {p['combined_cancel'] * 100:.2f}% "
                                        f"(each leg with at least {MIN_LEG_FLIGHTS_FOR_ONE_STOP} historical flights)"
                                    )
                                st.caption(
                                    "For these hub combinations, use the "
                                    "“Explore & compare → Airlines” tab to pick specific "
                                    "carriers on each leg."
                                )

                    # Model‑based alternatives (safer hours / airlines)
                    safer_hours = suggest_safer_hours(
                        travel_date,
                        airline,
                        origin,
                        dest,
                        distance,
                        route_meta,
                        airline_meta,
                        slot_meta,
                        daily_origin_meta,
                        daily_origin_airline_meta,
                        hub_origin_meta,
                        hub_dest_meta,
                        origin_weather_meta,
                        dest_weather_meta,
                        defaults,
                        delay_model,
                        dep_hour,
                        airport_to_state,
                    )
                    safer_airlines = suggest_safer_airlines(
                        travel_date,
                        dep_hour,
                        origin,
                        dest,
                        distance,
                        route_meta,
                        airline_meta,
                        slot_meta,
                        daily_origin_meta,
                        daily_origin_airline_meta,
                        hub_origin_meta,
                        hub_dest_meta,
                        origin_weather_meta,
                        dest_weather_meta,
                        defaults,
                        delay_model,
                        airline,
                        airport_to_state,
                    )

                    if safer_hours or safer_airlines:
                        st.markdown("##### Model‑based alternatives")

                    if safer_hours:
                        st.write(
                            "**Potentially safer departure times (same airline & route):**"
                        )
                        lines = [
                            f"- {h:02d}:00 – predicted delay risk ≈ {p * 100:.1f}%"
                            for (h, p) in safer_hours
                        ]
                        st.markdown("\n".join(lines))

                    if safer_airlines:
                        st.write(
                            f"**Airlines with lower predicted delay risk on "
                            f"{origin} airport → {dest} airport at {dep_hour:02d}:00:**"
                        )
                        lines = [
                            f"- {code} airline – delay risk ≈ {p * 100:.1f}% "
                            f"(based on ≈{int(rf):,} historical flights)"
                            for (code, p, rf) in safer_airlines
                        ]
                        st.markdown("\n".join(lines))

            else:
                st.info('Set the flight details on the left and click **"Predict risk"**.')

            st.caption(
                "Probabilities are based on long‑term historical patterns and "
                "typical monthly weather, **not** real‑time conditions."
            )

    # --------------------------------------------------------------
    # TAB 2: High‑risk hours (historical)
    # --------------------------------------------------------------
    with tab_whatif:
        st.subheader("High‑risk hours by departure time")
        st.caption(
            "Use this view to see which departure hours are historically risky for "
            "a given airport and/or airline, then feed safer hours back into the "
            "Single flight tab."
        )

        # Defaults in session state for presets
        if "hr_airport" not in st.session_state:
            st.session_state["hr_airport"] = "All airports"
        if "hr_airline" not in st.session_state:
            st.session_state["hr_airline"] = "All airlines"
        if "hr_month" not in st.session_state:
            st.session_state["hr_month"] = "All months"

        preset_col1, preset_col2, preset_col3 = st.columns(3)
        with preset_col1:
            if st.button("Global view (all airports & airlines)", key="preset_global"):
                st.session_state["hr_airport"] = "All airports"
                st.session_state["hr_airline"] = "All airlines"
                st.session_state["hr_month"] = "All months"
        with preset_col2:
            if st.button("Use my last origin airport", key="preset_origin"):
                if "last_origin" in st.session_state:
                    st.session_state["hr_airport"] = st.session_state["last_origin"]
                st.session_state["hr_airline"] = "All airlines"
        with preset_col3:
            if st.button("Use my last airline", key="preset_airline"):
                if "last_airline" in st.session_state:
                    st.session_state["hr_airline"] = st.session_state["last_airline"]
                st.session_state["hr_airport"] = "All airports"

        sel_col1, sel_col2, sel_col3 = st.columns(3)
        with sel_col1:
            airport_scope = st.selectbox(
                "Origin airport (optional)",
                ["All airports"] + origins,
                key="hr_airport",
            )
        with sel_col2:
            airline_scope = st.selectbox(
                "Airline (optional)",
                ["All airlines"] + airlines,
                key="hr_airline",
            )
        with sel_col3:
            month_options = ["All months"] + list(range(1, 13))
            month_scope = st.selectbox(
                "Month (optional)",
                month_options,
                index=0,
                format_func=lambda x: "All months"
                if x == "All months"
                else f"{int(x):02d}",
                key="hr_month",
            )

        with st.expander("Advanced options for stability", expanded=False):
            min_flights_hour = st.slider(
                "Minimum flights per departure hour (for stable estimates)",
                min_value=50,
                max_value=5000,
                value=200,
                step=50,
            )

        if st.button("Show high‑risk hours", type="primary"):
            # Base: flights where we know delay outcome
            mask_hr = df["Cancelled"].isin([0, 1]) & df["ArrDel15"].notna()

            if airport_scope != "All airports":
                mask_hr &= df["Origin"] == airport_scope
            if airline_scope != "All airlines":
                mask_hr &= df["Reporting_Airline"] == airline_scope
            if month_scope != "All months":
                mask_hr &= df["Month"] == int(month_scope)

            df_hr = df[mask_hr]
            total_flights = len(df_hr)

            if total_flights < min_flights_hour * 2:
                st.info(
                    f"This filter combination covers **{total_flights:,}** flights. "
                    "Hourly patterns would be quite noisy at that level. "
                    "For clearer guidance, try leaving airline or month as **All** "
                    "and/or reduce the minimum‑flights threshold."
                )
            else:
                hour_stats = (
                    df_hr.groupby("DepHour")
                    .agg(
                        Flights=("Cancelled", "size"),
                        DelayRate=("ArrDel15", "mean"),
                        CancelRate=("Cancelled", "mean"),
                    )
                    .reset_index()
                )

                hour_stats = hour_stats[hour_stats["Flights"] >= min_flights_hour]

                if hour_stats.empty:
                    st.info(
                        "With the current minimum‑flights setting, no hour reaches that "
                        "volume. Lower the threshold above to reveal more hours."
                    )
                else:
                    hour_stats["DelayRatePct"] = hour_stats["DelayRate"] * 100.0
                    hour_stats["CancelRatePct"] = hour_stats["CancelRate"] * 100.0

                    # Context sentence
                    context_parts = []
                    if airport_scope != "All airports":
                        context_parts.append(f"departing **{airport_scope} airport**")
                    if airline_scope != "All airlines":
                        context_parts.append(f"on **{airline_scope} airline**")
                    if month_scope != "All months":
                        context_parts.append(f"in month **{int(month_scope):02d}**")

                    if context_parts:
                        context_str = " for " + ", ".join(context_parts)
                    else:
                        context_str = " across **all airports and airlines**"

                    st.markdown(
                        f"Based on **{total_flights:,}** flights{context_str} in 2010–2024."
                    )

                    plot_df = hour_stats.set_index("DepHour")[
                        ["DelayRatePct", "CancelRatePct"]
                    ]
                    st.line_chart(plot_df, height=320)
                    st.caption(
                        "Lines show empirical delay and cancellation rates by scheduled "
                        "departure hour. Higher values indicate riskier hours."
                    )

                    # Hour × month heatmap when all months are included
                    if month_scope == "All months":
                        month_hour_stats = (
                            df_hr.groupby(["Month", "DepHour"])
                            .agg(
                                Flights=("Cancelled", "size"),
                                DelayRate=("ArrDel15", "mean"),
                            )
                            .reset_index()
                        )
                        month_hour_stats = month_hour_stats[
                            month_hour_stats["Flights"] >= min_flights_hour
                        ]
                        if not month_hour_stats.empty:
                            month_hour_stats["DelayRatePct"] = (
                                month_hour_stats["DelayRate"] * 100.0
                            )
                            st.markdown("##### Delay risk by hour and month")
                            heatmap = (
                                alt.Chart(month_hour_stats)
                                .mark_rect()
                                .encode(
                                    x=alt.X("DepHour:O", title="Departure hour"),
                                    y=alt.Y("Month:O", title="Month"),
                                    color=alt.Color(
                                        "DelayRatePct:Q", title="Delay rate (%)"
                                    ),
                                    tooltip=[
                                        alt.Tooltip("Month:O", title="Month"),
                                        alt.Tooltip("DepHour:O", title="Hour"),
                                        alt.Tooltip("Flights:Q", format=","),
                                        alt.Tooltip(
                                            "DelayRatePct:Q",
                                            title="Delay rate (%)",
                                            format=".1f",
                                        ),
                                    ],
                                )
                            )
                            st.altair_chart(heatmap, use_container_width=True)
                            st.caption(
                                "Darker tiles indicate hours and months with higher historical delay rates."
                            )

                    # Volume context
                    st.markdown("###### Flight volume by hour (for context)")
                    st.bar_chart(
                        hour_stats.set_index("DepHour")[["Flights"]], height=180
                    )

                    # Identify best / worst hours
                    worst_delay = hour_stats.sort_values(
                        "DelayRatePct", ascending=False
                    ).head(3)
                    best_delay = hour_stats.sort_values(
                        "DelayRatePct", ascending=True
                    ).head(3)
                    worst_cancel = hour_stats.sort_values(
                        "CancelRatePct", ascending=False
                    ).head(3)

                    best_delay_hours = [
                        int(h) for h in best_delay["DepHour"].tolist()
                    ]
                    worst_delay_hours = [
                        int(h) for h in worst_delay["DepHour"].tolist()
                    ]

                    if best_delay_hours:
                        safe_str = ", ".join(f"{h:02d}:00" for h in best_delay_hours)
                        st.success(
                            f"Safer departure hours based on delay risk: **{safe_str}**."
                        )
                    if worst_delay_hours:
                        risky_str = ", ".join(f"{h:02d}:00" for h in worst_delay_hours)
                        st.warning(
                            f"Hours to avoid if possible due to higher delay risk: **{risky_str}**."
                        )

                    col_hd, col_hc = st.columns(2)
                    with col_hd:
                        st.markdown("##### Hours with highest **delay** risk")
                        for _, row in worst_delay.iterrows():
                            st.write(
                                f"**{int(row['DepHour']):02d}:00** – delay rate ≈ "
                                f"{row['DelayRatePct']:.1f}% (n={int(row['Flights'])})"
                            )

                        st.markdown("##### Hours with **lower** delay risk")
                        for _, row in best_delay.iterrows():
                            st.write(
                                f"**{int(row['DepHour']):02d}:00** – delay rate ≈ "
                                f"{row['DelayRatePct']:.1f}% (n={int(row['Flights'])})"
                            )

                    with col_hc:
                        st.markdown("##### Hours with highest **cancellation** risk")
                        for _, row in worst_cancel.iterrows():
                            st.write(
                                f"**{int(row['DepHour']):02d}:00** – cancellation rate ≈ "
                                f"{row['CancelRatePct']:.2f}% (n={int(row['Flights'])})"
                            )

                    st.info(
                        "Use this view together with the *Single flight* and *Explore & compare* "
                        "tabs: first identify risky hours here, then plug a safer hour into the "
                        "single‑flight model for a personalised risk estimate."
                    )
        else:
            st.info(
                'Choose an airport and/or airline (or leave them as "All") and click '
                '**"Show high‑risk hours"** to see which departure times are safest.'
            )

    # --------------------------------------------------------------
    # TAB 3: Explore & compare (airlines / airports / states)
    # --------------------------------------------------------------
    with tab_explore:
        st.subheader("Explore and compare airlines and airports")
        st.caption(
            "Use this section to compare airlines and airports based on historical "
            "delay and cancellation rates, and to discover strong options by state."
        )

        subtab_airlines, subtab_airports, subtab_states = st.tabs(
            ["✈️ Airlines", "🛫 Airports", "🗺️ States"]
        )

        # -------------------- Airline comparison -------------------
        with subtab_airlines:
            st.markdown("#### Compare airlines on a specific route")

            cmp_col1, cmp_col2, cmp_col3 = st.columns(3)
            with cmp_col1:
                origin_cmp = st.selectbox(
                    "Origin airport", origins, key="cmp_origin"
                )
            with cmp_col2:
                dest_cmp = st.selectbox("Destination airport", dests, key="cmp_dest")
            with cmp_col3:
                month_options_cmp = ["All months"] + list(range(1, 13))
                month_choice = st.selectbox(
                    "Month",
                    month_options_cmp,
                    index=0,
                    format_func=lambda x: (
                        "All months"
                        if x == "All months"
                        else f"{int(x):02d}"
                    ),
                    key="cmp_month",
                )

            # Indicate if we are looking at last searched route
            if "last_origin" in st.session_state and "last_dest" in st.session_state:
                last_route = (
                    f"{st.session_state['last_origin']} airport → "
                    f"{st.session_state['last_dest']} airport"
                )
                current_route = f"{origin_cmp} airport → {dest_cmp} airport"
                if last_route == current_route:
                    st.caption(
                        f"Comparing airlines for your last searched route: **{current_route}**."
                    )
                else:
                    st.caption(
                        f"Your last searched route was **{last_route}**. "
                        f"Adjust origin/destination above to align the comparison if needed."
                    )

            min_flights_airline = st.slider(
                "Minimum flights per airline to include",
                min_value=50,
                max_value=2000,
                value=MIN_FLIGHTS_AIRLINE_COMPARE,
                step=50,
            )

            # Optional airline filter (multi-select)
            airline_filter = st.multiselect(
                "Limit to these airlines (optional)",
                airlines,
                default=[],
                help=(
                    "If left empty, all airlines on this route that meet the flights "
                    "threshold are shown."
                ),
            )

            if origin_cmp == dest_cmp:
                st.warning(
                    "Origin and destination airport can’t be the same. "
                    "Choose different airports to compare airlines on a real route."
                )
            else:
                mask_cmp = (df["Origin"] == origin_cmp) & (df["Dest"] == dest_cmp)
                if month_choice != "All months":
                    mask_cmp &= df["Month"] == int(month_choice)

                df_route = df[mask_cmp]

                if len(df_route) < min_flights_airline:
                    st.info(
                        "This route‑and‑month combination has relatively light traffic "
                        "in 2010–2024, so airline differences are harder to visualise. "
                        "Try selecting **All months** or exploring nearby airports in the "
                        "🛫 *Airports* tab to see a broader picture."
                    )
                if len(df_route) == 0:
                    # really no direct flights; still nothing sensible to plot
                    pass
                else:
                    stats = (
                        df_route.groupby("Reporting_Airline")
                        .agg(
                            Flights=("Cancelled", "size"),
                            DelayRate=("ArrDel15", "mean"),
                            CancelRate=("Cancelled", "mean"),
                        )
                        .reset_index()
                    )

                    # Keep airlines with enough traffic if possible; otherwise show all
                    stats_filtered = stats[stats["Flights"] >= min_flights_airline]
                    if stats_filtered.empty:
                        stats_to_show = stats.sort_values(
                            "Flights", ascending=False
                        ).head(10)
                        st.info(
                            "No carrier reaches the current minimum‑flights threshold on "
                            "this route, so we show the busiest airlines instead. "
                            "You can still lower the threshold above to include more."
                        )
                    else:
                        stats_to_show = stats_filtered

                    # Apply optional airline filter
                    if airline_filter:
                        filtered = stats_to_show[
                            stats_to_show["Reporting_Airline"].isin(airline_filter)
                        ]
                        if filtered.empty:
                            st.warning(
                                "None of the selected airlines meet the minimum‑flights "
                                "threshold on this route. Try lowering the threshold or "
                                "clearing the airline filter."
                            )
                            stats_to_show = pd.DataFrame()
                        else:
                            stats_to_show = filtered

                    if not stats_to_show.empty:
                        stats_to_show["DelayRatePct"] = (
                            stats_to_show["DelayRate"] * 100.0
                        )
                        stats_to_show["CancelRatePct"] = (
                            stats_to_show["CancelRate"] * 100.0
                        )

                        st.markdown(
                            f"Historical performance on **{origin_cmp} airport → {dest_cmp} airport** "
                            f"{'(all months)' if month_choice=='All months' else f'(month {int(month_choice):02d})'}."
                        )

                        # Route summary card
                        overall_mask = (
                            df_route["ArrDel15"].notna()
                            & (df_route["Cancelled"] == 0)
                        )
                        overall_delay = (
                            float(df_route.loc[overall_mask, "ArrDel15"].mean())
                            if overall_mask.any()
                            else np.nan
                        )
                        overall_cancel = float(df_route["Cancelled"].mean())
                        best_delay_row = stats_to_show.loc[
                            stats_to_show["DelayRate"].idxmin()
                        ]
                        best_cancel_row = stats_to_show.loc[
                            stats_to_show["CancelRate"].idxmin()
                        ]

                        sum_col1, sum_col2 = st.columns(2)
                        with sum_col1:
                            if not np.isnan(overall_delay):
                                st.metric(
                                    "Route‑wide delay rate",
                                    f"{overall_delay * 100:.1f}%",
                                )
                            st.write(
                                f"Best airline for delay: **{best_delay_row['Reporting_Airline']} airline** "
                                f"(≈ {best_delay_row['DelayRatePct']:.1f}% delays)"
                            )
                        with sum_col2:
                            st.metric(
                                "Route‑wide cancellation rate",
                                f"{overall_cancel * 100:.2f}%",
                            )
                            st.write(
                                f"Best airline for cancellations: "
                                f"**{best_cancel_row['Reporting_Airline']} airline** "
                                f"(≈ {best_cancel_row['CancelRatePct']:.2f}% cancellations)"
                            )

                        # Delay rate bar chart
                        delay_chart = (
                            alt.Chart(stats_to_show)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "Reporting_Airline:N",
                                    title="Airline",
                                    sort="-y",
                                ),
                                y=alt.Y(
                                    "DelayRatePct:Q", title="Delay rate (%)"
                                ),
                                tooltip=[
                                    "Reporting_Airline",
                                    "Flights",
                                    alt.Tooltip("DelayRatePct:Q", format=".1f"),
                                    alt.Tooltip("CancelRatePct:Q", format=".2f"),
                                ],
                            )
                        )
                        st.altair_chart(delay_chart, use_container_width=True)

                        # Cancellation rate bar chart
                        cancel_chart = (
                            alt.Chart(stats_to_show)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "Reporting_Airline:N",
                                    title="Airline",
                                    sort="-y",
                                ),
                                y=alt.Y(
                                    "CancelRatePct:Q", title="Cancellation rate (%)"
                                ),
                                tooltip=[
                                    "Reporting_Airline",
                                    "Flights",
                                    alt.Tooltip("DelayRatePct:Q", format=".1f"),
                                    alt.Tooltip("CancelRatePct:Q", format=".2f"),
                                ],
                            )
                        )
                        st.altair_chart(cancel_chart, use_container_width=True)

                        # Delay vs cancellation scatter
                        scatter = (
                            alt.Chart(stats_to_show)
                            .mark_circle(size=80)
                            .encode(
                                x=alt.X(
                                    "DelayRatePct:Q", title="Delay rate (%)"
                                ),
                                y=alt.Y(
                                    "CancelRatePct:Q",
                                    title="Cancellation rate (%)",
                                ),
                                size=alt.Size(
                                    "Flights:Q",
                                    title="Number of flights",
                                    scale=alt.Scale(range=[20, 400]),
                                ),
                                tooltip=[
                                    "Reporting_Airline",
                                    "Flights",
                                    alt.Tooltip("DelayRatePct:Q", format=".1f"),
                                    alt.Tooltip("CancelRatePct:Q", format=".2f"),
                                ],
                            )
                        )
                        st.markdown("##### Delay vs cancellation trade‑off")
                        st.altair_chart(scatter, use_container_width=True)

                        st.caption(
                            "Airlines towards the bottom‑left (low delay and low cancellation) "
                            "are typically the most reliable choices on this route."
                        )

        # -------------------- Airport comparison -------------------
        with subtab_airports:
            st.markdown("#### Compare airports")

            role = st.radio(
                "Treat airports as",
                ["Origin", "Destination"],
                horizontal=True,
                key="airport_role",
            )
            metric = st.radio(
                "Metric",
                ["Delay rate", "Cancellation rate"],
                horizontal=True,
                key="airport_metric",
            )

            month_options_a = ["All months"] + list(range(1, 13))
            month_choice_a = st.selectbox(
                "Month",
                month_options_a,
                index=0,
                format_func=lambda x: (
                    "All months" if x == "All months" else f"{int(x):02d}"
                ),
                key="airport_month",
            )

            # Focus airport (e.g., "safest destinations from my origin")
            if role == "Origin":
                focus_airport = st.selectbox(
                    "Focus on departures from this origin (optional)",
                    ["None"] + origins,
                    key="airport_focus_origin",
                )
            else:
                focus_airport = st.selectbox(
                    "Focus on arrivals into this destination (optional)",
                    ["None"] + dests,
                    key="airport_focus_dest",
                )

            with st.expander("Advanced airport ranking options", expanded=False):
                min_flights_airport = st.slider(
                    "Minimum flights per airport to include",
                    min_value=100,
                    max_value=5000,
                    value=MIN_FLIGHTS_AIRPORT_COMPARE,
                    step=100,
                )
                top_n = st.slider(
                    "Number of airports to display (when not filtering explicitly)",
                    min_value=5,
                    max_value=20,
                    value=10,
                    step=1,
                )
                show_mode = st.radio(
                    "Show",
                    ["Worst (highest risk)", "Best (lowest risk)"],
                    horizontal=True,
                    key="airport_show_mode",
                )

            group_col = "Origin" if role == "Origin" else "Dest"

            mask_a = df["Cancelled"].isin([0, 1])
            if month_choice_a != "All months":
                mask_a &= df["Month"] == int(month_choice_a)

            df_a = df[mask_a]

            # Decide grouping depending on focus
            if role == "Origin" and focus_airport != "None":
                df_subset = df_a[df_a["Origin"] == focus_airport]
                group_col_display = "Dest"
            elif role == "Destination" and focus_airport != "None":
                df_subset = df_a[df_a["Dest"] == focus_airport]
                group_col_display = "Origin"
            else:
                df_subset = df_a
                group_col_display = "Origin" if role == "Origin" else "Dest"

            stats_a = (
                df_subset.groupby(group_col_display)
                .agg(
                    Flights=("Cancelled", "size"),
                    DelayRate=("ArrDel15", "mean"),
                    CancelRate=("Cancelled", "mean"),
                )
                .reset_index()
            )

            stats_a = stats_a[stats_a["Flights"] >= min_flights_airport]

            # Optional airport filter (multi-select)
            if not stats_a.empty:
                airport_choices = stats_a[group_col_display].tolist()
                selected_airports = st.multiselect(
                    "Filter to specific airports (optional)",
                    airport_choices,
                    key="airport_filter",
                )
                if selected_airports:
                    stats_a = stats_a[
                        stats_a[group_col_display].isin(selected_airports)
                    ]

            if stats_a.empty:
                st.info(
                    "With the current filters and minimum‑flights setting, no airports "
                    "reach that volume. Reducing the minimum or choosing **All months** "
                    "will surface more airports to compare."
                )
            else:
                stats_a["DelayRatePct"] = stats_a["DelayRate"] * 100.0
                stats_a["CancelRatePct"] = stats_a["CancelRate"] * 100.0

                if metric == "Delay rate":
                    plot_col = "DelayRatePct"
                    y_label = "Delay rate (%)"
                else:
                    plot_col = "CancelRatePct"
                    y_label = "Cancellation rate (%)"

                ascending = show_mode.startswith("Best")

                if "selected_airports" in locals() and selected_airports:
                    stats_sorted = stats_a.sort_values(plot_col, ascending=ascending)
                else:
                    stats_sorted = stats_a.sort_values(
                        plot_col, ascending=ascending
                    ).head(top_n)

                if focus_airport != "None":
                    if role == "Origin":
                        context_label = f"destinations from **{focus_airport} airport**"
                    else:
                        context_label = f"origins into **{focus_airport} airport**"
                else:
                    context_label = f"{role.lower()} airports"

                st.markdown(
                    f"{'Worst' if not ascending else 'Best'} {context_label} "
                    f"by {metric.lower()} "
                    f"{'(all months)' if month_choice_a=='All months' else f'(month {int(month_choice_a):02d})'}."
                )

                chart_df = stats_sorted.set_index(group_col_display)[[plot_col]]

                st.bar_chart(chart_df, height=320)
                st.caption(
                    "Bars are based on historical performance in 2010–2024. "
                    "Use this view to spot particularly risky or reliable airports."
                )

        # -------------------- State-based recommendations -------------------
        with subtab_states:
            st.markdown("#### State‑based airport and airline recommendations")

            if not states or origin_state_col is None or dest_state_col is None:
                st.info(
                    "State information for airports is not available in this build, "
                    "so state‑level recommendations are disabled."
                )
            else:
                s_col1, s_col2, s_col3 = st.columns(3)
                with s_col1:
                    state_from = st.selectbox(
                        "Departure state", states, key="state_from"
                    )
                with s_col2:
                    state_to = st.selectbox(
                        "Destination state", states, key="state_to"
                    )
                with s_col3:
                    month_options_s = ["All months"] + list(range(1, 13))
                    month_choice_s = st.selectbox(
                        "Month (optional)",
                        month_options_s,
                        index=0,
                        format_func=lambda x: "All months"
                        if x == "All months"
                        else f"{int(x):02d}",
                        key="state_month",
                    )

                min_flights_state = st.slider(
                    "Minimum flights per itinerary (for reliable comparisons)",
                    min_value=50,
                    max_value=2000,
                    value=200,
                    step=50,
                )

                if st.button(
                    "Recommend airports and airlines",
                    type="primary",
                    key="state_recommend_btn",
                ):
                    # Direct (non-stop) flights between states
                    mask_state = df["Cancelled"].isin([0, 1]) & df["ArrDel15"].notna()
                    mask_state &= df[origin_state_col] == state_from
                    mask_state &= df[dest_state_col] == state_to
                    if month_choice_s != "All months":
                        mask_state &= df["Month"] == int(month_choice_s)

                    df_state = df[mask_state]

                    direct_group = (
                        df_state.groupby(["Origin", "Dest", "Reporting_Airline"])
                        .agg(
                            Flights=("Cancelled", "size"),
                            DelayRate=("ArrDel15", "mean"),
                            CancelRate=("Cancelled", "mean"),
                        )
                        .reset_index()
                    )

                    direct_group = direct_group[
                        direct_group["Flights"] >= min_flights_state
                    ]

                    if not direct_group.empty:
                        direct_group["DelayRatePct"] = (
                            direct_group["DelayRate"] * 100.0
                        )
                        direct_group["CancelRatePct"] = (
                            direct_group["CancelRate"] * 100.0
                        )
                        direct_group["CombinedRisk"] = (
                            direct_group["DelayRate"]
                            + ALPHA_CANCEL_WEIGHT * direct_group["CancelRate"]
                        )

                        best_direct = direct_group.sort_values(
                            "CombinedRisk", ascending=True
                        ).head(5)
                        top = best_direct.iloc[0]

                        st.success(
                            f"Recommended non‑stop option from **{state_from}** to **{state_to}**:\n\n"
                            f"Fly from **{top['Origin']} airport** to **{top['Dest']} airport** "
                            f"with **{top['Reporting_Airline']} airline**.\n\n"
                            f"Estimated delay risk ≈ {top['DelayRatePct']:.1f}%, "
                            f"cancellation risk ≈ {top['CancelRatePct']:.2f}%, "
                            f"based on about {int(top['Flights']):,} flights in 2010–2024."
                        )

                        if len(best_direct) > 1:
                            st.markdown("##### Other reliable non‑stop options")
                            for _, row in best_direct.iloc[1:].iterrows():
                                st.write(
                                    f"- **{row['Origin']} airport → {row['Dest']} airport** on "
                                    f"**{row['Reporting_Airline']} airline** – delay ≈ "
                                    f"{row['DelayRatePct']:.1f}%, cancellation ≈ "
                                    f"{row['CancelRatePct']:.2f}% "
                                    f"(n≈{int(row['Flights']):,})"
                                )

                        st.caption(
                            "These recommendations are based on state‑to‑state flights "
                            "observed in 2010–2024."
                        )
                    else:
                        # Build one-stop (hub) suggestions
                        orig_airports = [
                            a for a, s in airport_to_state.items() if s == state_from
                        ]
                        dest_airports = [
                            a for a, s in airport_to_state.items() if s == state_to
                        ]

                        routes_simple = (
                            df.groupby(["Origin", "Dest"])
                            .agg(
                                Flights=("Cancelled", "size"),
                                DelayRate=("ArrDel15", "mean"),
                                CancelRate=("Cancelled", "mean"),
                            )
                            .reset_index()
                        )

                        from_routes = routes_simple[
                            routes_simple["Origin"].isin(orig_airports)
                        ].rename(
                            columns={
                                "Origin": "OriginAirport",
                                "Dest": "Hub",
                                "Flights": "Flights_out",
                                "DelayRate": "DelayRate_out",
                                "CancelRate": "CancelRate_out",
                            }
                        )

                        to_routes = routes_simple[
                            routes_simple["Dest"].isin(dest_airports)
                        ].rename(
                            columns={
                                "Origin": "Hub",
                                "Dest": "DestAirport",
                                "Flights": "Flights_in",
                                "DelayRate": "DelayRate_in",
                                "CancelRate": "CancelRate_in",
                            }
                        )

                        transit = from_routes.merge(to_routes, on="Hub", how="inner")

                        if transit.empty:
                            st.info(
                                "Non‑stop options between these states were scarce, and "
                                "typical one‑stop patterns are highly fragmented. "
                                "Try selecting nearby large airports in each state and "
                                "use the other tabs to inspect specific routes."
                            )
                        else:
                            transit["MinFlights"] = transit[
                                ["Flights_out", "Flights_in"]
                            ].min(axis=1)
                            transit = transit[
                                transit["MinFlights"] >= min_flights_state
                            ]

                            if transit.empty:
                                st.info(
                                    "Historically, one‑stop connections between these "
                                    "states exist but are spread thinly across many "
                                    "combinations. Lowering the minimum‑flights setting "
                                    "will reveal more patterns."
                                )
                            else:
                                transit["CombinedDelay"] = (
                                    transit["DelayRate_out"]
                                    + transit["DelayRate_in"]
                                )
                                transit["CombinedCancel"] = (
                                    transit["CancelRate_out"]
                                    + transit["CancelRate_in"]
                                )
                                transit["RiskScore"] = (
                                    transit["CombinedDelay"]
                                    + ALPHA_CANCEL_WEIGHT
                                    * transit["CombinedCancel"]
                                )
                                transit = transit.sort_values(
                                    "RiskScore", ascending=True
                                ).head(5)

                                st.warning(
                                    "Non‑stop options between these states were limited "
                                    "in the 2010–2024 history. The one‑stop patterns "
                                    "below tended to perform relatively well."
                                )

                                for _, row in transit.iterrows():
                                    st.write(
                                        f"- **{row['OriginAirport']} airport → {row['Hub']} airport → {row['DestAirport']} airport** – "
                                        f"combined delay risk ≈ "
                                        f"{row['CombinedDelay'] * 100:.1f}%, "
                                        f"combined cancellation risk ≈ "
                                        f"{row['CombinedCancel'] * 100:.2f}% "
                                        f"(leg volumes ≈ {int(row['Flights_out']):,} and "
                                        f"{int(row['Flights_in']):,} flights)"
                                    )

                                st.caption(
                                    "For these airport combinations, use the "
                                    "“Explore & compare → Airlines” tab to pick specific "
                                    "carriers on each leg."
                                )


if __name__ == "__main__":
    main()

