from pathlib import Path


import pandas as pd

DATA_IN = Path("data/processed/bts_delay_2010_2024_balanced_research.parquet")
WEATHER_PATH = Path("data/external/airport_daily_weather.parquet")
AIRPORTS_USED_PATH = Path("data/external/weather_airports.csv")
DATA_OUT = Path("data/processed/bts_delay_2010_2024_balanced_research_weather.parquet")


def make_bad_weather_flag(temp_df, prefix: str) -> pd.DataFrame:
    """
    Given a df with columns like {prefix}_prcp, {prefix}_snow, {prefix}_wspd,
    create a boolean/int bad-weather flag based on simple thresholds.
    """
    prcp = temp_df[f"{prefix}_prcp"]
    snow = temp_df[f"{prefix}_snow"]
    wspd = temp_df[f"{prefix}_wspd"]

    bad = (
        (prcp >= 1.0)  # rain/snow >= 1mm
        | (snow > 0.0)  # any snow
        | (wspd >= 30.0)  # strong wind
    )

    return bad.astype("int8")


def main():
    print(f"[INFO] Loading main dataset from {DATA_IN}")
    df = pd.read_parquet(DATA_IN)

    print(f"[INFO] Loading airport daily weather from {WEATHER_PATH}")
    wx = pd.read_parquet(WEATHER_PATH)

    print(f"[INFO] Loading list of airports with weather from {AIRPORTS_USED_PATH}")
    airports_used = pd.read_csv(AIRPORTS_USED_PATH)
    weather_airports = airports_used["iata"].tolist()
    print(f"[INFO] Weather available for {len(weather_airports)} airports.")

    # Restrict to flights where both Origin and Dest are in the weather airports
    mask = df["Origin"].isin(weather_airports) & df["Dest"].isin(weather_airports)
    df = df[mask].copy()
    print(
        f"[INFO] After restricting to flights between weather airports: {df.shape[0]:,} rows"
    )

    # Ensure 'time' is date-only for join
    wx["time"] = pd.to_datetime(wx["time"]).dt.date

    # We will join on FlightDate (date only)
    df["FlightDate_date"] = pd.to_datetime(df["FlightDate"]).dt.date

    # ---- Origin weather ----
    origin_cols = ["Airport", "time"]
    for col in ["tavg", "prcp", "snow", "wspd"]:
        if col in wx.columns:
            origin_cols.append(col)

    origin_wx = wx[origin_cols].rename(
        columns={
            "Airport": "Origin",
            "time": "FlightDate_date",
            "tavg": "Origin_tavg",
            "prcp": "Origin_prcp",
            "snow": "Origin_snow",
            "wspd": "Origin_wspd",
        }
    )

    df = df.merge(
        origin_wx,
        on=["Origin", "FlightDate_date"],
        how="left",
    )

    # ---- Destination weather ----
    dest_wx = wx[origin_cols].rename(
        columns={
            "Airport": "Dest",
            "time": "FlightDate_date",
            "tavg": "Dest_tavg",
            "prcp": "Dest_prcp",
            "snow": "Dest_snow",
            "wspd": "Dest_wspd",
        }
    )

    df = df.merge(
        dest_wx,
        on=["Dest", "FlightDate_date"],
        how="left",
    )

    # Fill missing weather with neutral / average values
    mean_tavg = wx["tavg"].mean() if "tavg" in wx.columns else 15.0
    mean_wspd = wx["wspd"].mean() if "wspd" in wx.columns else 10.0

    for col in [
        "Origin_tavg",
        "Origin_prcp",
        "Origin_snow",
        "Origin_wspd",
        "Dest_tavg",
        "Dest_prcp",
        "Dest_snow",
        "Dest_wspd",
    ]:
        if col.endswith("_prcp") or col.endswith("_snow"):
            df[col] = df[col].fillna(0.0).astype("float32")
        elif col.endswith("_tavg"):
            df[col] = df[col].fillna(mean_tavg).astype("float32")
        elif col.endswith("_wspd"):
            df[col] = df[col].fillna(mean_wspd).astype("float32")

    # Bad-weather flags
    df["Origin_BadWeather"] = make_bad_weather_flag(df, "Origin")
    df["Dest_BadWeather"] = make_bad_weather_flag(df, "Dest")

    # Drop helper column
    df = df.drop(columns=["FlightDate_date"])

    print("[INFO] Weather columns added. Sample:")
    print(
        df[
            [
                "Origin",
                "Dest",
                "FlightDate",
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
        ].head()
    )

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_OUT, index=False)
    print(f"[OK] Saved dataset with weather to {DATA_OUT}")
    print(f"[INFO] Final shape: {df.shape}")


if __name__ == "__main__":
    main()
