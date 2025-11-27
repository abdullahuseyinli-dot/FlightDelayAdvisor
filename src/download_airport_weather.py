from datetime import datetime
from pathlib import Path

import pandas as pd
from airportsdata import load as load_airports
from meteostat import Point, Daily

DATA_PATH = Path("data/processed/bts_delay_2010_2024_balanced_research.parquet")
OUT_DIR = Path("data/external")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_OUT = OUT_DIR / "airport_daily_weather.parquet"
AIRPORTS_USED_OUT = OUT_DIR / "weather_airports.csv"

START_DATE = datetime(2010, 1, 1)
END_DATE = datetime(2024, 12, 31)

TOP_N_AIRPORTS = 100  # choose 50 or 100 as you like


def main():
    print(f"[INFO] Loading processed BTS dataset from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    # All airports present in your sample
    all_airports = pd.concat([df["Origin"], df["Dest"]]).dropna()
    airport_counts = all_airports.value_counts()
    print(f"[INFO] There are {len(airport_counts)} airports with flights in the sample")

    # Take top N by flight count
    top_iata = airport_counts.head(TOP_N_AIRPORTS).index.tolist()
    print(f"[INFO] Top {TOP_N_AIRPORTS} airports by traffic: {top_iata[:10]}...")

    # Load airports database (IATA keyed)
    airports_db = load_airports("IATA")  # dict like { 'JFK': {...}, ... }

    chosen_rows = []
    for code in top_iata:
        info = airports_db.get(code)
        if info is None:
            print(f"[WARN] No airport metadata for {code} in airportsdata, skipping.")
            continue

        # Restrict to US airports
        country = info.get("country")
        if country not in ("US", "USA"):
            print(f"[WARN] {code} is not in US (country={country}), skipping.")
            continue

        lat = info.get("lat")
        lon = info.get("lon")
        name = info.get("name", "")

        if lat is None or lon is None:
            print(f"[WARN] Missing lat/lon for {code}, skipping.")
            continue

        chosen_rows.append(
            {"iata": code, "name": name, "lat": float(lat), "lon": float(lon), "country": country}
        )

    if not chosen_rows:
        raise RuntimeError("No US airports with valid metadata found among top N airports.")

    airports_df = pd.DataFrame(chosen_rows).sort_values("iata")
    print(f"[INFO] Will fetch weather for {len(airports_df)} airports.")
    print(airports_df.head())

    # Save which airports we used (for transparency)
    airports_df.to_csv(AIRPORTS_USED_OUT, index=False)
    print(f"[OK] Saved airport list to {AIRPORTS_USED_OUT}")

    all_frames = []

    for row in airports_df.itertuples(index=False):
        iata = row.iata
        lat = row.lat
        lon = row.lon

        print(f"[INFO] Fetching daily weather for {iata} ({lat}, {lon})")

        # Meteostat uses UTC and metric units by default
        point = Point(lat, lon)
        daily = Daily(point, START_DATE, END_DATE)
        wx = daily.fetch()

        if wx.empty:
            print(f"[WARN] No weather data returned for {iata}, skipping.")
            continue

        wx = wx.reset_index()  # 'time' column
        wx["Airport"] = iata

        # Keep a small subset of useful variables
        keep_cols = ["Airport", "time"]
        for col in ["tavg", "prcp", "snow", "wspd"]:
            if col in wx.columns:
                keep_cols.append(col)

        wx = wx[keep_cols]
        all_frames.append(wx)

    if not all_frames:
        raise RuntimeError("No weather data downloaded for any airport!")

    weather = pd.concat(all_frames, ignore_index=True)

    print(f"[INFO] Combined weather rows: {len(weather):,}")
    print(weather.head())

    weather.to_parquet(WEATHER_OUT, index=False)
    print(f"[OK] Saved airport daily weather to {WEATHER_OUT}")


if __name__ == "__main__":
    main()

