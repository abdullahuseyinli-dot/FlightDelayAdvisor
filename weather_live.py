# src/weather_live.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Dict, Any

import math
import requests
from airportsdata import load as load_airports

# --------------------------------------------------------------------
# Airport metadata (same IATA base as used for historical weather)
# --------------------------------------------------------------------
_AIRPORTS = load_airports("IATA")


@dataclass
class DailyWeather:
    """
    Daily weather snapshot for an airport.

    Units and semantics are chosen to match the historical dataset created via
    download_airport_weather.py + add_weather_to_dataset.py:

      - tavg: °C                (Meteostat daily mean temperature)
      - prcp: mm                (daily total precipitation)
      - snow: mm                (daily snowfall / snow water equivalent)
      - wspd: km/h              (daily mean / max wind speed)
      - bad_weather: 0/1 flag   (same rule as `make_bad_weather_flag`)

    The bad_weather flag is defined as:

        bad = (prcp >= 1.0) or (snow > 0.0) or (wspd >= 30.0)
    """

    tavg: float        # °C, daily mean temperature
    prcp: float        # mm, daily total precip
    snow: float        # mm, daily total snowfall
    wspd: float        # km/h, daily windspeed (Open‑Meteo max; close enough)
    bad_weather: int   # 0/1 flag, same rule as training


class LiveWeatherError(Exception):
    """Domain‑specific error for live weather issues."""


# --------------------------------------------------------------------
# Helpers: airport lookup + Open‑Meteo call
# --------------------------------------------------------------------
def get_airport_latlon(iata: str) -> tuple[float, float]:
    """
    Resolve an IATA airport code to (lat, lon) using airportsdata.

    Raises LiveWeatherError if the code is unknown.
    """
    iata = iata.upper()
    try:
        rec = _AIRPORTS[iata]
    except KeyError as exc:  # noqa: BLE001
        raise LiveWeatherError(f"Unknown IATA airport code: {iata}") from exc

    lat = float(rec["lat"])
    lon = float(rec["lon"])
    return lat, lon


def _fetch_open_meteo_daily(lat: float, lon: float, day: date) -> Dict[str, Any]:
    """
    Call Open‑Meteo daily endpoint for a *single date*.

    We request daily aggregates which align with the features you trained on:

      - temperature_2m_mean -> tavg  (°C)
      - precipitation_sum   -> prcp  (mm)
      - snowfall_sum        -> snow  (mm)
      - windspeed_10m_max   -> wspd  (km/h)

    Open‑Meteo serves data in metric units by default, consistent with
    the historical Meteostat pipeline.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": (
            "temperature_2m_mean,"
            "precipitation_sum,"
            "snowfall_sum,"
            "windspeed_10m_max"
        ),
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _parse_daily_payload(payload: Dict[str, Any], day: date) -> DailyWeather:
    """
    Convert Open‑Meteo's daily payload into a DailyWeather instance
    using the same bad‑weather rule as add_weather_to_dataset.py:

        bad = (prcp >= 1.0) or (snow > 0.0) or (wspd >= 30.0)
    """
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        raise LiveWeatherError("Open‑Meteo daily payload missing 'time' field")

    target_str = day.isoformat()
    try:
        idx = times.index(target_str)
    except ValueError as exc:  # noqa: BLE001
        raise LiveWeatherError(f"No daily record for {target_str}") from exc

    # All of these are in metric units (°C, mm, km/h)
    tavg = float(daily.get("temperature_2m_mean", [math.nan])[idx])
    prcp = float(daily.get("precipitation_sum", [0.0])[idx])
    snow = float(daily.get("snowfall_sum", [0.0])[idx])
    wspd = float(daily.get("windspeed_10m_max", [0.0])[idx])

    # Same rule as make_bad_weather_flag in the training pipeline
    bad_weather = int((prcp >= 1.0) or (snow > 0.0) or (wspd >= 30.0))

    return DailyWeather(
        tavg=tavg,
        prcp=prcp,
        snow=snow,
        wspd=wspd,
        bad_weather=bad_weather,
    )


# --------------------------------------------------------------------
# Public API used by the app
# --------------------------------------------------------------------
def get_live_daily_weather_for_airport(
    iata: str,
    day: date,
) -> Optional[DailyWeather]:
    """
    DAILY INTERFACE (preferred by the app, and what the docs show).

    Returns:
        DailyWeather for the given airport and date,
        or None if no live data/forecast is available.

    This mirrors the semantics of the historical dataset:
    one weather row per (airport, date).
    """
    try:
        lat, lon = get_airport_latlon(iata)
        payload = _fetch_open_meteo_daily(lat, lon, day)
        return _parse_daily_payload(payload, day)
    except (LiveWeatherError, requests.RequestException):
        # Swallow errors: the app will fall back to climatology.
        return None


def get_live_weather_for_airport(airport: str, when: datetime) -> Dict[str, float]:
    """
    TIME‑OF‑DAY INTERFACE (used by the Streamlit adapter).

    Expected interface from the app:

        def get_live_weather_for_airport(airport: str, when: datetime) -> dict:
            # Return a mapping with at least:
            #   "tavg": float (°C, consistent with training)
            #   "prcp": float (mm of liquid equivalent)
            #   "snow": float (mm of snow)
            #   "wspd": float (wind speed, km/h)
            #   "bad_weather": float or bool (1/0 or True/False)

    Behaviour:
      - Uses the DAILY interface internally (so both paths stay consistent).
      - If live data/forecast is available for (airport, date), returns the dict.
      - If not, raises LiveWeatherError so the app can fall back cleanly.
    """
    # Accept both datetime and date objects gracefully
    if isinstance(when, datetime):
        day = when.date()
    elif isinstance(when, date):
        day = when
    else:
        raise TypeError(
            f"'when' must be a datetime or date, got {type(when).__name__}"
        )

    dw = get_live_daily_weather_for_airport(airport, day)
    if dw is None:
        raise LiveWeatherError(
            f"Live weather not available for airport={airport}, date={day}"
        )

    # Return plain floats; the app will cast as needed
    return {
        "tavg": float(dw.tavg),
        "prcp": float(dw.prcp),
        "snow": float(dw.snow),
        "wspd": float(dw.wspd),
        "bad_weather": float(dw.bad_weather),
    }


def get_live_weather(airport: str, when: datetime) -> Dict[str, float]:
    """
    Backwards‑compatible alias.

    Some app versions may call `weather_live.get_live_weather(...)`
    instead of `get_live_weather_for_airport(...)`.  Both return the
    same mapping.
    """
    return get_live_weather_for_airport(airport, when)


