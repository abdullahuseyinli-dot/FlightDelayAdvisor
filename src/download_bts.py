import os
import time
import calendar
from pathlib import Path

import requests

BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- CONFIGURE YOUR YEAR RANGE HERE ----
START_YEAR = 2010
END_YEAR = 2024
# ----------------------------------------


def download_month(year: int, month: int, sleep_seconds: float = 1.0) -> None:
    """Download one month's On-Time Performance zip file if not already present."""
    url = BASE_URL.format(year=year, month=month)
    fname = DATA_DIR / f"ontime_{year}_{month:02d}.zip"

    if fname.exists():
        print(f"[SKIP] {fname} already exists")
        return

    print(f"[INFO] Downloading {year}-{month:02d} from {url}")
    resp = requests.get(url, stream=True, timeout=60)

    # If the file doesn't exist on the server (e.g. future month), just skip it
    if resp.status_code == 404:
        print(f"[WARN] {year}-{month:02d}: file not found (404). Skipping.")
        return

    # Raise for any other HTTP errors
    resp.raise_for_status()

    with open(fname, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1_048_576):  # 1 MB
            if chunk:
                f.write(chunk)

    print(f"[OK] Saved to {fname}")
    time.sleep(sleep_seconds)  # be polite to the server


def download_range(start_year: int, end_year: int) -> None:
    """Download all months from start_year to end_year inclusive."""
    for year in range(start_year, end_year + 1):
        print(f"\n[INFO] ===== Year {year} =====")
        for month in range(1, 13):
            try:
                download_month(year, month)
            except Exception as e:
                month_name = calendar.month_name[month]
                print(f"[ERROR] Failed {year}-{month_name}: {e}")

    print("\n[DONE] Finished downloading all requested years.")


if __name__ == "__main__":
    # This will ONLY attempt 2010–2025 and then exit
    download_range(start_year=START_YEAR, end_year=END_YEAR)

