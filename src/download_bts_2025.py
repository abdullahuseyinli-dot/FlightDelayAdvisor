"""
Download 2025 BTS On-Time Performance data into data/raw_2025.

This script uses the same PREZIP endpoint as the historical BTS downloads:
    https://transtats.bts.gov/PREZIP

File naming convention (per BTS):
    On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{M}.zip

For 2025, this gives URLs like:
    .../On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_1.zip

The script:
- Loops over months in 2025 (1..current_month by default).
- Downloads any available monthly ZIPs.
- Verifies that the response is a real ZIP (Content-Type + PK header).
- Saves them into data/raw_2025 as:
      on_time_2025_MM.zip

Run from the repo root:

    (.project1venv) PS> python .\src\download_bts_2025.py

You can override year/months if needed:

    (.project1venv) PS> python .\src\download_bts_2025.py --year 2025 --start-month 1 --end-month 12

Note: You need the `requests` library:

    (.project1venv) PS> pip install requests
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable

import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw_2025"

BASE_URL = "https://transtats.bts.gov/PREZIP"
# Official BTS naming pattern for monthly PREZIP files:
FILENAME_TEMPLATE = (
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)


# ---------------------------------------------------------------------
# Core download helpers
# ---------------------------------------------------------------------
def month_iter(year: int, start_month: int, end_month: int) -> Iterable[tuple[int, int]]:
    """Yield (year, month) pairs from start_month..end_month inclusive."""
    for m in range(start_month, end_month + 1):
        if not (1 <= m <= 12):
            continue
        yield year, m


def build_remote_url(year: int, month: int) -> str:
    """Construct the BTS PREZIP URL for a given (year, month)."""
    filename = FILENAME_TEMPLATE.format(year=year, month=month)
    return f"{BASE_URL}/{filename}"


def build_local_path(out_dir: Path, year: int, month: int) -> Path:
    """Where to save the downloaded ZIP locally."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"on_time_{year}_{month:02d}.zip"


def is_zip_content(response: requests.Response, content: bytes) -> bool:
    """
    Heuristic check that the response is a real ZIP file.

    - Status code must be 200.
    - Content-Type header should contain 'zip' (if present).
    - First bytes should start with 'PK' (ZIP signature).
    """
    if response.status_code != 200:
        return False

    ctype = (response.headers.get("Content-Type") or "").lower()
    if "zip" not in ctype and "octet-stream" not in ctype:
        # Many BTS ZIPs use 'application/x-zip-compressed' or similar.
        # Allow generic binary as a fallback, but we'll still check magic bytes.
        pass

    # ZIP files start with 'PK' (0x50 0x4B).
    return content[:2] == b"PK"


def download_month(year: int, month: int, out_dir: Path, overwrite: bool = False) -> bool:
    """
    Download a single month's BTS PREZIP file.

    Returns True if a valid ZIP was saved, False otherwise.
    """
    url = build_remote_url(year, month)
    dest = build_local_path(out_dir, year, month)

    if dest.exists() and not overwrite:
        print(f"[SKIP] {dest.name} already exists.")
        return True

    print(f"[INFO] Downloading {url} -> {dest}")

    try:
        resp = requests.get(url, timeout=120)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Request failed for {url}: {exc}")
        return False

    if resp.status_code == 404:
        print(f"[INFO] 404 for {url} – likely not published yet.")
        return False

    content = resp.content
    if not content:
        print(f"[WARN] Empty response from {url}; not saving.")
        return False

    if not is_zip_content(resp, content):
        print(
            f"[ERROR] Response from {url} does not look like a valid ZIP file.\n"
            f"        Content-Type: {resp.headers.get('Content-Type')!r}\n"
            "        This is often an HTML error page (e.g. login or error). "
            "Skipping without saving."
        )
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(content)

    print(f"[OK] Saved {dest} ({len(content) / 1_048_576:.1f} MB)")
    return True


def download_range(
    year: int,
    start_month: int,
    end_month: int,
    out_dir: Path,
    overwrite: bool = False,
) -> None:
    """Download a range of months for a single year."""
    print(
        f"[INFO] Downloading BTS On-Time PREZIP files for {year}, "
        f"months {start_month:02d}-{end_month:02d} into {out_dir}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    attempts = 0

    for y, m in month_iter(year, start_month, end_month):
        attempts += 1
        if download_month(y, m, out_dir=out_dir, overwrite=overwrite):
            successes += 1

    print(
        f"[SUMMARY] Finished: {successes}/{attempts} months downloaded "
        f"successfully into {out_dir}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download 2025 BTS On-Time Performance monthly PREZIP files "
            "into data/raw_2025."
        )
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year to download (default: 2025).",
    )
    parser.add_argument(
        "--start-month",
        type=int,
        default=1,
        help="First month to download (1-12, default: 1).",
    )
    parser.add_argument(
        "--end-month",
        type=int,
        default=None,
        help=(
            "Last month to download (1-12). "
            "If omitted, uses the current month for the given year, "
            "or 12 if the year is in the past."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for ZIP files (default: data/raw_2025).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and overwrite existing local files.",
    )
    return parser.parse_args()


def resolve_month_range(year: int, start_month: int, end_month: int | None) -> tuple[int, int]:
    """Determine a sensible (start_month, end_month) pair for the given year."""
    today = dt.date.today()

    if end_month is None:
        if year < today.year:
            end_month = 12
        elif year == today.year:
            end_month = today.month
        else:
            raise ValueError(
                f"Year {year} is in the future relative to today ({today}); "
                "please specify --end-month explicitly."
            )

    if not (1 <= start_month <= 12) or not (1 <= end_month <= 12) or start_month > end_month:
        raise ValueError(
            f"Invalid month range: start={start_month}, end={end_month}. "
            "Months must be between 1 and 12 and start <= end."
        )

    return start_month, end_month


def main() -> None:
    args = parse_args()
    year = args.year
    out_dir = Path(args.out_dir).resolve()

    start_m, end_m = resolve_month_range(year, args.start_month, args.end_month)

    print("=" * 72)
    print(f"BTS 2025 FETCHER – Year {year}, months {start_m:02d}-{end_m:02d}")
    print("=" * 72)
    print(f"[PATH] Repo root:     {REPO_ROOT}")
    print(f"[PATH] Output folder: {out_dir}")
    print()

    download_range(
        year=year,
        start_month=start_m,
        end_month=end_m,
        out_dir=out_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
