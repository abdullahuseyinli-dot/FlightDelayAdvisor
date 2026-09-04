"""Small archive-feasibility audit, not a weather model or forecast-gain experiment.

IEM processed TAF output calls the validity start utc_taf_issue. Read the original
header instead to distinguish issue time, valid interval, and our retrieval time.
Never extend an expired TAF or replace it with a post-cutoff amendment.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .hashing import canonical_json_sha256, sha256_file, write_canonical_json

API = "https://mesonet.agron.iastate.edu/api/1"
STATIONS = ("KJFK", "KORD", "KDSM", "KCWA")
PILOT_CUTOFFS = tuple(f"2024-{month}-15T{hour}:00:00Z" for month in ("01", "07") for hour in ("06", "12"))


def _resolve_day(reference: pd.Timestamp, day: int, hour: int, minute: int = 0) -> pd.Timestamp:
    if not 1 <= day <= 31 or not 0 <= hour <= 24 or not 0 <= minute < 60 or (hour == 24 and minute):
        raise ValueError("invalid TAF date/time group")
    candidates = []
    for offset in (-1, 0, 1):
        month = reference.replace(day=1, hour=0, minute=0, second=0) + pd.DateOffset(months=offset)
        try:
            candidates.append(month.replace(day=day) + pd.Timedelta(hours=hour, minutes=minute))
        except ValueError:
            continue
    return min(candidates, key=lambda value: abs(value - reference))


def parse_taf_header(raw: str, station: str, product_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Z0-9]{4}", station) or not re.fullmatch(r"2024\d{8}-[A-Z0-9-]+", product_id):
        raise ValueError("pilot accepts only explicit 2024 NWS products and four-character stations")
    product_time = pd.to_datetime(product_id[:12], format="%Y%m%d%H%M", utc=True)
    match = re.search(rf"\b{re.escape(station)}\s+(\d{{6}})Z\s+(\d{{4}})/(\d{{4}})\b", raw)
    if match is None:
        raise ValueError("TAF header missing; no expiry is inferred from forecast segments")
    issue_group, start_group, end_group = match.groups()
    issue = _resolve_day(product_time, int(issue_group[:2]), int(issue_group[2:4]), int(issue_group[4:]))
    if abs(issue - product_time) > pd.Timedelta(hours=1):
        raise ValueError("TAF and NWS product timestamps disagree materially")
    start = _resolve_day(issue, int(start_group[:2]), int(start_group[2:]))
    end = _resolve_day(start + pd.Timedelta(hours=24), int(end_group[:2]), int(end_group[2:]))
    if not pd.Timedelta(0) < end - start <= pd.Timedelta(hours=36):
        raise ValueError("invalid TAF validity interval")
    return {
        "station": station, "product_id": product_id,
        "taf_header_issue_utc": issue.isoformat(), "nws_product_time_utc": product_time.isoformat(),
        "producer_time_bound_utc": max(issue, product_time).isoformat(),
        "valid_start_utc": start.isoformat(), "valid_end_utc": end.isoformat(),
        "taf_cancelled_or_nil": bool(re.search(r"\b(?:CNL|NIL)\b", raw[match.end():])),
    }


def covers_horizon(record: dict[str, Any], cutoff: str, horizon_hours: int, latency_minutes: int = 0) -> bool:
    at = pd.Timestamp(cutoff)
    if at.tzinfo is None or latency_minutes < 0 or horizon_hours < 0:
        raise ValueError("aware cutoff and nonnegative horizon/latency required")
    target = at + pd.Timedelta(hours=horizon_hours)
    return bool(
        not record["taf_cancelled_or_nil"]
        and pd.Timestamp(record["producer_time_bound_utc"]) + pd.Timedelta(minutes=latency_minutes) <= at
        and pd.Timestamp(record["valid_start_utc"]) <= target < pd.Timestamp(record["valid_end_utc"])
    )


def _fetch_preserved(session: requests.Session, url: str, path: Path, params: dict[str, str] | None = None) -> tuple[bytes, dict[str, Any]]:
    if path.exists():
        raise FileExistsError(f"refusing to replace source response {path}")
    response = session.get(url, params=params, timeout=(10, 30))
    # Preserve error bodies as well as successful source responses.
    with path.open("xb") as handle:
        handle.write(response.content)
    evidence = {"url": response.url, "http_status": response.status_code, "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_canonical_json(path.with_suffix(path.suffix + ".metadata.json"), evidence)
    response.raise_for_status()
    return response.content, evidence


def run_pilot(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite pilot {output_dir}")
    output_dir.mkdir(parents=True)
    write_canonical_json(output_dir / "intent.json", {
        "status": "STARTED_ARCHIVE_FEASIBILITY_PILOT", "stations": list(STATIONS), "cutoffs": list(PILOT_CUTOFFS),
        "selection": "fixed small seasonal station/time grid, not flight-weighted or representative",
        "outcomes_accessed": False, "maximum_requests": len(STATIONS) * len(PILOT_CUTOFFS) * 2,
    })
    records, failures, sources = [], [], []
    with requests.Session() as session:
        for station in STATIONS:
            for cutoff in PILOT_CUTOFFS:
                name = f"{station}_{pd.Timestamp(cutoff).strftime('%Y%m%dT%H%M')}"
                try:
                    content, source = _fetch_preserved(session, f"{API}/nws/taf.json", output_dir / f"{name}.json", {"station": station, "issued": cutoff})
                    sources.append(source)
                    rows = json.loads(content)["data"]
                    identifiers = {row["product_id"] for row in rows}
                    if len(identifiers) != 1:
                        raise ValueError("missing or ambiguous source TAF issuance")
                    identifier = identifiers.pop()
                    if not re.fullmatch(r"2024\d{8}-[A-Z0-9-]+", identifier):
                        raise ValueError("unauthorized product year or malformed ID")
                    raw, raw_source = _fetch_preserved(session, f"{API}/nwstext/{identifier}", output_dir / f"{name}.txt")
                    sources.append(raw_source)
                    record = parse_taf_header(raw.decode("utf-8"), station, identifier)
                    record["cutoff_utc"] = cutoff
                    record["issue_not_after_cutoff"] = pd.Timestamp(record["producer_time_bound_utc"]) <= pd.Timestamp(cutoff)
                    record["coverage"] = {f"h{horizon}_latency{latency}m": covers_horizon(record, cutoff, horizon, latency)
                                          for horizon in (24, 27) for latency in (0, 15, 60)}
                    records.append(record)
                except (ValueError, KeyError, requests.RequestException) as error:
                    failures.append({"station": station, "cutoff": cutoff, "exception_type": type(error).__name__, "message": str(error)})
                print(json.dumps({"station": station, "cutoff": cutoff, "audited": len(records), "failed": len(failures)}), flush=True)
    coverage = {station: {
        key: sum(record["coverage"][key] for record in records if record["station"] == station)
        for key in (f"h{horizon}_latency{latency}m" for horizon in (24, 27) for latency in (0, 15, 60))
    } for station in STATIONS}
    report: dict[str, Any] = {
        "schema_version": 1, "status": "COMPLETED_SMALL_ARCHIVE_FEASIBILITY_PILOT" if not failures else "PARTIAL_ARCHIVE_FEASIBILITY_PILOT",
        "requested_station_times": len(STATIONS) * len(PILOT_CUTOFFS), "audited_station_times": len(records),
        "coverage_counts_by_station": coverage, "records": records, "failures": failures, "sources": sources,
        "historical_consumer_receipt_evidenced": False, "latency_values_are_assumptions": True,
        "latency_sensitivity_reselects_older_product": False,
        "flight_outcomes_accessed": False, "predictive_gain_measured": False,
        "claim_limit": "Producer timestamps and expiry audited on a small grid. No historical receipt guarantee, population coverage, restriction coverage or accuracy gain is established.",
        "documentation": ["https://mesonet.agron.iastate.edu/api/1/docs", "https://aviationweather.gov/help/data/"],
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_pilot(args.output_dir)
    print(json.dumps({"status": result["status"], "audited_station_times": result["audited_station_times"]}))


if __name__ == "__main__":
    main()
