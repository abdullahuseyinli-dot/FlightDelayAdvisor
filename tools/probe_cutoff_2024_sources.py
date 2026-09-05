"""Preserve small, fixed 2024 source-access probes without flight outcome data.

HTTP success establishes retrieval today, never historical consumer receipt.
This is Gate A feasibility evidence, not a model or population-coverage pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from flightdelaybench.hashing import canonical_json_sha256, write_canonical_json

PROBES = (
    (
        "faa_listing_20240115.html",
        "https://www.fly.faa.gov/adv/adv_list?whichAdvisories=ATCSCC&advisoryCategory=All"
        "&date=2024-01-15&airflow=true&ctop=true&gStop=true&gDelay=true&route=true&other=true",
    ),
    (
        "faa_advisory_20240115_079.html",
        "https://www.fly.faa.gov/adv/adv_otherdis?adv_date=01152024&advn=79",
    ),
    (
        "iem_taf_overview_KDSM_20240115.json",
        "https://mesonet.agron.iastate.edu/api/1/nws/taf_overview.json"
        "?station=KDSM&sts=2024-01-15&ets=2024-01-16",
    ),
    (
        "iem_taf_KDSM_202401150532.txt",
        "https://mesonet.agron.iastate.edu/api/1/nwstext/202401150532-KDMX-FTUS43-TAFDSM",
    ),
)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def probe_sources(output_dir: Path) -> dict[str, Any]:
    """Fetch only the locked URLs; keep failed attempts and refuse reused output."""
    output_dir.mkdir(parents=True, exist_ok=False)
    intent = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "probes": [{"path": name, "url": url} for name, url in PROBES],
        "max_response_bytes": MAX_RESPONSE_BYTES,
        "scope": "Fixed 2024 FAA advisory and NWS weather source access only",
        "historical_consumer_receipt_established": False,
        "flight_labels_requested": False,
    }
    write_canonical_json(output_dir / "intent.json", intent)
    records: list[dict[str, Any]] = []
    for name, url in PROBES:
        record: dict[str, Any] = {"path": name, "url": url}
        try:
            request = Request(url, headers={"User-Agent": "FlightDelayAdvisor research source audit"})
            with urlopen(request, timeout=30) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                record.update({
                    "http_status": response.status,
                    "response_url": response.url,
                    "response_headers": dict(response.headers.items()),
                })
            # Even a capped/unexpected response remains evidence of this attempt.
            with (output_dir / name).open("xb") as handle:
                handle.write(payload)
            record.update({
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "response_truncated": len(payload) > MAX_RESPONSE_BYTES,
                "status": "RETRIEVED_FOR_REVIEW" if len(payload) <= MAX_RESPONSE_BYTES else "RESPONSE_LIMIT_EXCEEDED",
            })
        except Exception as error:
            record.update({"status": "FAILED_ACCESS", "exception_type": type(error).__name__, "message": str(error)})
        record["retrieved_at_utc"] = datetime.now(UTC).isoformat()
        record["historical_consumer_available_at_utc"] = None
        write_canonical_json(output_dir / f"{name}.metadata.json", record)
        records.append(record)
        print(json.dumps({"path": name, "status": record["status"]}), flush=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "SOURCE_ACCESS_RECORDED" if all(r["status"] == "RETRIEVED_FOR_REVIEW" for r in records) else "INCOMPLETE_SOURCE_ACCESS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "intent_sha256": canonical_json_sha256(intent),
        "records": records,
        "strict_availability_gate_passed": False,
        "flight_weighted_coverage_established": False,
        "historical_consumer_receipt_established": False,
        "flight_labels_requested": False,
        "claim_limit": "Source access and retained bytes only. Review issue/send/validity semantics and revisions separately; today's retrieval is not historical receipt.",
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = probe_sources(args.output_dir)
    if report["status"] != "SOURCE_ACCESS_RECORDED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
