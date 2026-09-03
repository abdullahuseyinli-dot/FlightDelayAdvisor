"""Create-only, checksummed acquisition of official BTS monthly archives."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import requests

from .hashing import canonical_json_sha256, sha256_file, write_canonical_json

BTS_BASE_URL = "https://transtats.bts.gov/PREZIP"
BTS_FILENAME = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
CONFIRMATION_YEAR = 2026


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    year: int
    month: int
    url: str
    local_path: str
    bytes: int
    sha256: str
    csv_member: str
    retrieved_at: str
    etag: str | None
    last_modified: str | None
    status: str = "VERIFIED"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def remote_url(year: int, month: int) -> str:
    if year < 1987 or not 1 <= month <= 12:
        raise ValueError(f"invalid BTS period: {year}-{month:02d}")
    return f"{BTS_BASE_URL}/{BTS_FILENAME.format(year=year, month=month)}"


def validate_bts_zip(path: Path) -> str:
    """Validate ZIP structure and return its single on-time CSV member."""

    try:
        with ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"ZIP CRC failed for {bad_member}")
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    except BadZipFile as exc:
        raise ValueError(f"not a valid ZIP archive: {path}") from exc
    if len(csv_members) != 1:
        raise ValueError(f"expected one CSV member in {path}, found {len(csv_members)}")
    return csv_members[0]


def validate_confirmation_lock(path: Path, expected_year: int) -> None:
    """Fail closed unless a confirmation lock explicitly authorises opening labels."""

    if not path.is_file():
        raise PermissionError(f"confirmation lock does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("confirmation_year") != expected_year:
        raise PermissionError("confirmation lock year does not match requested data")
    if payload.get("status") != "LOCKED" or payload.get("open_authorized") is not True:
        raise PermissionError("confirmation lock is not locked and authorised")
    recorded_hash = payload.get("manifest_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded_hash != canonical_json_sha256(unhashed):
        raise PermissionError("confirmation lock self-hash is invalid")


def _manifest_payload(records: list[ArchiveRecord], events: list[dict[str, Any]]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 1,
        "source": "US DOT BTS Reporting Carrier On-Time Performance",
        "records": [
            asdict(record) for record in sorted(records, key=lambda item: (item.year, item.month))
        ],
        "events": events,
    }
    body["manifest_sha256"] = canonical_json_sha256(body)
    return body


def _load_manifest(path: Path) -> tuple[list[ArchiveRecord], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded_hash = payload.pop("manifest_sha256", None)
    if recorded_hash != canonical_json_sha256(payload):
        raise ValueError(f"source manifest self-hash failed: {path}")
    records = [ArchiveRecord(**record) for record in payload.get("records", [])]
    events = list(payload.get("events", []))
    return records, events


def _record_verified_archive(
    *,
    archive_path: Path,
    repository_root: Path,
    year: int,
    month: int,
    retrieved_at: str,
    etag: str | None,
    last_modified: str | None,
) -> ArchiveRecord:
    csv_member = validate_bts_zip(archive_path)
    return ArchiveRecord(
        year=year,
        month=month,
        url=remote_url(year, month),
        local_path=archive_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        bytes=archive_path.stat().st_size,
        sha256=sha256_file(archive_path),
        csv_member=csv_member,
        retrieved_at=retrieved_at,
        etag=etag,
        last_modified=last_modified,
    )


def acquire_month(
    *,
    repository_root: Path,
    year: int,
    month: int,
    output_directory: Path,
    manifest_path: Path,
    confirmation_lock: Path | None = None,
    session: requests.Session | None = None,
) -> ArchiveRecord:
    """Acquire or re-verify one monthly archive without overwriting raw evidence."""

    if year == CONFIRMATION_YEAR:
        if confirmation_lock is None:
            raise PermissionError("confirmation acquisition requires an explicit lock path")
        validate_confirmation_lock(confirmation_lock, year)

    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"on_time_{year}_{month:02d}.zip"
    timestamp = utc_now()
    records, events = _load_manifest(manifest_path)

    if destination.exists():
        record = _record_verified_archive(
            archive_path=destination,
            repository_root=repository_root,
            year=year,
            month=month,
            retrieved_at=timestamp,
            etag=None,
            last_modified=None,
        )
        existing = {(item.year, item.month): item for item in records}
        prior = existing.get((year, month))
        if prior is not None and (prior.sha256 != record.sha256 or prior.bytes != record.bytes):
            raise ValueError(f"existing archive differs from recorded evidence: {destination}")
        existing[(year, month)] = record
        events.append(
            {"at": timestamp, "action": "REVERIFIED_EXISTING", "year": year, "month": month}
        )
        write_canonical_json(manifest_path, _manifest_payload(list(existing.values()), events))
        return record

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"unadjudicated partial download exists: {partial}")

    client = session or requests.Session()
    url = remote_url(year, month)
    try:
        with client.get(url, timeout=(30, 180), stream=True) as response:
            response.raise_for_status()
            with partial.open("xb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
            record = _record_verified_archive(
                archive_path=partial,
                repository_root=repository_root,
                year=year,
                month=month,
                retrieved_at=timestamp,
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
            )
        partial.replace(destination)
    except Exception as exc:
        events.append(
            {
                "at": timestamp,
                "action": "DOWNLOAD_FAILED",
                "year": year,
                "month": month,
                "url": url,
                "partial_path": partial.as_posix() if partial.exists() else None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_canonical_json(manifest_path, _manifest_payload(records, events))
        raise

    record = ArchiveRecord(
        **{
            **asdict(record),
            "local_path": destination.resolve().relative_to(repository_root.resolve()).as_posix(),
        }
    )
    existing = {(item.year, item.month): item for item in records}
    existing[(year, month)] = record
    events.append(
        {"at": timestamp, "action": "DOWNLOADED_AND_VERIFIED", "year": year, "month": month}
    )
    write_canonical_json(manifest_path, _manifest_payload(list(existing.values()), events))
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--confirmation-lock", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_directory or root / "data" / "raw_bts" / str(args.year)
    manifest = args.manifest or root / "manifests" / f"raw_bts_{args.year}.json"
    for month in range(args.start_month, args.end_month + 1):
        record = acquire_month(
            repository_root=root,
            year=args.year,
            month=month,
            output_directory=output,
            manifest_path=manifest,
            confirmation_lock=args.confirmation_lock,
        )
        print(f"verified {record.year}-{record.month:02d}: {record.sha256} ({record.bytes} bytes)")


if __name__ == "__main__":
    main()
