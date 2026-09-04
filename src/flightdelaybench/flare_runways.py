"""Build a versioned FLARE-24 runway-heading catalog from an FAA NASR archive."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .flare_aviation import runway_heading_from_identifier
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

FAA_ARCHIVE_URL = (
    "https://aeronav.faa.gov/aero_data/28DaySub/2024-10-03/"
    "28DaySubscription_Effective_2024-10-03.zip"
)


def _read_csv_member(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    try:
        payload = archive.read(name)
    except KeyError as error:
        raise ValueError(f"FAA NASR archive is missing {name}") from error
    return pd.read_csv(io.BytesIO(payload), low_memory=False)


def _catalog_airports(path: Path) -> list[str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("airports")
    if not isinstance(records, list) or not records:
        raise ValueError("airport catalog contains no records")
    airports = sorted({str(record["iata"]) for record in records})
    if len(airports) != len(records):
        raise ValueError("airport catalog IATA identifiers must be unique")
    return airports


def build_runway_catalog(
    *,
    archive_path: Path,
    airport_catalog_path: Path,
    output_path: Path,
    manifest_path: Path,
    minimum_runway_length_ft: float = 4_000.0,
) -> dict[str, Any]:
    """Extract true runway-end alignments for the frozen airport cohort."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite a FLARE-24 runway artifact")
    if not np.isfinite(minimum_runway_length_ft) or minimum_runway_length_ft <= 0.0:
        raise ValueError("minimum runway length must be finite and positive")
    airports = _catalog_airports(airport_catalog_path)
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"FAA NASR archive CRC failed: {bad_member}")
        airport_base = _read_csv_member(archive, "CSV/APT_BASE.csv")
        runways = _read_csv_member(archive, "CSV/APT_RWY.csv")
        runway_ends = _read_csv_member(archive, "CSV/APT_RWY_END.csv")
    required_base = {"ARPT_ID", "ICAO_ID"}
    required_runways = {"ARPT_ID", "RWY_ID", "RWY_LEN"}
    required_ends = {"ARPT_ID", "RWY_ID", "RWY_END_ID", "TRUE_ALIGNMENT"}
    for name, frame, required in (
        ("APT_BASE", airport_base, required_base),
        ("APT_RWY", runways, required_runways),
        ("APT_RWY_END", runway_ends, required_ends),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"FAA {name} is missing columns: {missing}")
    base = airport_base.loc[:, ["ARPT_ID", "ICAO_ID"]].copy()
    base["ARPT_ID"] = base["ARPT_ID"].astype("string")
    base["ICAO_ID"] = base["ICAO_ID"].astype("string")
    base["IATA"] = base["ARPT_ID"]
    # For continental US airports ICAO is typically K + IATA; this fallback is
    # also harmless for Alaska/Hawaii/territories whose ARPT_ID is already IATA.
    unresolved = ~base["IATA"].isin(airports)
    icao_suffix = base["ICAO_ID"].str[-3:]
    base.loc[unresolved & icao_suffix.isin(airports), "IATA"] = icao_suffix
    base = base.loc[base["IATA"].isin(airports)].drop_duplicates("ARPT_ID")

    runway_table = runways.loc[:, ["ARPT_ID", "RWY_ID", "RWY_LEN"]].copy()
    runway_table["RWY_LEN"] = pd.to_numeric(runway_table["RWY_LEN"], errors="coerce")
    runway_table = runway_table.loc[
        runway_table["RWY_LEN"].ge(minimum_runway_length_ft)
    ]
    ends = runway_ends.loc[:, ["ARPT_ID", "RWY_ID", "RWY_END_ID", "TRUE_ALIGNMENT"]].copy()
    merged = (
        ends.merge(runway_table, on=["ARPT_ID", "RWY_ID"], how="inner", validate="many_to_one")
        .merge(base.loc[:, ["ARPT_ID", "IATA"]], on="ARPT_ID", how="inner", validate="many_to_one")
    )
    merged["TRUE_ALIGNMENT"] = pd.to_numeric(merged["TRUE_ALIGNMENT"], errors="coerce")
    fallback = merged["RWY_END_ID"].map(
        lambda value: runway_heading_from_identifier(str(value))
    )
    merged["HEADING"] = merged["TRUE_ALIGNMENT"].fillna(fallback)
    merged.loc[merged["HEADING"].eq(0.0), "HEADING"] = 360.0
    if (~merged["HEADING"].between(0.0, 360.0, inclusive="right")).any():
        raise ValueError("FAA runway headings are outside (0, 360]")

    records: list[dict[str, Any]] = []
    for airport in airports:
        airport_rows = merged.loc[merged["IATA"].eq(airport)]
        headings = sorted(
            {round(float(value), 3) for value in airport_rows["HEADING"].dropna()}
        )
        records.append(
            {
                "iata": airport,
                "headings_true_degrees": headings,
                "eligible_runway_end_count": len(headings),
                "minimum_eligible_runway_length_ft": (
                    float(airport_rows["RWY_LEN"].min()) if not airport_rows.empty else None
                ),
                "missing": not headings,
            }
        )
    missing_airports = [record["iata"] for record in records if record["missing"]]
    output: dict[str, Any] = {
        "schema_version": 1,
        "effective_date": "2024-10-03",
        "heading_reference": "true north",
        "selection": (
            "FAA runway ends on runways at least the configured length; inference later "
            "uses a wind-optimal envelope and does not infer the realised active runway"
        ),
        "minimum_runway_length_ft": minimum_runway_length_ft,
        "airports": records,
    }
    output["catalog_sha256"] = canonical_json_sha256(output)
    write_canonical_json(output_path, output)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_STATIC_COVARIATE_NO_FLIGHT_OUTCOMES_ACCESSED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "FAA 28-Day NASR Subscription",
        "source_url": FAA_ARCHIVE_URL,
        "effective_date": "2024-10-03",
        "raw_archive": {
            "path": archive_path.as_posix(),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
        "airport_catalog": {
            "path": airport_catalog_path.as_posix(),
            "sha256": sha256_file(airport_catalog_path),
        },
        "output": {
            "path": output_path.as_posix(),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "self_hash": output["catalog_sha256"],
        },
        "airport_count": len(records),
        "airports_missing_eligible_runways": missing_airports,
        "minimum_runway_length_ft": minimum_runway_length_ft,
        "members_used": ["CSV/APT_BASE.csv", "CSV/APT_RWY.csv", "CSV/APT_RWY_END.csv"],
        "provenance": capture_provenance(
            (Path(__file__), Path(__file__).with_name("flare_aviation.py"))
        ),
        "claim_limit": (
            "This dated static snapshot supplies runway geometry only. It does not identify "
            "the active runway, NOTAM closures, or operational runway configuration."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def load_runway_headings(path: Path) -> dict[str, tuple[float, ...]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("catalog_sha256")
    body = {key: value for key, value in payload.items() if key != "catalog_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"runway catalog self-hash failed: {path}")
    result: dict[str, tuple[float, ...]] = {}
    for record in payload.get("airports", []):
        headings = tuple(float(value) for value in record["headings_true_degrees"])
        if any(not np.isfinite(value) or value <= 0.0 or value > 360.0 for value in headings):
            raise ValueError(f"invalid runway headings for {record['iata']}")
        result[str(record["iata"])] = headings
    if not result:
        raise ValueError("runway catalog contains no airports")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--airport-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-runway-length-ft", type=float, default=4_000.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_runway_catalog(
        archive_path=args.archive,
        airport_catalog_path=args.airport_catalog,
        output_path=args.output,
        manifest_path=args.manifest,
        minimum_runway_length_ft=args.minimum_runway_length_ft,
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest.as_posix(),
                "airports": result["airport_count"],
                "missing": result["airports_missing_eligible_runways"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
