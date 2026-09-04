"""Build the static airport-resource catalog for the FLARE capacity hypergraph.

The legacy runway catalog intentionally retained only unique headings.  This
extension preserves physical runways, parallel multiplicity, dimensions,
instrumented ends, and coarse annual-operation metadata.  It remains a static
NASR snapshot and never claims the realised active runway configuration.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .flare_aviation import runway_heading_from_identifier
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json
from .provenance import capture_provenance

ANNUAL_OPERATION_COLUMNS = (
    "COMMERCIAL_OPS",
    "COMMUTER_OPS",
    "AIR_TAXI_OPS",
    "LOCAL_OPS",
    "ITNRNT_OPS",
    "MIL_ACFT_OPS",
)


def _read_csv_member(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    try:
        payload = archive.read(name)
    except KeyError as error:
        raise ValueError(f"FAA NASR archive is missing {name}") from error
    return pd.read_csv(io.BytesIO(payload), low_memory=False)


def _airport_ids(path: Path) -> tuple[str, ...]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("airports")
    if not isinstance(records, list) or not records:
        raise ValueError("airport catalog contains no records")
    airports = tuple(sorted(str(record["iata"]) for record in records))
    if len(airports) != len(set(airports)):
        raise ValueError("airport catalog contains duplicate IATA identifiers")
    return airports


def _axis_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 180.0
    return min(difference, 180.0 - difference)


def _orientation_summary(
    axes: list[float], *, parallel_tolerance_degrees: float
) -> tuple[int, int]:
    if not axes:
        return 0, 0
    parent = list(range(len(axes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(axes)):
        for right in range(left + 1, len(axes)):
            if _axis_distance(axes[left], axes[right]) <= parallel_tolerance_degrees:
                union(left, right)
    sizes: dict[int, int] = {}
    for index in range(len(axes)):
        root = find(index)
        sizes[root] = sizes.get(root, 0) + 1
    return len(sizes), max(sizes.values())


def _is_instrumented(value: object) -> bool:
    if pd.isna(value):
        return False
    token = str(value).strip().upper()
    return token not in {"", "N", "NONE", "NO", "N/A", "NA"}


def _as_optional_float(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def build_resource_catalog(
    *,
    archive_path: Path,
    airport_catalog_path: Path,
    output_path: Path,
    manifest_path: Path,
    minimum_runway_length_ft: float = 4_000.0,
    parallel_tolerance_degrees: float = 12.0,
) -> dict[str, Any]:
    """Extract cycle-dated static resources without reading flight outcomes."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite a capacity resource artifact")
    if not math.isfinite(minimum_runway_length_ft) or minimum_runway_length_ft <= 0.0:
        raise ValueError("minimum runway length must be finite and positive")
    if not math.isfinite(parallel_tolerance_degrees) or not (
        0.0 < parallel_tolerance_degrees < 45.0
    ):
        raise ValueError("parallel tolerance must be in (0, 45) degrees")

    airports = _airport_ids(airport_catalog_path)
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"FAA NASR archive CRC failed: {bad_member}")
        base_raw = _read_csv_member(archive, "CSV/APT_BASE.csv")
        runway_raw = _read_csv_member(archive, "CSV/APT_RWY.csv")
        end_raw = _read_csv_member(archive, "CSV/APT_RWY_END.csv")

    required_base = {"ARPT_ID", "ICAO_ID", "EFF_DATE", *ANNUAL_OPERATION_COLUMNS}
    required_runway = {
        "ARPT_ID",
        "RWY_ID",
        "RWY_LEN",
        "RWY_WIDTH",
        "SURFACE_TYPE_CODE",
        "RWY_LGT_CODE",
    }
    required_end = {
        "ARPT_ID",
        "RWY_ID",
        "RWY_END_ID",
        "TRUE_ALIGNMENT",
        "ILS_TYPE",
        "APCH_LGT_SYSTEM_CODE",
        "LAT_DECIMAL",
        "LONG_DECIMAL",
    }
    for source, frame, required in (
        ("APT_BASE", base_raw, required_base),
        ("APT_RWY", runway_raw, required_runway),
        ("APT_RWY_END", end_raw, required_end),
    ):
        schema_missing = sorted(required - set(frame.columns))
        if schema_missing:
            raise ValueError(f"FAA {source} is missing columns: {schema_missing}")

    base_columns = [
        "ARPT_ID",
        "ICAO_ID",
        "EFF_DATE",
        "ANNUAL_OPS_DATE",
        *ANNUAL_OPERATION_COLUMNS,
    ]
    base = base_raw.loc[:, base_columns].copy()
    base["ARPT_ID"] = base["ARPT_ID"].astype("string")
    base["ICAO_ID"] = base["ICAO_ID"].astype("string")
    base["IATA"] = base["ARPT_ID"]
    unresolved = ~base["IATA"].isin(airports)
    suffix = base["ICAO_ID"].str[-3:]
    base.loc[unresolved & suffix.isin(airports), "IATA"] = suffix
    base = base.loc[base["IATA"].isin(airports)].drop_duplicates("ARPT_ID")
    for column in ANNUAL_OPERATION_COLUMNS:
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base["ANNUAL_OPERATIONS"] = base[list(ANNUAL_OPERATION_COLUMNS)].sum(
        axis=1, min_count=1
    )

    runway_columns = [
        "ARPT_ID",
        "RWY_ID",
        "RWY_LEN",
        "RWY_WIDTH",
        "SURFACE_TYPE_CODE",
        "RWY_LGT_CODE",
    ]
    runways = runway_raw.loc[:, runway_columns].copy()
    runways["RWY_LEN"] = pd.to_numeric(runways["RWY_LEN"], errors="coerce")
    runways["RWY_WIDTH"] = pd.to_numeric(runways["RWY_WIDTH"], errors="coerce")
    runways = runways.loc[runways["RWY_LEN"].ge(minimum_runway_length_ft)]
    runways = runways.merge(
        base.loc[:, ["ARPT_ID", "IATA"]],
        on="ARPT_ID",
        how="inner",
        validate="many_to_one",
    )

    end_columns = [
        "ARPT_ID",
        "RWY_ID",
        "RWY_END_ID",
        "TRUE_ALIGNMENT",
        "ILS_TYPE",
        "APCH_LGT_SYSTEM_CODE",
        "LAT_DECIMAL",
        "LONG_DECIMAL",
    ]
    ends = end_raw.loc[:, end_columns].copy()
    ends = ends.merge(
        runways.loc[:, ["ARPT_ID", "RWY_ID", "IATA"]],
        on=["ARPT_ID", "RWY_ID"],
        how="inner",
        validate="many_to_one",
    )
    ends["TRUE_ALIGNMENT"] = pd.to_numeric(ends["TRUE_ALIGNMENT"], errors="coerce")
    fallback = ends["RWY_END_ID"].map(lambda value: runway_heading_from_identifier(str(value)))
    ends["HEADING"] = ends["TRUE_ALIGNMENT"].fillna(fallback)
    ends.loc[ends["HEADING"].eq(0.0), "HEADING"] = 360.0
    if (~ends["HEADING"].between(0.0, 360.0, inclusive="right")).any():
        raise ValueError("FAA runway headings are outside (0, 360]")

    effective_dates = sorted(
        {
            pd.Timestamp(value).date().isoformat()
            for value in base["EFF_DATE"].dropna().unique()
        }
    )
    if len(effective_dates) != 1:
        raise ValueError(f"expected one NASR effective date, observed {effective_dates}")

    records: list[dict[str, Any]] = []
    missing_airports: list[str] = []
    for airport in airports:
        airport_base = base.loc[base["IATA"].eq(airport)]
        airport_runways = runways.loc[runways["IATA"].eq(airport)].copy()
        airport_ends = ends.loc[ends["IATA"].eq(airport)].copy()
        runway_records: list[dict[str, Any]] = []
        axes: list[float] = []
        for runway_id, runway_rows in airport_runways.groupby("RWY_ID", sort=True):
            row = runway_rows.iloc[0]
            runway_ends = airport_ends.loc[airport_ends["RWY_ID"].eq(runway_id)]
            headings = [float(value) for value in runway_ends["HEADING"].dropna()]
            if headings:
                axes.append(float(headings[0] % 180.0))
            end_records = [
                {
                    "runway_end_id": str(end_row["RWY_END_ID"]),
                    "heading_true_degrees": float(end_row["HEADING"]),
                    "ils_type": (
                        None if pd.isna(end_row["ILS_TYPE"]) else str(end_row["ILS_TYPE"])
                    ),
                    "approach_light_system": (
                        None
                        if pd.isna(end_row["APCH_LGT_SYSTEM_CODE"])
                        else str(end_row["APCH_LGT_SYSTEM_CODE"])
                    ),
                    "latitude": _as_optional_float(end_row["LAT_DECIMAL"]),
                    "longitude": _as_optional_float(end_row["LONG_DECIMAL"]),
                }
                for _, end_row in runway_ends.sort_values("RWY_END_ID").iterrows()
            ]
            runway_records.append(
                {
                    "runway_id": str(runway_id),
                    "length_ft": float(row["RWY_LEN"]),
                    "width_ft": _as_optional_float(row["RWY_WIDTH"]),
                    "surface_type": (
                        None
                        if pd.isna(row["SURFACE_TYPE_CODE"])
                        else str(row["SURFACE_TYPE_CODE"])
                    ),
                    "lighting_code": (
                        None if pd.isna(row["RWY_LGT_CODE"]) else str(row["RWY_LGT_CODE"])
                    ),
                    "ends": end_records,
                }
            )
        orientation_families, max_parallel = _orientation_summary(
            axes,
            parallel_tolerance_degrees=parallel_tolerance_degrees,
        )
        end_count = sum(len(record["ends"]) for record in runway_records)
        instrumented_count = sum(
            _is_instrumented(end["ils_type"])
            for runway in runway_records
            for end in runway["ends"]
        )
        airport_missing = not runway_records
        if airport_missing:
            missing_airports.append(airport)
        records.append(
            {
                "iata": airport,
                "physical_runway_count": len(runway_records),
                "eligible_runway_end_count": end_count,
                "orientation_family_count": orientation_families,
                "max_parallel_runways": max_parallel,
                "ils_end_fraction": (
                    float(instrumented_count / end_count) if end_count else None
                ),
                "minimum_runway_length_ft": (
                    float(min(record["length_ft"] for record in runway_records))
                    if runway_records
                    else None
                ),
                "maximum_runway_length_ft": (
                    float(max(record["length_ft"] for record in runway_records))
                    if runway_records
                    else None
                ),
                "annual_operations": (
                    _as_optional_float(airport_base["ANNUAL_OPERATIONS"].iloc[0])
                    if not airport_base.empty
                    else None
                ),
                "annual_operations_date": (
                    None
                    if airport_base.empty or pd.isna(airport_base["ANNUAL_OPS_DATE"].iloc[0])
                    else str(airport_base["ANNUAL_OPS_DATE"].iloc[0])
                ),
                "runways": runway_records,
                "missing": airport_missing,
            }
        )

    output: dict[str, Any] = {
        "schema_version": 1,
        "method": "FLARE-24-CC-RTH-STATIC-RESOURCE-CATALOG",
        "effective_date": effective_dates[0],
        "heading_reference": "true north",
        "minimum_runway_length_ft": minimum_runway_length_ft,
        "parallel_family_tolerance_degrees": parallel_tolerance_degrees,
        "airports": records,
        "claim_limit": (
            "Cycle-dated static geometry and published annual-operation metadata only; "
            "no active configuration, runway independence, closure, declared rate, gate, "
            "taxiway, deicing, or realised throughput is inferred."
        ),
    }
    output["catalog_sha256"] = canonical_json_sha256(output)
    write_canonical_json(output_path, output)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_STATIC_RESOURCE_CATALOG_NO_FLIGHT_OUTCOMES_ACCESSED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "FAA 28-Day NASR Subscription",
        "effective_date": effective_dates[0],
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
        "members_used": ["CSV/APT_BASE.csv", "CSV/APT_RWY.csv", "CSV/APT_RWY_END.csv"],
        "outcome_columns_read": [],
        "confirmation_outcomes_accessed": False,
        "provenance": capture_provenance((Path(__file__),)),
        "claim_limit": output["claim_limit"],
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(manifest_path, manifest)
    return manifest


def load_resource_catalog(path: Path) -> dict[str, dict[str, Any]]:
    """Load and self-validate a static resource catalog by IATA code."""

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("catalog_sha256")
    body = {key: value for key, value in payload.items() if key != "catalog_sha256"}
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"resource catalog self-hash failed: {path}")
    records = payload.get("airports")
    if not isinstance(records, list) or not records:
        raise ValueError("resource catalog contains no airports")
    result = {str(record["iata"]): dict(record) for record in records}
    if len(result) != len(records):
        raise ValueError("resource catalog contains duplicate airports")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--airport-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-runway-length-ft", type=float, default=4_000.0)
    parser.add_argument("--parallel-tolerance-degrees", type=float, default=12.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_resource_catalog(
        archive_path=args.archive,
        airport_catalog_path=args.airport_catalog,
        output_path=args.output,
        manifest_path=args.manifest,
        minimum_runway_length_ft=args.minimum_runway_length_ft,
        parallel_tolerance_degrees=args.parallel_tolerance_degrees,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "airports": result["airport_count"],
                "missing": result["airports_missing_eligible_runways"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
