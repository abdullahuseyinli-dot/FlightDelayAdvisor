from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from flightdelaybench.flare_resource_catalog import build_resource_catalog, load_resource_catalog


def _write_csv(archive: zipfile.ZipFile, name: str, frame: pd.DataFrame) -> None:
    archive.writestr(name, frame.to_csv(index=False))


def test_resource_catalog_preserves_parallel_runways_and_missing_operations(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "nasr.zip"
    base = pd.DataFrame(
        {
            "EFF_DATE": ["2024-10-03"],
            "ARPT_ID": ["AAA"],
            "ICAO_ID": ["KAAA"],
            "ANNUAL_OPS_DATE": [None],
            "COMMERCIAL_OPS": [None],
            "COMMUTER_OPS": [None],
            "AIR_TAXI_OPS": [None],
            "LOCAL_OPS": [None],
            "ITNRNT_OPS": [None],
            "MIL_ACFT_OPS": [None],
        }
    )
    runway = pd.DataFrame(
        {
            "ARPT_ID": ["AAA", "AAA"],
            "RWY_ID": ["09L/27R", "09R/27L"],
            "RWY_LEN": [9_000, 8_500],
            "RWY_WIDTH": [150, 150],
            "SURFACE_TYPE_CODE": ["ASPH", "ASPH"],
            "RWY_LGT_CODE": ["HIGH", "HIGH"],
        }
    )
    ends = pd.DataFrame(
        {
            "ARPT_ID": ["AAA"] * 4,
            "RWY_ID": ["09L/27R", "09L/27R", "09R/27L", "09R/27L"],
            "RWY_END_ID": ["09L", "27R", "09R", "27L"],
            "TRUE_ALIGNMENT": [90.0, 270.0, 91.0, 271.0],
            "ILS_TYPE": ["ILS", "ILS", None, None],
            "APCH_LGT_SYSTEM_CODE": [None] * 4,
            "LAT_DECIMAL": [1.0, 1.1, 1.01, 1.11],
            "LONG_DECIMAL": [2.0, 2.1, 2.01, 2.11],
        }
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        _write_csv(archive, "CSV/APT_BASE.csv", base)
        _write_csv(archive, "CSV/APT_RWY.csv", runway)
        _write_csv(archive, "CSV/APT_RWY_END.csv", ends)
    airport_catalog = tmp_path / "airports.json"
    airport_catalog.write_text(json.dumps({"airports": [{"iata": "AAA"}]}))
    output = tmp_path / "resources.json"
    manifest = tmp_path / "manifest.json"
    build_resource_catalog(
        archive_path=archive_path,
        airport_catalog_path=airport_catalog,
        output_path=output,
        manifest_path=manifest,
    )
    record = load_resource_catalog(output)["AAA"]
    assert record["physical_runway_count"] == 2
    assert record["eligible_runway_end_count"] == 4
    assert record["orientation_family_count"] == 1
    assert record["max_parallel_runways"] == 2
    assert record["ils_end_fraction"] == 0.5
    assert record["annual_operations"] is None
