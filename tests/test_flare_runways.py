from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from flightdelaybench.flare_runways import build_runway_catalog, load_runway_headings


def _write_csv(archive: zipfile.ZipFile, name: str, frame: pd.DataFrame) -> None:
    archive.writestr(name, frame.to_csv(index=False))


def test_build_and_load_faa_runway_catalog(tmp_path: Path) -> None:
    archive_path = tmp_path / "nasr.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        _write_csv(
            archive,
            "CSV/APT_BASE.csv",
            pd.DataFrame({"ARPT_ID": ["AAA"], "ICAO_ID": ["KAAA"]}),
        )
        _write_csv(
            archive,
            "CSV/APT_RWY.csv",
            pd.DataFrame(
                {
                    "ARPT_ID": ["AAA", "AAA"],
                    "RWY_ID": ["09/27", "01/19"],
                    "RWY_LEN": [8_000, 3_000],
                }
            ),
        )
        _write_csv(
            archive,
            "CSV/APT_RWY_END.csv",
            pd.DataFrame(
                {
                    "ARPT_ID": ["AAA", "AAA", "AAA"],
                    "RWY_ID": ["09/27", "09/27", "01/19"],
                    "RWY_END_ID": ["09", "27", "01"],
                    "TRUE_ALIGNMENT": [0.0, 271.2, 10.0],
                }
            ),
        )
    airport_catalog = tmp_path / "airports.json"
    airport_catalog.write_text(
        json.dumps({"airports": [{"iata": "AAA"}]}), encoding="utf-8"
    )
    output = tmp_path / "runways.json"
    manifest = tmp_path / "manifest.json"
    result = build_runway_catalog(
        archive_path=archive_path,
        airport_catalog_path=airport_catalog,
        output_path=output,
        manifest_path=manifest,
    )
    assert result["airports_missing_eligible_runways"] == []
    assert load_runway_headings(output) == {"AAA": (271.2, 360.0)}
