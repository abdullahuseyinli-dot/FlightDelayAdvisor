from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.census_recent import (
    attach_census_recent_features,
    build_flight_recent_tables,
)
from flightdelaybench.census_recent_validation import validate_flight_recent_features
from flightdelaybench.hashing import canonical_json_sha256, sha256_file, write_canonical_json
from flightdelaybench.recent import attach_recent_features, build_recent_feature_tables


def _write_source(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    source = pd.DataFrame(
        {
            "sample_id": ["d1-a", "d1-b", "d2-a", "d2-b", "d3-a"],
            "Year": [2020] * 5,
            "FlightDate": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"]
            ),
            "Route": ["A_B", "A_C", "A_B", "B_A", "A_B"],
            "Reporting_Airline": ["XX", "XX", "XX", "YY", "XX"],
            "ScheduledFlightId": ["XX_1", "XX_2", "XX_1", "YY_3", "XX_1"],
            "Origin": ["A", "A", "A", "B", "A"],
            "Dest": ["B", "C", "B", "A", "B"],
            "ArrDel15": [1.0, 0.0, 0.0, np.nan, 1.0],
            "Cancelled": [0, 0, 0, 1, 0],
            "delay_label_observed": [1, 1, 1, 0, 1],
        }
    )
    source_path = tmp_path / "data" / "year=2020.parquet"
    source_path.parent.mkdir(parents=True)
    source.to_parquet(source_path, index=False)
    manifest: dict[str, object] = {
        "feature_variant": "unit_test",
        "outputs": [
            {
                "path": "data/year=2020.parquet",
                "rows": len(source),
                "sha256": sha256_file(source_path),
            }
        ],
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = tmp_path / "manifests" / "features.json"
    write_canonical_json(manifest_path, manifest)
    return manifest_path, source


def test_recent_builder_excludes_target_day_and_attachment_preserves_rows(tmp_path: Path) -> None:
    source_manifest, source = _write_source(tmp_path)
    output_dir = tmp_path / "recent"
    output_manifest = tmp_path / "manifests" / "recent.json"
    manifest = build_recent_feature_tables(
        feature_manifest=source_manifest,
        output_dir=output_dir,
        output_manifest=output_manifest,
    )
    hash_payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    declared_hash = hash_payload.pop("manifest_sha256")
    assert canonical_json_sha256(hash_payload) == declared_hash
    assert manifest["target_day_outcomes_excluded"] is True

    route = pd.read_parquet(output_dir / "level=route" / "year=2020.parquet")
    day_two = route.loc[
        route["Route"].eq("A_B") & route["FlightDate"].eq(pd.Timestamp("2020-01-02"))
    ].iloc[0]
    # The A_B target-day outcome is on time.  A closed-left feature can only see
    # its delayed flight from January 1.  With alpha=50 and a global prior of 0.5,
    # the empirical-Bayes result is (1 + 50*0.5) / (1 + 50).
    assert day_two["recent_route_delay_rate_7d"] == pytest.approx(26 / 51)

    to_attach = source.copy()
    for level in ("global", "route", "airline", "origin", "dest"):
        to_attach[f"prior_{level}_delay_rate"] = 0.2
        to_attach[f"prior_{level}_cancel_rate"] = 0.01
    attached = attach_recent_features(to_attach, recent_dir=output_dir)
    assert attached["sample_id"].tolist() == source["sample_id"].tolist()
    assert attached.loc[0, "recent_global_delay_rate_7d"] == pytest.approx(0.2)
    assert attached.loc[2, "recent_global_delay_rate_7d"] == pytest.approx(0.5)
    assert attached.loc[2, "recent_route_delay_rate_7d"] == pytest.approx(26 / 51)
    assert "Origin" in attached and "Dest" in attached
    assert np.isfinite(
        attached.filter(regex=r"^recent_").to_numpy(dtype=np.float64)
    ).all()


def test_census_attachment_uses_recent_global_fallback_and_flight_history(
    tmp_path: Path,
) -> None:
    source_manifest, source = _write_source(tmp_path)
    common_dir = tmp_path / "recent"
    build_recent_feature_tables(
        feature_manifest=source_manifest,
        output_dir=common_dir,
        output_manifest=tmp_path / "manifests" / "recent.json",
    )
    flight_dir = tmp_path / "flight_recent"
    build_flight_recent_tables(
        feature_manifest=source_manifest,
        output_dir=flight_dir,
        output_manifest=tmp_path / "manifests" / "flight_recent.json",
    )

    attached = attach_census_recent_features(
        source,
        recent_dir=common_dir,
        flight_recent_dir=flight_dir,
    )

    assert attached["sample_id"].tolist() == source["sample_id"].tolist()
    assert attached.loc[0, "recent_global_delay_rate_7d"] == pytest.approx(0.20)
    assert attached.loc[0, "recent_flight_delay_rate_7d"] == pytest.approx(0.20)
    assert attached.loc[2, "recent_flight_count_log1p_7d"] == pytest.approx(np.log1p(1))
    assert np.isfinite(
        attached.filter(regex=r"^recent_").to_numpy(dtype=np.float64)
    ).all()

    validation = validate_flight_recent_features(
        manifest_path=tmp_path / "manifests" / "flight_recent.json",
        report_path=tmp_path / "flight_recent_validation.json",
        check_dates=("2020-01-03",),
    )
    assert validation["status"] == "PASS"
    assert len(validation["independent_cutoff_checks"]) == 3


def test_duckdb_flight_history_matches_pandas_reference(tmp_path: Path) -> None:
    source_manifest, _ = _write_source(tmp_path)
    pandas_dir = tmp_path / "pandas_flight"
    duckdb_dir = tmp_path / "duckdb_flight"
    build_flight_recent_tables(
        feature_manifest=source_manifest,
        output_dir=pandas_dir,
        output_manifest=tmp_path / "manifests" / "pandas_flight.json",
        backend="pandas",
    )
    build_flight_recent_tables(
        feature_manifest=source_manifest,
        output_dir=duckdb_dir,
        output_manifest=tmp_path / "manifests" / "duckdb_flight.json",
        backend="duckdb",
        duckdb_memory_limit="512MB",
    )
    pandas_table = pd.read_parquet(pandas_dir / "level=flight" / "year=2020.parquet")
    duckdb_table = pd.read_parquet(duckdb_dir / "level=flight" / "year=2020.parquet")
    keys = ["FlightDate", "ScheduledFlightId"]
    pandas_table = pandas_table.sort_values(keys).reset_index(drop=True)
    duckdb_table = duckdb_table.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        pandas_table,
        duckdb_table,
        check_dtype=False,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )
