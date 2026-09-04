from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from flightdelaybench.flare_boundary_validation import (
    BOUNDARY_METHOD,
    PARTIAL_STATUS,
    validate_boundary_feature_manifest,
)
from flightdelaybench.flare_capacity_contracts import CAPACITY_ALL_FEATURES
from flightdelaybench.hashing import canonical_json_sha256, sha256_file


def _self_hashed(path: Path, payload: dict[str, object]) -> None:
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def test_partial_boundary_feature_validation_preserves_target_ids(tmp_path: Path) -> None:
    target_dir = tmp_path / "targets"
    target_path = target_dir / "year=2024" / "month=01.parquet"
    target_path.parent.mkdir(parents=True)
    pd.DataFrame({"sample_id": ["a", "b"]}).to_parquet(target_path, index=False)

    values = {name: [0.5, 0.5] for name in CAPACITY_ALL_FEATURES}
    for side in ("origin", "dest"):
        for scenario in ("constrained", "marginal", "good"):
            values[f"ccrth_{side}_capacity_scenario_{scenario}_probability"] = [
                1.0 / 3.0,
                1.0 / 3.0,
            ]
    feature_path = tmp_path / "features.parquet"
    pd.DataFrame({"sample_id": ["a", "b"], **values}).to_parquet(feature_path, index=False)
    context_path = tmp_path / "context.json"
    _self_hashed(
        context_path,
        {
            "outcome_columns_read": [],
            "tail_number_read": False,
            "confirmation_outcomes_accessed": False,
        },
    )
    catalog_path = tmp_path / "airports.json"
    catalog_path.write_text("{}", encoding="utf-8")
    frontier_path = tmp_path / "frontier.parquet"
    pd.DataFrame({"airport": ["AAA"]}).to_parquet(frontier_path, index=False)
    safe_input = tmp_path / "schedule.parquet"
    pd.DataFrame({"sample_id": ["a", "b"]}).to_parquet(safe_input, index=False)

    manifest_path = tmp_path / "manifest.json"
    manifest: dict[str, object] = {
        "status": PARTIAL_STATUS,
        "method": BOUNDARY_METHOD,
        "artifact_mode": "features-only",
        "features": list(CAPACITY_ALL_FEATURES),
        "target_years": [2024],
        "target_months": [1],
        "rows": 2,
        "outcome_columns_read": [],
        "target_tail_number_read": False,
        "confirmation_outcomes_accessed": False,
        "scored_population": "unchanged frozen top100-to-top100 sample ids",
        "context_schedule_manifest": {
            **_record(context_path),
            "self_hash": json.loads(context_path.read_text())["manifest_sha256"],
        },
        "context_airport_catalog": _record(catalog_path),
        "schedule_inputs": [{**_record(safe_input), "columns_read": ["sample_id", "FlightDate"]}],
        "weather_feature_manifest": _record(context_path),
        "resource_catalog_manifest": _record(context_path),
        "rotation_manifest": _record(context_path),
        "frontier_outputs": [
            {
                **_record(frontier_path),
                "history_year": 2023,
                "target_year": 2024,
            }
        ],
        "outputs": [
            {
                "year": 2024,
                "month": 1,
                "features": _record(feature_path),
                "flight_nodes": None,
                "resource_nodes": None,
                "incidence_edges": None,
                "rotation_edges": None,
            }
        ],
        "diagnostics": [
            {
                "year": 2024,
                "month": 1,
                "resource_airport_count": 100,
                "target_events": 4,
                "context_events_after_resource_filter": 5,
                "context_events_before_resource_filter": 6,
                "rotation_context_only_edges": 1,
                "rotation_context_only_predecessor_states": 1,
            }
        ],
        "provenance": {"source_files": []},
    }
    _self_hashed(manifest_path, manifest)

    report = validate_boundary_feature_manifest(
        manifest_path,
        target_census_dir=target_dir,
        require_complete=False,
    )
    assert report["target_id_set_match"] is True
    assert report["context_only_rotation_edges"] == 1
