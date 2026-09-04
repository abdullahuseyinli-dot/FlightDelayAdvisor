from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from test_cutoff_experiments import FEATURES, synthetic_dataset
from test_cutoff_history import _observations, _targets

from flightdelaybench.cutoff_dataset import (
    build_history_files,
    guard_parquet_years,
    load_dataset,
    validate_dataset_manifest,
)
from flightdelaybench.hashing import canonical_json_sha256, sha256_file, write_canonical_json


def _record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    frame = synthetic_dataset()
    partition = tmp_path / "synthetic.parquet"
    frame.to_parquet(partition, index=False)
    source = tmp_path / "synthetic_source.json"
    write_canonical_json(source, {"evidence_class": "SYNTHETIC_SOFTWARE_ONLY"})
    audit = tmp_path / "synthetic_audit.json"
    write_canonical_json(audit, {
        "status": "PASS_CUTOFF_AVAILABILITY_AUDIT", "post_cutoff_values": 0,
        "features": list(FEATURES), "source_timestamps_synthesized": False,
        "label_availability_validated": True,
        "partition_sha256": [sha256_file(partition)],
        "claim_limit": "Synthetic fixture, not real historical source evidence",
    })
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "PASS_CUTOFF_DATASET", "target_years": [2024],
        "evidence_class": "SYNTHETIC_SOFTWARE_ONLY",
        "source_evidence": [{**_record(source), "source_id": "fixture", "timestamp_basis": "observed_event_publication_and_availability"}],
        "features_by_context": {context: list(FEATURES) for context in ("baseline", "induced", "boundary")},
        "feature_evidence": {feature: ["fixture"] for feature in FEATURES},
        "label_evidence": {task: ["fixture"] for task in ("cancellation", "delay")},
        "availability_audits": [_record(audit)],
        "partitions": [{**_record(partition), "year": 2024, "rows": len(frame)}],
    }
    path = tmp_path / "manifest.json"
    _save(path, manifest)
    return path, manifest


def _save(path: Path, manifest: dict[str, Any]) -> None:
    manifest["manifest_sha256"] = canonical_json_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    write_canonical_json(path, manifest)


def test_synthetic_requires_explicit_software_test_opt_in(tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path)
    with pytest.raises(ValueError, match="synthetic timestamps"):
        load_dataset(path)
    frame, manifest = load_dataset(path, allow_synthetic=True)
    assert len(frame) == len(synthetic_dataset())
    assert manifest["evidence_class"] == "SYNTHETIC_SOFTWARE_ONLY"


def test_source_tampering_is_rejected(tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path)
    write_canonical_json(tmp_path / "synthetic_source.json", {"modified": True})
    with pytest.raises(ValueError, match="hash/size mismatch"):
        load_dataset(path, allow_synthetic=True)


def test_missing_feature_lineage_is_rejected(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["feature_evidence"].pop("Distance")
    _save(path, manifest)
    with pytest.raises(ValueError, match="lacks reviewed"):
        validate_dataset_manifest(path, allow_synthetic=True)


def test_year_guard_precedes_outcome_reads(tmp_path: Path) -> None:
    partition = tmp_path / "forbidden_synthetic.parquet"
    pd.DataFrame({"FlightDate": pd.to_datetime(["2026-01-01"]), "Cancelled": [0]}).to_parquet(partition)
    with pytest.raises(ValueError, match="unauthorized operating year"):
        guard_parquet_years(partition, {2024})


def test_stale_availability_audit_cannot_authorize_other_data(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    frame = synthetic_dataset()
    frame["Distance"] += 1
    partition = tmp_path / "synthetic.parquet"
    frame.to_parquet(partition, index=False)
    manifest["partitions"][0].update(_record(partition))
    _save(path, manifest)
    with pytest.raises(ValueError, match="different data partitions"):
        load_dataset(path, allow_synthetic=True)


def test_history_file_builder_is_create_only_and_discloses_limits(tmp_path: Path) -> None:
    targets, observations = tmp_path / "targets.parquet", tmp_path / "observations.parquet"
    _targets().to_parquet(targets, index=False)
    _observations().to_parquet(observations, index=False)
    output = tmp_path / "history"
    report = build_history_files(targets=targets, observations=observations, output_dir=output)
    assert report["status"] == "PASS_OBSERVATION_TIME_HISTORY"
    assert report["historical_source_authenticity_established_by_this_builder"] is False
    assert (output / "history.parquet").is_file()
    with pytest.raises(FileExistsError):
        build_history_files(targets=targets, observations=observations, output_dir=output)
