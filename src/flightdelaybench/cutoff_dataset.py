"""Hash-bound entry point for corrected data; legacy feature caches are not inputs.

A checksum establishes integrity, not the historical truth of supplied timestamps.
Source provenance must be reviewed separately. This contract binds that review and
rejects assumed timing, synthetic fixtures, and confirmation years in research mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .contracts import CUTOFF_HISTORY_FEATURES
from .cutoff_history import build_cutoff_history
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json


def artifact_path(root: Path, record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    if not path.is_absolute():
        path = root / path
    return path


def checked_artifact(root: Path, record: dict[str, Any]) -> Path:
    path = artifact_path(root, record)
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"artifact missing or hash/size mismatch: {path}")
    return path


def guard_parquet_years(path: Path, allowed: set[int]) -> None:
    """Inspect operating-date statistics before reading any outcome column."""
    metadata = pq.read_metadata(path)
    index = metadata.schema.names.index("FlightDate")
    for row_group in range(metadata.num_row_groups):
        stats = metadata.row_group(row_group).column(index).statistics
        if stats is None or not stats.has_min_max or stats.null_count:
            raise ValueError("complete FlightDate min/max statistics required before data access")
        low, high = pd.Timestamp(stats.min), pd.Timestamp(stats.max)
        if not set(range(low.year, high.year + 1)) <= allowed:
            raise ValueError("partition metadata includes an unauthorized operating year")


def validate_dataset_manifest(path: Path, *, allow_synthetic: bool = False) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_json_sha256(body):
        raise ValueError("invalid dataset manifest self-hash")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS_CUTOFF_DATASET" or manifest.get("target_years") != [2024]:
        raise ValueError("incomplete dataset contract or unauthorized target year")
    permitted = {"REVIEWED_OBSERVATION_TIMESTAMPS"}
    if allow_synthetic:
        permitted.add("SYNTHETIC_SOFTWARE_ONLY")
    if manifest.get("evidence_class") not in permitted:
        raise ValueError("assumed or synthetic timestamps are not corrected research evidence")
    sources = manifest.get("source_evidence", [])
    if not sources or len({source["source_id"] for source in sources}) != len(sources):
        raise ValueError("unique hash-bound source evidence is required")
    for source in sources:
        checked_artifact(path.parent, source)
        if source.get("timestamp_basis") != "observed_event_publication_and_availability":
            raise ValueError("source timestamps cannot be inferred from operating dates")
    from .cutoff_experiments import validate_new_features
    contexts = manifest.get("features_by_context", {})
    if set(contexts) != {"baseline", "induced", "boundary"}:
        raise ValueError("baseline, induced and boundary feature contracts are required")
    for features in contexts.values():
        validate_new_features(tuple(features))
    if not set(contexts["baseline"]) <= set(contexts["induced"]) <= set(contexts["boundary"]):
        raise ValueError("context ablations must retain baseline predictors")
    evidence = manifest.get("feature_evidence", {})
    source_ids = {source["source_id"] for source in sources}
    for feature in contexts["boundary"]:
        if not evidence.get(feature) or not set(evidence[feature]) <= source_ids:
            raise ValueError(f"feature lacks reviewed, hash-bound source evidence: {feature}")
    label_evidence = manifest.get("label_evidence", {})
    for task in ("cancellation", "delay"):
        if not label_evidence.get(task) or not set(label_evidence[task]) <= source_ids:
            raise ValueError(f"label availability lacks hash-bound source evidence: {task}")
    audits = manifest.get("availability_audits", [])
    if not audits:
        raise ValueError("reviewed availability audit required")
    for record in audits:
        audit = json.loads(checked_artifact(path.parent, record).read_text(encoding="utf-8"))
        if audit.get("status") != "PASS_CUTOFF_AVAILABILITY_AUDIT" or audit.get("post_cutoff_values") != 0:
            raise ValueError("availability audit did not pass")
        if set(audit.get("features", [])) != set(contexts["boundary"]):
            raise ValueError("audit does not cover the entire predictor set")
        if audit.get("source_timestamps_synthesized") is not False:
            raise ValueError("availability audit uses synthesized timestamps")
        if audit.get("label_availability_validated") is not True:
            raise ValueError("label availability was not reviewed")
    partitions = manifest.get("partitions", [])
    if not partitions or any(record.get("year") != 2024 for record in partitions):
        raise ValueError("only explicit 2024 partitions are authorized")
    for record in partitions:
        data_path = artifact_path(path.parent, record)
        # Read only operating-date metadata before hashing the payload, too.
        guard_parquet_years(data_path, {2024})
        checked_artifact(path.parent, record)
    # Bind the review to these exact partitions; a passing audit for other data
    # must never authorize a substituted dataset.
    expected_hashes = sorted(record["sha256"] for record in partitions)
    for record in audits:
        audit = json.loads(checked_artifact(path.parent, record).read_text(encoding="utf-8"))
        if sorted(audit.get("partition_sha256", [])) != expected_hashes:
            raise ValueError("availability audit belongs to different data partitions")
    return manifest


def load_dataset(path: Path, *, allow_synthetic: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = validate_dataset_manifest(path, allow_synthetic=allow_synthetic)
    frames = []
    for record in manifest["partitions"]:
        frame = pd.read_parquet(checked_artifact(path.parent, record))
        if len(frame) != record["rows"]:
            raise ValueError("partition row count mismatch")
        frames.append(frame)
    from .cutoff_experiments import validate_dataset
    result = validate_dataset(pd.concat(frames, ignore_index=True), tuple(manifest["features_by_context"]["boundary"]))
    return result, manifest


def build_history_files(*, targets: Path, observations: Path, output_dir: Path) -> dict[str, Any]:
    """Materialize small/batched histories; provenance review remains a separate gate."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite history evidence {output_dir}")
    guard_parquet_years(targets, {2024})
    guard_parquet_years(observations, set(range(1987, 2025)))
    history, audit = build_cutoff_history(pd.read_parquet(targets), pd.read_parquet(observations))
    output_dir.mkdir(parents=True)
    destination = output_dir / "history.parquet"
    history.to_parquet(destination, index=False)
    audit.update({
        "targets_sha256": sha256_file(targets), "observations_sha256": sha256_file(observations),
        "output_sha256": sha256_file(destination), "features": list(CUTOFF_HISTORY_FEATURES),
        "historical_source_authenticity_established_by_this_builder": False,
        "claim_limit": "Checks supplied times only; source review and dataset audit still required.",
    })
    audit["report_sha256"] = canonical_json_sha256(audit)
    write_canonical_json(output_dir / "audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_history_files(targets=args.targets, observations=args.observations, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
