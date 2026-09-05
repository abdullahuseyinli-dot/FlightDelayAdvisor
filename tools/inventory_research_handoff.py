"""Inventory selected handoff artifacts without downloading or decoding outcome/model data.

Only named metadata sections and their explicitly listed files are inspected. The
default 8-GiB hash budget prioritizes the Git-index LFS allowlist, then raw BTS,
then historical derived artifacts. Unhashed files are never marked verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

OLD_DATA_ROOT = "D:/FlightDelayAdvisorResearchData"
MAX_METADATA_BYTES = 16 * 1024 * 1024
LFS_ALLOWLIST = frozenset({
    "data/processed/bts_delay_2010_2024_balanced_research_weather.parquet",
    "models/catboost_delay15_calibrated.joblib",
    "models/lgbm_cancel_calibrated.joblib",
    *(f"data/raw_2025/on_time_2025_{month:02d}.zip" for month in range(1, 13)),
})
RAW_MANIFESTS = {
    **{f"manifests/raw_bts_census_{year}.json": ("records",) for year in range(2018, 2025)},
    "manifests/raw_bts_2025.json": ("records",),
}
SELECTED_METADATA = {
    **RAW_MANIFESTS,
    "manifests/census_top100_2018_2025_v2.json": ("outputs",),
    "manifests/census_recent_prior_day_v2.json": ("outputs",),
    "manifests/census_flight_recent_v2_duckdb.json": ("outputs",),
    "manifests/census_graph_clsgmp_v1.json": ("outputs",),
    "manifests/forecast24_openmeteo_gfs_v1.json": ("requests", "airport_catalog"),
    "manifests/forecast24_features_v1.json": ("outputs",),
    "manifests/flare24_openmeteo_gfs_day2_v1.json": ("requests",),
    "manifests/flare24_weather_cube_v2.json": ("outputs",),
    "manifests/flare24_features_v1.json": ("outputs",),
    "manifests/flare24_rotation_features_v5.json": ("outputs", "models"),
    "manifests/flare24_ccrth_features_v3.json": ("outputs", "frontier_outputs"),
    "manifests/flare24_boundary_context_v1.json": ("outputs",),
    "manifests/flare24_boundary_rotations_v1.json": ("models",),
    "manifests/flare24_boundary_pot_v1.json": ("outputs", "frontier_outputs"),
    "reports/experiments/flare24_2025_audit_v1_recovered.json": (
        "frozen_method", "aggregate_refit", "aggregate_forecast_artifacts", "prediction_artifacts",
    ),
    "reports/experiments/flare24_ccrth_metastack_v1.json": (
        "model_artifacts", "retrospective_prediction_artifacts", "q4_input",
    ),
    "reports/experiments/flare24_boundary_pot_v6_recovered.json": (
        "model_artifacts", "calibration_artifacts", "raw_selection_prediction_artifacts",
        "selection_crossfit_prediction_artifact", "retrospective_prediction_artifacts",
    ),
}
POINTER = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\r?\n"
    rb"oid sha256:([0-9a-f]{64})\r?\nsize ([0-9]+)\r?\n?"
)
YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
MODEL_SUFFIXES = {".joblib", ".cbm", ".pkl", ".pickle", ".pt", ".pth"}


def _git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], input=input_bytes,
        check=True, capture_output=True,
    ).stdout


def parse_pointer(payload: bytes) -> dict[str, Any] | None:
    match = POINTER.fullmatch(payload)
    if match is None:
        return None
    return {"sha256": match[1].decode("ascii"), "bytes": int(match[2])}


def indexed_lfs(root: Path) -> list[dict[str, Any]]:
    """Read pointer expectations from the index; never treat ls-files '*' as proof."""
    names = _git(root, "ls-files", "-z")
    attributes = _git(root, "check-attr", "--cached", "-z", "--stdin", "filter", input_bytes=names)
    fields = attributes.split(b"\0")
    result = []
    for offset in range(0, len(fields) - 1, 3):
        name, _, value = fields[offset:offset + 3]
        if value != b"lfs":
            continue
        relative = name.decode("utf-8")
        record: dict[str, Any] = {"recorded_path": relative, "expectation_basis": "git_index_pointer"}
        if relative not in LFS_ALLOWLIST:
            record["state"] = "EXCLUDED_NOT_IN_HISTORICAL_LFS_ALLOWLIST"
        else:
            size = int(_git(root, "cat-file", "-s", f":{relative}"))
            pointer = parse_pointer(_git(root, "cat-file", "blob", f":{relative}")) if size <= 1024 else None
            if pointer is None:
                record["state"] = "INVALID_GIT_INDEX_LFS_POINTER"
            else:
                record.update(expected_sha256=pointer["sha256"], expected_bytes=pointer["bytes"])
        result.append(record)
    return sorted(result, key=lambda record: record["recorded_path"])


def _normal(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def _has_future_year(value: str) -> bool:
    return any(int(year) > 2025 for year in YEAR.findall(value))


def historical_reference(record: dict[str, Any]) -> bool:
    """Reject future paths/operating-date metadata before stat or payload reads."""
    path = str(record.get("path", record.get("local_path", "")))
    if _has_future_year(path):
        return False
    for key in ("year", "target_year", "history_year", "first_date", "last_date", "start_date", "end_date"):
        if key in record and _has_future_year(str(record[key])):
            return False
    return True


def relocation_map(data_root: Path, values: list[str]) -> list[tuple[str, Path]]:
    mappings = {OLD_DATA_ROOT: data_root.resolve()}
    for value in values:
        if "=" not in value:
            raise ValueError("--relocate requires RECORDED_PREFIX=LOCAL_PREFIX")
        source, target = value.split("=", 1)
        source = _normal(source)
        if not source or not target or ".." in PurePosixPath(source).parts:
            raise ValueError("relocation prefixes must be nonempty and contain no '..'")
        if source.casefold() == OLD_DATA_ROOT.casefold() and Path(target).resolve() != data_root.resolve():
            raise ValueError("relocation conflicts with the explicit --data-root mapping")
        if any(key.casefold() == source.casefold() for key in mappings):
            if mappings.get(source) != Path(target).resolve():
                raise ValueError("conflicting relocation prefixes")
        mappings[source] = Path(target).resolve()
    return sorted(mappings.items(), key=lambda item: (-len(item[0]), item[0]))


def resolve_reference(recorded: str, root: Path, mappings: list[tuple[str, Path]]) -> tuple[Path | None, str]:
    normalized = _normal(recorded)
    if ".." in PurePosixPath(normalized).parts or "://" in normalized:
        return None, "EXCLUDED_UNSAFE_PATH"
    for prefix, target in mappings:
        if normalized.casefold() == prefix.casefold() or normalized.casefold().startswith(prefix.casefold() + "/"):
            path = target / normalized[len(prefix):].lstrip("/")
            boundary = target
            basis = f"explicit_relocation:{prefix}"
            break
    else:
        if PurePosixPath(normalized).is_absolute() or re.match(r"^[A-Za-z]:", normalized):
            return None, "UNMAPPED_ABSOLUTE_PATH"
        path = root / normalized
        boundary = root
        basis = "repository_relative"
    if not path.resolve().is_relative_to(boundary.resolve()):
        return None, "EXCLUDED_PATH_ESCAPES_DECLARED_ROOT"
    if _has_future_year(_normal(str(path.relative_to(boundary)))):
        return None, "EXCLUDED_FUTURE_OUTCOME_PATH"
    return path, basis


@dataclass
class HashBudget:
    limit: int
    used: int = 0
    cache: dict[tuple[str, int, int], str] = field(default_factory=dict)

    def digest(self, path: Path) -> str | None:
        before = path.stat()
        identity = (str(path.resolve()), before.st_size, before.st_mtime_ns)
        if identity in self.cache:
            return self.cache[identity]
        if self.used + before.st_size > self.limit:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                self.used += len(chunk)
                if self.used > self.limit:
                    raise OSError("file grew beyond the remaining hash budget")
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError("file changed while hashing; no integrity result accepted")
        self.cache[identity] = digest.hexdigest()
        return digest.hexdigest()


def inspect_file(path: Path, expected: dict[str, Any], budget: HashBudget) -> dict[str, Any]:
    """Inspect bytes only; this function never imports a table/model deserializer."""
    result: dict[str, Any] = {"resolved_path": str(path), "byte_integrity_verified": False}
    try:
        if not path.is_file():
            result["state"] = "MISSING" if not path.exists() else "NOT_A_FILE"
            return result
        size = path.stat().st_size
        result["observed_bytes"] = size
        if size <= 1024:
            pointer = parse_pointer(path.read_bytes())
            if pointer is not None:
                result.update(state="LFS_POINTER_ONLY", working_pointer=pointer)
                if any(expected.get(key) is not None and expected[key] != pointer[key] for key in ("bytes", "sha256")):
                    result["state"] = "LFS_POINTER_EXPECTATION_MISMATCH"
                return result
        if expected.get("bytes") is not None and size != expected["bytes"]:
            result["state"] = "SIZE_MISMATCH"
            return result
        digest = budget.digest(path)
        if digest is None:
            result["state"] = "MATERIALIZED_UNVERIFIED_HASH_BUDGET"
        else:
            result["observed_sha256"] = digest
            if expected.get("sha256") is None:
                result["state"] = "MATERIALIZED_HASHED_WITHOUT_EXPECTED_HASH"
            elif digest != expected["sha256"]:
                result["state"] = "HASH_MISMATCH"
            else:
                result.update(state="MATERIALIZED_VERIFIED_SHA256", byte_integrity_verified=True)
    except OSError as error:
        result.update(state="IO_ERROR", error=str(error))
    return result


def artifact_references(value: Any, location: str, inherited: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Extract path metadata, excluding score tables and model contents from output."""
    context = dict(inherited or {})
    records = []
    if isinstance(value, dict):
        for key in ("year", "month", "target_year", "history_year", "first_date", "last_date"):
            if key in value:
                context[key] = value[key]
        path = value.get("path", value.get("local_path"))
        if isinstance(path, str):
            context.update({key: value[key] for key in ("bytes", "sha256", "rows", "url", "csv_member") if key in value})
            records.append({**context, "path": path, "metadata_location": location})
        else:
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    records.extend(artifact_references(child, f"{location}.{key}", context))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(artifact_references(child, f"{location}[{index}]", context))
    return records


def _category(record: dict[str, Any], metadata_path: str) -> str:
    path = record["path"]
    if metadata_path in RAW_MANIFESTS:
        return "PUBLIC_BTS_RAW_SOURCE"
    if "openmeteo" in metadata_path:
        return "PUBLIC_WEATHER_SOURCE_LICENSE_REVIEW_REQUIRED"
    if PurePosixPath(path).suffix.lower() in MODEL_SUFFIXES:
        return "HISTORICAL_MODEL_OR_CALIBRATOR"
    return "HISTORICAL_DERIVED_ARTIFACT"


def build_inventory(root: Path, data_root: Path, mappings: list[tuple[str, Path]], max_hash_bytes: int) -> dict[str, Any]:
    if max_hash_bytes < 0:
        raise ValueError("hash budget must be nonnegative")
    root = root.resolve()
    budget = HashBudget(max_hash_bytes)
    lfs = indexed_lfs(root)
    for record in lfs:
        if "state" in record:
            continue
        path, basis = resolve_reference(record["recorded_path"], root, [])
        record["resolution_basis"] = basis
        if path is None:
            record["state"] = basis
        else:
            record.update(inspect_file(path, {"bytes": record["expected_bytes"], "sha256": record["expected_sha256"]}, budget))
    metadata_records = []
    artifacts = []
    for relative, sections in SELECTED_METADATA.items():
        source = root / relative
        metadata: dict[str, Any] = {"path": relative, "selected_sections": list(sections)}
        metadata_records.append(metadata)
        if not source.is_file():
            metadata["state"] = "MISSING_METADATA"
            continue
        if source.stat().st_size > MAX_METADATA_BYTES:
            metadata["state"] = "EXCLUDED_METADATA_SIZE_LIMIT"
            continue
        payload = source.read_bytes()
        metadata.update(bytes=len(payload), file_sha256=hashlib.sha256(payload).hexdigest())
        try:
            contents = json.loads(payload)
            if not isinstance(contents, dict):
                raise ValueError("metadata must be a JSON object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            metadata.update(state="INVALID_METADATA", error=str(error))
            continue
        metadata.update(
            state="METADATA_READ_ONLY", declared_status=contents.get("status"),
            declared_canonical_self_hash=contents.get("manifest_sha256", contents.get("report_sha256")),
            canonical_self_hash_revalidated=False,
            absent_selected_sections=[section for section in sections if section not in contents],
        )
        count_before = len(artifacts)
        for section in sections:
            for reference in artifact_references(contents.get(section), section):
                record = {
                    "metadata_path": relative, "category": _category(reference, relative),
                    "recorded_path": reference["path"], "metadata_location": reference["metadata_location"],
                    "expected_bytes": reference.get("bytes"), "expected_sha256": reference.get("sha256"),
                    "byte_integrity_verified": False,
                    **{key: reference[key] for key in ("year", "month", "target_year", "history_year", "rows", "url") if key in reference},
                }
                artifacts.append(record)
                if not historical_reference(reference):
                    record["state"] = "EXCLUDED_FUTURE_OUTCOME_REFERENCE"
                    continue
                path, basis = resolve_reference(reference["path"], root, mappings)
                record["resolution_basis"] = basis
                if path is None:
                    record["state"] = basis
                else:
                    record.update(inspect_file(path, reference, budget))
                record["reconstruction_status"] = (
                    "PRESERVED_BYTES_VERIFIED_LEGACY_ONLY" if record["byte_integrity_verified"]
                    else "PRESENCE_ONLY_REQUIRES_HASH_VERIFICATION" if str(record["state"]).startswith("MATERIALIZED")
                    else "REACQUIRE_OR_TRANSFER_VERSIONED_SOURCE" if record["category"].startswith("PUBLIC_")
                    else "TRANSFER_OR_REBUILD_IN_NEW_GENERATION"
                )
        metadata["artifact_reference_count"] = len(artifacts) - count_before
    raw = [record for record in artifacts if record["category"] == "PUBLIC_BTS_RAW_SOURCE"]
    raw_coverage = []
    for year in range(2018, 2026):
        annual = [record for record in raw if record.get("year") == year]
        months = [record.get("month") for record in annual]
        raw_coverage.append({
            "year": year, "expected_months": list(range(1, 13)), "manifest_record_count": len(annual),
            "unlisted_months": [month for month in range(1, 13) if month not in months],
            "duplicate_months": sorted(month for month, count in Counter(months).items() if count > 1),
            "states": dict(sorted(Counter(record["state"] for record in annual).items())),
            "expected_bytes": sum(record["expected_bytes"] or 0 for record in annual),
            "verified_bytes": sum(record["observed_bytes"] for record in annual if record["byte_integrity_verified"]),
        })
    return {
        "schema_version": 1, "status": "INVENTORY_COMPLETE_GAPS_RECORDED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository_root": str(root), "data_root": str(data_root.resolve()),
        "git_head": _git(root, "rev-parse", "HEAD").decode().strip(),
        "git_branch": _git(root, "branch", "--show-current").decode().strip(),
        "relocations": [{"recorded_prefix": old, "local_prefix": str(new)} for old, new in mappings],
        "scope": {
            "selected_metadata": list(SELECTED_METADATA), "lfs_payload_allowlist": sorted(LFS_ALLOWLIST),
            "external_traversal": "none; only explicitly listed artifact paths are probed",
            "hash_order": "allowlisted index LFS, 2018-2025 raw sources, selected derived/model references",
            "metadata_file_bytes_are_sha256_hashed_outside_payload_budget": True,
            "hash_budget_bytes": budget.limit, "payload_bytes_hashed": budget.used,
            "table_rows_or_model_contents_decoded": False, "confirmation_outcomes_accessed": False,
            "inventory_is_exhaustive_for_external_workspace": False,
            "schema_rows_ids_and_date_coverage_revalidated": False,
            "duplicate_references_are_retained_but_hash_reads_are_cached": True,
            "unselected_material": "Other generations, failures, confirmation locks, unlisted sources and run directories are preserved and uninspected.",
        },
        "data_root_disk": {"free_bytes": shutil.disk_usage(data_root).free, "total_bytes": shutil.disk_usage(data_root).total} if data_root.exists() else None,
        "metadata": metadata_records, "git_lfs": lfs, "artifact_references": artifacts,
        "raw_bts_year_coverage": raw_coverage,
        "summary": {
            "lfs_states": dict(sorted(Counter(record["state"] for record in lfs).items())),
            "artifact_states": dict(sorted(Counter(record["state"] for record in artifacts).items())),
            "categories": dict(sorted(Counter(record["category"] for record in artifacts).items())),
            "unique_artifact_paths": len({record.get("resolved_path", record["recorded_path"]) for record in artifacts}),
        },
        "strict_t24_gate": {
            "status": "BLOCKED_AUTHENTIC_TIMESTAMP_EVIDENCE_NOT_ESTABLISHED_BY_INVENTORY",
            "inventory_can_satisfy_source_timing_gate": False,
            "missing_evidence": [
                "Authentic historical event, publication and consumer-availability timestamps and revision policy",
                "Authentic cancellation/delay label availability at training and stopping cutoffs",
                "Reviewed advance schedule snapshot equivalence and feature-level source lineage",
            ],
            "claim_limit": "Hashes prove byte identity only. Present historical caches/models remain retrospective proxies and cannot be relabelled corrected T-24 inputs. No corrected real-data trial is authorized by this report.",
        },
    }


def write_create_only(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, required=True, help="Explicit replacement for D:/FlightDelayAdvisorResearchData; recorded separately")
    parser.add_argument("--relocate", action="append", default=[], metavar="RECORDED_PREFIX=LOCAL_PREFIX", help="Additional explicit prefix mapping; no implicit drive searches")
    parser.add_argument("--max-hash-bytes", type=int, default=8 * 1024 ** 3)
    parser.add_argument("--output", type=Path, required=True, help="Create-only JSON output")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("inventory output is create-only")
    report = build_inventory(args.repo_root, args.data_root, relocation_map(args.data_root, args.relocate), args.max_hash_bytes)
    write_create_only(args.output, report)
    print(json.dumps({"status": report["status"], "summary": report["summary"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
