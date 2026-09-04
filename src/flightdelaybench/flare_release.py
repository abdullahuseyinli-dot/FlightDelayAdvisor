"""Build a create-only FLARE-24 release-candidate evidence ledger."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from . import __version__
from .hashing import canonical_json_sha256, sha256_file, write_canonical_json

CORE_FILES = (
    "README.md",
    "CHANGELOG.md",
    "CITATION.cff",
    ".zenodo.json",
    "pyproject.toml",
    "uv.lock",
    "configs/flare24_v1.toml",
    "configs/flare24_boundary_pot_v1.toml",
    "configs/flare24_boundary_rotation_v1.toml",
    "docs/DATA_CARD.md",
    "docs/BCPOTR_ACCEPTANCE_GATES.md",
    "docs/BCPOTR_METHOD.md",
    "docs/BCPOTR_REPRODUCIBILITY.md",
    "docs/BCPOTR_RESULTS.md",
    "docs/FLARE24_METHOD.md",
    "docs/FLARE24_REPRODUCIBILITY.md",
    "docs/FLARE24_TECHNICAL_REPORT.md",
    "docs/LITERATURE_AND_NOVELTY.md",
    "docs/MODEL_CARD.md",
    "docs/RESULTS.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/RELEASE_NOTES_FLARE24_RC1.md",
    "reports/experiments/flare24_2024_selection_v2.json",
    "manifests/flare24_method_lock_v1.json",
    "reports/experiments/flare24_2025_audit_v1_recovered.json",
    "reports/experiments/flare24_2025_nested_ablation_v1.json",
    "manifests/flare24_publication_bundle_v2.json",
    "manifests/confirmation_lock_v1.json",
    "reports/validation/flare24_2024_selection_v2.json",
    "reports/validation/flare24_method_lock_v1.json",
    "reports/validation/flare24_2025_audit_v1_recovered.json",
    "reports/validation/flare24_2025_nested_ablation_v1.json",
    "reports/validation/flare24_publication_bundle_v2.json",
    "reports/validation/confirmation_lock_v1.json",
    "manifests/flare24_boundary_context_v1.json",
    "manifests/flare24_boundary_rotations_v1.json",
    "manifests/flare24_boundary_pot_rotation_v1.json",
    "reports/validation/flare24_boundary_rotations_v1.json",
    "reports/validation/flare24_boundary_rotation_smoke_v1.json",
    "reports/validation/flare24_boundary_pot_rotation_v1.json",
    "manifests/failures/flare24_boundary_pot_study_v6_saturated_contrast.json",
    "reports/experiments/flare24_boundary_pot_v6_recovered.json",
    "reports/validation/flare24_boundary_pot_study_v6_recovered_v2.json",
    "manifests/flare24_boundary_publication_assets_v6_recovered_v2.json",
    "reports/validation/flare24_boundary_publication_assets_v6_recovered_v2.json",
    "manifests/failures/flare24_release_candidate_v1_superseded_by_bcpotr.json",
    "manifests/failures/flare24_release_candidate_v2_superseded_metadata_review.json",
    "reports/validation/flare24_boundary_pot_study_v6_recovered_wheel_v1.json",
    "reports/validation/flare24_boundary_publication_assets_v6_recovered_wheel_v1.json",
)

REQUIRED_WHEEL_MODULES = (
    "flightdelaybench/flare_ablation.py",
    "flightdelaybench/flare_aggregate.py",
    "flightdelaybench/flare_audit_recovery.py",
    "flightdelaybench/flare_confirmation.py",
    "flightdelaybench/flare_evaluation.py",
    "flightdelaybench/flare_features.py",
    "flightdelaybench/flare_modeling.py",
    "flightdelaybench/flare_reconciliation.py",
    "flightdelaybench/flare_release.py",
    "flightdelaybench/flare_reporting.py",
    "flightdelaybench/flare_rotation.py",
    "flightdelaybench/flare_rotation_features.py",
    "flightdelaybench/flare_smoke.py",
    "flightdelaybench/flare_study.py",
    "flightdelaybench/flare_validation.py",
    "flightdelaybench/flare_weather.py",
    "flightdelaybench/flare_weather_acquisition.py",
    "flightdelaybench/boundary_context.py",
    "flightdelaybench/boundary_rotation_models.py",
    "flightdelaybench/boundary_rotation_smoke_validation.py",
    "flightdelaybench/boundary_rotation_validation.py",
    "flightdelaybench/flare_boundary_contracts.py",
    "flightdelaybench/flare_boundary_modeling.py",
    "flightdelaybench/flare_boundary_recovery.py",
    "flightdelaybench/flare_boundary_reporting.py",
    "flightdelaybench/flare_boundary_reporting_validation.py",
    "flightdelaybench/flare_boundary_study.py",
    "flightdelaybench/flare_boundary_study_validation.py",
    "flightdelaybench/flare_boundary_validation.py",
)


def _load_self_hashed(path: Path) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    keys = [key for key in ("report_sha256", "manifest_sha256") if key in payload]
    if len(keys) != 1:
        raise ValueError(f"expected one self-hash field: {path}")
    key = keys[0]
    body = {name: value for name, value in payload.items() if name != key}
    if payload[key] != canonical_json_sha256(body):
        raise ValueError(f"self-hash mismatch: {path}")
    return payload


def _record(root: Path, path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        display_path = resolved.as_posix()
    return {
        "role": role,
        "path": display_path,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def inspect_release_archives(wheel_path: Path, sdist_path: Path) -> dict[str, Any]:
    """Require complete FLARE code and reject bundled large-data payloads."""

    if wheel_path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("wheel unexpectedly exceeds 10 MiB")
    if sdist_path.stat().st_size > 25 * 1024 * 1024:
        raise ValueError("source archive unexpectedly exceeds 25 MiB")
    with zipfile.ZipFile(wheel_path) as archive:
        wheel_members = set(archive.namelist())
    missing = sorted(set(REQUIRED_WHEEL_MODULES) - wheel_members)
    if missing:
        raise ValueError(f"wheel is missing FLARE-24 modules: {missing}")

    with tarfile.open(sdist_path, mode="r:gz") as archive:
        sdist_members = [member.name.replace("\\", "/") for member in archive.getmembers()]
    forbidden_suffixes = (".parquet", ".joblib", ".zip")
    forbidden = [
        name
        for name in sdist_members
        if name.lower().endswith(forbidden_suffixes)
        or "/data/raw/" in name.lower()
        or "/models/" in name.lower()
    ]
    if forbidden:
        raise ValueError(f"source archive contains forbidden large payloads: {forbidden[:10]}")
    return {
        "wheel_member_count": len(wheel_members),
        "sdist_member_count": len(sdist_members),
        "required_flare_modules": len(REQUIRED_WHEEL_MODULES),
        "missing_flare_modules": missing,
        "forbidden_sdist_payloads": forbidden,
    }


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def build_release_manifest(
    *,
    root: Path,
    quality_report_path: Path,
    smoke_report_path: Path,
    wheel_path: Path,
    sdist_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite release manifest: {output_path}")
    withdrawal = root / "manifests/failures/flare24_release_candidate_v3_withdrawn_cutoff_audit.json"
    if withdrawal.is_file():
        raise ValueError(
            "release promotion blocked: cutoff audit withdrawal requires corrected "
            "research evidence and a newly reviewed release contract"
        )
    if __version__ != "0.1.0rc1":
        raise ValueError(f"unexpected package version: {__version__}")

    quality = _load_self_hashed(quality_report_path)
    smoke = _load_self_hashed(smoke_report_path)
    if quality.get("status") != "PASS_FLARE24_SOURCE_RELEASE_CHECKS":
        raise ValueError("source release checks did not pass")
    if smoke.get("status") != "PASS_FLARE24_FRESH_WHEEL_INFERENCE_SMOKE":
        raise ValueError("fresh-wheel inference smoke did not pass")
    if smoke.get("package", {}).get("version") != __version__:
        raise ValueError("fresh-wheel smoke used a different package version")
    if smoke.get("outcomes_accessed", {}).get("2026") is not False:
        raise ValueError("fresh-wheel smoke outcome boundary is invalid")

    core_records: list[dict[str, Any]] = []
    for relative in CORE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing release file: {path}")
        core_records.append(_record(root, path, "release_core"))
    source_records = [
        _record(root, path, "source")
        for parent in (root / "src" / "flightdelaybench", root / "tools")
        for path in sorted(parent.glob("*.py"))
    ]
    archive_audit = inspect_release_archives(wheel_path, sdist_path)
    git_status = _git(root, "status", "--short")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "READY_FOR_GITHUB_ZENODO_DEPOSIT_PENDING_RELEASE_COMMIT_TAG_AND_UPLOAD",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "release": {
            "name": "FLARE-24 + BC-POT-R 0.1.0-rc.1 candidate v3",
            "package_version": __version__,
            "external_publication_performed": False,
            "doi_assigned": False,
        },
        "git": {
            "head": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "worktree_dirty": bool(git_status),
            "status_snapshot": git_status.splitlines(),
            "release_commit_created": False,
            "release_tag_created": False,
        },
        "quality_report": _record(root, quality_report_path, "source_release_checks"),
        "smoke_report": _record(root, smoke_report_path, "fresh_wheel_inference_smoke"),
        "distributions": [
            _record(root, wheel_path, "wheel"),
            _record(root, sdist_path, "source_distribution"),
        ],
        "archive_audit": archive_audit,
        "core_files": core_records,
        "source_files": source_records,
        "evidence_boundary": {
            "selection_year": 2024,
            "retrospective_audit_year": 2025,
            "confirmation_window": ["2026-01-01", "2026-06-30"],
            "confirmation_outcomes_accessed": False,
        },
        "claim_limit": (
            "This ledger marks a locally validated deposit candidate. It does not claim an "
            "external release, DOI, clean release commit/tag, blind confirmation, causal effect, "
            "production readiness, safety, fairness, or universal state of the art."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_canonical_json(output_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = build_release_manifest(
        root=args.root,
        quality_report_path=args.quality_report,
        smoke_report_path=args.smoke_report,
        wheel_path=args.wheel,
        sdist_path=args.sdist,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
