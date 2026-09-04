"""Check the evidence bundle and bind identical wheel bytes to clean inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PureWindowsPath

from flightdelaybench.flare_release import inspect_release_archives
from flightdelaybench.hashing import canonical_json_sha256, sha256_file, write_canonical_json

ROOT = Path(__file__).resolve().parents[1]


def resolve_inference_evidence(
    record: dict[str, str], *, root: Path, override: Path | None = None,
) -> Path:
    """Locate relocated evidence without changing its original hash binding."""
    inference = override if override is not None else Path(record["path"])
    if override is None and not inference.is_file():
        basename = PureWindowsPath(record["path"]).name
        if not basename or basename in {".", ".."}:
            raise ValueError("invalid recorded inference filename")
        inference = root / "reports" / "validation" / basename
    if sha256_file(inference) != record["sha256"]:
        raise ValueError("clean inference evidence changed")
    return inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--clean-wheel-report", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, help="Relocated inference report; must match the recorded SHA-256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("bundle audit is create-only")
    wheel = args.artifact_dir / "flightdelaybench-0.1.0rc1-py3-none-any.whl"
    sdist = args.artifact_dir / "flightdelaybench-0.1.0rc1.tar.gz"
    audit = inspect_release_archives(wheel, sdist)
    clean = json.loads(args.clean_wheel_report.read_text(encoding="utf-8"))
    if clean.get("report_sha256") != canonical_json_sha256({key: value for key, value in clean.items() if key != "report_sha256"}):
        raise ValueError("invalid clean-wheel evidence hash")
    if clean["status"] != "PASS_CUTOFF_CLEAN_WHEEL_VALIDATION" or clean["wheel"]["sha256"] != sha256_file(wheel):
        raise ValueError("new wheel differs; a new clean-environment check is required")
    inference = resolve_inference_evidence(
        clean["smoke"], root=ROOT, override=args.inference_report,
    )
    checked = []
    with tarfile.open(sdist, "r:gz") as archive:
        for path in sorted((ROOT / "data/external/cutoff_taf_pilot_v1").iterdir()):
            if not path.is_file():
                continue
            member = f"flightdelaybench-0.1.0rc1/{path.relative_to(ROOT).as_posix()}"
            handle = archive.extractfile(member)
            if handle is None or hashlib.sha256(handle.read()).hexdigest() != sha256_file(path):
                raise ValueError(f"source evidence omitted or changed: {member}")
            checked.append(member)
    if not checked:
        raise ValueError("pilot evidence absent")
    # Bind the current reader-facing account, not just the executable package.
    document_paths = [
        *ROOT.glob("*.md"),
        *(ROOT / "docs").rglob("*.md"),
        ROOT / "CITATION.cff",
        ROOT / ".zenodo.json",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / ".github/workflows/tests.yml",
        ROOT / "tools/validate_documentation.py",
        ROOT / "tests/test_documentation.py",
    ]
    verified_documents = []
    with tarfile.open(sdist, "r:gz") as archive:
        for path in sorted(document_paths):
            member = f"flightdelaybench-0.1.0rc1/{path.relative_to(ROOT).as_posix()}"
            handle = archive.extractfile(member)
            if handle is None or hashlib.sha256(handle.read()).hexdigest() != sha256_file(path):
                raise ValueError(f"current documentation omitted or changed: {member}")
            verified_documents.append(member)
    report = {
        "schema_version": 1, "status": "PASS_CUTOFF_EVIDENCE_BUNDLE_AUDIT",
        "wheel_sha256": sha256_file(wheel), "sdist_sha256": sha256_file(sdist),
        "clean_wheel_report_sha256": sha256_file(args.clean_wheel_report),
        "clean_inference_report": {"path": inference.resolve().as_posix(), "sha256": sha256_file(inference)},
        "clean_inference_reused_for_identical_wheel_bytes": True,
        "pilot_files_byte_verified": checked, "archive_audit": audit,
        "documentation_files_byte_verified": verified_documents,
        "publication_authorized": False, "corrected_research_results_established": False,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(args.output, report)
    print(json.dumps({"status": report["status"], "pilot_files_checked": len(checked)}))


if __name__ == "__main__":
    main()
