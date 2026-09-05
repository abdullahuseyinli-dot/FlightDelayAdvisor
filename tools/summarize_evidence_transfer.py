"""Publish a small, path-portable record of an already verified private transfer.

This verifies the input reports' bindings, not the large payloads again. It never
converts file-integrity evidence into historical availability certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def summary(package: Path) -> dict[str, Any]:
    manifest_bytes = (package / "PACKAGE.json").read_bytes()
    verification_bytes = (package / "VERIFICATION.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    verification = json.loads(verification_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    verification_sha = hashlib.sha256(verification_bytes).hexdigest()
    detached = (package / "VERIFICATION.sha256").read_text(encoding="utf-8").split()
    if detached != [verification_sha, "VERIFICATION.json"]:
        raise ValueError("verification report checksum mismatch")
    if verification.get("package_manifest_sha256") != manifest_sha:
        raise ValueError("package manifest binding mismatch")
    if verification.get("status") != "PASS_TRANSFER_READY_STRICT_CUTOFF_RETRAINING_BLOCKED":
        raise ValueError("expected the completed, scientifically blocked transfer record")
    if verification.get("historical_availability_certified") is not False:
        raise ValueError("transfer may not certify historical availability")
    if not verification.get("full_size_archive_verification"):
        raise ValueError("original report lacks full archive verification")
    files = manifest["files"]
    total_bytes = sum(record["bytes"] for record in files)
    archives = manifest["archives"]
    if (verification["payload_files"], verification["payload_bytes"], verification["archives"]) != (
        len(files), total_bytes, len(archives),
    ):
        raise ValueError("transfer counts differ from verification")
    return {
        "schema_version": 1,
        "evidence_class": "RECORD_OF_COMPLETED_PRIVATE_TRANSFER_VERIFICATION",
        "status": "TRANSFER_BYTES_VERIFIED_STRICT_CUTOFF_RESEARCH_BLOCKED",
        "source_commit": manifest["source"]["commit"],
        "source_branch": manifest["source"]["branch"],
        "recorded_verification_time_utc": verification["checked_at_utc"],
        "payload_files": len(files), "payload_bytes": total_bytes,
        "archive_count": len(archives),
        "archives": [{key: record[key] for key in ("path", "bytes", "sha256")} for record in archives],
        "input_records": [
            {"path": "PACKAGE.json", "bytes": len(manifest_bytes), "sha256": manifest_sha},
            {"path": "VERIFICATION.json", "bytes": len(verification_bytes), "sha256": verification_sha},
        ],
        "original_integrity_status": verification["integrity_status"],
        "original_restore_test_exit_code": verification["synthetic_restore_tests"]["returncode"],
        "full_size_extraction_performed_on_source": verification["full_size_local_extraction_performed"],
        "large_payloads_rehashed_by_summary": False,
        "local_absolute_paths_published": False,
        "historical_availability_certified": False,
        "corrected_real_data_performance_established": False,
        "later_source_investigation_in_this_snapshot": False,
        "claim_limit": "Original transfer verification plus checked report bindings. Restore bytes separately; retain the newer checkout and the stopped source investigation. This does not supply missing receipt/label-availability evidence or authorize training/publication.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("transfer summary is create-only")
    result = summary(args.package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Transfer summary created: {result['payload_files']} previously verified payload files.")


if __name__ == "__main__":
    main()
