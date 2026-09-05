import hashlib
import json
from pathlib import Path

import pytest

from tools import summarize_evidence_transfer as transfer


def fixture_package(root: Path, *, certified: bool = False, count: int = 1) -> None:
    manifest = {
        "source": {"commit": "synthetic", "branch": "synthetic"},
        "files": [{"path": "synthetic.bin", "bytes": 3}],
        "archives": [{"path": "payload-001.zip", "bytes": 3, "sha256": "0" * 64}],
    }
    body = json.dumps(manifest).encode()
    (root / "PACKAGE.json").write_bytes(body)
    report = {
        "status": "PASS_TRANSFER_READY_STRICT_CUTOFF_RETRAINING_BLOCKED",
        "package_manifest_sha256": hashlib.sha256(body).hexdigest(),
        "historical_availability_certified": certified,
        "full_size_archive_verification": True,
        "payload_files": count, "payload_bytes": 3, "archives": 1,
        "checked_at_utc": "synthetic-fixture-time",
        "integrity_status": "PASS_PACKAGE_INTEGRITY",
        "synthetic_restore_tests": {"returncode": 0},
        "full_size_local_extraction_performed": False,
    }
    encoded = json.dumps(report).encode()
    (root / "VERIFICATION.json").write_bytes(encoded)
    (root / "VERIFICATION.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  VERIFICATION.json\n", encoding="utf-8",
    )


def test_summary_distinguishes_recorded_integrity_from_new_verification(tmp_path: Path) -> None:
    fixture_package(tmp_path)
    result = transfer.summary(tmp_path)
    assert result["payload_files"] == 1
    assert result["large_payloads_rehashed_by_summary"] is False
    assert result["historical_availability_certified"] is False
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize("modified", ["PACKAGE.json", "VERIFICATION.json"])
def test_summary_rejects_modified_input_binding(tmp_path: Path, modified: str) -> None:
    fixture_package(tmp_path)
    path = tmp_path / modified
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="mismatch"):
        transfer.summary(tmp_path)


def test_summary_rejects_availability_promotion(tmp_path: Path) -> None:
    fixture_package(tmp_path, certified=True)
    with pytest.raises(ValueError, match="may not certify"):
        transfer.summary(tmp_path)


def test_summary_rejects_inconsistent_count(tmp_path: Path) -> None:
    fixture_package(tmp_path, count=42)
    with pytest.raises(ValueError, match="counts differ"):
        transfer.summary(tmp_path)
