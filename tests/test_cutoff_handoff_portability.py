from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from tools import audit_cutoff_bundle, validate_cutoff_clean_wheel


@pytest.mark.parametrize(
    ("platform", "relative"),
    [("win32", "Scripts/python.exe"), ("linux", "bin/python"), ("darwin", "bin/python")],
)
def test_clean_environment_python_matches_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, relative: str,
) -> None:
    monkeypatch.setattr(validate_cutoff_clean_wheel.sys, "platform", platform)
    assert validate_cutoff_clean_wheel.venv_python(tmp_path) == tmp_path / relative


@pytest.mark.parametrize("override_models", [False, True])
def test_model_remapping_preserves_frozen_hash_checks(tmp_path: Path, override_models: bool) -> None:
    frozen = {
        "models": {
            "delay": {"path": "D:/old-machine/delay.cbm", "sha256": "1" * 64},
            "cancellation": {"path": "D:/old-machine/cancel.cbm", "sha256": "2" * 64},
        },
    }
    original = copy.deepcopy(frozen)
    delay = tmp_path / "delay.cbm" if override_models else None
    cancellation = tmp_path / "cancel.cbm" if override_models else None
    command = validate_cutoff_clean_wheel.inference_command(
        tmp_path / "python", tmp_path / "data", tmp_path / "smoke.json", frozen,
        delay_model=delay, cancellation_model=cancellation,
    )
    for role, override in [("delay", delay), ("cancellation", cancellation)]:
        path_option = command.index(f"--{role}-model")
        hash_option = command.index(f"--{role}-model-sha256")
        expected_path = str(override.resolve()) if override is not None else frozen["models"][role]["path"]
        assert command[path_option + 1] == expected_path
        assert command[hash_option + 1] == frozen["models"][role]["sha256"]
    assert frozen == original
    assert command[command.index("--execution-context") + 1] == "fresh_wheel"


@pytest.mark.parametrize("recorded_path", ["Z:/old-machine/report.json", r"Z:\old-machine\report.json"])
def test_bundle_audit_resolves_relocated_windows_report(tmp_path: Path, recorded_path: str) -> None:
    evidence = tmp_path / "reports" / "validation" / "report.json"
    evidence.parent.mkdir(parents=True)
    payload = b'{"synthetic":true}\r\n'
    evidence.write_bytes(payload)
    record = {"path": recorded_path, "sha256": hashlib.sha256(payload).hexdigest()}
    original = record.copy()
    assert audit_cutoff_bundle.resolve_inference_evidence(record, root=tmp_path) == evidence
    assert record == original


def test_bundle_audit_accepts_explicit_relocation_with_identical_bytes(tmp_path: Path) -> None:
    evidence = tmp_path / "renamed.json"
    payload = b'{"synthetic":true}\n'
    evidence.write_bytes(payload)
    record = {"path": "Z:/old-machine/report.json", "sha256": hashlib.sha256(payload).hexdigest()}
    assert audit_cutoff_bundle.resolve_inference_evidence(
        record, root=tmp_path, override=evidence,
    ) == evidence


@pytest.mark.parametrize("use_override", [False, True])
def test_bundle_audit_rejects_relocated_evidence_with_changed_bytes(tmp_path: Path, use_override: bool) -> None:
    evidence = tmp_path / "reports" / "validation" / "report.json"
    evidence.parent.mkdir(parents=True)
    original = b'{"synthetic":true}\r\n'
    evidence.write_bytes(original.replace(b"\r\n", b"\n"))
    record = {"path": "Z:/old-machine/report.json", "sha256": hashlib.sha256(original).hexdigest()}
    with pytest.raises(ValueError, match="clean inference evidence changed"):
        audit_cutoff_bundle.resolve_inference_evidence(
            record, root=tmp_path, override=evidence if use_override else None,
        )


def test_bundle_audit_does_not_mask_changed_original_or_missing_override(tmp_path: Path) -> None:
    original = tmp_path / "original" / "report.json"
    original.parent.mkdir()
    original.write_bytes(b"changed")
    fallback = tmp_path / "reports" / "validation" / "report.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"frozen")
    record = {"path": str(original), "sha256": hashlib.sha256(b"frozen").hexdigest()}
    with pytest.raises(ValueError, match="clean inference evidence changed"):
        audit_cutoff_bundle.resolve_inference_evidence(record, root=tmp_path)
    with pytest.raises(FileNotFoundError):
        audit_cutoff_bundle.resolve_inference_evidence(
            record, root=tmp_path, override=tmp_path / "missing.json",
        )
