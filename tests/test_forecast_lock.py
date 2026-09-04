from __future__ import annotations

import json

import pytest

from flightdelaybench.forecast_lock import LOCK_STATUS, validate_forecast_method_lock
from flightdelaybench.hashing import canonical_json_sha256, sha256_file, write_canonical_json


def test_forecast_lock_validates_artifacts_and_fails_closed(tmp_path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"frozen-model")
    path = tmp_path / "lock.json"
    payload = {
        "status": LOCK_STATUS,
        "audit_plan": {"evaluation_year": 2025},
        "confirmation_gate": {"authorized": False},
        "frozen_artifacts": [
            {"path": artifact.as_posix(), "sha256": sha256_file(artifact)}
        ],
    }
    payload["lock_sha256"] = canonical_json_sha256(payload)
    write_canonical_json(path, payload)
    assert validate_forecast_method_lock(path)["status"] == LOCK_STATUS

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["audit_plan"]["evaluation_year"] = 2026
    write_canonical_json(path, tampered)
    with pytest.raises(ValueError, match="self-hash"):
        validate_forecast_method_lock(path)


def test_forecast_lock_never_authorizes_confirmation(tmp_path) -> None:
    path = tmp_path / "lock.json"
    payload = {
        "status": LOCK_STATUS,
        "audit_plan": {"evaluation_year": 2025},
        "confirmation_gate": {"authorized": True},
        "frozen_artifacts": [],
    }
    payload["lock_sha256"] = canonical_json_sha256(payload)
    write_canonical_json(path, payload)
    with pytest.raises(PermissionError, match="confirmation"):
        validate_forecast_method_lock(path)
