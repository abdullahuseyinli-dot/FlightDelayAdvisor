from __future__ import annotations

import json
from pathlib import Path

from flightdelaybench.flare_confirmation import write_flare24_confirmation_lock
from flightdelaybench.hashing import canonical_json_sha256, sha256_file


def _write_self_hashed(path: Path, status: str, **extra: object) -> Path:
    payload: dict[str, object] = {"status": status, **extra}
    key = "manifest_sha256" if "lock" in path.stem or "bundle" in path.stem else "report_sha256"
    payload[key] = canonical_json_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_confirmation_lock_does_not_open_2026(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.toml"
    protocol_path.write_text(
        """
[information_boundary]
confirmation_year = 2026
confirmation_gate_opened = false

[evaluation]
primary_metrics = ["joint_log_loss", "multiclass_brier"]
bootstrap_repetitions = 2000
bootstrap_seed = 7
""".strip(),
        encoding="utf-8",
    )
    selection_path = _write_self_hashed(
        tmp_path / "selection.json",
        "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY",
    )
    selection_hash = sha256_file(selection_path)
    method_path = _write_self_hashed(
        tmp_path / "method-lock.json",
        "LOCKED_FLARE24_METHOD_BEFORE_2025_RETROSPECTIVE_AUDIT",
        selection_report={"sha256": selection_hash},
        frozen_choices={"ensemble_weights": {"rotation_structural": 1.0}},
    )
    audit_path = _write_self_hashed(
        tmp_path / "audit.json",
        "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION",
        method_lock={"sha256": sha256_file(method_path)},
        outcomes_accessed={"2026_accessed": False},
    )
    bundle_path = _write_self_hashed(
        tmp_path / "bundle.json",
        "COMPLETE_FLARE24_PUBLICATION_BUNDLE",
    )

    output_path = tmp_path / "confirmation-lock.json"
    result = write_flare24_confirmation_lock(
        protocol_path=protocol_path,
        selection_report_path=selection_path,
        method_lock_path=method_path,
        audit_report_path=audit_path,
        publication_bundle_path=bundle_path,
        output_path=output_path,
    )

    assert result["confirmation_gate"]["opened"] is False
    assert result["outcomes_accessed_at_lock"]["2026"] is False
