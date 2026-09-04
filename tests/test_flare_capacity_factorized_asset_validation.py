from __future__ import annotations

from pathlib import Path

import pytest

from flightdelaybench.flare_capacity_factorized_asset_validation import _verify_record
from flightdelaybench.hashing import sha256_file


def test_asset_record_allows_optional_byte_count(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("bound by checksum\n", encoding="utf-8")

    resolved = _verify_record(
        {"path": artifact.as_posix(), "sha256": sha256_file(artifact)},
        role="test artifact",
    )

    assert resolved == artifact


def test_asset_record_checks_byte_count_when_present(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("bound by checksum\n", encoding="utf-8")

    with pytest.raises(ValueError, match="byte count differs"):
        _verify_record(
            {
                "path": artifact.as_posix(),
                "bytes": artifact.stat().st_size + 1,
                "sha256": sha256_file(artifact),
            },
            role="test artifact",
        )
