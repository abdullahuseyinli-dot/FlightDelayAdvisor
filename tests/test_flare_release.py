from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from flightdelaybench.flare_release import (
    REQUIRED_WHEEL_MODULES,
    build_release_manifest,
    inspect_release_archives,
)


def _write_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        for name in REQUIRED_WHEEL_MODULES:
            archive.writestr(name, "# test\n")


def _write_sdist(path: Path, member_name: str = "package/README.md") -> None:
    payload = b"test\n"
    info = tarfile.TarInfo(member_name)
    info.size = len(payload)
    with tarfile.open(path, mode="w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))


def test_release_archive_inspection_accepts_curated_archives(tmp_path: Path) -> None:
    wheel = tmp_path / "release.whl"
    sdist = tmp_path / "release.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    result = inspect_release_archives(wheel, sdist)

    assert result["required_flare_modules"] == len(REQUIRED_WHEEL_MODULES)
    assert result["missing_flare_modules"] == []
    assert result["forbidden_sdist_payloads"] == []


def test_release_archive_inspection_rejects_data_payload(tmp_path: Path) -> None:
    wheel = tmp_path / "release.whl"
    sdist = tmp_path / "release.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist, "package/data/raw/flights.parquet")

    with pytest.raises(ValueError, match="forbidden large payloads"):
        inspect_release_archives(wheel, sdist)


def test_cutoff_withdrawal_blocks_new_release_even_if_old_quality_report_passed(tmp_path: Path) -> None:
    withdrawal = tmp_path / "manifests/failures/flare24_release_candidate_v3_withdrawn_cutoff_audit.json"
    withdrawal.parent.mkdir(parents=True)
    withdrawal.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="release promotion blocked"):
        build_release_manifest(
            root=tmp_path, quality_report_path=tmp_path / "quality.json",
            smoke_report_path=tmp_path / "smoke.json", wheel_path=tmp_path / "package.whl",
            sdist_path=tmp_path / "package.tar.gz", output_path=tmp_path / "new.json",
        )
    assert not (tmp_path / "new.json").exists()
