from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from flightdelaybench.acquisition import (
    acquire_month,
    remote_url,
    validate_bts_zip,
    validate_confirmation_lock,
)
from flightdelaybench.hashing import canonical_json_sha256, write_canonical_json


def _valid_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("On_Time_2025_1.csv", "Year,Month,Cancelled\n2025,1,0\n")


def test_remote_url_and_zip_validation(tmp_path: Path) -> None:
    archive = tmp_path / "month.zip"
    _valid_zip(archive)
    assert remote_url(2025, 1).endswith("1987_present_2025_1.zip")
    assert validate_bts_zip(archive) == "On_Time_2025_1.csv"
    with pytest.raises(ValueError, match="invalid BTS period"):
        remote_url(2025, 13)


def test_existing_archive_is_recorded_without_replacement(tmp_path: Path) -> None:
    output = tmp_path / "raw"
    output.mkdir()
    archive = output / "on_time_2025_01.zip"
    _valid_zip(archive)
    before = archive.read_bytes()
    record = acquire_month(
        repository_root=tmp_path,
        year=2025,
        month=1,
        output_directory=output,
        manifest_path=tmp_path / "manifest.json",
    )
    assert archive.read_bytes() == before
    assert record.status == "VERIFIED"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["records"][0]["sha256"] == record.sha256
    assert manifest["events"][0]["action"] == "REVERIFIED_EXISTING"


def test_confirmation_requires_valid_self_hashed_lock(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    with pytest.raises(PermissionError, match="does not exist"):
        validate_confirmation_lock(lock, 2026)

    payload = {
        "schema_version": 1,
        "confirmation_year": 2026,
        "status": "LOCKED",
        "open_authorized": True,
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    write_canonical_json(lock, payload)
    validate_confirmation_lock(lock, 2026)

    payload["open_authorized"] = False
    write_canonical_json(lock, payload)
    with pytest.raises(PermissionError):
        validate_confirmation_lock(lock, 2026)


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.headers = {"ETag": "test", "Last-Modified": "now"}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [
            self.content[index : index + chunk_size]
            for index in range(0, len(self.content), chunk_size)
        ]


class _FakeSession:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def get(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        return _FakeResponse(self.content)


def test_new_download_is_atomic_and_manifested(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", "Year,Month\n2025,2\n")
    output = tmp_path / "raw"
    record = acquire_month(
        repository_root=tmp_path,
        year=2025,
        month=2,
        output_directory=output,
        manifest_path=tmp_path / "manifest.json",
        session=_FakeSession(buffer.getvalue()),  # type: ignore[arg-type]
    )
    assert (output / "on_time_2025_02.zip").is_file()
    assert not (output / "on_time_2025_02.zip.part").exists()
    assert record.etag == "test"
