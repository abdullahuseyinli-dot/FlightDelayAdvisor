from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import ClassVar
from urllib.error import URLError
from urllib.request import Request

import pytest

from tools import probe_cutoff_2024_sources as probe


class Response(io.BytesIO):
    status = 200
    url = "https://example.test/2024"
    headers: ClassVar[dict[str, str]] = {"Content-Type": "text/plain"}


def test_access_failure_preserves_other_source_bytes_and_unknown_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe, "PROBES", (("good.txt", "https://example.test/good"),
                                          ("failed.txt", "https://example.test/failed")))

    def fetch(request: Request, timeout: int) -> Response:
        if request.full_url.endswith("failed"):
            raise URLError("synthetic access failure")
        return Response(b"original 2024 source bytes\r\n")

    monkeypatch.setattr(probe, "urlopen", fetch)
    destination = tmp_path / "run"
    report = probe.probe_sources(destination)
    assert report["status"] == "INCOMPLETE_SOURCE_ACCESS"
    assert report["strict_availability_gate_passed"] is False
    assert report["historical_consumer_receipt_established"] is False
    assert report["records"][0]["sha256"] == hashlib.sha256((destination / "good.txt").read_bytes()).hexdigest()
    assert report["records"][1]["status"] == "FAILED_ACCESS"
    assert json.loads((destination / "failed.txt.metadata.json").read_text())["exception_type"] == "URLError"
    assert all(record["historical_consumer_available_at_utc"] is None for record in report["records"])


def test_reused_output_refused_before_network_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = tmp_path / "intent.json"
    original.write_bytes(b"preserved failure")

    def fetch(*args: object, **kwargs: object) -> None:
        pytest.fail("reused output must fail before any network request")

    monkeypatch.setattr(probe, "urlopen", fetch)
    with pytest.raises(FileExistsError):
        probe.probe_sources(tmp_path)
    assert original.read_bytes() == b"preserved failure"


def test_oversized_source_is_preserved_but_never_reported_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe, "PROBES", (("large.txt", "https://example.test/2024"),))
    monkeypatch.setattr(probe, "MAX_RESPONSE_BYTES", 3)
    monkeypatch.setattr(probe, "urlopen", lambda *args, **kwargs: Response(b"123456"))
    report = probe.probe_sources(tmp_path / "run")
    assert report["status"] == "INCOMPLETE_SOURCE_ACCESS"
    assert report["records"][0]["response_truncated"] is True
    assert (tmp_path / "run" / "large.txt").read_bytes() == b"1234"
