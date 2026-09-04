from __future__ import annotations

from pathlib import Path

import pytest

import flightdelaybench.provenance as provenance


def test_capture_provenance_binds_exact_source_bytes() -> None:
    source = Path(provenance.__file__).resolve()

    result = provenance.capture_provenance((source,))

    assert result["source_files"] == [
        {
            "path": source.relative_to(provenance.PROJECT_ROOT).as_posix(),
            "bytes": source.stat().st_size,
            "sha256": provenance.sha256_file(source),
        }
    ]
    assert isinstance(result["git_worktree_dirty"], bool)
    assert isinstance(result["git_status_short"], list)


def test_capture_provenance_rejects_external_paths(tmp_path: Path) -> None:
    source = tmp_path / "outside.py"
    source.write_text("pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the repository"):
        provenance.capture_provenance((source,))
