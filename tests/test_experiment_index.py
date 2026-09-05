import json
from pathlib import Path

import pytest

from tools import build_experiment_index as index


def make_report(root: Path, name: str = "trial.json") -> Path:
    path = root / "reports/experiments" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "STOPPED", "metric": None}) + "\n", encoding="utf-8")
    return path


def test_repository_index_includes_every_experiment() -> None:
    assert index.check_index(index.ROOT) >= 20


def test_index_roundtrip_preserves_stopped_status_and_original_bytes(tmp_path: Path) -> None:
    path = make_report(tmp_path, "handoff_continuation_example.json")
    before = path.read_bytes()
    assert index.write_index(tmp_path) == index.check_index(tmp_path) == 1
    record = json.loads((tmp_path / index.MANIFEST).read_text(encoding="utf-8"))["files"][0]
    assert record["recorded_status"] == "STOPPED"
    assert "no real-data trials" in record["evidence_class"]
    assert record["bytes"] == len(before)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["bytes", "new_record", "markdown"])
def test_stale_index_is_rejected(tmp_path: Path, change: str) -> None:
    path = make_report(tmp_path)
    index.write_index(tmp_path)
    if change == "bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "new_record":
        make_report(tmp_path, "new.json")
    else:
        (tmp_path / index.MARKDOWN).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        index.check_index(tmp_path)


def test_existing_index_is_not_overwritten(tmp_path: Path) -> None:
    make_report(tmp_path)
    index.write_index(tmp_path)
    original = (tmp_path / index.MANIFEST).read_bytes()
    with pytest.raises(FileExistsError, match="create-only"):
        index.write_index(tmp_path)
    assert (tmp_path / index.MANIFEST).read_bytes() == original


def test_index_excludes_unparsed_console_logs(tmp_path: Path) -> None:
    make_report(tmp_path)
    (tmp_path / "reports/experiments/console.log").write_bytes(b"\xff")
    assert index.write_index(tmp_path) == 1
