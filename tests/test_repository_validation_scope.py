from pathlib import Path

import pytest

from tools import validate_repository


def test_clean_environment_dependencies_are_not_public_project_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validate_repository, "ROOT", tmp_path)
    (tmp_path / ".gitignore").write_text(".local/\n", encoding="utf-8")
    environment = tmp_path / ".local" / "validation_environment"
    environment.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
    (environment / "third_party.py").write_text("c:" + chr(92) + "users", encoding="utf-8")
    validate_repository.validate_public_text()


def test_execution_folder_without_venv_marker_is_still_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validate_repository, "ROOT", tmp_path)
    (tmp_path / ".gitignore").write_text("", encoding="utf-8")
    directory = tmp_path / ".local" / "project_source"
    directory.mkdir(parents=True)
    (directory / "source.py").write_text("c:" + chr(92) + "users", encoding="utf-8")
    with pytest.raises(SystemExit, match="contains"):
        validate_repository.validate_public_text()
