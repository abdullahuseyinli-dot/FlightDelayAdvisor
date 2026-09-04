from pathlib import Path

import pytest

from tools import validate_documentation as docs


def test_documentation_matches_preserved_results() -> None:
    docs.validate_result_tables(docs.ROOT)
    docs.validate_metadata_and_status(docs.ROOT)


def test_all_documentation_local_links_resolve() -> None:
    assert docs.validate_links(docs.ROOT) > 100


def test_broken_relative_link_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[missing](docs/absent.md)", encoding="utf-8")
    with pytest.raises(ValueError, match="broken link"):
        docs.validate_links(tmp_path)


def test_external_links_and_fragments_do_not_need_local_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "[web](https://example.org) [anchor](#section)", encoding="utf-8"
    )
    assert docs.validate_links(tmp_path) == 0


def test_links_resolve_relative_to_the_document(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (tmp_path / "README.md").write_text("[guide](docs/guide.md)", encoding="utf-8")
    (folder / "guide.md").write_text("[back](../README.md)", encoding="utf-8")
    assert docs.validate_links(tmp_path) == 2


def test_link_outside_repository_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[outside](../README.md)", encoding="utf-8")
    with pytest.raises(ValueError, match="nonportable"):
        docs.validate_links(tmp_path)
