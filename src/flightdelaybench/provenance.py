"""Compact code and Git provenance for material research artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .hashing import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def capture_provenance(paths: tuple[Path, ...]) -> dict[str, Any]:
    """Bind a report to exact source bytes even when the worktree is dirty."""

    records: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"provenance source file is missing: {path}")
        try:
            relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as error:
            raise ValueError(f"provenance path lies outside the repository: {path}") from error
        records.append(
            {
                "path": relative,
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    status = _git("status", "--short")
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_worktree_dirty": bool(status),
        "git_status_short": [] if not status else status.splitlines(),
        "source_files": records,
    }
