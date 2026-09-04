#!/usr/bin/env python3
"""Run and record the source-only FLARE-24 release checks without overwriting evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flightdelaybench.hashing import canonical_json_sha256, write_canonical_json

ROOT = Path(__file__).resolve().parents[1]


def _run(name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    displayed_command = list(command)
    if Path(displayed_command[0]).resolve() == Path(sys.executable).resolve():
        displayed_command[0] = "python"
    elif Path(displayed_command[0]).name.lower() in {"uv", "uv.exe"}:
        displayed_command[0] = "uv"
    return {
        "name": name,
        "command": displayed_command,
        "elapsed_seconds": time.perf_counter() - started,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite release-check evidence: {output_path}")

    python = sys.executable
    uv = shutil.which("uv")
    checks = [
        (
            "version_consistency",
            [
                python,
                "-c",
                (
                    "from importlib.metadata import version; "
                    "from flightdelaybench import __version__; "
                    "assert version('flightdelaybench') == __version__ == '0.1.0rc1'"
                ),
            ],
        ),
        ("compileall", [python, "-m", "compileall", "-q", "app.py", "src", "tools", "tests"]),
        ("ruff", [python, "-m", "ruff", "check", "src/flightdelaybench", "tests"]),
        ("mypy", [python, "-m", "mypy", "src/flightdelaybench"]),
        (
            "source_only_tests",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-m",
                "not integration and not slow and not confirmation",
            ],
        ),
        ("repository_evidence", [python, "tools/validate_repository.py"]),
        ("documentation", [python, "tools/validate_documentation.py"]),
    ]
    if uv is not None:
        checks.insert(0, ("lock_consistency", [uv, "lock", "--check", "--offline"]))

    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    records = [_run(name, command) for name, command in checks]
    passed = all(record["status"] == "PASS" for record in records)
    withdrawn = (ROOT / "manifests/failures/flare24_release_candidate_v3_withdrawn_cutoff_audit.json").is_file()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": ("PASS_FLARE24_WITHDRAWN_SOURCE_CHECKS" if withdrawn else "PASS_FLARE24_SOURCE_RELEASE_CHECKS") if passed else "FAIL_FLARE24_SOURCE_RELEASE_CHECKS",
        "release_withdrawn": withdrawn,
        "publication_authorized_by_this_report": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started_at,
        "elapsed_seconds": time.perf_counter() - started,
        "checks": records,
        "environment": {
            "python": platform.python_version(),
            "python_executable_name": Path(sys.executable).name,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "flightdelaybench": importlib.metadata.version("flightdelaybench"),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "pytest": importlib.metadata.version("pytest"),
            "ruff": importlib.metadata.version("ruff"),
            "mypy": importlib.metadata.version("mypy"),
        },
        "resource_disclosure": {
            "historical_pipeline_peak_memory_instrumented": False,
            "reason": (
                "The completed research runs recorded wall-clock and per-model fit times but did "
                "not instrument process-tree peak RSS; this is disclosed rather than reconstructed."
            ),
        },
        "outcomes_accessed": {"2026": False},
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(output_path, report)
    print(json.dumps({"status": report["status"], "output": str(output_path)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
