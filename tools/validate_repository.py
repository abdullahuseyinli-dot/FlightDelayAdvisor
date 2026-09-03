#!/usr/bin/env python3
"""Validate the lightweight release contract without loading model artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "app.py",
    "config/model_paths.yml",
    "reports/metrics_summary.txt",
    "reports/backtest_2025_metrics.txt",
    "models/catboost_delay15_calibrated.joblib",
    "models/lgbm_cancel_calibrated.joblib",
    "data/processed/bts_delay_2010_2024_balanced_research_weather.parquet",
)

EXPECTED_IN_PERIOD = (
    "Context   : delay15 – CatBoost (calibrated, t=0.5)",
    "ROC-AUC   : 0.6864",
    "PR-AUC    : 0.3819",
    "Brier     : 0.1587",
    "Context   : cancel – LGBM (calibrated, t=0.5)",
    "ROC-AUC   : 0.7061",
    "PR-AUC    : 0.1089",
    "Brier     : 0.0171",
)

EXPECTED_BACKTEST = (
    "196,693",
    "0.6469",
    "0.3460",
    "0.1703",
    "200,000",
    "0.6543",
    "0.0368",
    "0.0165",
)

TEXT_SUFFIXES = {".md", ".py", ".toml", ".txt", ".yml", ".yaml"}
FORBIDDEN_PUBLIC_MARKERS = (
    "c:" + chr(92) + "users",
    "[screenshot " + "placeholder",
)
IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "htmlcov",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"validation failed: {message}")


def validate_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    require(not missing, f"missing required files: {', '.join(missing)}")


def validate_metrics() -> None:
    in_period = (ROOT / "reports/metrics_summary.txt").read_text(encoding="utf-8")
    backtest = (ROOT / "reports/backtest_2025_metrics.txt").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for value in EXPECTED_IN_PERIOD:
        require(value in in_period, f"in-period evidence is missing {value!r}")
    for value in EXPECTED_BACKTEST:
        require(value in backtest, f"backtest evidence is missing {value!r}")
        require(value in readme, f"README is missing promoted value {value!r}")


def validate_markdown_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", readme):
        target = target.strip().split(maxsplit=1)[0].strip("<>")
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        local = unquote(target.split("#", 1)[0])
        require((ROOT / local).exists(), f"broken README link: {target}")


def validate_public_text() -> None:
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or any(part in IGNORED_DIRECTORIES for part in path.parts)
            or path.suffix.lower() not in TEXT_SUFFIXES
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        for marker in FORBIDDEN_PUBLIC_MARKERS:
            require(marker not in text, f"{path.relative_to(ROOT)} contains {marker!r}")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    require("```" not in gitignore, ".gitignore contains Markdown fences")


def main() -> None:
    validate_required_files()
    validate_metrics()
    validate_markdown_links()
    validate_public_text()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()
