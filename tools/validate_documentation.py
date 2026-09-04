#!/usr/bin/env python3
"""Validate current documentation against preserved reports without loading outcomes."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

from flightdelaybench.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
STUDY = "reports/experiments/flare24_boundary_pot_v6_recovered.json"
STUDY_SHA256 = "adaea6d8fb83d834f5a3b190f471990c7dab7fe218719dce39fadaf491c07279"
METHOD_LABELS = {
    "schedule_baseline": "Schedule/history baseline",
    "flare24": "FLARE-24 structural rotation",
    "capacity_gated_simplex": "CC-RTH gated model",
    "meta_current": "Regularized cancellation meta-stack",
    "boundary_only": "Boundary-complete model",
    "counterfactual_residual": "Counterfactual-residual model",
    "boundary_ensemble": "Global boundary ensemble",
    "boundary_gated_ensemble": "Q4-selected boundary ensemble",
}
README_METHODS = {
    "schedule_baseline", "flare24", "meta_current", "boundary_only", "boundary_gated_ensemble",
}
CURRENT_PAGES = (
    "README.md", "docs/README.md", "docs/CURRENT_RESULTS.md", "docs/BENCHMARK_CARD.md",
    "docs/PROJECT_STATUS.md", "docs/ARTIFACTS.md", "docs/USAGE.md", "docs/LIMITATIONS.md",
    "docs/DATA_CARD.md", "docs/MODEL_CARD.md", "docs/LEGACY_APPLICATION.md",
    "docs/RESEARCH_STANDARDS.md", "docs/RELEASE_CHECKLIST.md", "THIRD_PARTY_NOTICES.md",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def markdown_files(root: Path) -> list[Path]:
    return sorted([*root.glob("*.md"), *(root / "docs").rglob("*.md")])


def validate_links(root: Path) -> int:
    count = 0
    for page in markdown_files(root):
        content = page.read_text(encoding="utf-8")
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            target = match.group(1).strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith("#") or urlsplit(target).scheme in {"http", "https", "mailto"}:
                continue
            relative = unquote(target.split("#", 1)[0])
            resolved = (page.parent / relative).resolve()
            require(resolved.is_relative_to(root.resolve()), f"nonportable link in {page.name}: {target}")
            require(resolved.exists(), f"broken link in {page.relative_to(root)}: {target}")
            count += 1
    return count


def validate_result_tables(root: Path) -> None:
    source = root / STUDY
    require(sha256_file(source) == STUDY_SHA256, "historical source report changed")
    report = json.loads(source.read_text(encoding="utf-8"))
    methods = report["primary_evaluation"]["methods"]
    decisions = report["primary_argmax_decision_metrics"]
    for file, keys in (("docs/CURRENT_RESULTS.md", METHOD_LABELS), ("README.md", README_METHODS)):
        content = (root / file).read_text(encoding="utf-8")
        for key in keys:
            metrics = methods[key]["joint"]
            require(metrics["n"] == decisions[key]["n"] == 5_754_266, "cohort mismatch")
            expected = (
                f"| {METHOD_LABELS[key]} | {100 * decisions[key]['accuracy']:.4f} | "
                f"{metrics['log_loss']:.6f} | {metrics['multiclass_brier']:.6f} |"
            )
            require(content.count(expected) == 1, f"missing/incorrect/duplicate result row in {file}: {key}")
    content = (root / "docs/CURRENT_RESULTS.md").read_text(encoding="utf-8")
    for key, prevalence in report["primary_class_prevalence"].items():
        require(f"{100 * prevalence:.4f}%" in content, f"missing prevalence: {key}")
    delta = decisions["boundary_gated_ensemble"]["accuracy_difference_vs_reference"]
    for field in ("estimate", "lower", "upper"):
        require(f"{100 * delta[field]:+.4f}" in content, f"incorrect percentage-point {field}")


def validate_metadata_and_status(root: Path) -> None:
    for name in CURRENT_PAGES:
        require((root / name).is_file(), f"missing current document: {name}")
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    require(project["version"] == "0.1.0rc1", "unexpected package version")
    require(citation["version"] == zenodo["version"] == "0.1.0-rc.1", "citation/archive version mismatch")
    require(citation["title"] == zenodo["title"], "citation/archive title mismatch")
    require("doi" not in citation and "doi" not in zenodo, "unassigned project DOI")
    require(citation["license"] == zenodo["license"] == "MIT", "license mismatch")
    for name in ("README.md", "docs/PROJECT_STATUS.md", "docs/BENCHMARK_CARD.md", "docs/MODEL_CARD.md"):
        content = (root / name).read_text(encoding="utf-8").lower()
        require("withdrawn" in content, f"missing withdrawal in {name}")
    require("withdrawn" in citation["message"].lower(), "citation omits withdrawal")
    require("withdrawn" in zenodo["description"].lower(), "archive draft omits withdrawal")
    results = (root / "docs/CURRENT_RESULTS.md").read_text(encoding="utf-8").lower()
    for phrase in ("confounded", "2026 outcomes", "remain unopened", "retrospective proxy", "not met"):
        require(phrase in results, f"missing result boundary: {phrase}")


def main() -> None:
    count = validate_links(ROOT)
    validate_result_tables(ROOT)
    validate_metadata_and_status(ROOT)
    print(f"Documentation validation passed: {len(markdown_files(ROOT))} Markdown files, "
          f"{count} local links, report-backed result tables and metadata.")
    print("Historical scores unchanged; corrected forecast claims remain withdrawn.")


if __name__ == "__main__":
    main()
