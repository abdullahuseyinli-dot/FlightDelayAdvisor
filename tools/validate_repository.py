#!/usr/bin/env python3
"""Validate the lightweight release contract without loading model artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any, cast
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
    "configs/flare24_v1.toml",
    "docs/DATA_CARD.md",
    "docs/FLARE24_METHOD.md",
    "docs/FLARE24_REPRODUCIBILITY.md",
    "docs/FLARE24_TECHNICAL_REPORT.md",
    "docs/LITERATURE_AND_NOVELTY.md",
    "docs/MODEL_CARD.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/RELEASE_NOTES_FLARE24_RC1.md",
    "docs/RESULTS.md",
    "manifests/confirmation_lock_v1.json",
    "manifests/flare24_method_lock_v1.json",
    "manifests/flare24_publication_bundle_v2.json",
    "reports/experiments/flare24_2024_selection_v2.json",
    "reports/experiments/flare24_2025_audit_v1_recovered.json",
    "reports/experiments/flare24_2025_nested_ablation_v1.json",
    "reports/figures/flare24_v2/flare24_2025_monthly_stability.png",
    "reports/validation/confirmation_lock_v1.json",
    "reports/validation/flare24_2025_audit_v1_recovered.json",
    "reports/validation/flare24_2025_nested_ablation_v1.json",
    "reports/validation/flare24_method_lock_v1.json",
    "reports/validation/flare24_publication_bundle_v2.json",
    "reports/validation/flare24_release_assets_v1.json",
    "reports/validation/flare24_source_release_checks_v3.json",
    "reports/validation/flare24_fresh_wheel_smoke_v1.json",
)

EXPECTED_IN_PERIOD = (
    "Context   : delay15 – CatBoost (calibrated, t=0.5)",  # noqa: RUF001
    "ROC-AUC   : 0.6864",
    "PR-AUC    : 0.3819",
    "Brier     : 0.1587",
    "Context   : cancel – LGBM (calibrated, t=0.5)",  # noqa: RUF001
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

EXPECTED_FLARE_FILE_SHA256 = {
    "reports/experiments/flare24_2024_selection_v2.json": (
        "13fef0024e7a08139786ccdfc150a0c3db61208558d31be2a673f960b701e089"
    ),
    "manifests/flare24_method_lock_v1.json": (
        "bac1cf84e9b58be00a4e5ccf511071b4b16697e84f36a0a62b1d4ee4906b8ecc"
    ),
    "reports/experiments/flare24_2025_audit_v1_recovered.json": (
        "8a4e88070d1a55cb30d3292629c1615057a92c661c2998df9045fc601d204171"
    ),
    "reports/experiments/flare24_2025_nested_ablation_v1.json": (
        "d6bc3cb00c226d1dfebb79c0a80dd8b29b91106c362090bda27a9e386fc88a9c"
    ),
    "manifests/flare24_publication_bundle_v2.json": (
        "04e7ed7f6770c34c41af67549323c1e5c38a420643f7c96635f059d6408ec896"
    ),
    "manifests/confirmation_lock_v1.json": (
        "daecde5d7c6690d21012031755dba1fbf72edd2fda6e8d395c2f1c42e252d037"
    ),
    "reports/validation/flare24_release_assets_v1.json": (
        "0035f9dbf814346ad61b649d5df204d0c41995ef98a68ebaf6f23704a11e3517"
    ),
    "reports/validation/flare24_source_release_checks_v3.json": (
        "8c9c1092200e0e0f89a87572ee9ce1f82cdebec27d5f4ff92a0a2c9ece903c88"
    ),
    "reports/validation/flare24_fresh_wheel_smoke_v1.json": (
        "00479cd841f133ed648491f3b4e5dbe7f0f1984da0b3a4a21e1cfee29a640e7a"
    ),
}

EXPECTED_FLARE_RESULTS = (
    "5,829,666",
    "5,814,817",
    "0.555476",
    "0.539225",
    "0.337528",
    "0.327076",
    "-0.016251",
    "-0.010452",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    body = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def load_self_hashed_json(path: str) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads((ROOT / path).read_text(encoding="utf-8")))
    hash_fields = [name for name in ("manifest_sha256", "report_sha256") if name in payload]
    require(len(hash_fields) == 1, f"{path} must have exactly one self-hash field")
    field = hash_fields[0]
    recorded = payload[field]
    body = {name: value for name, value in payload.items() if name != field}
    require(recorded == canonical_json_sha256(body), f"{path} self-hash mismatch")
    return payload


def validate_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    require(not missing, f"missing required files: {', '.join(missing)}")


def validate_metrics() -> None:
    in_period = (ROOT / "reports/metrics_summary.txt").read_text(encoding="utf-8")
    backtest = (ROOT / "reports/backtest_2025_metrics.txt").read_text(encoding="utf-8")
    legacy = (ROOT / "docs/LEGACY_APPLICATION.md").read_text(encoding="utf-8")
    results = (ROOT / "docs/CURRENT_RESULTS.md").read_text(encoding="utf-8")

    for value in EXPECTED_IN_PERIOD:
        require(value in in_period, f"in-period evidence is missing {value!r}")
    for value in EXPECTED_BACKTEST:
        require(value in backtest, f"backtest evidence is missing {value!r}")
        require(value in legacy, f"legacy guide is missing recorded value {value!r}")
    for value in EXPECTED_FLARE_RESULTS:
        require(value in results, f"results guide is missing FLARE-24 value {value!r}")


def validate_flare_evidence() -> None:
    for path, expected in EXPECTED_FLARE_FILE_SHA256.items():
        require(sha256_file(ROOT / path) == expected, f"frozen FLARE-24 file changed: {path}")

    selection = load_self_hashed_json("reports/experiments/flare24_2024_selection_v2.json")
    lock = load_self_hashed_json("manifests/flare24_method_lock_v1.json")
    audit = load_self_hashed_json("reports/experiments/flare24_2025_audit_v1_recovered.json")
    nested = load_self_hashed_json("reports/experiments/flare24_2025_nested_ablation_v1.json")
    bundle = load_self_hashed_json("manifests/flare24_publication_bundle_v2.json")
    confirmation = load_self_hashed_json("manifests/confirmation_lock_v1.json")
    source_checks = load_self_hashed_json(
        "reports/validation/flare24_source_release_checks_v3.json"
    )
    smoke = load_self_hashed_json("reports/validation/flare24_fresh_wheel_smoke_v1.json")

    require(
        selection.get("status") == "COMPLETE_2024_FLARE24_SELECTION_NOT_CONFIRMATORY",
        "unexpected FLARE-24 selection status",
    )
    require(
        lock.get("status") == "LOCKED_FLARE24_METHOD_BEFORE_2025_RETROSPECTIVE_AUDIT",
        "unexpected FLARE-24 method-lock status",
    )
    choices = lock["frozen_choices"]
    require(
        choices["ensemble_weights"]
        == {"baseline": 0.0, "rotation_risk": 0.0, "rotation_structural": 1.0, "weather": 0.0},
        "FLARE-24 ensemble lock changed",
    )
    require(choices["reconciliation_enabled"] is False, "reconciliation must remain locked off")
    require(
        all(item["selected_method"] == "identity" for item in choices["calibrator_artifacts"]),
        "a frozen FLARE-24 calibrator is no longer identity",
    )

    require(
        audit.get("status") == "COMPLETE_2025_FLARE24_RETROSPECTIVE_AUDIT_NOT_BLIND_CONFIRMATION",
        "unexpected FLARE-24 audit status",
    )
    require(audit["outcomes_accessed"]["2026_accessed"] is False, "2026 audit access detected")
    methods = audit["primary_evaluation"]["methods"]
    require(
        math.isclose(methods["baseline"]["joint"]["log_loss"], 0.555476137916928),
        "FLARE-24 baseline joint log loss changed",
    )
    require(
        math.isclose(methods["rotation_structural"]["joint"]["log_loss"], 0.5392251622713801),
        "FLARE-24 selected joint log loss changed",
    )
    require(
        math.isclose(
            methods["rotation_structural"]["joint"]["multiclass_brier"],
            0.32707623623444126,
        ),
        "FLARE-24 selected joint Brier score changed",
    )
    paired = audit["primary_evaluation"]["paired_date_cluster_comparisons"][
        "rotation_structural_minus_baseline"
    ]
    require(paired["joint_log_loss"]["upper"] < 0.0, "joint log-loss interval is not favorable")
    require(paired["multiclass_brier"]["upper"] < 0.0, "joint Brier interval is not favorable")
    require(paired["joint_log_loss"]["clusters"] == 365, "paired audit must contain 365 dates")
    require(paired["joint_log_loss"]["repetitions"] == 2000, "unexpected bootstrap count")

    require(
        nested.get("status") == "COMPLETE_2025_FLARE24_NESTED_ABLATION_ANALYSIS",
        "unexpected nested-ablation status",
    )
    require(nested["confirmation_outcomes_accessed"] is False, "nested analysis opened confirmation")
    require(
        bundle.get("status") == "COMPLETE_FLARE24_PUBLICATION_BUNDLE",
        "unexpected publication-bundle status",
    )
    require(len(bundle["artifacts"]) == 12, "FLARE-24 publication bundle must contain 12 artifacts")
    require(
        confirmation.get("status")
        == "LOCKED_2026_FLARE24_CONFIRMATION_PROTOCOL_OUTCOMES_UNOPENED",
        "unexpected confirmation-lock status",
    )
    require(confirmation["outcomes_accessed_at_lock"]["2026"] is False, "2026 outcomes were opened")
    require(
        source_checks.get("status") == "PASS_FLARE24_SOURCE_RELEASE_CHECKS",
        "source release checks did not pass",
    )
    require(
        source_checks.get("environment", {}).get("flightdelaybench") == "0.1.0rc1",
        "source-check package version changed",
    )
    require(
        smoke.get("status") == "PASS_FLARE24_FRESH_WHEEL_INFERENCE_SMOKE",
        "fresh-wheel inference smoke did not pass",
    )
    require(smoke.get("package", {}).get("version") == "0.1.0rc1", "smoke version changed")
    require(smoke.get("sample", {}).get("rows") == 128, "unexpected smoke sample size")
    require(
        smoke.get("sample", {}).get("outcome_columns_used_for_prediction") is False,
        "smoke prediction used outcome columns",
    )
    require(
        smoke.get("outcomes_accessed", {}).get("2026") is False,
        "fresh-wheel smoke opened 2026 outcomes",
    )
    require(
        smoke.get("prediction_checks", {}).get("maximum_simplex_error", 1.0) <= 1e-12,
        "fresh-wheel smoke probabilities violate the simplex",
    )

    validation_paths = (
        "reports/validation/flare24_method_lock_v1.json",
        "reports/validation/flare24_2025_audit_v1_recovered.json",
        "reports/validation/flare24_2025_nested_ablation_v1.json",
        "reports/validation/flare24_publication_bundle_v2.json",
        "reports/validation/confirmation_lock_v1.json",
        "reports/validation/flare24_release_assets_v1.json",
    )
    for path in validation_paths:
        payload = load_self_hashed_json(path)
        require(str(payload.get("status", "")).startswith("PASS"), f"validation did not pass: {path}")


def validate_release_metadata() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    require(version == "0.1.0rc1", "unexpected package release-candidate version")
    init_text = (ROOT / "src/flightdelaybench/__init__.py").read_text(encoding="utf-8")
    require('__version__ = "0.1.0rc1"' in init_text, "package __version__ disagrees")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    require("version: 0.1.0-rc.1" in citation, "CITATION.cff version disagrees")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    require(zenodo.get("version") == "0.1.0-rc.1", ".zenodo.json version disagrees")
    require("doi" not in zenodo, "Zenodo metadata must not contain a placeholder DOI")


def validate_release_ledger_if_present() -> None:
    v1_relative = "manifests/flare24_release_candidate_v1.json"
    v2_relative = "manifests/flare24_release_candidate_v2.json"
    v3_relative = "manifests/flare24_release_candidate_v3.json"
    supersession_relative = (
        "manifests/failures/flare24_release_candidate_v1_superseded_by_bcpotr.json"
    )
    v2_supersession_relative = (
        "manifests/failures/flare24_release_candidate_v2_superseded_metadata_review.json"
    )
    v1_path = ROOT / v1_relative
    v2_path = ROOT / v2_relative
    v3_path = ROOT / v3_relative
    if not v1_path.is_file() and not v2_path.is_file() and not v3_path.is_file():
        return
    if v1_path.is_file():
        historical = load_self_hashed_json(v1_relative)
        if (ROOT / supersession_relative).is_file():
            supersession = json.loads(
                (ROOT / supersession_relative).read_text(encoding="utf-8")
            )
            bound = supersession.get("superseded_ledger", {})
            require(
                supersession.get("status")
                == "SUPERSEDED_BEFORE_EXTERNAL_PUBLICATION_BY_BCPOTR_RELEASE_CANDIDATE_V2",
                "invalid FLARE-24 v1 supersession status",
            )
            require(
                bound.get("sha256") == sha256_file(v1_path)
                and bound.get("bytes") == v1_path.stat().st_size
                and bound.get("self_hash") == historical.get("manifest_sha256"),
                "FLARE-24 v1 supersession does not bind the historical ledger",
            )
            require(
                supersession.get("external_publication_performed_for_v1") is False
                and supersession.get("doi_assigned_for_v1") is False
                and supersession.get("2026_outcomes_accessed") is False,
                "FLARE-24 v1 supersession overstates its evidence boundary",
            )
    if v2_path.is_file() and (ROOT / v2_supersession_relative).is_file():
        historical_v2 = load_self_hashed_json(v2_relative)
        supersession_v2 = json.loads(
            (ROOT / v2_supersession_relative).read_text(encoding="utf-8")
        )
        bound_v2 = supersession_v2.get("superseded_ledger", {})
        require(
            supersession_v2.get("status")
            == "SUPERSEDED_BEFORE_EXTERNAL_PUBLICATION_AFTER_BCPOTR_METADATA_REVIEW",
            "invalid FLARE-24 v2 supersession status",
        )
        require(
            bound_v2.get("sha256") == sha256_file(v2_path)
            and bound_v2.get("bytes") == v2_path.stat().st_size
            and bound_v2.get("self_hash") == historical_v2.get("manifest_sha256"),
            "FLARE-24 v2 supersession does not bind the historical ledger",
        )
        require(
            supersession_v2.get("external_publication_performed_for_v2") is False
            and supersession_v2.get("doi_assigned_for_v2") is False
            and supersession_v2.get("research_results_changed") is False
            and supersession_v2.get("2026_outcomes_accessed") is False,
            "FLARE-24 v2 supersession overstates its evidence boundary",
        )
    relative = (
        v3_relative
        if v3_path.is_file()
        else v2_relative
        if v2_path.is_file()
        else v1_relative
    )
    if relative == v1_relative and (ROOT / supersession_relative).is_file():
        return
    if relative == v2_relative and (ROOT / v2_supersession_relative).is_file():
        return
    ledger = load_self_hashed_json(relative)
    withdrawal_path = ROOT / "manifests/failures/flare24_release_candidate_v3_withdrawn_cutoff_audit.json"
    if relative == v3_relative and withdrawal_path.is_file():
        withdrawal = json.loads(withdrawal_path.read_text(encoding="utf-8"))
        bound = withdrawal.get("historical_ledger", {})
        require(
            withdrawal.get("status") == "WITHDRAWN_PENDING_CUTOFF_CORRECTION_AND_NEW_RELEASE_VALIDATION"
            and bound.get("path") == v3_relative
            and bound.get("sha256") == sha256_file(v3_path)
            and bound.get("bytes") == v3_path.stat().st_size
            and bound.get("self_hash") == ledger.get("manifest_sha256"),
            "invalid cutoff withdrawal or historical ledger binding",
        )
        require(
            withdrawal.get("confirmation_outcomes_accessed") is False
            and withdrawal.get("corrected_performance_established") is False
            and withdrawal.get("old_artifacts_retained") is True,
            "cutoff withdrawal misstates evidence status",
        )
        print("Release candidate v3 WITHDRAWN; historical ledger preserved, new release blocked.")
        return
    require(
        ledger.get("status")
        == "READY_FOR_GITHUB_ZENODO_DEPOSIT_PENDING_RELEASE_COMMIT_TAG_AND_UPLOAD",
        "unexpected FLARE-24 release-ledger status",
    )
    release = ledger.get("release", {})
    require(release.get("package_version") == "0.1.0rc1", "release-ledger version changed")
    require(release.get("external_publication_performed") is False, "external release misclaimed")
    require(release.get("doi_assigned") is False, "unassigned DOI misclaimed")
    require(
        ledger.get("evidence_boundary", {}).get("confirmation_outcomes_accessed") is False,
        "release ledger opened confirmation outcomes",
    )
    audit = ledger.get("archive_audit", {})
    require(audit.get("required_flare_modules") == 29, "release wheel module count changed")
    require(audit.get("missing_flare_modules") == [], "release wheel is incomplete")
    require(audit.get("forbidden_sdist_payloads") == [], "release sdist contains forbidden payloads")

    records: list[dict[str, Any]] = []
    records.extend(ledger.get("core_files", []))
    records.extend(ledger.get("source_files", []))
    records.extend(ledger.get("distributions", []))
    records.extend([ledger.get("quality_report", {}), ledger.get("smoke_report", {})])
    require(all(isinstance(record, dict) for record in records), "invalid ledger record")
    for record in records:
        recorded_path = Path(str(record.get("path", "")))
        target = recorded_path if recorded_path.is_absolute() else ROOT / recorded_path
        require(target.is_file(), f"release-ledger target missing: {recorded_path}")
        require(
            target.stat().st_size == record.get("bytes"),
            f"release-ledger size mismatch: {recorded_path}",
        )
        require(
            sha256_file(target) == record.get("sha256"),
            f"release-ledger hash mismatch: {recorded_path}",
        )


def validate_markdown_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", readme):
        target = target.strip().split(maxsplit=1)[0].strip("<>")
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        local = unquote(target.split("#", 1)[0])
        require((ROOT / local).exists(), f"broken README link: {target}")


def validate_public_text() -> None:
    # New isolated validation environments are not project source. Recognize
    # actual venv markers in the execution workspace, rather than exempting all
    # similarly named source folders or changing third-party package contents.
    execution_environments = [marker.parent for marker in ROOT.glob(".*/*/pyvenv.cfg")]
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or any(part in IGNORED_DIRECTORIES for part in path.parts)
            or any(path.is_relative_to(environment) for environment in execution_environments)
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
    validate_flare_evidence()
    validate_release_metadata()
    validate_release_ledger_if_present()
    validate_markdown_links()
    validate_public_text()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()
