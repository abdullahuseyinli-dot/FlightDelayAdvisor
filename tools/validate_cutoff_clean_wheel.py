"""Install a new wheel in a new environment and preserve an inference-only audit."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flightdelaybench.flare_release import inspect_release_archives
from flightdelaybench.hashing import canonical_json_sha256, sha256_file, write_canonical_json

ROOT = Path(__file__).resolve().parents[1]


def venv_python(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def inference_command(
    python: Path,
    data: Path,
    smoke: Path,
    frozen: dict[str, Any],
    *,
    delay_model: Path | None = None,
    cancellation_model: Path | None = None,
) -> list[str]:
    """Remap model locations while retaining the recorded model-byte requirements."""
    return [str(python), "-I", "-B", "-m", "flightdelaybench.flare_smoke",
            "--census-dir", str(data / "normalized_top100_census_v2"),
            "--recent-dir", str(data / "derived/census_recent_prior_day_v2"),
            "--flight-recent-dir", str(data / "derived/census_flight_recent_v2_duckdb"),
            "--graph-dir", str(data / "derived/census_graph_clsgmp_v1"),
            "--weather-feature-dir", str(data / "derived/flare24_features_v1"),
            "--rotation-feature-dir", str(data / "derived/flare24_rotation_features_v5"),
            "--delay-model", str(delay_model.resolve()) if delay_model is not None else frozen["models"]["delay"]["path"],
            "--cancellation-model", str(cancellation_model.resolve()) if cancellation_model is not None else frozen["models"]["cancellation"]["path"],
            "--delay-model-sha256", frozen["models"]["delay"]["sha256"],
            "--cancellation-model-sha256", frozen["models"]["cancellation"]["sha256"],
            "--execution-context", "fresh_wheel", "--limit", "128", "--output", str(smoke)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--venv-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--delay-model", type=Path, help="Relocated frozen delay model; the recorded SHA-256 still applies")
    parser.add_argument("--cancellation-model", type=Path, help="Relocated frozen cancellation model; the recorded SHA-256 still applies")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.venv_dir.exists():
        raise FileExistsError("clean-wheel validation requires new output and environment paths")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to create the isolated environment")
    wheel = args.artifact_dir.resolve() / "flightdelaybench-0.1.0rc1-py3-none-any.whl"
    sdist = args.artifact_dir.resolve() / "flightdelaybench-0.1.0rc1.tar.gz"
    archive_audit = inspect_release_archives(wheel, sdist)
    required = [f"flightdelaybench/{name}.py" for name in ("cutoff_history", "cutoff_dataset", "cutoff_experiments", "cutoff_weather_pilot")]
    with zipfile.ZipFile(wheel) as archive:
        missing = sorted(set(required) - set(archive.namelist()))
    if missing:
        raise ValueError(f"wheel omits corrected modules: {missing}")
    environment = args.venv_dir.resolve()
    python = venv_python(environment)
    smoke = args.output.resolve().with_name(args.output.stem + "_inference.json")
    frozen = json.loads((ROOT / "reports/validation/flare24_fresh_wheel_smoke_v3_bcpotr.json").read_text(encoding="utf-8"))
    data = args.data_root.resolve()
    command = inference_command(
        python, data, smoke, frozen,
        delay_model=args.delay_model, cancellation_model=args.cancellation_model,
    )
    checks: list[dict[str, Any]] = []
    phases = [
        ("create_clean_environment", [uv, "venv", "--python", sys.executable, str(environment)]),
        ("install_wheel_and_models", [uv, "pip", "install", "--python", str(python), "--no-cache", f"{wheel}[models]"]),
        ("isolated_import", [str(python), "-I", "-B", "-c", "from pathlib import Path; import sys; import flightdelaybench.cutoff_experiments as m; assert Path(m.__file__).is_relative_to(Path(sys.prefix)); print(m.__file__)"]),
        ("inference_smoke", command),
    ]
    for name, phase in phases:
        started = time.perf_counter()
        completed = subprocess.run(phase, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        checks.append({"name": name, "command": phase, "exit_code": completed.returncode,
                       "elapsed_seconds": time.perf_counter() - started, "stdout": completed.stdout, "stderr": completed.stderr})
        print(json.dumps({"phase": name, "exit_code": completed.returncode}), flush=True)
        if completed.returncode:
            break
    passed = len(checks) == len(phases) and all(check["exit_code"] == 0 for check in checks)
    report: dict[str, Any] = {
        "schema_version": 1, "status": "PASS_CUTOFF_CLEAN_WHEEL_VALIDATION" if passed else "FAIL_CUTOFF_CLEAN_WHEEL_VALIDATION",
        "created_at_utc": datetime.now(UTC).isoformat(), "checks": checks,
        "wheel": {"path": wheel.as_posix(), "bytes": wheel.stat().st_size, "sha256": sha256_file(wheel)},
        "sdist": {"path": sdist.as_posix(), "bytes": sdist.stat().st_size, "sha256": sha256_file(sdist)},
        "archive_audit": archive_audit, "required_cutoff_modules": required, "missing_cutoff_modules": missing,
        "research_performance_reestimated": False, "publication_authorized": False,
        "claim_limit": "Fresh environment, wheel imports, archive payload and legacy-model inference only; corrected dataset and research experiments remain separate gates.",
        "confirmation_outcomes_accessed": False,
    }
    if smoke.is_file():
        report["smoke"] = {"path": smoke.as_posix(), "sha256": sha256_file(smoke)}
    report["report_sha256"] = canonical_json_sha256(report)
    write_canonical_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": args.output.as_posix()}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
