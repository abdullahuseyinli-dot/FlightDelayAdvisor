import subprocess
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_evaluate_models_script_runs():
    script = PROJECT_ROOT / "src" / "evaluate_models.py"
    result = subprocess.run(
        ["python", str(script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"evaluate_models.py failed:\n{result.stderr}"


@pytest.mark.slow
def test_generate_plots_script_runs():
    script = PROJECT_ROOT / "src" / "generate_plots_only.py"
    result = subprocess.run(
        ["python", str(script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"generate_plots_only.py failed:\n{result.stderr}"
