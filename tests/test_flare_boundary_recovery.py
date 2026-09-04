from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from flightdelaybench.flare_boundary_recovery import (
    _assert_probability_reproduction,
    _inventory_completed_run,
    _validate_locked_model_parameters,
)
from flightdelaybench.flare_boundary_study import CANDIDATES, SELECTION_MONTHS
from flightdelaybench.flare_study import TASKS


class _Estimator:
    tree_count_ = 7

    def __init__(self, parameters: dict[str, Any]) -> None:
        self._parameters = parameters

    def get_params(self) -> dict[str, Any]:
        return self._parameters


class _Model:
    candidate = "boundary_only"

    def __init__(self, parameters: dict[str, Any]) -> None:
        self.estimator = _Estimator(parameters)


def _complete_inventory(root: Path) -> None:
    paths = [root / "method_lock.json"]
    paths.extend(
        root / "models" / f"{candidate}_{task}.joblib"
        for candidate in CANDIDATES
        for task in TASKS
    )
    paths.extend(
        root / "calibrators" / f"{candidate}_{task}_identity.joblib"
        for candidate in CANDIDATES
        for task in TASKS
    )
    paths.extend(
        root / "predictions" / f"raw_selection_2024_{month:02d}.parquet"
        for month in SELECTION_MONTHS
    )
    paths.append(root / "predictions" / "selection_2024_crossfit_joint.parquet")
    paths.extend(
        root / "predictions" / f"retrospective_2025_{month:02d}.parquet"
        for month in range(1, 13)
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_completed_run_inventory_is_exact(tmp_path: Path) -> None:
    _complete_inventory(tmp_path)
    calibrators = _inventory_completed_run(tmp_path)
    assert len(calibrators) == 4
    extra = tmp_path / "predictions" / "unadjudicated.part"
    extra.touch()
    with pytest.raises(ValueError, match="artifact inventory differs"):
        _inventory_completed_run(tmp_path)


def test_probability_reproduction_has_a_hard_tolerance() -> None:
    expected = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
    assert _assert_probability_reproduction(
        expected,
        expected + 1e-8,
        role="test",
    ) == pytest.approx(1e-8)
    with pytest.raises(ValueError, match="predictions differ"):
        _assert_probability_reproduction(
            expected,
            expected + 1e-3,
            role="test",
        )


def test_locked_model_parameters_ignore_only_predeclared_iteration_cap() -> None:
    locked = {
        "iterations": 1_200,
        "learning_rate": 0.02,
        "boosting_type": "Plain",
    }
    actual = {
        **locked,
        "iterations": 7,
        "random_seed": 20260903,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "bootstrap_type": "Bayesian",
        "task_type": "GPU",
        "devices": "0",
        "verbose": False,
        "allow_writing_files": False,
    }
    model = _Model(actual)
    result = _validate_locked_model_parameters(
        model,  # type: ignore[arg-type]
        task="delay",
        lock={"model_parameters": {"delay": locked}},
    )
    assert result["iterations"] == 7
    actual["learning_rate"] = 0.03
    with pytest.raises(ValueError, match="learning_rate"):
        _validate_locked_model_parameters(
            model,  # type: ignore[arg-type]
            task="delay",
            lock={"model_parameters": {"delay": locked}},
        )
