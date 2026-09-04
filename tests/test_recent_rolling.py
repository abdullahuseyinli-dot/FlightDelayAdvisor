from __future__ import annotations

import json

import pytest

from flightdelaybench.hashing import canonical_json_sha256
from flightdelaybench.recent_rolling import _frozen_parameters, _verify_tuning_report


def test_frozen_parameters_convert_zero_based_best_iteration() -> None:
    result = _frozen_parameters(
        {
            "best_iteration": 41,
            "best_parameters": {"depth": 8, "learning_rate": 0.04},
        }
    )
    assert result["iterations"] == 42
    assert result["bootstrap_type"] == "Bayesian"


def test_tuning_report_requires_valid_hash_and_2018(tmp_path) -> None:
    path = tmp_path / "tuning.json"
    body = {
        "schema_version": 1,
        "task": "delay",
        "validation_year": 2018,
        "best_iteration": 10,
        "best_parameters": {"depth": 7},
    }
    payload = {**body, "result_sha256": canonical_json_sha256(body)}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _verify_tuning_report(path)["task"] == "delay"

    payload["best_iteration"] = 11
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        _verify_tuning_report(path)
