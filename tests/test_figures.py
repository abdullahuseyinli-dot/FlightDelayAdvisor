from __future__ import annotations

from flightdelaybench.figures import _task


def test_task_extracts_exactly_one_result() -> None:
    report = {"task_results": [{"task": "delay", "value": 1}]}
    assert _task(report, "delay")["value"] == 1
