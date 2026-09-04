from __future__ import annotations

import numpy as np
import pytest

from flightdelaybench.flare_audit_recovery import normalize_persisted_joint


def test_normalize_persisted_joint_repairs_bounded_float32_drift() -> None:
    probabilities = np.asarray(
        [[0.1, 0.2, 0.7], [0.73, 0.26, 0.01]],
        dtype=np.float32,
    )

    normalized, audit = normalize_persisted_joint(probabilities, method="candidate")

    assert np.allclose(normalized.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    assert audit["storage_dtype"] == "float32"
    assert audit["maximum_absolute_row_sum_error_before_normalization"] <= 1e-6
    assert audit["rows_over_recovery_1e_6_absolute_tolerance"] == 0


def test_normalize_persisted_joint_rejects_material_incoherence() -> None:
    probabilities = np.asarray([[0.1, 0.2, 0.6]], dtype=np.float32)

    with pytest.raises(ValueError, match="exceeds the float32 recovery bound"):
        normalize_persisted_joint(probabilities, method="candidate")
