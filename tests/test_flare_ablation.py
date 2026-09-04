from __future__ import annotations

import numpy as np
import pytest

from flightdelaybench.flare_ablation import binary_loss_rows


def test_binary_loss_rows_matches_hand_calculation() -> None:
    log_rows, brier_rows = binary_loss_rows([0, 1], [0.25, 0.8])

    assert log_rows == pytest.approx([-np.log(0.75), -np.log(0.8)])
    assert brier_rows == pytest.approx([0.25**2, 0.2**2])


def test_binary_loss_rows_rejects_nonbinary_labels() -> None:
    with pytest.raises(ValueError, match="invalid"):
        binary_loss_rows([0, 2], [0.2, 0.8])
