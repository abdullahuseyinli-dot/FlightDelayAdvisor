from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightdelaybench.frontier import TABM_CATEGORICAL_FEATURES, TabMPreprocessor, fit_tabm
from flightdelaybench.modeling import (
    CATEGORICAL_FEATURES,
    CLIMATOLOGY_FEATURES,
    PRIOR_RATE_FEATURES,
    PRIOR_SUPPORT_FEATURES,
    REGISTERED_MODEL_INPUTS,
)


def _frontier_frame(rows: int = 96) -> pd.DataFrame:
    index = np.arange(rows)
    values: dict[str, object] = {}
    for column in REGISTERED_MODEL_INPUTS:
        if column in CATEGORICAL_FEATURES:
            values[column] = np.where(index % 2, "B", "A")
        elif column in PRIOR_RATE_FEATURES:
            values[column] = 0.05 + 0.4 * ((index % 11) / 10)
        elif column in PRIOR_SUPPORT_FEATURES:
            values[column] = 10.0 + index
        elif column in CLIMATOLOGY_FEATURES and column.endswith("_missing"):
            values[column] = index % 2
        else:
            values[column] = 1.0 + (index % 17)
    frame = pd.DataFrame(values)
    frame["sample_id"] = [f"sample-{value}" for value in index]
    frame["FlightDate"] = pd.date_range("2017-01-01", periods=rows)
    labels = (index % 4 == 0).astype(np.int64)
    frame["ArrDel15"] = labels
    frame["Cancelled"] = 0
    frame["delay_label_observed"] = 1
    frame["joint_label_observed"] = 1
    frame["disruption_state"] = labels
    return frame


def test_tabm_preprocessor_reserves_zero_for_unseen_categories() -> None:
    train = _frontier_frame()
    preprocessor = TabMPreprocessor.fit(train, seed=7, quantile_subsample=64)
    unseen = train.iloc[[0]].copy()
    unseen["Origin"] = "UNSEEN"
    numeric, categorical = preprocessor.transform(unseen)
    origin_index = TABM_CATEGORICAL_FEATURES.index("Origin")
    assert numeric.dtype == np.float32
    assert np.isfinite(numeric).all()
    assert categorical.dtype == np.int64
    assert categorical[0, origin_index] == 0
    assert all(value >= 3 for value in preprocessor.categorical_cardinalities)


def test_tabm_one_epoch_smoke_returns_coherent_probabilities() -> None:
    pytest.importorskip("tabm")
    pytest.importorskip("torch")
    frame = _frontier_frame()
    train = frame.iloc[:64].reset_index(drop=True)
    validation = frame.iloc[64:].reset_index(drop=True)
    model = fit_tabm(
        train,
        train["ArrDel15"].to_numpy(dtype=np.int64),
        task="delay",
        params={
            "device": "cpu",
            "inference_device": "cpu",
            "amp": False,
            "max_epochs": 1,
            "patience": 0,
            "batch_size": 32,
            "eval_batch_size": 32,
            "k": 2,
            "n_blocks": 1,
            "d_block": 16,
            "d_embedding": 4,
            "n_bins": 4,
            "quantile_subsample": 64,
            "bin_sample_rows": 64,
        },
        validation_frame=validation,
        validation_labels=validation["ArrDel15"].to_numpy(dtype=np.int64),
    )
    probabilities = model.predict_proba(validation)
    assert probabilities.shape == (len(validation), 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert model.best_epoch == 1
