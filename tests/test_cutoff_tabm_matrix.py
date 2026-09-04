from __future__ import annotations

import numpy as np
import pandas as pd

from flightdelaybench.frontier import TabMPreprocessor


def test_current_feature_tabm_preprocessor_uses_train_only_columns_and_imputation() -> None:
    train = pd.DataFrame({
        "Origin": ["JFK", "LAX", "JFK", "LAX"],
        "asof_global_delay_rate_7d": [0.1, 0.2, 0.3, np.nan],
        "Cancelled": [0, 0, 0, 1],
    })
    features = ("Origin", "asof_global_delay_rate_7d")
    processor = TabMPreprocessor.fit(
        train, seed=3, quantile_subsample=100,
        categorical_features=("Origin",), feature_columns=features,
    )
    assert processor.numeric_features == ("asof_global_delay_rate_7d",)
    assert processor.numeric_fill_values == {"asof_global_delay_rate_7d": 0.2}
    test = pd.DataFrame({"Origin": ["UNSEEN"], "asof_global_delay_rate_7d": [np.nan]})
    numeric, categories = processor.transform(test)
    assert np.isfinite(numeric).all()
    assert categories.tolist() == [[0]]
    assert processor.categorical_cardinalities == [3]
