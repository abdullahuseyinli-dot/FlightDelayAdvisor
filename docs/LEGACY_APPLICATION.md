# Legacy application and sampled benchmark

[Documentation index](README.md)

This guide preserves the original Streamlit application track. It uses a sampled
2010–2024 dataset and legacy model artifacts, not the complete-context FLARE/CC-RTH
research pipeline. Its historical scores use different cohorts and protocols and
must not be mixed into the current matched results table.

The previous main snapshot is preserved on
[legacy/streamlit-baseline-20260905](https://github.com/abdullahuseyinli-dot/FlightDelayAdvisor/tree/legacy/streamlit-baseline-20260905)
at `34ac793a6e681a95fdd0ec2044918cb615d146aa`. The application files also remain
available in current main for reproducibility; they were not deleted or relabelled
as corrected research. See [versioning](VERSIONING.md).

## Recorded scores

Source: [in-period summary](../reports/metrics_summary.txt).

| Target | Selected model | ROC-AUC | Average precision | Brier |
|---|---|---:|---:|---:|
| Arrival delay at least 15 minutes | CatBoost | 0.6864 | 0.3819 | 0.1587 |
| Cancellation | LightGBM | 0.7061 | 0.1089 | 0.0171 |

Source: [sampled 2025 check](../reports/backtest_2025_metrics.txt).

| Target | Rows | Base rate | ROC-AUC | Average precision | Brier | Top-decile rate |
|---|---:|---:|---:|---:|---:|---:|
| Arrival delay at least 15 minutes | 196,693 | 0.2287 | 0.6469 | 0.3460 | 0.1703 | 0.4190 |
| Cancellation | 200,000 | 0.0165 | 0.6543 | 0.0368 | 0.0165 | 0.0435 |

The retained earlier 500,000-row backtest log is a different run and is not the
source of this table. The 2025 results are historical out-of-time evidence, not a
new blind holdout for subsequent research.

## Run the demo

In a source checkout with Git LFS installed:

```bash
git lfs install
git lfs pull
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use a dedicated activated environment. Model and data locations are defined in
[model_paths.yml](../config/model_paths.yml). The first load constructs route,
airline, congestion and monthly-weather aggregates from the materialized Parquet
file; subsequent interactions use the application cache.

The interface supports single-flight historical risk estimates, departure-time and
airline comparisons, airport summaries and route alternatives. It is an educational
historical-risk demo, not a live forecast or travel guarantee. Monthly climatology
does not represent a particular storm or current operational disruption.

## Legacy pipeline locations

The download, preparation, weather join, training and evaluation entry points are
`src/download_bts.py`, `src/prepare_dataset.py`,
`src/download_airport_weather.py`, `src/add_weather_to_dataset.py`,
`src/train_models.py` and `src/evaluate_models.py`.
The 2025 scripts are `src/prepare_bts_2025_for_backtest.py` and
`src/backtest_2025_from_processed.py`.

Several scripts use fixed output paths and can overwrite tracked historical
reports. Do not execute them in the preserved evidence checkout. Reproduction
requires an isolated working copy and a reviewed output-path plan; these scripts
are not the create-only corrected research runner.

The [original design document](FlightDelayAdvisor_Documentation.md) retains the
application's contributor attribution and design history. Current source usage,
research limitations and release status are in the [main documentation](README.md).
