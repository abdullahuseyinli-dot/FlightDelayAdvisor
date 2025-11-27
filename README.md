# ✈️ FlightDelayAdvisor

FlightDelayAdvisor is a data-driven tool to **estimate delay and cancellation risk for US domestic flights**.

It combines:

- **2010–2024 US BTS On-Time Performance data**
- **Route- and airline-level reliability statistics**
- **Airport congestion patterns by hour of day**
- **Monthly climatological weather at origin/destination airports**
- **Modern tabular ML models (CatBoost & LightGBM) with probability calibration**

and exposes them through an interactive **Streamlit app** for “what-if” analysis and decision support.

---

## 1. Project structure

```text
.
├── src/
│   ├── app.py                   # Streamlit app (main user entrypoint)
│   ├── download_bts.py          # Script to download raw BTS On-Time data (2010–2024)
│   ├── prepare_dataset.py       # Build sampled, feature-engineered BTS dataset
│   ├── download_airport_weather.py  # Download daily airport weather for selected airports
│   ├── add_weather_to_dataset.py    # Join airport weather onto BTS dataset
│   ├── train_models.py          # Train delay (ArrDel15) and cancellation models
│   ├── evaluate_models.py       # Evaluate trained models (ROC-AUC, PR-AUC, Brier, etc.)
│   └── generate_plots_only.py   # Regenerate CatBoost diagnostics (ROC/PR/Calib/CM/Feature importance)
│
├── data/
│   ├── raw/                     # Raw BTS zip files (download_bts.py)               [gitignored]
│   ├── external/                # External sources (e.g. airport_daily_weather)     [gitignored]
│   └── processed/               # Engineered parquet datasets                       [gitignored]
│
├── models/                      # Trained model artefacts (.joblib, .pt, calibrators)  [gitignored]
│
├── tests/
│   └── test_flight_delay_pipeline.py
│       # Regression tests:
│       # - schema consistency (FEATURE_COLS)
│       # - feature builder sanity
│       # - model probability range
│       # - minimum ROC-AUC for delay & cancellation models
│
├── reports/
│   ├── metrics_summary.txt      # Consolidated metrics from train_models.py (multiple models + ablations)
│   └── figures/                 # Diagnostic plots (ROC, PR, calibration, CM, feature importance)
│
├── notebooks/                   # Optional: EDA / calibration / ablation notebooks   [not required]
│
├── requirements.txt             # Project dependencies for the .project1venv
├── README.md                    # This file
└── .gitignore                   # Ignore venv, data, models, caches, etc.
