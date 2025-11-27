# ✈️ FlightDelayAdvisor

FlightDelayAdvisor is a data-driven tool to **estimate delay and cancellation risk for US domestic flights**.

It combines:

- 2010–2024 US BTS On‑Time Performance data
- Route- and airline-level reliability statistics
- Airport congestion patterns by hour
- Monthly climatological airport weather
- Gradient-boosted tree models (CatBoost & LightGBM) with probability calibration

and exposes them through an interactive **Streamlit app**.

---

## 1. Project structure

```text
.
├── src/
│   ├── app.py                 # Streamlit app (main entrypoint for users)
│   ├── train_models.py        # (optional) training script for delay/cancel models
│   ├── prepare_dataset.py     # (optional) data engineering pipeline
│   └── evaluate_models.py     # script to reproduce key metrics
├── data/
│   ├── raw/                   # raw BTS data (gitignored)
│   └── processed/             # engineered dataset used by models (gitignored)
├── models/                    # trained model files (.joblib) (gitignored)
├── notebooks/                 # EDA / calibration / ablation notebooks (optional)
├── tests/                     # unit tests for feature/building & guidance
├── reports/
│   └── model_evaluation.txt   # saved evaluation summary (ROC-AUC, PR-AUC, etc.)
├── requirements.txt
├── README.md
└── .gitignore