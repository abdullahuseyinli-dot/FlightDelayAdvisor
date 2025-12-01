import pandas as pd
from pathlib import Path

# 1. Point this to your FULL weather dataset (wherever you have it locally)
SRC_PATH = Path("data/processed/bts_delay_2010_2024_balanced_research_weather.parquet")

# 2. Path for the SMALL version that only the app will use
DEST_PATH = Path("data/processed/bts_delay_2010_2024_balanced_research_weather_sample.parquet")

# 3. How many rows you want to keep for the app
#    Tweak this if needed. 1–2 million is usually okay on free tiers.
TARGET_ROWS = 2_000_000

print(f"Loading full dataset from {SRC_PATH} ...")
df = pd.read_parquet(SRC_PATH)
n_rows = len(df)
print(f"Full dataset has {n_rows:,} rows")

if n_rows > TARGET_ROWS:
    print(f"Sampling {TARGET_ROWS:,} rows (random, without replacement)...")
    df_sample = df.sample(n=TARGET_ROWS, random_state=42)
else:
    print("Dataset already small enough, using full data")
    df_sample = df

print(f"Saving sample to {DEST_PATH} ...")
DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
df_sample.to_parquet(DEST_PATH, index=False)
print("Done.")
