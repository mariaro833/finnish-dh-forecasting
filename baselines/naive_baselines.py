
"""

Naive baselines aligned EXACTLY to XGB v3 indexing.



Two reference forecasts:

  (a) "Persistence-from-base": for each horizon h, predict heat_load[t+h]

      using the same base lag the XGB-resid model uses:

        h=1  -> heat_lag_1h[t]

        h=2  -> heat_lag_2h[t]

        h>=3 -> heat_lag_24h[t]

      (i.e., XGB-resid with all residual predictions set to zero)

  (b) "Seasonal naive 24h": for every horizon h, predict heat_load[t+h]

      using heat_lag_24h[t] (same hour previous day).



Same row filter as XGB v3: year==2024, all targets and base lags non-null.

"""

import json

import numpy as np

import pandas as pd



CSV = "data_full_real.csv"

TARGET = "heat_load"

H = 24

TRAIN_YEARS = list(range(2016, 2023))

VAL_YEAR = 2023

TEST_YEAR = 2024



print("Loading data...")

df = pd.read_csv(CSV)

print(f"  {len(df):,} rows")



# Build per-horizon targets

for h in range(1, H + 1):

    df[f"y_h{h}"] = df[TARGET].shift(-h)



target_cols = [f"y_h{h}" for h in range(1, H + 1)]

base_lag_cols = ["heat_lag_1h", "heat_lag_2h", "heat_lag_24h"]



valid_mask = (df[target_cols].notna().all(axis=1)

              & df[base_lag_cols].notna().all(axis=1)

              & df["year"].notna())



test_df = df[(df["year"] == TEST_YEAR) & valid_mask].copy()

print(f"  Test rows (2024): {len(test_df):,}")



# Per-horizon naive errors

mae_persistence = []

mae_seasonal = []



for h in range(1, H + 1):

    y_true = test_df[f"y_h{h}"].values



    # Persistence-from-base, matching XGB-resid base by horizon

    if h == 1:

        base_col = "heat_lag_1h"

    elif h == 2:

        base_col = "heat_lag_2h"

    else:

        base_col = "heat_lag_24h"

    y_persist = test_df[base_col].values



    # Seasonal naive 24h: always heat_lag_24h

    y_seasonal = test_df["heat_lag_24h"].values



    mae_p = float(np.mean(np.abs(y_true - y_persist)))

    mae_s = float(np.mean(np.abs(y_true - y_seasonal)))

    mae_persistence.append(mae_p)

    mae_seasonal.append(mae_s)



    print(f"  h={h:2d}: persistence-from-base ({base_col}) MAE={mae_p:7.2f}, seasonal-24h MAE={mae_s:7.2f}")



results = {

    "n_test_windows": int(len(test_df)),

    "persistence_from_base": {

        "description": "Predict heat_load[t+h] using same base lag XGB-resid uses",

        "base_h1": "heat_lag_1h",

        "base_h2": "heat_lag_2h",

        "base_h3plus": "heat_lag_24h",

        "mae_per_horizon": mae_persistence,

        "mae_mean": float(np.mean(mae_persistence)),

        "mae_h1": mae_persistence[0],

        "mae_h2": mae_persistence[1],

    },

    "seasonal_naive_24h": {

        "description": "Predict heat_load[t+h] using heat_lag_24h[t] for all horizons",

        "mae_per_horizon": mae_seasonal,

        "mae_mean": float(np.mean(mae_seasonal)),

        "mae_h1": mae_seasonal[0],

    },

}



with open("results_naive.json", "w") as f:

    json.dump(results, f, indent=2)



print()

print("Summary:")

pb = results["persistence_from_base"]

sn = results["seasonal_naive_24h"]

print(f"  Persistence-from-base: mean MAE={pb['mae_mean']:7.2f}, h=1={pb['mae_h1']:6.2f}, h=2={pb['mae_h2']:6.2f}")

print(f"  Seasonal naive 24h:    mean MAE={sn['mae_mean']:7.2f}, h=1={sn['mae_h1']:6.2f}")

print()

print("Saved: results_naive.json")

