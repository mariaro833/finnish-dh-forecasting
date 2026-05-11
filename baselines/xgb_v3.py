"""
XGBoost baseline v3 for national CHP heat forecasting.

Improvements over v2:
  1. Graded persistence base by horizon (matches the closest available lag):
        h=1  -> heat_lag_1h
        h=2  -> heat_lag_2h
        h>=3 -> heat_lag_24h  (smallest causal-and-cleanly-anchored lag for >2h)
     Rationale: residual = heat(t+h) - heat_lag_X(t). The base must be
     stationary across train/test. Closer lags carry more info but for h>=3
     we only have heat_lag_{24,48,168} (the smaller ones describe values
     INSIDE the forecast window relative to the prediction target).

  2. Per-horizon feature pruning: at horizon h, drop heat_lag_kh features
     where k < h, since those describe heat values inside the prediction
     window and are not meaningful predictors for that target. Rolling
     features (computed up to row t) remain available for all horizons.

  3. Same Espoo static-feature drop as v2.

Same 3-way split, same quantile loss, same metrics format as v2.
"""

import json
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

H = 24
QUANTILES = [0.1, 0.5, 0.9]
TRAIN_YEARS = list(range(2016, 2023))
VAL_YEAR = 2023
TEST_YEAR = 2024

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 30,
}

DROP_STATIC = ["building_year", "floor_area_m2", "dh_share", "population"]
DROP_META = ["time", "group", "year_month", "time_idx"]
TARGET = "heat_load"

# All heat lag feature names available in dataset
HEAT_LAG_FEATS = {1: "heat_lag_1h", 2: "heat_lag_2h", 24: "heat_lag_24h",
                  48: "heat_lag_48h", 168: "heat_lag_168h"}


def base_lag_for_horizon(h):
    """Persistence base used for residual target at horizon h."""
    if h == 1:
        return "heat_lag_1h"
    if h == 2:
        return "heat_lag_2h"
    return "heat_lag_24h"


def features_for_horizon(h, all_features):
    """Drop heat_lag_kh features where k < h (stale at issue time for predicting t+h).

    Rolling features (heat_roll*) are causal at row t, so available for all h.
    All other features (weather, calendar, prices) are available unconditionally.
    """
    drop = set()
    for k, name in HEAT_LAG_FEATS.items():
        if k < h:
            drop.add(name)
    return [f for f in all_features if f not in drop]


print("Loading data...")
df = pd.read_csv("data_full_real.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)
print(f"  {len(df):,} rows")

all_feature_cols = [
    c for c in df.columns
    if c not in DROP_META + [TARGET] + DROP_STATIC + ["year", "month", "hour", "dayofweek"]
]
print(f"  Feature universe ({len(all_feature_cols)} cols): drops {DROP_STATIC} + raw date ints")

print("Building horizon targets and residual targets...")
for h in range(1, H + 1):
    df[f"y_h{h}"] = df[TARGET].shift(-h)
    base_col = base_lag_for_horizon(h)
    df[f"resid_h{h}"] = df[f"y_h{h}"] - df[base_col]

target_cols = [f"y_h{h}" for h in range(1, H + 1)]
resid_cols = [f"resid_h{h}" for h in range(1, H + 1)]
valid_mask = df[target_cols].notna().all(axis=1) & df[resid_cols].notna().all(axis=1)

train_df = df[df["year"].isin(TRAIN_YEARS) & valid_mask].copy()
val_df = df[(df["year"] == VAL_YEAR) & valid_mask].copy()
test_df = df[(df["year"] == TEST_YEAR) & valid_mask].copy()
print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

n_test = len(test_df)
preds_q = np.zeros((n_test, H, len(QUANTILES)))
y_true = test_df[target_cols].values

feat_imp_global = {f: 0.0 for f in all_feature_cols}
feat_imp_count = {f: 0 for f in all_feature_cols}

print(f"Training {H} horizons x {len(QUANTILES)} quantiles = {H * len(QUANTILES)} models")
print("  Target = heat_load(t+h) - base_lag(t)  (residual)")
print()

for h in range(1, H + 1):
    h_features = features_for_horizon(h, all_feature_cols)
    X_train = train_df[h_features].values
    X_val = val_df[h_features].values
    X_test = test_df[h_features].values
    base_col = base_lag_for_horizon(h)
    base_test = test_df[base_col].values

    y_train_h = train_df[f"resid_h{h}"].values
    y_val_h = val_df[f"resid_h{h}"].values

    for qi, q in enumerate(QUANTILES):
        model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **XGB_PARAMS)
        model.fit(X_train, y_train_h, eval_set=[(X_val, y_val_h)], verbose=False)
        resid_pred = model.predict(X_test)
        preds_q[:, h - 1, qi] = base_test + resid_pred
        if q == 0.5:
            for f, imp in zip(h_features, model.feature_importances_):
                feat_imp_global[f] += float(imp)
                feat_imp_count[f] += 1

    h_mae = float(np.mean(np.abs(preds_q[:, h - 1, 1] - y_true[:, h - 1])))
    print(f"  Horizon {h:>2}h done  (n_features={len(h_features)}, base={base_col}, MAE={h_mae:.2f})")

feat_imp_avg = {
    f: (feat_imp_global[f] / feat_imp_count[f]) if feat_imp_count[f] > 0 else 0.0
    for f in all_feature_cols
}

print()
print("Computing metrics...")
median_idx = QUANTILES.index(0.5)
y_pred = preds_q[:, :, median_idx]
mae = float(np.mean(np.abs(y_pred - y_true)))
rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
nonzero = y_true > 1.0
mape = float(np.mean(np.abs((y_pred[nonzero] - y_true[nonzero]) / y_true[nonzero])) * 100)
smape = float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)) * 100)
lo = preds_q[:, :, 0]
hi = preds_q[:, :, -1]
pi_coverage = float(np.mean((y_true >= lo) & (y_true <= hi)) * 100)
pi_width = float(np.mean(hi - lo))

mae_per_h = np.mean(np.abs(y_pred - y_true), axis=0).tolist()
rmse_per_h = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0)).tolist()

imp_items = sorted(feat_imp_avg.items(), key=lambda kv: kv[1], reverse=True)
top_features = [(f, imp) for f, imp in imp_items[:10]]

print()
print("=" * 60)
print("XGBOOST v3 RESULTS (graded residual target, held-out 2024)")
print("=" * 60)
print(f"  MAE:           {mae:.2f} MWh/h")
print(f"  RMSE:          {rmse:.2f} MWh/h")
print(f"  MAPE:          {mape:.2f} %")
print(f"  sMAPE:         {smape:.2f} %")
print(f"  80% PI cover:  {pi_coverage:.2f} %")
print(f"  80% PI width:  {pi_width:.2f} MWh/h")
print("=" * 60)
print("Per-horizon MAE:")
for h_i, m in enumerate(mae_per_h, 1):
    print(f"  h={h_i:>2d}  MAE={m:7.2f}")
print("=" * 60)
print("Top 10 features (avg importance across horizons):")
for name, imp in top_features:
    print(f"  {name:<30s}  {imp:.4f}")

results = {
    "test_mae_mwh_h": mae,
    "test_rmse_mwh_h": rmse,
    "test_mape_pct": mape,
    "test_smape_pct": smape,
    "test_pi80_coverage_pct": pi_coverage,
    "test_pi80_width_mwh_h": pi_width,
    "n_test_windows": int(n_test),
    "mae_per_horizon": mae_per_h,
    "rmse_per_horizon": rmse_per_h,
    "top_10_features": top_features,
    "config": {
        "target": "graded_residual_from_persistence",
        "base_h1": "heat_lag_1h",
        "base_h2": "heat_lag_2h",
        "base_h3plus": "heat_lag_24h",
        "dropped_static": DROP_STATIC,
    },
}
with open("results_xgb_v3.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez_compressed(
    "predictions_xgb_v3.npz",
    y_true=y_true,
    quantiles=preds_q,
    quantile_levels=np.array(QUANTILES),
    feature_names=np.array(all_feature_cols),
    feature_importance=np.array([feat_imp_avg[f] for f in all_feature_cols]),
)
print()
print("Saved: results_xgb_v3.json, predictions_xgb_v3.npz")
print("Done.")