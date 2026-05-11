
import os, time, requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

FINGRID_API_KEY = os.environ.get("FINGRID_API_KEY", "")



DATASET_ID      = 201
START_DATE      = "2016-01-01"
END_DATE        = "2024-12-31"
PAGE_SIZE       = 20000

if not FINGRID_API_KEY:
    raise SystemExit("Set FINGRID_API_KEY environment variable first")

def fetch_fingrid(dataset_id, start, end, api_key):
    base_url = f"https://data.fingrid.fi/api/datasets/{dataset_id}/data"
    headers  = {"x-api-key": api_key, "Accept": "application/json"}
    all_rows, page = [], 1
    while True:
        params = {
            "startTime": f"{start}T00:00:00Z",
            "endTime":   f"{end}T23:59:59Z",
            "format": "json", "oneRowPerTimePeriod": False,
            "pageSize": PAGE_SIZE, "page": page,
            "sortBy": "startTime", "sortOrder": "asc",
        }

        print(f"  Page {page} ...", end=" ", flush=True)
        resp = requests.get(base_url, headers=headers, params=params, timeout=60)

        if resp.status_code == 429:
            print("rate-limited, waiting 10s ...")
            time.sleep(10)
            continue
        resp.raise_for_status()
        rows = resp.json().get("data", [])

        print(f"{len(rows)} rows  (total: {len(all_rows)+len(rows):,})")

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < PAGE_SIZE:
            break

        page += 1
        time.sleep(0.5)

    return all_rows


print(f"Fetching Fingrid dataset {DATASET_ID} ({START_DATE} to {END_DATE}) ...")
records = fetch_fingrid(DATASET_ID, START_DATE, END_DATE, FINGRID_API_KEY)
print(f"Total records: {len(records):,}")

df = pd.DataFrame(records)
df["time"]      = pd.to_datetime(df["startTime"], utc=True).dt.tz_convert("Europe/Helsinki")
df["heat_load"] = pd.to_numeric(df["value"], errors="coerce")
df = df[["time","heat_load"]].dropna().sort_values("time").reset_index(drop=True)
print(f"Raw rows (3-min): {len(df):,}")

df_hourly = (df.set_index("time").resample("1h")["heat_load"]
               .mean().dropna().reset_index())
print(f"After hourly resample: {len(df_hourly):,} rows")
print(f"Mean: {df_hourly['heat_load'].mean():.1f}  Max: {df_hourly['heat_load'].max():.1f} MWh/h")

df_hourly["year"]       = df_hourly["time"].dt.year
df_hourly["month"]      = df_hourly["time"].dt.month
df_hourly["hour"]       = df_hourly["time"].dt.hour
df_hourly["dayofweek"]  = df_hourly["time"].dt.dayofweek
df_hourly["year_month"] = (df_hourly["year"].astype(str) + "-" +
                            df_hourly["month"].astype(str).str.zfill(2))

for lag in [1, 2, 24, 48, 168]:
    df_hourly[f"heat_lag_{lag}h"] = df_hourly["heat_load"].shift(lag)

df_hourly["heat_roll24h_mean"] = df_hourly["heat_load"].rolling(24, min_periods=1).mean().round(2)
df_hourly["heat_roll24h_std"]  = df_hourly["heat_load"].rolling(24, min_periods=1).std().round(2)
df_hourly.to_csv("fingrid_hourly_raw.csv", index=False)
print(f"Saved: fingrid_hourly_raw.csv  ({len(df_hourly):,} rows)")

monthly = (df_hourly.groupby(["year_month","year","month"])
           .agg(total_heat_mwh=("heat_load","sum"),
                mean_heat_load_mwh_h=("heat_load","mean"),
                hours_of_data=("heat_load","count"))
           .reset_index().sort_values(["year","month"]))

monthly["data_completeness_pct"] = (monthly["hours_of_data"] / 720 * 100).round(1)
monthly["dc_share"] = 0.0
monthly["dc_waste_heat_share_pct"] = 0.0
monthly["dc_heat_mwh"] = 0.0
monthly["hp_base_mwh"] = (monthly["total_heat_mwh"] * 0.45).round(0)
monthly["fossil_mwh"]  = (monthly["total_heat_mwh"] - monthly["hp_base_mwh"]).round(0)
monthly["co2_saved_tonnes"] = 0.0
monthly.to_csv("fingrid_monthly.csv", index=False)

print(f"Saved: fingrid_monthly.csv  ({len(monthly)} rows)")
print(monthly[["year_month","total_heat_mwh","mean_heat_load_mwh_h"]].tail(6).to_string(index=False))
