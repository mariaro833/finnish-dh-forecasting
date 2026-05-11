
import requests

import pandas as pd

import numpy as np

import time



START_DATE = "2016-01-01"

END_DATE   = "2024-12-31"



LOCATIONS = [

    {"name": "Helsinki",  "lat": 60.1699, "lon": 24.9384},

    {"name": "Espoo",     "lat": 60.2052, "lon": 24.6559},

    {"name": "Vantaa",    "lat": 60.2934, "lon": 25.0378},

]



VARIABLES = [

    "temperature_2m","wind_speed_10m","relative_humidity_2m",

    "precipitation","snowfall","surface_pressure",

    "shortwave_radiation","apparent_temperature",

]



BASE_URL = "https://archive-api.open-meteo.com/v1/archive"



def fetch_location(name, lat, lon):

    print(f"\nFetching {name} ({lat}, {lon}) ...")

    params = {

        "latitude": lat, "longitude": lon,

        "start_date": START_DATE, "end_date": END_DATE,

        "hourly": ",".join(VARIABLES),

        "timezone": "Europe/Helsinki",

        "wind_speed_unit": "ms",

    }

    for attempt in range(5):

        try:

            resp = requests.get(BASE_URL, params=params, timeout=120)

            resp.raise_for_status()

            data = resp.json()

            break

        except Exception as e:

            print(f"  Attempt {attempt+1} failed: {e}")

            time.sleep(5 * (attempt + 1))

    else:

        raise RuntimeError(f"Failed to fetch {name}")

    df = pd.DataFrame(data["hourly"])

    df["time"] = pd.to_datetime(df["time"])

    df["location"] = name

    print(f"  {len(df):,} rows")

    return df



dfs = []

for loc in LOCATIONS:

    dfs.append(fetch_location(loc["name"], loc["lat"], loc["lon"]))

    time.sleep(1)



print("\nAveraging across three locations...")

all_df = pd.concat(dfs)

numeric_cols = [c for c in all_df.columns if c not in ["time","location"]]

avg_df = all_df.groupby("time")[numeric_cols].mean().reset_index().sort_values("time")



avg_df["hdh"]           = (17 - avg_df["temperature_2m"]).clip(lower=0).round(2)

avg_df["hour"]          = avg_df["time"].dt.hour

avg_df["month"]         = avg_df["time"].dt.month

avg_df["year"]          = avg_df["time"].dt.year

avg_df["dayofweek"]     = avg_df["time"].dt.dayofweek

avg_df["is_weekend"]    = (avg_df["dayofweek"] >= 5).astype(int)

avg_df["hour_sin"]      = np.sin(2 * np.pi * avg_df["hour"]      / 24)

avg_df["hour_cos"]      = np.cos(2 * np.pi * avg_df["hour"]      / 24)

avg_df["month_sin"]     = np.sin(2 * np.pi * avg_df["month"]     / 12)

avg_df["month_cos"]     = np.cos(2 * np.pi * avg_df["month"]     / 12)

avg_df["dow_sin"]       = np.sin(2 * np.pi * avg_df["dayofweek"] / 7)

avg_df["dow_cos"]       = np.cos(2 * np.pi * avg_df["dayofweek"] / 7)

avg_df["temp_roll24h"]  = avg_df["temperature_2m"].rolling(24,  min_periods=1).mean().round(2)

avg_df["temp_roll168h"] = avg_df["temperature_2m"].rolling(168, min_periods=1).mean().round(2)

avg_df["temp_change_1h"]  = avg_df["temperature_2m"].diff(1).round(2)

avg_df["temp_change_24h"] = avg_df["temperature_2m"].diff(24).round(2)

avg_df["morning_ramp"]  = ((avg_df["hour"].between(6,9))  & (avg_df["dayofweek"] < 5)).astype(int)

avg_df["evening_peak"]  = ((avg_df["hour"].between(16,20)) & (avg_df["dayofweek"] < 5)).astype(int)

avg_df["daylight_hours"] = avg_df["month"].map(lambda m: [6,7,9,12,15,17,18,17,14,11,8,6][m-1])



for lag in [1, 2, 3, 6, 12, 24, 48, 168]:

    avg_df[f"temp_lag_{lag}h"] = avg_df["temperature_2m"].shift(lag)



avg_df = avg_df.round(4)

avg_df.to_csv("weather_raw.csv", index=False)

print(f"\nSaved: weather_raw.csv  ({len(avg_df):,} rows, {len(avg_df.columns)} cols)")

print(f"Temp range: {avg_df['temperature_2m'].min():.1f} to {avg_df['temperature_2m'].max():.1f} C")

