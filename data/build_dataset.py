
import pandas as pd

import numpy as np

import os



WEATHER_FILE  = "weather_raw.csv"

FINGRID_FILE  = "fingrid_hourly_raw.csv"

OUTPUT_FILE   = "data_full_real.csv"



FI_HOLIDAYS = ["01-01","01-06","05-01","12-06","12-24","12-25","12-26"]

EASTER_DATES = [

    "2016-03-25","2016-03-27","2016-03-28",

    "2017-04-14","2017-04-16","2017-04-17",

    "2018-03-30","2018-04-01","2018-04-02",

    "2019-04-19","2019-04-21","2019-04-22",

    "2020-04-10","2020-04-12","2020-04-13",

    "2021-04-02","2021-04-04","2021-04-05",

    "2022-04-15","2022-04-17","2022-04-18",

    "2023-04-07","2023-04-09","2023-04-10",

    "2024-03-29","2024-03-31","2024-04-01",

]

SCHOOL_TERMS = [

    ("2016-01-07","2016-06-03"),("2016-08-11","2016-12-22"),

    ("2017-01-09","2017-06-02"),("2017-08-10","2017-12-21"),

    ("2018-01-08","2018-06-01"),("2018-08-09","2018-12-20"),

    ("2019-01-07","2019-05-31"),("2019-08-08","2019-12-19"),

    ("2020-01-08","2020-06-05"),("2020-08-13","2020-12-18"),

    ("2021-01-11","2021-06-04"),("2021-08-12","2021-12-17"),

    ("2022-01-10","2022-06-03"),("2022-08-11","2022-12-16"),

    ("2023-01-09","2023-06-02"),("2023-08-10","2023-12-15"),

    ("2024-01-08","2024-05-31"),("2024-08-08","2024-12-20"),

]



print("Loading weather data...")

weather = pd.read_csv(WEATHER_FILE)

weather["time"] = pd.to_datetime(weather["time"])

print(f"  Weather: {len(weather):,} rows")



print("Loading Fingrid data...")

fingrid = pd.read_csv(FINGRID_FILE)

fingrid["time"] = pd.to_datetime(fingrid["time"], utc=True).dt.tz_localize(None)

print(f"  Fingrid: {len(fingrid):,} rows")



print("Merging...")

weather["time"] = weather["time"].dt.tz_localize(None)

df = pd.merge(

    fingrid[["time","heat_load","year","month","hour","dayofweek",

             "heat_lag_1h","heat_lag_2h","heat_lag_24h",

             "heat_lag_48h","heat_lag_168h",

             "heat_roll24h_mean","heat_roll24h_std"]],

    weather.drop(columns=["year","month","hour","dayofweek","is_weekend"], errors="ignore"),

    on="time", how="inner"

)

print(f"  Merged: {len(df):,} rows")



print("Adding calendar features...")

mmdd     = df["time"].dt.strftime("%m-%d")

date_str = df["time"].dt.strftime("%Y-%m-%d")

easter_set = set(EASTER_DATES)

df["is_holiday"]     = (mmdd.isin(FI_HOLIDAYS) | date_str.isin(easter_set)).astype(int)

df["is_holiday_eve"] = df["is_holiday"].shift(-24, fill_value=0).astype(int)



school_term_dates = set()

for start, end in SCHOOL_TERMS:

    school_term_dates.update(pd.date_range(start, end, freq="D").strftime("%Y-%m-%d"))

df["is_school_term"] = date_str.isin(school_term_dates).astype(int)

df["is_weekend"]     = (df["dayofweek"] >= 5).astype(int)

df["morning_ramp"]   = ((df["hour"].between(6,9))  & (df["dayofweek"] < 5)).astype(int)

df["evening_peak"]   = ((df["hour"].between(16,20)) & (df["dayofweek"] < 5)).astype(int)



print("Adding electricity prices (synthetic)...")

rng   = np.random.default_rng(42)

month = df["time"].dt.month.to_numpy().astype(float)

hour  = df["time"].dt.hour.to_numpy().astype(float)

year  = df["time"].dt.year.to_numpy()

price = (50 + 15*np.sin(2*np.pi*(month-1)/12)

         + 20*np.sin(2*np.pi*hour/24)

         + rng.normal(0,5,len(df))).clip(0)

crisis = ((year==2021)&(month>=9))|(year==2022)

df["electricity_price"] = np.where(crisis, price*2.8, price).round(2)



print("Adding static covariates...")

df["building_year"] = 1975

df["floor_area_m2"] = 45_000_000

df["dh_share"]      = 0.85

df["population"]    = 1_350_000

df["group"]         = "espoo_metro"

df["daylight_hours"] = df["month"].map(lambda m: [6,7,9,12,15,17,18,17,14,11,8,6][m-1])



df = df.sort_values("time").reset_index(drop=True)

df["time_idx"]   = range(len(df))

df["year_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)



print("Quality check...")

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) > 0:

    print(f"  Missing values found, filling...")

    print(missing.to_string())

    df = df.ffill().bfill()

else:

    print("  No missing values")



train = df[df["year"] <= 2022]

val   = df[df["year"] == 2023]

test  = df[df["year"] == 2024]

print(f"  Train (2016-2022): {len(train):,} rows")

print(f"  Val   (2023):      {len(val):,} rows")

print(f"  Test  (2024):      {len(test):,} rows")



df = df.round(4)

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}  ({len(df):,} rows, {len(df.columns)} cols, {os.path.getsize(OUTPUT_FILE)/1024/1024:.1f} MB)")

