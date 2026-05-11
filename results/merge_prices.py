
import pandas as pd



DATA_DIR    = "/scratch/project_2001220/kaukolampo"

PRICE_CACHE = f"{DATA_DIR}/entsoe_prices_fi.csv"

MAIN        = f"{DATA_DIR}/data_full_real.csv"



print("Loading prices...")

prices = pd.read_csv(PRICE_CACHE, parse_dates=["timestamp"])

prices["timestamp"] = pd.to_datetime(prices["timestamp"]).dt.tz_localize(None)



print("Loading main dataset...")

main = pd.read_csv(MAIN, parse_dates=["time"])

main["time"] = pd.to_datetime(main["time"]).dt.tz_localize(None)



print(f"  Main: {len(main)} rows")

print(f"  Prices: {len(prices)} rows")

print(f"  Old electricity_price — mean: {main['electricity_price'].mean():.1f}, max: {main['electricity_price'].max():.1f}")



# Drop old synthetic price

main = main.drop(columns=["electricity_price"])



# Merge on time

merged = main.merge(prices.rename(columns={"timestamp":"time","spot_price_eur_mwh":"electricity_price"}),

                    on="time", how="left")



missing = merged["electricity_price"].isna().sum()

if missing > 0:

    median = merged["electricity_price"].median()

    merged["electricity_price"] = merged["electricity_price"].fillna(median)

    print(f"  Filled {missing} missing with median {median:.1f} EUR/MWh")



print(f"\nNew electricity_price stats:")

print(merged["electricity_price"].describe().round(1))



merged.to_csv(MAIN, index=False)

print(f"\nSaved to {MAIN}")

