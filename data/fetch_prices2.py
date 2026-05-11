
from dotenv import load_dotenv
import os, requests, xml.etree.ElementTree as ET, pandas as pd, time

load_dotenv()

API_KEY = os.environ.get("ENTSO_E_API_KEY", "")

BASE_URL    = "https://web-api.tp.entsoe.eu/api"
DOMAIN      = "10YFI-1--------U"
DATA_DIR    = "/scratch/project_2001220/kaukolampo"
PRICE_CACHE = f"{DATA_DIR}/entsoe_prices_fi.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/xml, text/xml, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

def fetch_chunk(start, end):
    r = requests.get(BASE_URL, params={
        "securityToken": API_KEY, "documentType": "A44",
        "in_Domain": DOMAIN, "out_Domain": DOMAIN,
        "periodStart": start.strftime("%Y%m%d%H%M"),
        "periodEnd":   end.strftime("%Y%m%d%H%M"),
    }, headers=HEADERS, timeout=60)

    if r.status_code == 503:
        print(f"  503 server busy — waiting 60s...")
        time.sleep(60)
        return None

    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:200]}")
        return None

    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
    root = ET.fromstring(r.content)
    prices = {}

    for ts in root.findall(".//ns:TimeSeries", ns):
        period = ts.find("ns:Period", ns)
        if period is None: continue
        dt_start = datetime.strptime(
            period.find("ns:timeInterval/ns:start", ns).text, "%Y-%m-%dT%H:%MZ")
        for point in period.findall("ns:Point", ns):
            pos   = int(point.find("ns:position", ns).text)
            price = float(point.find("ns:price.amount", ns).text)
            prices[dt_start + (pos-1)*timedelta(hours=1)] = price
    return prices

all_prices = {}

for year in range(2016, 2025):
    print(f"Fetching {year}...", end=" ", flush=True)
    success = False
    for attempt in range(5):
        chunk = fetch_chunk(datetime(year,1,1), datetime(year+1,1,1))
        if chunk:
            all_prices.update(chunk)
            print(f"{len(chunk)} rows OK")
            success = True
            break
        print(f"retry {attempt+1}...", end=" ")
        time.sleep(20)

    if not success:
        print(f"FAILED {year} — skipping")
    time.sleep(5)

if not all_prices:
    print("No data fetched — ENTSO-E API may be down. Try again later.")
    exit(1)

df = pd.DataFrame([{"timestamp": k, "spot_price_eur_mwh": v}
                   for k,v in sorted(all_prices.items())])
df.to_csv(PRICE_CACHE, index=False)
print(f"\nSaved {len(df)} rows to {PRICE_CACHE}")
print(df["spot_price_eur_mwh"].describe().round(1))
