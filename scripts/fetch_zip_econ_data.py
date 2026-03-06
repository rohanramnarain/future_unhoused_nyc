import json
import os
import sys

import pandas as pd
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings


def _load_nyc_modzcta_zips(modzcta_path: str) -> list[str]:
    with open(modzcta_path, "r") as f:
        data = json.load(f)
    zips = sorted(
        {
            str(feat.get("properties", {}).get("modzcta", "")).zfill(5)
            for feat in data.get("features", [])
            if str(feat.get("properties", {}).get("modzcta", "")).isdigit()
            and str(feat.get("properties", {}).get("modzcta", "")) != "99999"
        }
    )
    return zips


def _fetch_acs_zip_baseline(nyc_zips: list[str], census_api_key: str, year: int = 2023) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME,B23025_003E,B23025_005E,B19013_001E,B25064_001E,B25070_001E,B25070_007E,B25070_008E,B25070_009E,B25070_010E,B17001_001E,B17001_002E",
        "for": "zip code tabulation area:*",
        "key": census_api_key,
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns={"zip code tabulation area": "zipcode"})
    df = df[df["zipcode"].isin(nyc_zips)].copy()

    numeric = [
        "B23025_003E",
        "B23025_005E",
        "B19013_001E",
        "B25064_001E",
        "B25070_001E",
        "B25070_007E",
        "B25070_008E",
        "B25070_009E",
        "B25070_010E",
        "B17001_001E",
        "B17001_002E",
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame()
    out["zipcode"] = df["zipcode"]
    out["year"] = int(year)
    out["unemp_rate"] = df["B23025_005E"] / df["B23025_003E"]
    out["med_income"] = df["B19013_001E"]
    out["med_rent"] = df["B25064_001E"]
    out["rent_burden"] = (
        df["B25070_007E"] + df["B25070_008E"] + df["B25070_009E"] + df["B25070_010E"]
    ) / df["B25070_001E"]
    out["poverty_rate"] = df["B17001_002E"] / df["B17001_001E"]

    out = out.dropna(subset=["unemp_rate", "med_income", "med_rent", "rent_burden", "poverty_rate"]).copy()
    out = out[(out["unemp_rate"] >= 0.0) & (out["unemp_rate"] <= 1.0)].copy()
    out = out[(out["poverty_rate"] >= 0.0) & (out["poverty_rate"] <= 1.0)].copy()
    return out.sort_values("zipcode").reset_index(drop=True)


def _fetch_macro_forecast() -> pd.DataFrame:
    latest_resp = requests.get(
        "https://data.cityofnewyork.us/resource/xatq-cxeq.json",
        params={"$select": "max(pub_dt) as max_pub"},
        timeout=60,
    )
    latest_resp.raise_for_status()
    latest_pub = latest_resp.json()[0]["max_pub"]

    where = (
        f'pub_dt="{latest_pub}" '
        'AND ref_yr IN("2026","2027","2028","2029") '
        'AND ind IN("US Unemployment Rate","NYC Personal Income","NYC Consumer Price Index")'
    )

    resp = requests.get(
        "https://data.cityofnewyork.us/resource/xatq-cxeq.json",
        params={
            "$select": "ref_yr,ind,value",
            "$where": where,
            "$order": "ind,ref_yr",
            "$limit": 200,
        },
        timeout=60,
    )
    resp.raise_for_status()

    df = pd.DataFrame(resp.json())
    if df.empty:
        raise RuntimeError("OMB forecast query returned no rows.")

    df["ref_yr"] = pd.to_numeric(df["ref_yr"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["ref_yr", "value"]).copy()

    piv = df.pivot_table(index="ref_yr", columns="ind", values="value", aggfunc="first").sort_index()
    required = ["US Unemployment Rate", "NYC Personal Income", "NYC Consumer Price Index"]
    missing = [c for c in required if c not in piv.columns]
    if missing:
        raise RuntimeError(f"Missing OMB indicators: {', '.join(missing)}")

    years = [2027, 2028, 2029]

    def pct_growth(series: pd.Series, year: int) -> float:
        prev = float(series.loc[year - 1])
        cur = float(series.loc[year])
        if prev == 0:
            return 0.0
        return (cur - prev) / abs(prev)

    out = pd.DataFrame({"year": years})
    out["city_unemp_growth"] = [pct_growth(piv["US Unemployment Rate"], y) for y in years]
    out["city_income_growth"] = [pct_growth(piv["NYC Personal Income"], y) for y in years]
    out["city_rent_growth"] = [pct_growth(piv["NYC Consumer Price Index"], y) for y in years]
    out["city_rent_burden_growth"] = 0.0
    return out


def main() -> None:
    if not settings.census_api_key:
        raise RuntimeError("CENSUS_API_KEY is required in .env to fetch ZIP baseline data.")

    os.makedirs(settings.external_dir, exist_ok=True)
    modzcta_path = os.path.join(settings.external_dir, "modzcta.geojson")
    if not os.path.exists(modzcta_path):
        raise FileNotFoundError(f"Missing MODZCTA file at {modzcta_path}")

    nyc_zips = _load_nyc_modzcta_zips(modzcta_path)
    baseline = _fetch_acs_zip_baseline(nyc_zips, settings.census_api_key, year=2023)
    macro = _fetch_macro_forecast()

    baseline_path = os.path.join(settings.external_dir, "zip_econ_baseline_hist.csv")
    macro_path = os.path.join(settings.external_dir, "nyc_macro_forecast_2027_2029.csv")

    baseline.to_csv(baseline_path, index=False)
    macro.to_csv(macro_path, index=False)

    print(f"Wrote {baseline_path} rows={len(baseline)} zips={baseline['zipcode'].nunique()}")
    print(f"Wrote {macro_path} rows={len(macro)} years={sorted(macro['year'].unique().tolist())}")


if __name__ == "__main__":
    main()
