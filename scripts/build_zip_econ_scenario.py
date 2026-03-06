import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings


BASE_REQUIRED = ["zipcode", "year", "unemp_rate", "med_income", "med_rent", "rent_burden", "poverty_rate"]
MACRO_REQUIRED = ["year", "city_unemp_growth", "city_income_growth", "city_rent_growth"]


def _normalize_zip(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.extract(r"(\d+)")[0]
        .fillna("")
        .str[:5]
        .str.zfill(5)
    )


def _load_csv(path: str, required: list[str], name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing columns: {', '.join(missing)}. Present: {', '.join(df.columns)}"
        )
    return df


def build_zip_scenario(
    baseline_path: str,
    macro_path: str,
    out_path: str,
    years: list[int],
) -> pd.DataFrame:
    base = _load_csv(baseline_path, BASE_REQUIRED, "ZIP baseline")
    macro = _load_csv(macro_path, MACRO_REQUIRED, "NYC macro")

    base = base.copy()
    base["zipcode"] = _normalize_zip(base["zipcode"])
    base["year"] = pd.to_numeric(base["year"], errors="coerce")
    for col in ["unemp_rate", "med_income", "med_rent", "rent_burden", "poverty_rate"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base = base.dropna(subset=["zipcode", "year", "unemp_rate", "med_income", "med_rent", "rent_burden", "poverty_rate"]).copy()
    base["poverty_rate"] = base["poverty_rate"].clip(lower=0.0, upper=1.0)
    base["year"] = base["year"].astype(int)

    macro = macro.copy()
    macro["year"] = pd.to_numeric(macro["year"], errors="coerce").astype("Int64")
    for col in ["city_unemp_growth", "city_income_growth", "city_rent_growth"]:
        macro[col] = pd.to_numeric(macro[col], errors="coerce")
    if "city_rent_burden_growth" in macro.columns:
        macro["city_rent_burden_growth"] = pd.to_numeric(macro["city_rent_burden_growth"], errors="coerce")
    else:
        macro["city_rent_burden_growth"] = 0.0
    macro = macro.dropna(subset=["year", "city_unemp_growth", "city_income_growth", "city_rent_growth"]).copy()
    macro["year"] = macro["year"].astype(int)

    target_years = sorted(set(int(y) for y in years))
    macro = macro[macro["year"].isin(target_years)].copy()
    if macro.empty:
        raise ValueError(f"Macro file has no rows for target years {target_years}")

    latest_year = int(base["year"].max())
    latest = base[base["year"] == latest_year].copy()
    if latest.empty:
        raise ValueError("ZIP baseline has no usable latest-year rows")

    # Sensitivity profile: lower-income / higher-poverty ZIPs absorb larger citywide shocks.
    # We intentionally make this nonlinear so high-income ZIPs dampen shocks while
    # low-income ZIPs amplify them.
    med_unemp = latest["unemp_rate"].median() or 1.0
    med_income = latest["med_income"].median() or 1.0
    med_rent_burden = latest["rent_burden"].median() or 1.0
    med_poverty = latest["poverty_rate"].median() or 0.15

    poverty_line = 15060.0  # 2024 one-person federal poverty guideline
    triple_poverty = 3.0 * poverty_line
    latest["income_to_3x_poverty"] = latest["med_income"] / triple_poverty
    latest["vuln_income"] = (1.0 / latest["income_to_3x_poverty"].clip(lower=0.25, upper=6.0)).pow(1.25)
    latest["vuln_income"] = latest["vuln_income"].clip(lower=0.35, upper=3.75)

    latest["vuln_poverty"] = (latest["poverty_rate"] / max(med_poverty, 1e-6)).pow(0.90)
    latest["vuln_poverty"] = latest["vuln_poverty"].clip(lower=0.45, upper=3.50)

    latest["vuln_total"] = (
        0.50 * latest["vuln_income"]
        + 0.30 * latest["vuln_poverty"]
        + 0.20 * (latest["rent_burden"] / max(med_rent_burden, 1e-6)).clip(lower=0.40, upper=2.60)
    ).clip(lower=0.35, upper=3.80)

    latest["sens_unemp"] = (latest["unemp_rate"] / med_unemp).clip(lower=0.70, upper=1.50)
    latest["sens_income"] = (med_income / latest["med_income"].clip(lower=1.0)).clip(lower=0.55, upper=2.20)
    latest["sens_rent"] = (latest["rent_burden"] / med_rent_burden).clip(lower=0.70, upper=1.50)

    rows = []
    for macro_row in macro.itertuples(index=False):
        y = int(macro_row.year)
        for z in latest.itertuples(index=False):
            vulnerability = float(z.vuln_total)
            unemp_shock = float(macro_row.city_unemp_growth) * float(z.sens_unemp) * vulnerability
            income_shock = float(macro_row.city_income_growth) * float(z.sens_income) / max(vulnerability, 1e-6)
            rent_shock = float(macro_row.city_rent_growth) * float(z.sens_rent) * vulnerability
            rent_burden_shock = (
                0.75 * rent_shock
                + 0.65 * unemp_shock
                - 0.35 * income_shock
                + float(macro_row.city_rent_burden_growth)
            )

            mult_n311 = 1.0 + 0.85 * rent_burden_shock + 0.45 * unemp_shock - 0.12 * income_shock
            mult_nhpd = 1.0 + 1.00 * rent_shock + 0.75 * rent_burden_shock + 0.35 * unemp_shock - 0.12 * income_shock
            mult_nevict = 1.0 + 0.95 * unemp_shock + 0.50 * rent_shock + 0.25 * rent_burden_shock - 0.12 * income_shock
            mult_nfiled = 1.0 + 1.05 * unemp_shock + 0.55 * rent_burden_shock + 0.25 * rent_shock - 0.12 * income_shock

            rows.append(
                {
                    "zipcode": z.zipcode,
                    "year": y,
                    "unemp_shock": unemp_shock,
                    "income_shock": income_shock,
                    "rent_shock": rent_shock,
                    "rent_burden_shock": rent_burden_shock,
                    "mult_n311": min(2.60, max(0.30, mult_n311)),
                    "mult_nhpd": min(2.90, max(0.30, mult_nhpd)),
                    "mult_nevict": min(2.80, max(0.30, mult_nevict)),
                    "mult_nfiled": min(3.00, max(0.30, mult_nfiled)),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["zipcode", "year"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build ZIP-year economic scenario multipliers for 2027-2029 from baseline ZIP data + citywide macro forecasts."
    )
    parser.add_argument(
        "--baseline",
        default=os.path.join(settings.external_dir, "zip_econ_baseline_hist.csv"),
        help="CSV with columns: zipcode,year,unemp_rate,med_income,med_rent,rent_burden,poverty_rate",
    )
    parser.add_argument(
        "--macro",
        default=os.path.join(settings.external_dir, "nyc_macro_forecast_2027_2029.csv"),
        help="CSV with columns: year,city_unemp_growth,city_income_growth,city_rent_growth[,city_rent_burden_growth]",
    )
    parser.add_argument(
        "--years",
        default="2027,2028,2029",
        help="Comma-separated target years for scenario generation",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(settings.interim_dir, "zip_econ_scenario_2027_2029.csv"),
        help="Output ZIP-year scenario CSV consumed by src/data/features.py",
    )
    args = parser.parse_args()

    years = [int(part.strip()) for part in str(args.years).split(",") if part.strip()]
    out = build_zip_scenario(args.baseline, args.macro, args.out, years)
    print(
        f"Wrote {args.out} rows={len(out)} zips={out['zipcode'].nunique() if len(out) else 0} years={sorted(out['year'].unique().tolist()) if len(out) else []}"
    )


if __name__ == "__main__":
    main()
