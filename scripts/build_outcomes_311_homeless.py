import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.outcomes_311 import build_hex_year_outcomes, HOMELESS_311_TYPES
from src.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Build a real outcome table from NYC 311 homeless-related requests (aggregated to H3 hex-year)."
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--horizon", type=int, default=1, help="Predict year+horizon from feature year")
    parser.add_argument(
        "--target-col",
        default="future_homeless_311",
        help="Column name for the label in the output table.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(settings.interim_dir, "outcomes_311_homeless.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = build_hex_year_outcomes(
        start_year=args.start_year,
        end_year=args.end_year,
        horizon_years=args.horizon,
        target_col=args.target_col,
        complaint_types=HOMELESS_311_TYPES,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} rows={len(df)} years={sorted(df.year.unique().tolist()) if len(df) else []}")


if __name__ == "__main__":
    main()
