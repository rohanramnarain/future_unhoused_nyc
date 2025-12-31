import argparse
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.features import build_features
from src.models.baselines import train_model, TRAINER_REGISTRY
from src.models.predict import make_predictions


def main():
    parser = argparse.ArgumentParser(description="Train a model and emit per-hex predictions.")
    parser.add_argument("--model", default="lgbm", choices=sorted(TRAINER_REGISTRY.keys()), help="Which model to train")
    parser.add_argument(
        "--target-col",
        default=None,
        help="Target column to predict (defaults to env MODEL_TARGET or 'risk_proxy').",
    )
    parser.add_argument(
        "--outcomes-path",
        default=None,
        help="Optional outcomes file (CSV/Parquet) keyed by (hex, year).",
    )
    parser.add_argument(
        "--feature-years",
        default=None,
        help="Optional comma-separated years to generate features for (overrides PREDICT_YEARS/outcome years).",
    )
    args = parser.parse_args()

    years = None
    if args.feature_years:
        years = [int(x.strip()) for x in str(args.feature_years).split(",") if x.strip()]

    feats = build_features(outcomes_path=args.outcomes_path, model_target=args.target_col, years=years)
    model, mae = train_model(feats, model_name=args.model, target_col=args.target_col)
    out = make_predictions(model_name=args.model, features_df=feats, target_col=args.target_col)
    print(f"{args.model.upper()} MAE: {mae:.4f}")
    print(f"Predictions written to: {out}")


if __name__ == "__main__":
    main()
