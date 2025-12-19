import argparse
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.features import build_features
from src.models.baselines import train_model, TRAINER_REGISTRY
from src.models.predict import make_predictions


def main():
    parser = argparse.ArgumentParser(description="Train a model and emit per-hex predictions.")
    parser.add_argument("--model", default="lgbm", choices=sorted(TRAINER_REGISTRY.keys()), help="Which model to train")
    args = parser.parse_args()

    feats = build_features()
    model, mae = train_model(feats, model_name=args.model)
    out = make_predictions(model_name=args.model, features_df=feats)
    print(f"{args.model.upper()} MAE: {mae:.4f}")
    print(f"Predictions written to: {out}")


if __name__ == "__main__":
    main()
