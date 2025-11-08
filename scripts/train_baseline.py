import os
from src.data.features import build_features
from src.models.baselines import train_baseline
from src.models.predict import make_predictions


if __name__ == "__main__":
    feats = build_features()
    model = train_baseline(feats)
    out = make_predictions()
    print(f"Predictions written to: {out}")