import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.features import build_features
from src.models.baselines import train_baseline
from src.models.predict import make_predictions

def main():
    feats = build_features()
    _ = train_baseline(feats)
    out = make_predictions()
    print(f"Predictions written to: {out}")

if __name__ == "__main__":
    main()
