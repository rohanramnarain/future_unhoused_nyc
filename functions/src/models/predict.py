import os
import joblib
import pandas as pd
from ..config import settings
from .evaluate import aggregate_to_predictions


def make_predictions(features_path: str | None = None):
    if features_path is None:
        features_path = os.path.join(settings.interim_dir, "features.parquet")
    df = pd.read_parquet(features_path)

    model_path = os.path.join(settings.models_dir, "baseline_lgbm.joblib")
    model = joblib.load(model_path)

    preds = aggregate_to_predictions(df, model)
    out = os.path.join(settings.processed_dir, "predictions_2026_2029.csv")
    preds.to_csv(out, index=False)
    return out