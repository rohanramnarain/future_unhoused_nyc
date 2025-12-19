import os
import joblib
import pandas as pd
from ..config import settings
from .evaluate import aggregate_to_predictions


MODEL_REGISTRY = {
    "lgbm": "baseline_lgbm.joblib",
    "xgb": "baseline_xgb.joblib",
    "rf": "baseline_rf.joblib",
}


def _load_features(features_path: str | None, features_df: pd.DataFrame | None) -> pd.DataFrame:
    if features_df is not None:
        return features_df.copy()
    path = features_path or os.path.join(settings.interim_dir, "features.parquet")
    return pd.read_parquet(path)


def _load_model(model_name: str):
    key = (model_name or "lgbm").lower()
    filename = MODEL_REGISTRY.get(key)
    if not filename:
        raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {', '.join(MODEL_REGISTRY.keys())}.")
    model_path = os.path.join(settings.models_dir, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact missing at {model_path}. Train the model first.")
    return joblib.load(model_path), key


def make_predictions(features_path: str | None = None, model_name: str = "lgbm", features_df: pd.DataFrame | None = None):
    df = _load_features(features_path, features_df)
    model, key = _load_model(model_name)

    preds = aggregate_to_predictions(df, model)
    out = os.path.join(settings.processed_dir, f"predictions_{key}_2026_2029.csv")
    preds.to_csv(out, index=False)
    return out