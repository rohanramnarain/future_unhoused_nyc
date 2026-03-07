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


def make_predictions(
    features_path: str | None = None,
    model_name: str = "lgbm",
    features_df: pd.DataFrame | None = None,
    *,
    target_col: str | None = None,
):
    df = _load_features(features_path, features_df)
    model, key = _load_model(model_name)

    preds_all = aggregate_to_predictions(df, model, target_col=target_col)

    predict_years = settings.predict_years or [2026, 2027, 2028, 2029]
    preds = preds_all[preds_all["year"].isin(predict_years)].copy()

    # Keep backward-compatible filename pattern (used by the Dash app).
    y0, y1 = min(predict_years), max(predict_years)
    out = os.path.join(settings.processed_dir, f"predictions_{key}_{y0}_{y1}.csv")
    preds.to_csv(out, index=False)
    return out