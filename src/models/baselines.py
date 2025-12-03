import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from ..config import settings
from ..utils.logging import get_logger


logger = get_logger()


FEATURE_COLS = ["n311_y", "nhpd_y", "nevict_y", "nfiled_y"]
TARGET = "risk_proxy"


def train_baseline(features: pd.DataFrame):
    df = features.copy()
    X = df[FEATURE_COLS]
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=settings.random_seed)

    model = LGBMRegressor(random_state=settings.random_seed, n_estimators=500, learning_rate=0.05, max_depth=-1)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    logger.info(f"Baseline MAE: {mae:.4f}")

    os.makedirs(settings.models_dir, exist_ok=True)
    model_path = os.path.join(settings.models_dir, "baseline_lgbm.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    return model