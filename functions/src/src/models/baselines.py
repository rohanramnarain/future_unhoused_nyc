import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from ..config import settings
from ..utils.logging import get_logger

logger = get_logger()


FEATURE_COLS = [
    "n311_y",
    "nhpd_y",
    "nevict_y",
    "nfiled_y",
    "n_dcp_units",
    "n_dcp_aff_units",
    "n_dcp_expiring5yr",
    "n_dcp_expired",
    "dcp_status_median",
]
TARGET = "risk_proxy"


def _train_lgbm(Xtr, ytr):
    model = LGBMRegressor(random_state=settings.random_seed, n_estimators=500, learning_rate=0.05, max_depth=-1)
    model.fit(Xtr, ytr)
    return model


def _train_xgb(Xtr, ytr):
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=settings.random_seed,
        tree_method="hist",
        n_jobs=4,
    )
    model.fit(Xtr, ytr)
    return model


def _train_rf(Xtr, ytr):
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=settings.random_seed,
        n_jobs=4,
    )
    model.fit(Xtr, ytr)
    return model


TRAINER_REGISTRY = {
    "lgbm": _train_lgbm,
    "xgb": _train_xgb,
    "rf": _train_rf,
}


def train_model(features: pd.DataFrame, model_name: str = "lgbm"):
    name = (model_name or "lgbm").lower()
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {', '.join(TRAINER_REGISTRY.keys())}.")

    df = features.copy()
    X = df[FEATURE_COLS]
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=settings.random_seed)

    trainer = TRAINER_REGISTRY[name]
    model = trainer(Xtr, ytr)

    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    logger.info(f"{name.upper()} MAE: {mae:.4f}")

    os.makedirs(settings.models_dir, exist_ok=True)
    model_path = os.path.join(settings.models_dir, f"baseline_{name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    return model, mae


def train_baseline(features: pd.DataFrame):
    # Backward-compatible wrapper for the original LightGBM baseline
    model, _ = train_model(features, model_name="lgbm")
    return model