import os
import pandas as pd
from ..config import settings
from .baselines import FEATURE_COLS


def aggregate_to_predictions(
    df: pd.DataFrame,
    model,
    alpha: float = 0.1,
    *,
    target_col: str | None = None,
) -> pd.DataFrame:
    from .calibration import split_conformal_intervals

    feats = df[["hex", "year", *FEATURE_COLS]].copy()
    mu = model.predict(feats[FEATURE_COLS])

    # Calibrate conformal intervals using any rows that have ground-truth labels.
    # In the bootstrap demo, this is `risk_proxy`. For a real model, set MODEL_TARGET
    # and provide OUTCOMES_PATH so `target_col` exists for historical years.
    target_col = (target_col or settings.model_target or "risk_proxy").strip() or "risk_proxy"
    if target_col in df.columns:
        y_series = pd.to_numeric(df[target_col], errors="coerce")
        mask = y_series.notna()
    else:
        mask = pd.Series(False, index=df.index)

    if mask.any():
        y_true = y_series[mask].values
        y_pred = pd.Series(mu, index=df.index)[mask].values
        lo_mu, hi_mu, q = split_conformal_intervals(df.loc[mask], y_true, y_pred, alpha=alpha)
        # Apply the calibrated quantile to all rows (future years included)
        lo = mu - q
        hi = mu + q
    else:
        # No labels available -> no uncertainty calibration
        lo = mu
        hi = mu

    out = feats.copy()
    out["pred"] = mu
    out["lo"] = lo
    out["hi"] = hi

    # Normalize within each year using percentile ranks so every slider position spans 0-1
    def _percentile(col: pd.Series) -> pd.Series:
        if len(col) <= 1:
            return pd.Series(0.0, index=col.index)
        if col.nunique() == 1:
            return pd.Series(0.5, index=col.index)
        ranks = col.rank(method="average") - 1.0
        return ranks / (len(col) - 1.0)

    for c in ["pred", "lo", "hi"]:
        out[c] = out.groupby("year")[c].transform(_percentile)

    return out