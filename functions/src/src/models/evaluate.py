import os
import pandas as pd
from ..config import settings
from .baselines import FEATURE_COLS


def aggregate_to_predictions(df: pd.DataFrame, model, alpha: float = 0.1) -> pd.DataFrame:
    from .calibration import split_conformal_intervals

    feats = df[["hex", "year", *FEATURE_COLS]].copy()
    mu = model.predict(feats[FEATURE_COLS])

    # Build a fake y_true for conformal using risk_proxy (demo)
    y_true = df["risk_proxy"].values
    lo, hi, q = split_conformal_intervals(df, y_true, mu, alpha=0.1)

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