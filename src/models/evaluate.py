import os
import numpy as np
import pandas as pd
from ..config import settings


def aggregate_to_predictions(df: pd.DataFrame, model, alpha: float = 0.1) -> pd.DataFrame:
    from .calibration import split_conformal_intervals

    feats = df[["hex", "year", "n311_y", "nhpd_y", "nevict_y"]].copy()
    mu = model.predict(feats[["n311_y", "nhpd_y", "nevict_y"]])

    # Build a fake y_true for conformal using risk_proxy (demo)
    y_true = df["risk_proxy"].values
    lo, hi, q = split_conformal_intervals(df, y_true, mu, alpha=0.1)

    out = feats.copy()
    out["pred"] = mu
    out["lo"] = lo
    out["hi"] = hi

    # Normalize to [0,1]
    for c in ["pred", "lo", "hi"]:
        mn, mx = out[c].min(), out[c].max()
        out[c] = (out[c] - mn) / (mx - mn + 1e-9)

    return out