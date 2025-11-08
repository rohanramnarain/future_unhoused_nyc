import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Simple split conformal prediction for regression


def split_conformal_intervals(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1):
    # Use residual quantile from calibration set to build symmetric interval
    resid = np.abs(y_true - y_pred)
    q = np.quantile(resid, 1 - alpha)
    lo = y_pred - q
    hi = y_pred + q
    return lo, hi, q


def borough_reconcile(pred_df: pd.DataFrame, group_col: str = "borough", target_sum_col: str | None = None):
    """If you have known aggregate totals by borough/year, scale predictions to match.
    In demo mode, we simply return the input.
    """
    return pred_df