import pandas as pd


# Simple scenario engine: scale features and recompute predicted risk proxy


def apply_scenario(df: pd.DataFrame, rent_pct: float = 0.0, eviction_pct: float = 0.0):
    adj = df.copy()
    # Approximate effects: rent -> HPD +311, eviction -> evictions
    adj["nhpd_y"] *= (1 + 0.5 * rent_pct)
    adj["n311_y"] *= (1 + 0.25 * rent_pct)
    adj["nevict_y"] *= (1 + eviction_pct)
    return adj