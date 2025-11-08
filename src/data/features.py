import os
import numpy as np
import pandas as pd
from h3 import h3
from .hexgrid import build_hex_index
from .preprocess import load_sample
from ..config import settings
from ..utils.logging import get_logger


logger = get_logger()


YEARS = [2026, 2027, 2028, 2029]


def _h3_index_points(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude", res: int = 9):
    mask = df[lat_col].notna() & df[lon_col].notna()
    idx = [h3.geo_to_h3(lat, lon, res) if m else None for lat, lon, m in zip(df[lat_col], df[lon_col], mask)]
    return pd.Series(idx, name="hex")


def build_features():
    os.makedirs(settings.interim_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)

    hexes = build_hex_index()
    features = []

    # 311 complaints (basic density proxy)
    try:
        df311 = load_sample("311")
        df311["hex"] = _h3_index_points(df311)
        g311 = df311.groupby("hex").size().rename("n311").reset_index()
    except Exception as e:
        logger.error(f"Error loading 311 data: {e}")
        g311 = pd.DataFrame({"hex": [], "n311": []})

    try:
        hpd = load_sample("hpd_complaints")
        if {"latitude", "longitude"}.issubset(hpd.columns):
            hpd["hex"] = _h3_index_points(hpd)
            gh = hpd.groupby("hex").size().rename("nhpd").reset_index()
        else:
            gh = pd.DataFrame({"hex": [], "nhpd": []})
    except Exception as e:
        logger.error(f"Error loading HPD data: {e}")
        gh = pd.DataFrame({"hex": [], "nhpd": []})

    try:
        ev = load_sample("evictions")
        if {"latitude", "longitude"}.issubset(ev.columns):
            ev["hex"] = _h3_index_points(ev)
            gev = ev.groupby("hex").size().rename("nevict").reset_index()
        else:
            gev = pd.DataFrame({"hex": [], "nevict": []})
    except Exception as e:
        logger.error(f"Error loading eviction data: {e}")
        gev = pd.DataFrame({"hex": [], "nevict": []})

    # Initialize base DataFrame with hexes
    base = hexes.copy()
    feat = base.merge(g311, on="hex", how="left").merge(gh, on="hex", how="left").merge(gev, on="hex", how="left")
    for c in ["n311", "nhpd", "nevict"]:
        feat[c] = feat[c].fillna(0.0)

    feats = []
    for y in YEARS:
        dfy = feat.copy()
        dfy["year"] = y
        growth = 1 + 0.03 * (y - 2025)
        for c in ["n311", "nhpd", "nevict"]:
            dfy[f"{c}_y"] = dfy[c] * growth
        feats.append(dfy)

    X = pd.concat(feats, ignore_index=True)

    # Target proxy for demo: risk index (to be learned). Here use weighted sum + noise.
    X["risk_proxy"] = 0.5 * X["nhpd_y"] + 0.3 * X["n311_y"] + 0.2 * X["nevict_y"]
    X["risk_proxy"] = (X["risk_proxy"] - X["risk_proxy"].min()) / (X["risk_proxy"].max() - X["risk_proxy"].min() + 1e-9)

    out = os.path.join(settings.interim_dir, "features.parquet")
    X.to_parquet(out, index=False)
    logger.info(f"Saved features to {out}")
    return X