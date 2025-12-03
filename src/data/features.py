# src/data/features.py
import json
import os
import pandas as pd
import h3
from shapely.geometry import Polygon, shape
from shapely.prepared import prep

from .hexgrid import build_hex_index, h3_polygon_coords
from .preprocess import load_sample
from ..config import settings
from ..utils.logging import get_logger

logger = get_logger()
YEARS = [2026, 2027, 2028, 2029]


def _hex_polygon(cell: str) -> Polygon:
    coords = [(pt["lng"], pt["lat"]) for pt in h3_polygon_coords(cell)]
    return Polygon(coords)


def _load_modzcta_polygons() -> dict[str, Polygon]:
    path = os.path.join(settings.external_dir, "modzcta.geojson")
    if not os.path.exists(path):
        logger.warning("MODZCTA boundary file missing at %s; filed-eviction redistribution skipped.", path)
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read MODZCTA boundaries (%s); filed-eviction redistribution skipped.", exc)
        return {}

    geoms: dict[str, Polygon] = {}
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        zip_code = str(props.get("modzcta") or props.get("zcta") or "").strip()
        if not zip_code:
            continue
        try:
            geom = shape(feature.get("geometry"))
        except Exception:
            continue
        geoms[zip_code.zfill(5)] = geom

    if not geoms:
        logger.warning("Loaded MODZCTA file but no usable ZIP polygons were found.")
    return geoms


def _zip_hex_weights(hexes: list[str]) -> dict[str, list[tuple[str, float]]]:
    zip_geoms = _load_modzcta_polygons()
    if not zip_geoms:
        return {}

    hex_polys = {cell: _hex_polygon(cell) for cell in hexes}
    weights: dict[str, list[tuple[str, float]]] = {}

    for zip_code, geom in zip_geoms.items():
        prepared = prep(geom)
        overlaps: list[tuple[str, float]] = []
        for cell, poly in hex_polys.items():
            if not prepared.intersects(poly):
                continue
            try:
                area = poly.intersection(geom).area
            except Exception:
                continue
            if area <= 0:
                continue
            overlaps.append((cell, area))

        if not overlaps:
            continue
        total = sum(a for _, a in overlaps) or 1.0
        weights[zip_code] = [(cell, area / total) for cell, area in overlaps]

    if weights:
        logger.info("Prepared ZIP-to-hex weights for %s ZIP codes", len(weights))
    else:
        logger.warning("MODZCTA overlaps computed but no ZIPs matched the H3 grid.")
    return weights

def _latlng_to_cell(lat: float, lon: float, res: int) -> str | None:
    if lat is None or lon is None:
        return None
    if hasattr(h3, "latlng_to_cell"):        # v4
        return h3.latlng_to_cell(lat, lon, res)
    if hasattr(h3, "geo_to_h3"):             # v3
        return h3.geo_to_h3(lat, lon, res)
    raise AttributeError("No H3 point->cell function found.")

def _h3_index_points(df: pd.DataFrame, lat_col="latitude", lon_col="longitude", res: int = 9):
    mask = df[lat_col].notna() & df[lon_col].notna()
    return pd.Series(
        [_latlng_to_cell(lat, lon, res) if m else None
         for lat, lon, m in zip(df[lat_col], df[lon_col], mask)],
        name="hex"
    )

def build_features():
    os.makedirs(settings.interim_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)

    hexes = build_hex_index()
    zip_hex_weights = _zip_hex_weights(hexes)
    # --- 311
    try:
        df311 = load_sample("311")
        df311["hex"] = _h3_index_points(df311)
        g311 = df311.groupby("hex").size().rename("n311").reset_index()
    except Exception:
        g311 = pd.DataFrame({"hex": [], "n311": []})
    # --- HPD
    try:
        hpd = load_sample("hpd_complaints")
        if {"latitude","longitude"}.issubset(hpd.columns):
            hpd["hex"] = _h3_index_points(hpd)
            gh = hpd.groupby("hex").size().rename("nhpd").reset_index()
        else:
            gh = pd.DataFrame({"hex": [], "nhpd": []})
    except Exception:
        gh = pd.DataFrame({"hex": [], "nhpd": []})
    # --- Evictions (executed notices)
    try:
        ev = load_sample("evictions")
        if {"latitude","longitude"}.issubset(ev.columns):
            ev["hex"] = _h3_index_points(ev)
            gev = ev.groupby("hex").size().rename("nevict").reset_index()
        else:
            gev = pd.DataFrame({"hex": [], "nevict": []})
    except Exception:
        gev = pd.DataFrame({"hex": [], "nevict": []})

    # --- Filed evictions (housing court filings)
    try:
        filed = load_sample("filed_evictions")
        if {"latitude", "longitude"}.issubset(filed.columns):
            filed["hex"] = _h3_index_points(filed)
            gfiled = filed.groupby("hex").size().rename("nfiled").reset_index()
        elif "zipcode" in filed.columns and zip_hex_weights:
            zip_series = filed["zipcode"].astype(str).str.extract(r"(\d+)")[0].fillna("")
            zip_series = zip_series.str[:5].str.zfill(5)
            filed_zip = filed.assign(zipcode=zip_series)
            filed_zip = filed_zip[filed_zip["zipcode"].str.fullmatch(r"\d{5}")]
            metric_col = None
            for candidate in ("nyc_filings", "filings", "total_filings"):
                if candidate in filed_zip.columns:
                    metric_col = candidate
                    break
            if metric_col is None:
                numeric_cols = filed_zip.select_dtypes(include=["number"]).columns.tolist()
                metric_col = numeric_cols[0] if numeric_cols else None

            if metric_col is None:
                logger.warning("Filed evictions CSV lacks a numeric filings column; skipping redistribution.")
                gfiled = pd.DataFrame({"hex": [], "nfiled": []})
            else:
                zip_totals = filed_zip.groupby("zipcode")[metric_col].sum()
                rows = []
                for zip_code, value in zip_totals.items():
                    weights = zip_hex_weights.get(zip_code)
                    if not weights:
                        continue
                    for cell, weight in weights:
                        rows.append((cell, float(value) * weight))
                if rows:
                    gfiled = (
                        pd.DataFrame(rows, columns=["hex", "nfiled"])
                        .groupby("hex")
                        .sum()
                        .reset_index()
                    )
                else:
                    logger.warning("Filed evictions ZIPs had no overlap with the hex grid; defaults to zero.")
                    gfiled = pd.DataFrame({"hex": [], "nfiled": []})
        else:
            if "zipcode" in filed.columns and not zip_hex_weights:
                logger.warning("Have ZIP-level filings but missing MODZCTA weights; defaults to zero counts.")
            gfiled = pd.DataFrame({"hex": [], "nfiled": []})
    except Exception as exc:
        logger.warning("Failed to process filed evictions sample: %s", exc)
        gfiled = pd.DataFrame({"hex": [], "nfiled": []})

    base = pd.DataFrame({"hex": hexes})
    feat = (
        base
        .merge(g311, on="hex", how="left")
        .merge(gh, on="hex", how="left")
        .merge(gev, on="hex", how="left")
        .merge(gfiled, on="hex", how="left")
    )
    for c in ["n311","nhpd","nevict", "nfiled"]:
        feat[c] = feat[c].fillna(0.0)

    # Yearly rows (placeholder trend so the app has data even if samples are sparse)
    feats = []
    for y in YEARS:
        dfy = feat.copy()
        dfy["year"] = y
        growth = 1 + 0.03 * (y - 2025)
        for c in ["n311", "nhpd", "nevict", "nfiled"]:
            dfy[f"{c}_y"] = dfy[c] * growth
        feats.append(dfy)
    X = pd.concat(feats, ignore_index=True)

    # Simple proxy target for demo
    X["risk_proxy"] = (
        0.4 * X["nhpd_y"]
        + 0.25 * X["n311_y"]
        + 0.2 * X["nevict_y"]
        + 0.15 * X["nfiled_y"]
    )
    X["risk_proxy"] = (X["risk_proxy"] - X["risk_proxy"].min()) / (X["risk_proxy"].max() - X["risk_proxy"].min() + 1e-9)

    out = os.path.join(settings.interim_dir, "features.parquet")
    X.to_parquet(out, index=False)
    logger.info(f"Saved features to {out}")
    return X
