# src/data/features.py
import json
import os
from typing import Optional

import pandas as pd
import h3
from shapely.geometry import Polygon, shape
from shapely.prepared import prep

try:  # Optional geocoder; only used if ENABLE_DCP_GEOCODER=1
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
except Exception:  # pragma: no cover - runtime optional dependency
    Nominatim = None
    RateLimiter = None

from .hexgrid import build_hex_index, h3_polygon_coords
from .preprocess import load_sample
from ..config import settings
from ..utils.logging import get_logger

logger = get_logger()
YEARS = [2026, 2027, 2028, 2029]
DCP_CSV_FILENAME = "dcp_housing.csv"
DCP_GEOCODE_CACHE = os.path.join(settings.interim_dir, "dcp_geocode_cache.json")
HPD_HEX_COUNTS = os.path.join(settings.interim_dir, "hpd_hex_counts.json")
MAX_DCP_GEOCODES = int(os.getenv("MAX_DCP_GEOCODES", "150"))


def _series_from_candidates(df: pd.DataFrame, candidates: list[str], default=None):
    for name in candidates:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def _load_geocode_cache() -> dict:
    if not os.path.exists(DCP_GEOCODE_CACHE):
        return {}
    try:
        with open(DCP_GEOCODE_CACHE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_geocode_cache(cache: dict):
    os.makedirs(settings.interim_dir, exist_ok=True)
    with open(DCP_GEOCODE_CACHE, "w") as f:
        json.dump(cache, f)


def _format_dcp_address(row: pd.Series) -> Optional[str]:
    parts = []
    for key in ("house_number", "house_num", "stnumber"):
        val = row.get(key) if key in row else None
        if val and str(val).strip():
            parts.append(str(val).strip())
            break
    street = row.get("street_name") or row.get("street") or row.get("streetname")
    if street and str(street).strip():
        parts.append(str(street).strip())
    borough = row.get("borough") or row.get("boro") or row.get("borough_name")
    if borough:
        parts.append(str(borough).title())
    if not parts:
        return None
    parts.append("NYC")
    return ", ".join(parts)


def _geocode_missing_projects(df: pd.DataFrame) -> pd.DataFrame:
    missing = df[df["latitude"].isna() | df["longitude"].isna()]
    if missing.empty:
        return df
    if not settings.enable_dcp_geocoder:
        logger.info("DCP geocoder disabled; dropping %s rows without coordinates.", len(missing))
        return df.dropna(subset=["latitude", "longitude"])
    if Nominatim is None or RateLimiter is None:
        logger.warning("geopy not installed; cannot geocode missing DCP rows (dropping %s rows).", len(missing))
        return df.dropna(subset=["latitude", "longitude"])

    user_agent = settings.geocoder_user_agent or "future-unhoused-nyc"
    geocoder = Nominatim(user_agent=user_agent, timeout=10)
    limiter = RateLimiter(geocoder.geocode, min_delay_seconds=1, swallow_exceptions=True)
    cache = _load_geocode_cache()
    updated = False
    requests_made = 0

    for idx, row in missing.iterrows():
        key = str(row.get("bbl") or row.get("project_id") or idx)
        lat_lon = cache.get(key)
        if not lat_lon:
            addr = _format_dcp_address(row)
            if not addr:
                continue
            if requests_made >= MAX_DCP_GEOCODES:
                break
            loc = limiter(addr)
            requests_made += 1
            if not loc:
                continue
            lat_lon = (loc.latitude, loc.longitude)
            cache[key] = lat_lon
            updated = True
        df.at[idx, "latitude"] = lat_lon[0]
        df.at[idx, "longitude"] = lat_lon[1]

    if updated:
        _save_geocode_cache(cache)

    return df.dropna(subset=["latitude", "longitude"])


def _load_dcp_projects() -> pd.DataFrame:
    path = os.path.join(settings.raw_dir, DCP_CSV_FILENAME)
    if not os.path.exists(path):
        logger.info("DCP Housing CSV missing at %s; skipping DCP feature engineering.", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.warning("Failed to read DCP Housing CSV: %s", exc)
        return pd.DataFrame()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["latitude"] = pd.to_numeric(_series_from_candidates(df, ["latitude", "lat"]), errors="coerce")
    df["longitude"] = pd.to_numeric(_series_from_candidates(df, ["longitude", "lon", "lng"]), errors="coerce")
    df["units_total"] = pd.to_numeric(_series_from_candidates(df, ["units_total", "total_units", "project_units", "tot_units"], default=0), errors="coerce").fillna(0)
    df["units_affordable"] = pd.to_numeric(_series_from_candidates(df, ["units_affordable", "affordable_units", "aff_units"], default=0), errors="coerce").fillna(0)
    df["exp_year"] = pd.to_numeric(_series_from_candidates(df, ["exp_year", "expiration_year", "reg_agreement_expiration_year"], default=None), errors="coerce")
    df["regulatory_status"] = _series_from_candidates(df, ["regulatory_status", "status", "regulatorystatus"], default="")
    df["bbl"] = _series_from_candidates(df, ["bbl"], default="").astype(str)
    df = _geocode_missing_projects(df)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    if df.empty:
        return df
    df["hex"] = _h3_index_points(df, res=9)
    df = df[df["hex"].notna()]
    return df


def _load_hpd_hex_counts() -> pd.DataFrame:
    if os.path.exists(HPD_HEX_COUNTS):
        try:
            df = pd.read_json(HPD_HEX_COUNTS)
        except Exception as exc:
            logger.warning("Failed to read aggregated HPD counts at %s: %s", HPD_HEX_COUNTS, exc)
        else:
            if {"hex", "nhpd"}.issubset(df.columns):
                df = df.copy()
                df = df[df["hex"].notna()]
                df["hex"] = df["hex"].astype(str)
                df["nhpd"] = pd.to_numeric(df["nhpd"], errors="coerce").fillna(0.0)
                return df[["hex", "nhpd"]]
            logger.warning("Aggregated HPD file missing required columns; falling back to raw sample.")

    try:
        hpd = load_sample("hpd_complaints")
    except Exception as exc:
        logger.warning("Failed to load HPD complaints sample: %s", exc)
        return pd.DataFrame({"hex": [], "nhpd": []})

    if {"latitude", "longitude"}.issubset(hpd.columns):
        hpd = hpd.copy()
        hpd["hex"] = _h3_index_points(hpd)
        return (
            hpd.groupby("hex")
            .size()
            .rename("nhpd")
            .reset_index()
        )

    logger.warning("HPD complaints sample lacks coordinates; returning empty counts.")
    return pd.DataFrame({"hex": [], "nhpd": []})


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
    dcp_projects = _load_dcp_projects()
    # --- 311
    try:
        df311 = load_sample("311")
        df311["hex"] = _h3_index_points(df311)
        g311 = df311.groupby("hex").size().rename("n311").reset_index()
    except Exception:
        g311 = pd.DataFrame({"hex": [], "n311": []})
    # --- HPD
    gh = _load_hpd_hex_counts()
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
    if not dcp_projects.empty:
        totals = (
            dcp_projects.groupby("hex")[["units_total", "units_affordable"]]
            .sum()
            .rename(columns={
                "units_total": "n_dcp_units",
                "units_affordable": "n_dcp_aff_units",
            })
            .reset_index()
        )
        statuses = sorted([s for s in dcp_projects["regulatory_status"].dropna().unique() if str(s).strip()])
        status_map = {status: idx + 1 for idx, status in enumerate(statuses)}
        dcp_projects["_status_code"] = dcp_projects["regulatory_status"].map(status_map).fillna(0.0)
        status_frame = (
            dcp_projects.groupby("hex")["_status_code"].median().rename("dcp_status_median").reset_index()
        )
        feat = (
            feat
            .merge(totals, on="hex", how="left")
            .merge(status_frame, on="hex", how="left")
        )
    else:
        feat["n_dcp_units"] = 0.0
        feat["n_dcp_aff_units"] = 0.0
        feat["dcp_status_median"] = 0.0

    for c in ["n311","nhpd","nevict", "nfiled", "n_dcp_units", "n_dcp_aff_units", "dcp_status_median"]:
        feat[c] = feat[c].fillna(0.0)

    # Yearly rows (placeholder trend so the app has data even if samples are sparse)
    feats = []
    for y in YEARS:
        dfy = feat.copy()
        dfy["year"] = y
        growth = 1 + 0.03 * (y - 2025)
        for c in ["n311", "nhpd", "nevict", "nfiled"]:
            dfy[f"{c}_y"] = dfy[c] * growth
        if not dcp_projects.empty:
            expiring = (
                dcp_projects[
                    dcp_projects["exp_year"].notna()
                    & dcp_projects["exp_year"].between(y, y + 5, inclusive="both")
                ]
                .groupby("hex")["units_total"]
                .sum()
            )
            expired = (
                dcp_projects[
                    dcp_projects["exp_year"].notna()
                    & (dcp_projects["exp_year"] < y)
                ]
                .groupby("hex")["units_total"]
                .sum()
            )
            dfy["n_dcp_expiring5yr"] = dfy["hex"].map(expiring).fillna(0.0)
            dfy["n_dcp_expired"] = dfy["hex"].map(expired).fillna(0.0)
        else:
            dfy["n_dcp_expiring5yr"] = 0.0
            dfy["n_dcp_expired"] = 0.0
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
