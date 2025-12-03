# src/data/download.py
import json
import os
import time
import requests
from urllib.parse import urlencode
from ..config import settings
from ..utils.logging import get_logger

logger = get_logger()
NYC_OD_BASE = "https://data.cityofnewyork.us/resource"

DATASETS = {
    "311": ("erm2-nwe9.json", {"$limit": 1500, "$select": "unique_key,created_date,latitude,longitude,complaint_type"}),
    "evictions": ("6z8x-wfk4.json", {"$limit": 2000}),
    "hpd_complaints": ("uwyv-629c.json", {"$limit": 1500}),
}

HEADERS = {"Accept": "application/json"}
if settings.socrata_app_token:
    HEADERS["X-App-Token"] = settings.socrata_app_token

HPD_PLACEHOLDER = [
    {"complaint_id": "hpd-placeholder-1", "latitude": 40.815, "longitude": -73.941, "borough": "MANHATTAN"},
    {"complaint_id": "hpd-placeholder-2", "latitude": 40.700, "longitude": -73.920, "borough": "BROOKLYN"},
    {"complaint_id": "hpd-placeholder-3", "latitude": 40.844, "longitude": -73.864, "borough": "BRONX"},
    {"complaint_id": "hpd-placeholder-4", "latitude": 40.580, "longitude": -74.150, "borough": "STATEN ISLAND"},
]

def _url_with_token(endpoint: str, params: dict) -> str:
    p = dict(params)
    if settings.socrata_app_token:
        p["$$app_token"] = settings.socrata_app_token
    return f"{NYC_OD_BASE}/{endpoint}?{urlencode(p)}"

def _download(endpoint: str, params: dict, out_path: str) -> bool:
    def attempt(pl):
        url = _url_with_token(endpoint, pl)
        logger.info(f"Downloading {url}")
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 429:
            logger.warning("429 Too Many Requests for %s — backing off 2s and retrying once.", endpoint)
            time.sleep(2)
            r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code in (403, 429):
            return r.status_code, None
        r.raise_for_status()
        return 200, r.text

    status, body = attempt(params)
    if status == 403 and "$limit" in params:
        small = dict(params); small["$limit"] = min(int(params["$limit"]), 500)
        logger.warning("403 for %s — retrying with smaller $limit=%s", endpoint, small["$limit"])
        status, body = attempt(small)

    if status != 200 or body is None:
        logger.warning("Skipping %s due to status %s. Set SOCRATA_APP_TOKEN to avoid rate limits.", endpoint, status)
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(body)
    return True


def _write_hpd_placeholder(out_path: str):
    logger.warning("HPD complaints endpoint restricted; writing placeholder sample to %s", out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(HPD_PLACEHOLDER, f)

def download_small_samples():
    os.makedirs(settings.raw_dir, exist_ok=True)
    for name, (endpoint, params) in DATASETS.items():
        out = os.path.join(settings.raw_dir, f"{name}.json")
        ok = _download(endpoint, params, out)
        if not ok and name == "hpd_complaints":
            _write_hpd_placeholder(out)

    # Boundary is optional; our code falls back to NYC bbox if 404
    boundary_url = "https://raw.githubusercontent.com/NYCPlanning/labs-layers/master/layers/city/city.geojson"
    out = os.path.join(settings.external_dir, "nyc_boundary.geojson")
    os.makedirs(settings.external_dir, exist_ok=True)
    try:
        r = requests.get(boundary_url, timeout=60)
        r.raise_for_status()
        with open(out, "wb") as f:
            f.write(r.content)
        logger.info("Saved NYC boundary geojson")
    except Exception as e:
        logger.warning(f"Boundary download failed (will use bbox fallback): {e}")

    modzcta_url = "https://data.cityofnewyork.us/api/geospatial/pri4-ifjk?method=export&format=GeoJSON"
    modzcta_path = os.path.join(settings.external_dir, "modzcta.geojson")
    try:
        r = requests.get(modzcta_url, timeout=120)
        r.raise_for_status()
        with open(modzcta_path, "wb") as f:
            f.write(r.content)
        logger.info("Saved MODZCTA boundaries")
    except Exception as e:
        logger.warning(f"MODZCTA download failed (ZIP lookups will be skipped): {e}")
