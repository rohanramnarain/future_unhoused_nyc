import os
import json
import requests
from typing import Optional
from urllib.parse import urlencode
from tqdm import tqdm
from ..config import settings
from ..utils.logging import get_logger


logger = get_logger()


NYC_OD_BASE = "https://data.cityofnewyork.us/resource"


DATASETS = {
# small sample pulls (tweak $limit as needed)
"311": ("erm2-nwe9.json", {"$limit": 5000, "$select": "unique_key,created_date,latitude,longitude,complaint_type"}),
"evictions": ("6z8x-wfk4.json", {"$limit": 20000}),
"hpd_complaints": ("uwyv-629c.json", {"$limit": 20000}),
}


HEADERS = {}
if settings.socrata_app_token:
	HEADERS["X-App-Token"] = settings.socrata_app_token


def _download(endpoint: str, params: dict, out_path: str):
	url = f"{NYC_OD_BASE}/{endpoint}?{urlencode(params)}"
	logger.info(f"Downloading {url}")
	r = requests.get(url, headers=HEADERS, timeout=60)
	r.raise_for_status()
	with open(out_path, "w") as f:
		f.write(r.text)


def download_small_samples():
	os.makedirs(settings.raw_dir, exist_ok=True)
	for name, (endpoint, params) in DATASETS.items():
		out = os.path.join(settings.raw_dir, f"{name}.json")
		_download(endpoint, params, out)
	# NYC boundary geojson (lightweight open file hosted by nyc planning)
	boundary_url = "https://raw.githubusercontent.com/nycplanning/labs-boundaries/master/geojson/city/city.geojson"
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