import json
import os
import pandas as pd
from shapely.geometry import shape
from ..config import settings
from ..utils.geo import NYC_BBOX


BOUNDARY_PATH = os.path.join(settings.external_dir, "nyc_boundary.geojson")




def load_nyc_boundary_polygon():
	if os.path.exists(BOUNDARY_PATH):
		with open(BOUNDARY_PATH, "r") as f:
			gj = json.load(f)
			geom = shape(gj["features"][0]["geometry"]) if gj.get("features") else shape(gj["geometry"])
			return geom
	return NYC_BBOX




def load_sample(name: str) -> pd.DataFrame:
    path = os.path.join(settings.raw_dir, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing raw sample: {path}")
    df = pd.read_json(path)
    return df