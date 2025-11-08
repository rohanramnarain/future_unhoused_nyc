import json
from typing import List, Dict
from h3 import h3
from shapely.geometry import mapping
from .preprocess import load_nyc_boundary_polygon


# Build H3 hexes covering NYC at resolution 9 (~0.1 km^2)


def build_hex_index(res: int = 9) -> List[str]:
    nyc_poly = load_nyc_boundary_polygon()
    geo = {
    "type": "Polygon",
    "coordinates": [list(mapping(nyc_poly)["coordinates"][0])], 
    }
    return list(h3.polyfill_geojson(geo, res))




def h3_polygon_coords(h: str):
    boundary = h3.h3_to_geo_boundary(h, geo_json=True)
    # deck.gl expects lng/lat dicts
    return [{"lng": lon, "lat": lat} for lat, lon in boundary]




def hex_geojson(hexes: List[str]) -> Dict:
    features = []
    for h in hexes:
        coords = h3.h3_to_geo_boundary(h, geo_json=True)
        ring = [[lon, lat] for lat, lon in coords]
        ring.append(ring[0])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"hex": h},
        })
    return {"type": "FeatureCollection", "features": features}