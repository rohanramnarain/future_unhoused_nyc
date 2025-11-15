# src/data/hexgrid.py
from typing import List, Dict, Iterable
import h3
from shapely.geometry import mapping, Polygon, Point
from shapely.ops import unary_union
from .preprocess import load_nyc_boundary_polygon

def _cells_from_geojson(geo: dict, res: int) -> List[str]:
    # Prefer geo_to_cells (your build hinted at this), then other variants
    for fn_name in ("geo_to_cells", "geojson_to_cells", "polygon_to_cells", "polyfill"):
        fn = getattr(h3, fn_name, None)
        if fn is None:
            continue
        try:
            cells = fn(geo, res)  # some variants return sets/lists
            return list(cells)
        except Exception:
            continue
    # If all else fails, raise and let caller try fallback
    raise RuntimeError("No H3 polygon-fill function accepted the input.")

def _cells_by_sampling(poly: Polygon, res: int, step_deg: float = 0.003) -> List[str]:
    """
    Robust fallback: sample a lat/lon grid inside the polygon and convert to cells.
    step_deg≈0.003 gives ~300m spacing; good enough for res=9 coverage.
    """
    minx, miny, maxx, maxy = poly.bounds  # (lon, lat)
    cells = set()
    # version-agnostic point->cell
    latlng_to_cell = getattr(h3, "latlng_to_cell", None) or getattr(h3, "geo_to_h3", None)
    if latlng_to_cell is None:
        raise RuntimeError("No H3 point->cell function found (latlng_to_cell/geo_to_h3).")

    lat = miny
    while lat <= maxy:
        lon = minx
        while lon <= maxx:
            p = Point(lon, lat)  # shapely uses (x=lon, y=lat)
            if poly.contains(p):
                cells.add(latlng_to_cell(lat, lon, res))
            lon += step_deg
        lat += step_deg
    return list(cells)

def build_hex_index(res: int = 9) -> List[str]:
    nyc_poly = load_nyc_boundary_polygon()  # shapely Polygon
    # Build a simple, closed GeoJSON polygon (lon/lat order)
    coords = list(mapping(nyc_poly)["coordinates"][0])
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    geo = {"type": "Polygon", "coordinates": [coords]}

    try:
        return _cells_from_geojson(geo, res)
    except Exception:
        # Fallback: robust sampler
        return _cells_by_sampling(nyc_poly, res)

def h3_polygon_coords(cell: str):
    # Version-agnostic boundary fetch
    if hasattr(h3, "cell_to_boundary"):               # v4
        b = h3.cell_to_boundary(cell)                 # returns [(lat, lon), ...]
    elif hasattr(h3, "h3_to_geo_boundary"):           # v3
        b = h3.h3_to_geo_boundary(cell, geo_json=True)
    else:
        raise AttributeError("No H3 boundary function found.")
    # deck.gl expects {lng, lat}
    return [{"lng": lon, "lat": lat} for lat, lon in b]

def hex_geojson(hexes: List[str]) -> Dict:
    features = []
    for cell in hexes:
        b = h3_polygon_coords(cell)
        ring = [[pt["lng"], pt["lat"]] for pt in b]
        ring.append(ring[0])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"hex": cell},
        })
    return {"type": "FeatureCollection", "features": features}
