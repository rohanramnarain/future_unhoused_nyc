from shapely.geometry import Polygon


# Rough NYC bounding polygon fallback (if boundary file not present)
NYC_BBOX = Polygon([
(-74.255, 40.496),
(-73.700, 40.496),
(-73.700, 40.915),
(-74.255, 40.915),
])