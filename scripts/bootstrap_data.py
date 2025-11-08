import os
from src.data.download import download_small_samples
from src.data.hexgrid import build_hex_index, hex_geojson
from src.config import settings


if __name__ == "__main__":
    os.makedirs(settings.raw_dir, exist_ok=True)
    os.makedirs(settings.external_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)

    download_small_samples()

    # Create hex geojson covering NYC
    hexes = build_hex_index(res=9)
    gj = hex_geojson(hexes)
    out = os.path.join(settings.processed_dir, "hexes.geojson")
    import json
    with open(out, "w") as f:
        json.dump(gj, f)
    print(f"Wrote {out}")