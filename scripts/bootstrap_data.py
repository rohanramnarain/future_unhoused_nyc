import os, sys, json
# ensure project root is on the path BEFORE importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.download import download_small_samples, download_dcp_housing
from src.data.hexgrid import build_hex_index, hex_geojson
from src.config import settings

def main():
    os.makedirs(settings.raw_dir, exist_ok=True)
    os.makedirs(settings.external_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)

    download_small_samples()
    download_dcp_housing()

    # Create hex geojson covering NYC
    hexes = build_hex_index(res=9)
    gj = hex_geojson(hexes)
    out = os.path.join(settings.processed_dir, "hexes.geojson")
    with open(out, "w") as f:
        json.dump(gj, f)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
