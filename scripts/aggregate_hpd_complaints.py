import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import h3
import pandas as pd

# Ensure repo root on path before importing project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger()


def _series_from_candidates(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return df[name]
    return pd.Series([None] * len(df), index=df.index)


def _latlng_to_cell(lat: float, lon: float, res: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, res)
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lon, res)
    raise AttributeError("No H3 point-to-cell function available.")


def _detect_default_source() -> Path:
    raw_dir = Path(settings.raw_dir)
    direct = raw_dir / "hpd_complaints.csv"
    if direct.exists():
        return direct
    matches = sorted(raw_dir.glob("Housing_Maintenance*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find an HPD CSV in data/raw. Provide --source explicitly."
    )


def aggregate_hpd(source: Path, output: Path, chunk_size: int, resolution: int) -> None:
    if not source.exists():
        raise FileNotFoundError(f"HPD CSV not found: {source}")

    os.makedirs(output.parent, exist_ok=True)
    groups: list[pd.DataFrame] = []

    chunk_iter = pd.read_csv(
        source,
        chunksize=chunk_size,
        low_memory=False,
    )

    for idx, chunk in enumerate(chunk_iter, start=1):
        chunk.columns = [c.strip().lower().replace(" ", "_") for c in chunk.columns]
        lat = pd.to_numeric(_series_from_candidates(chunk, ("latitude", "lat")), errors="coerce")
        lon = pd.to_numeric(_series_from_candidates(chunk, ("longitude", "lon", "lng")), errors="coerce")
        mask = lat.notna() & lon.notna()
        if not mask.any():
            continue
        lat = lat[mask]
        lon = lon[mask]
        hexes = [_latlng_to_cell(la, lo, resolution) for la, lo in zip(lat, lon)]
        counts = (
            pd.Series(hexes)
            .value_counts()
            .rename_axis("hex")
            .reset_index(name="nhpd")
        )
        groups.append(counts)
        logger.info("Processed chunk %s (%s rows with coordinates)", idx, mask.sum())

    if not groups:
        agg = pd.DataFrame({"hex": [], "nhpd": []})
    else:
        agg = pd.concat(groups, ignore_index=True)
        agg = agg.groupby("hex", as_index=False)["nhpd"].sum()

    output.write_text(agg.to_json(orient="records"))
    logger.info("Wrote %s hex records to %s", len(agg), output)


def main():
    parser = argparse.ArgumentParser(description="Aggregate HPD complaints into H3 hex counts.")
    parser.add_argument(
        "--source",
        type=Path,
        default=_detect_default_source(),
        help="Path to the HPD complaints CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.interim_dir) / "hpd_hex_counts.json",
        help="Path where the aggregated JSON will be written",
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--resolution", type=int, default=9)
    args = parser.parse_args()

    aggregate_hpd(args.source, args.output, args.chunk_size, args.resolution)


if __name__ == "__main__":
    main()
