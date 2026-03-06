import time
from typing import Iterable

import pandas as pd
import requests

from ..config import settings
from ..utils.logging import get_logger

logger = get_logger()
NYC_OD_BASE = "https://data.cityofnewyork.us/resource"


HOMELESS_311_TYPES = [
    "Homeless Person Assistance",
    "Homeless Encampment",
]


def _headers() -> dict:
    headers = {"Accept": "application/json"}
    if settings.socrata_app_token:
        headers["X-App-Token"] = settings.socrata_app_token
    return headers


def _auth():
    if settings.socrata_username and settings.socrata_password:
        return (settings.socrata_username, settings.socrata_password)
    if settings.socrata_api_id and settings.socrata_api_secret:
        return (settings.socrata_api_id, settings.socrata_api_secret)
    return None


def _where_clause(
    *,
    complaint_types: Iterable[str],
    start_date: str,
    end_date: str,
) -> str:
    # Socrata SODA expects ISO-ish timestamps; created_date is a datetime.
    # Escape single quotes for the SoQL string literal.
    type_list = ",".join(["'" + str(t).replace("'", "''") + "'" for t in complaint_types])
    return (
        f"complaint_type in({type_list}) "
        f"AND created_date >= '{start_date}T00:00:00.000' "
        f"AND created_date < '{end_date}T00:00:00.000' "
        "AND latitude IS NOT NULL AND longitude IS NOT NULL"
    )


def download_311_rows(
    *,
    start_date: str,
    end_date: str,
    complaint_types: list[str] | None = None,
    page_size: int = 50000,
    max_pages: int | None = None,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    """Download filtered 311 rows from NYC Open Data (Socrata).

    This is intended for building outcome labels. It pulls only the columns needed
    for spatial/year aggregation.
    """

    complaint_types = complaint_types or HOMELESS_311_TYPES

    endpoint = "erm2-nwe9.json"
    url = f"{NYC_OD_BASE}/{endpoint}"

    where = _where_clause(complaint_types=complaint_types, start_date=start_date, end_date=end_date)

    all_rows: list[dict] = []
    offset = 0
    pages = 0

    while True:
        params = {
            "$select": "created_date,latitude,longitude,complaint_type",
            "$where": where,
            "$limit": page_size,
            "$offset": offset,
        }
        r = requests.get(url, params=params, headers=_headers(), timeout=120, auth=_auth())
        if r.status_code == 429:
            logger.warning("429 Too Many Requests; sleeping 2s then retry")
            time.sleep(2)
            r = requests.get(url, params=params, headers=_headers(), timeout=120, auth=_auth())
        r.raise_for_status()
        rows = r.json() or []
        if not rows:
            break

        all_rows.extend(rows)
        offset += len(rows)
        pages += 1

        logger.info("Downloaded %s rows (pages=%s offset=%s)", len(all_rows), pages, offset)

        if max_pages is not None and pages >= max_pages:
            logger.warning("Reached max_pages=%s; stopping early", max_pages)
            break

        if sleep_s:
            time.sleep(sleep_s)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["created_date", "latitude", "longitude"]).copy()

    return df


def build_hex_year_outcomes(
    *,
    start_year: int,
    end_year: int,
    horizon_years: int = 1,
    target_col: str = "future_homeless_311",
    complaint_types: list[str] | None = None,
    res: int = 9,
) -> pd.DataFrame:
    """Build (hex, year) outcome table for next-year homeless-related 311 requests.

    Produces rows keyed by (hex, year) where `year` corresponds to the *feature year*.
    The label value is the count in (year + horizon_years).
    """

    from .features import _h3_index_points  # reuse existing helper

    # Pull years [start_year, end_year] for outcomes; then align to feature year by shifting.
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year + 1}-01-01"

    df = download_311_rows(
        start_date=start_date,
        end_date=end_date,
        complaint_types=complaint_types,
    )
    if df.empty:
        return pd.DataFrame(columns=["hex", "year", target_col])

    df["year"] = df["created_date"].dt.year.astype(int)
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

    df["hex"] = _h3_index_points(df, res=res)
    df = df.dropna(subset=["hex"]).copy()

    counts = df.groupby(["hex", "year"]).size().rename("_count").reset_index()

    # Shift outcome year back to feature-year index
    counts["year"] = counts["year"] - int(horizon_years)
    counts = counts.rename(columns={"_count": target_col})

    # Keep only feature years we can train on
    counts = counts[(counts["year"] >= start_year) & (counts["year"] <= end_year)].copy()

    return counts
