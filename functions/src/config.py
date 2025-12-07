# src/config.py
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency handled gracefully
    load_dotenv = None

if load_dotenv:
    load_dotenv()

@dataclass
class Settings:
    data_dir: str = os.path.join("data")
    raw_dir: str = os.path.join("data", "raw")
    interim_dir: str = os.path.join("data", "interim")
    processed_dir: str = os.path.join("data", "processed")
    external_dir: str = os.path.join("data", "external")

    models_dir: str = os.path.join("models", "artifacts")
    reports_dir: str = os.path.join("models", "reports")

    mapbox_token: str = os.getenv("MAPBOX_TOKEN") or os.getenv("MAPBOX_API_TOKEN", "")
    census_api_key: str = os.getenv("CENSUS_API_KEY", "")
    socrata_app_token: str = os.getenv("SOCRATA_APP_TOKEN", "")
    socrata_api_id: str = os.getenv("SOCRATA_API_ID", "")
    socrata_api_secret: str = os.getenv("SOCRATA_API_SECRET", "")
    socrata_username: str = os.getenv("SOCRATA_USERNAME", "")
    socrata_password: str = os.getenv("SOCRATA_PASSWORD", "")
    filed_evictions_dataset: str = os.getenv(
        "FILED_EVICTIONS_DATASET",
        "https://raw.githubusercontent.com/housing-data-coalition/rtc-eviction-viz/main/csv/filings_by_zip_since_032320_pulled_120120.csv",
    )
    nycdb_cli: str = os.getenv("NYCDB_CLI", "nycdb")
    nycdb_database_url: str = os.getenv("NYCDB_DATABASE_URL", "")
    dcp_housing_limit: int | None = (
        int(os.getenv("DCP_HOUSING_LIMIT"))
        if os.getenv("DCP_HOUSING_LIMIT") not in (None, "")
        else 100000
    )
    enable_dcp_geocoder: bool = bool(int(os.getenv("ENABLE_DCP_GEOCODER", "0")))
    geocoder_user_agent: str = os.getenv("GEOCODER_USER_AGENT", "future-unhoused-nyc")

    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    advanced_model: bool = bool(int(os.getenv("ADVANCED_MODEL", "0")))

    app_port: int = int(os.getenv("APP_PORT", "8050"))
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")

settings = Settings()
