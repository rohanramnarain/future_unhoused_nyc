import os
from dataclasses import dataclass


@dataclass
class Settings:
    data_dir: str = os.path.join("data")
    raw_dir: str = os.path.join("data", "raw")
    interim_dir: str = os.path.join("data", "interim")
    processed_dir: str = os.path.join("data", "processed")
    external_dir: str = os.path.join("data", "external")


    models_dir: str = os.path.join("models", "artifacts")
    reports_dir: str = os.path.join("models", "reports")


    mapbox_token: str = os.getenv("MAPBOX_TOKEN", "")
    census_api_key: str = os.getenv("CENSUS_API_KEY", "")
    socrata_app_token: str = os.getenv("SOCRATA_APP_TOKEN", "")


    random_seed: int = int(os.getenv("RANDOM_SEED", 42))
    advanced_model: bool = bool(int(os.getenv("ADVANCED_MODEL", "0")))


    app_port: int = int(os.getenv("APP_PORT", 8050))
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")


    settings = Settings()