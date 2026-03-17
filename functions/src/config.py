# src/config.py
import os
from dataclasses import dataclass, field


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return default
    raw = str(raw).strip()
    if not raw:
        return default
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in ("1", "true", "yes", "y", "on"):
        return True
    if val in ("0", "false", "no", "n", "off"):
        return False
    return default

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

    # Modeling configuration
    # MODEL_TARGET should be a column present in the features table.
    # Default stays on the demo target to preserve backward compatibility.
    model_target: str = os.getenv("MODEL_TARGET", "risk_proxy")

    # Optional (for "real" training): a table keyed by (hex, year) containing the target.
    # The target column name defaults to MODEL_TARGET unless OUTCOMES_VALUE_COL is set.
    outcomes_path: str = os.getenv("OUTCOMES_PATH", "")
    outcomes_hex_col: str = os.getenv("OUTCOMES_HEX_COL", "hex")
    outcomes_year_col: str = os.getenv("OUTCOMES_YEAR_COL", "year")
    outcomes_value_col: str = os.getenv("OUTCOMES_VALUE_COL", "")

    # Year ranges
    # TRAIN_YEARS is optional; if omitted, we train on rows where the target is non-null.
    train_years: list[int] = field(default_factory=lambda: _parse_int_list(os.getenv("TRAIN_YEARS"), []))
    # PREDICT_YEARS controls what years are written to predictions CSVs and shown in the app.
    predict_years: list[int] = field(default_factory=lambda: _parse_int_list(os.getenv("PREDICT_YEARS"), [2026, 2027, 2028, 2029]))

    # Optional ZIP-year economic scenario used to drive future-only feature multipliers.
    # If present, these are applied for ZIP_ECON_SCENARIO_YEARS instead of flat growth.
    use_zip_econ_scenario: bool = _parse_bool(os.getenv("USE_ZIP_ECON_SCENARIO", "1"), True)
    zip_econ_scenario_path: str = os.getenv(
        "ZIP_ECON_SCENARIO_PATH",
        os.path.join("data", "interim", "zip_econ_scenario_2027_2029.csv"),
    )
    zip_econ_scenario_years: list[int] = field(
        default_factory=lambda: _parse_int_list(os.getenv("ZIP_ECON_SCENARIO_YEARS"), [2027, 2028, 2029])
    )

    app_port: int = int(os.getenv("APP_PORT", "8050"))
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")

settings = Settings()
