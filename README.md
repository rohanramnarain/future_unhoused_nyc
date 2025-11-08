# NYC Homelessness Forecast (2026–2029)


Equity-aware, spatiotemporal forecasting and an interactive Dash + deck.gl/Mapbox app to visualize predicted homelessness risk by H3 hex in NYC.


## Features
- H3 hex tiling of NYC; year-by-year features (311, HPD complaints, evictions, ACS, transit).*
- Baseline ML (LightGBM) with lagged spatial features.
- Conformal prediction for calibrated uncertainty bands; borough-level reconciliation.
- Dash app: zoomable street map, time slider (2026–2029), scenario toggles, equity scorecard.


> *The repo includes a **small bootstrap sample** to run end-to-end quickly. Swap in full NYC datasets later.


## API Keys / Environment
Create `.env` from `.env.sample`:


- `MAPBOX_TOKEN` (required): for map tiles.
- `CENSUS_API_KEY` (optional but recommended): to pull ACS.
- `SOCRATA_APP_TOKEN` (optional): to avoid throttling NYC Open Data.


## Quickstart
```bash
# 1) Setup
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.sample .env # then edit with your keys


# 2) Bootstrap (downloads small samples + NYC boundary)
python scripts/bootstrap_data.py


# 3) Build features & train
python scripts/train_baseline.py # creates models/artifacts and data/processed/predictions_2026_2029.csv


# 4) Run the app
python scripts/run_app.py # http://127.0.0.1:8050


# Or with Makefile shortcuts
make setup && make data && make train && make app

---
## .env.sample
```bash
# Map rendering
MAPBOX_TOKEN=YOUR_MAPBOX_PUBLIC_TOKEN


# Optional accelerators
CENSUS_API_KEY=
SOCRATA_APP_TOKEN=


# App
APP_PORT=8050
APP_HOST=127.0.0.1


# Modeling
RANDOM_SEED=42
ADVANCED_MODEL=0