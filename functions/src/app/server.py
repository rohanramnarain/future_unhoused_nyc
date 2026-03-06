# src/app/server.py
import os, json, re, glob
from typing import Dict, List
import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree

import dash
from dash import html, dcc, Output, Input, State, ctx
import dash_bootstrap_components as dbc

from ..config import settings
from ..utils.logging import get_logger

# Files produced by the pipeline
_PRED_YEARS = settings.predict_years or [2026, 2027, 2028, 2029]
_Y0, _Y1 = min(_PRED_YEARS), max(_PRED_YEARS)
PRED_BASE = os.path.join(settings.processed_dir, f"predictions_{_Y0}_{_Y1}.csv")
PRED_PATTERN = os.path.join(settings.processed_dir, f"predictions_*_{_Y0}_{_Y1}.csv")
HEX_GJ_PATH = os.path.join(settings.processed_dir, "hexes.geojson")
MODZCTA_PATH = os.path.join(settings.external_dir, "modzcta.geojson")

MODEL_LABELS = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random Forest",
}

logger = get_logger()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LITERA],
    assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    assets_url_path="/assets",
    title="The Future of the Unhoused",
)
server = app.server


def _build_zip_assets():
    if not os.path.exists(MODZCTA_PATH):
        logger.warning("ZIP boundary file missing at %s; tooltips will omit ZIP codes.", MODZCTA_PATH)
        return None, {}
    try:
        with open(MODZCTA_PATH, "r") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load MODZCTA boundaries (%s); tooltips will omit ZIP codes.", exc)
        return None, {}

    geoms, attrs = [], []
    centroids: dict[str, dict[str, float]] = {}
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        props = feat.get("properties", {})
        try:
            poly = shape(geom)
        except Exception:
            continue
        if poly.is_empty:
            continue
        zip_code = str(props.get("modzcta") or "").strip()
        pop_est = props.get("pop_est")
        pop_est = int(pop_est) if str(pop_est).isdigit() else None
        if not zip_code or zip_code == "99999" or not zip_code.isdigit() or pop_est == 0:
            continue
        geoms.append(poly)
        attrs.append({
            "zip": zip_code,
            "zip_label": (props.get("label") or "").strip(),
        })
        rep_point = poly.representative_point()
        centroids[zip_code] = {"lat": rep_point.y, "lon": rep_point.x}
    if not geoms:
        logger.warning("Loaded MODZCTA boundaries but none were usable; tooltips will omit ZIP codes.")
        return None, {}

    tree = STRtree(geoms)

    def lookup(point):
        for idx in tree.query(point):
            geom = geoms[idx]
            if geom.covers(point):
                return attrs[idx]
        return None

    logger.info("Loaded %s MODZCTA polygons for ZIP enrichment", len(geoms))
    return lookup, centroids


ZIP_LOOKUP, ZIP_CENTROIDS = _build_zip_assets()


def _normalize_zip(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) < 5:
        return None
    return digits[:5]


def _zip_focus_view_state(value: str | None) -> dict | None:
    zip_code = _normalize_zip(value)
    if not zip_code:
        return None
    centroid = ZIP_CENTROIDS.get(zip_code)
    if not centroid:
        return None
    return {
        "latitude": centroid["lat"],
        "longitude": centroid["lon"],
        "zoom": 13.5,
        "pitch": 0,
        "bearing": 0,
        "transitionDuration": 600,
    }


def _load_hex_features() -> List[dict]:
    if not os.path.exists(HEX_GJ_PATH):
        raise FileNotFoundError(
            "Missing hex grid. Run:\n"
            "  python scripts/bootstrap_data.py\n"
            "to generate data/processed/ files."
        )
    with open(HEX_GJ_PATH, "r") as f:
        gj = json.load(f)
    if "features" not in gj:
        raise ValueError("Hex GeoJSON missing features array")
    return gj["features"]


def _build_zip_props(hex_features: List[dict]):
    zip_props_by_hex = {}
    for feat in hex_features:
        hx = feat["properties"]["hex"]
        info = {}
        if ZIP_LOOKUP:
            try:
                centroid = shape(feat["geometry"]).representative_point()
                lookup = ZIP_LOOKUP(centroid)
                if lookup:
                    info = lookup.copy()
            except Exception as exc:
                logger.debug("ZIP lookup failed for hex %s: %s", hx, exc)
        zip_props_by_hex[hx] = info
    return zip_props_by_hex


def build_geojson_blob(pred_path: str, hex_features: List[dict], zip_props_by_hex: dict):
    preds = pd.read_csv(pred_path)
    props_by_hex_year = {
        (r.hex, int(r.year)): {"pred": float(r.pred), "lo": float(r.lo), "hi": float(r.hi)}
        for r in preds.itertuples()
    }

    years = sorted(preds["year"].unique())
    feat_out = []
    for feat in hex_features:
        hx = feat["properties"]["hex"]
        for year in years:
            p = props_by_hex_year.get((hx, int(year)))
            if p is None:
                continue
            zip_info = zip_props_by_hex.get(hx, {})
            feat_out.append({
                "type": "Feature",
                "geometry": feat["geometry"],
                "properties": {
                    "hex": hx,
                    "year": int(year),
                    "zip": zip_info.get("zip"),
                    "zip_label": zip_info.get("zip_label"),
                    **p,
                },
            })
    return {"type": "FeatureCollection", "features": feat_out}


def _discover_prediction_files():
    files: Dict[str, str] = {}
    if os.path.exists(PRED_BASE):
        files["lgbm"] = PRED_BASE
    for path in glob.glob(PRED_PATTERN):
        name = os.path.basename(path)
        if not name.startswith("predictions_") or not name.endswith(f"_{_Y0}_{_Y1}.csv"):
            continue
        suffix = name[len("predictions_"):-len(f"_{_Y0}_{_Y1}.csv")]
        if not suffix:
            continue
        key = suffix.lower()
        files[key] = path

    if not files:
        raise FileNotFoundError(
            "Missing predictions or hexes. Run:\n"
            "  python scripts/bootstrap_data.py\n"
            "  python scripts/train_baseline.py\n"
            "to generate data/processed/ files."
        )
    return files


def build_all_geojson_blobs():
    pred_files = _discover_prediction_files()
    hex_features = _load_hex_features()
    zip_props_by_hex = _build_zip_props(hex_features)

    blobs: Dict[str, dict] = {}
    for key, path in pred_files.items():
        blobs[key] = build_geojson_blob(path, hex_features, zip_props_by_hex)

    ordered_keys = []
    if "lgbm" in pred_files:
        ordered_keys.append("lgbm")
    for k in sorted(pred_files.keys()):
        if k != "lgbm":
            ordered_keys.append(k)

    model_options = [{"label": MODEL_LABELS.get(k, k.upper()), "value": k} for k in ordered_keys]
    default_model = ordered_keys[0]
    return blobs, model_options, default_model


GJ_BY_MODEL, MODEL_OPTIONS, DEFAULT_MODEL = build_all_geojson_blobs()


def make_deck_spec(geojson: Dict, year: int, color_metric: str = "pred", focus_view_state: dict | None = None):
    brewer_stops = [
        {"max": 0.0, "color": [0, 0, 0, 0]},
        {"max": 0.4, "color": [255, 255, 204, 120]},
        {"max": 0.6, "color": [255, 237, 160, 170]},
        {"max": 0.8, "color": [254, 178, 76, 210]},
        {"max": 0.9, "color": [253, 141, 60, 235]},
        {"max": 1.0, "color": [189, 0, 38, 255]},
    ]
    spec = {
        "initialViewState": {"latitude": 40.7128, "longitude": -74.0060, "zoom": 10},
        "controller": True,
        "mapStyle": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "layers": [{
            "@@type": "GeoJsonLayer",
            "id": "hex-choropleth",
            "data": geojson,
            "colorMetric": color_metric,
            "colorStops": brewer_stops,
            "pickable": True,
            "autoHighlight": True,
            "highlightColor": [255, 255, 255, 200],
            "stroked": True,
            "filled": True,
            "opacity": 0.35,
            "getLineColor": [245, 245, 245, 120],
            "lineWidthMinPixels": 0.6,
        }],
    }
    if focus_view_state:
        spec["focusViewState"] = focus_view_state
    return spec


ANALYSIS_COPY = html.Div(className="fhf-prose", children=[
    html.H6("Acronym dictionary (used below)", className="fhf-section-title"),
    html.Ul([
        html.Li([html.B("NYC"), " — New York City."]),
        html.Li([html.B("311"), " — NYC's non-emergency service request system."]),
        html.Li([html.B("HPD"), " — NYC Department of Housing Preservation and Development."]),
        html.Li([html.B("DCP"), " — NYC Department of City Planning."]),
        html.Li([html.B("H3"), " — Uber's hexagonal geospatial indexing system used to divide the map into hexes."]),
        html.Li([html.B("MODZCTA"), " — Modified ZIP Code Tabulation Area (NYC's ZIP-like boundary geography)."]),
        html.Li([html.B("ACS"), " — American Community Survey (U.S. Census program)."]),
        html.Li([html.B("CSV"), " — Comma-Separated Values file format."]),
        html.Li([html.B("JSON"), " — JavaScript Object Notation file format."]),
        html.Li([html.B("ZIP"), " — Postal ZIP code area (approximate here when mapped to hexes)."]),
    ]),
    html.H6("What are we predicting, exactly?", className="fhf-section-title"),
    html.Ul([
        html.Li("Target column: the model predicts a single outcome column for each hex-year (configured as MODEL_TARGET in the pipeline). In this deployment, that outcome is next-year homeless-related 311 activity at the hex level."),
        html.Li("How this becomes relative risk: the model first outputs raw predicted levels, then src/models/evaluate.py rescales those values within each year into percentile-style ranks on a 0-1 scale (shown as pred). Higher rank = relatively higher predicted risk versus other hexes that year."),
    ]),
    html.H6("What those column names mean (plain English)", className="fhf-section-title"),
    html.Ul([
        html.Li([
            html.Code("n311_y"),
            " — How many relevant 311 service requests came from that area for that year (from NYC 311 open data).",
        ]),
        html.Li([
            html.Code("nhpd_y"),
            " — How many HPD housing complaints were counted in that area (from HPD Complaint Problems data).",
        ]),
        html.Li([
            html.Code("nevict_y"),
            " — How many executed evictions were recorded there (from NYC Residential Evictions).",
        ]),
        html.Li([
            html.Code("nfiled_y"),
            " — How many eviction cases were filed there, even if not yet executed (from filed-eviction dataset).",
        ]),
        html.Li([
            html.Code("n_dcp_units"),
            " — Total housing units in DCP-tracked projects in that area (from DCP housing program data).",
        ]),
        html.Li([
            html.Code("n_dcp_aff_units"),
            " — Affordable units among those DCP-tracked units (same DCP source).",
        ]),
        html.Li([
            html.Code("n_dcp_expiring5yr"),
            " — Number of DCP-tracked units whose affordability/regulatory status is expected to expire within ~5 years.",
        ]),
        html.Li([
            html.Code("n_dcp_expired"),
            " — Number of DCP-tracked units whose affordability/regulatory period is already expired.",
        ]),
        html.Li([
            html.Code("dcp_status_median"),
            " — A median summary score of project status in that area (from DCP status fields, converted to numeric categories).",
        ]),
    ]),
    html.P([
        html.B("Interpreting the color · "),
        "Scores are normalized between 0 and 1 inside each year for the selected model, so 1.0 means \"highest relative risk across hexes this year for this model\" rather than a literal count of future shelter placements.",
    ]),
    html.H6("Source links", className="fhf-section-title"),
    html.Ul([
        html.Li(html.A("NYC 311 Service Requests", href="https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9", target="_blank")),
        html.Li(html.A("HPD Complaint Problems", href="https://data.cityofnewyork.us/Housing-Development/HPD-Complaint-Problems/uwyv-629c", target="_blank")),
        html.Li(html.A("Residential Evictions", href="https://data.cityofnewyork.us/City-Government/Eviction-Notices/6z8x-wfk4", target="_blank")),
        html.Li(html.A("MODZCTA Boundaries", href="https://data.cityofnewyork.us/City-Government/Modified-Zip-Code-Tabulation-Areas-MODZCTA-/pri4-ifjk", target="_blank")),
        html.Li(html.A("American Community Survey 5-year", href="https://www.census.gov/data/developers/data-sets/acs-5year.html", target="_blank")),
        html.Li(html.A("H3 Index", href="https://h3geo.org/", target="_blank")),
    ]),
])


TECHNICAL_DETAILS_COPY = html.Div(className="fhf-prose", children=[
    html.P("This deployment includes the freshest datasets we finished ingesting on December 7, 2025:"),
    html.Ul([
        html.Li("data/raw/311.json — service requests already geocoded to lat/lon."),
        html.Li("data/interim/hpd_hex_counts.json — ≈6k pre-aggregated HPD complaint totals per H3 hex, produced by scripts/aggregate_hpd_complaints.py."),
        html.Li("data/raw/evictions.json and data/raw/filed_evictions.json — executed and filed eviction events for nevict/nfiled."),
        html.Li("data/processed/hexes.geojson — the NYC H3 grid we render and join against."),
        html.Li("data/processed/predictions_<model>_2026_2029.csv — per-model outputs with conformal lower/upper bands and per-year percentile scaling."),
    ]),
    html.H6("Pipeline checkpoints", className="fhf-section-title"),
    html.P([
        html.B("Ingestion · "),
        "scripts/bootstrap_data.py grabs 311, eviction, and MODZCTA samples, then scripts/aggregate_hpd_complaints.py streams the large HPD CSV into the lightweight hex counts listed above.",
    ]),
    html.P([
        html.B("Feature engineering · "),
        "src/data/features.py reads the aggregated HPD counts plus 311/eviction events and DCP housing program data, maps everything to the hex grid, and clones the table for each forecast year with a modest 3% growth proxy (*_y columns).",
    ]),
    html.P([
        html.B("Model + bands · "),
        "src/models/baselines.py fits your selected model on nine engineered signals (n311_y, nhpd_y, nevict_y, nfiled_y, n_dcp_units, n_dcp_aff_units, n_dcp_expiring5yr, n_dcp_expired, dcp_status_median), while src/models/evaluate.py wraps the predictions with symmetric conformal intervals so each hex gets pred, lo, and hi.",
    ]),
    html.P("Need higher fidelity? Swap the bootstrap inputs for full NYC feeds, rerun scripts/aggregate_hpd_complaints.py and scripts/train_baseline.py, then redeploy."),
])


MODEL_COPY = {
    "lgbm": html.Div(className="fhf-prose", children=[
        html.P("LightGBM = gradient boosting (leaf-wise trees)."),
        html.Img(src="/assets/model_diagram_lgbm.svg?v=4", className="model-diagram", alt="LightGBM diagram"),
        html.Div("Diagram: boosting adds trees sequentially; LightGBM often grows leaf-wise.", className="model-diagram-note"),
        html.P("Light Gradient Boosting uses hundreds of tiny decision trees trained sequentially; each tree focuses on the residual mistakes of prior trees."),
        html.Ul([
            html.Li("Fast on sparse tabular data and handles nonlinear jumps in complaint/eviction patterns."),
            html.Li("Same nine engineered signals (n311_y, nhpd_y, nevict_y, nfiled_y, n_dcp_units, n_dcp_aff_units, n_dcp_expiring5yr, n_dcp_expired, dcp_status_median)."),
            html.Li("Conformal bands add a give-or-take range without extra retraining."),
        ]),
    ]),
    "xgb": html.Div(className="fhf-prose", children=[
        html.P("XGBoost = gradient boosting (level-wise trees)."),
        html.Img(src="/assets/model_diagram_xgb.svg?v=4", className="model-diagram", alt="XGBoost diagram"),
        html.Div("Diagram: each tree is an incremental correction to the score.", className="model-diagram-note"),
        html.P("Another gradient-boosted tree ensemble; uses histogram splits for speed and strong performance on tabular problems."),
        html.Ul([
            html.Li("Captures sharp thresholds (e.g., sudden eviction spikes) while staying fast enough for frequent re-trains."),
            html.Li("Same feature set and per-year percentile scaling as LightGBM so colors stay comparable within the model."),
            html.Li("Choose this to test a stronger regularized boosting baseline."),
        ]),
    ]),
    "rf": html.Div(className="fhf-prose", children=[
        html.P("Random Forest = many trees voting/averaging."),
        html.Img(src="/assets/model_diagram_rf.svg?v=4", className="model-diagram", alt="Random Forest diagram"),
        html.Div("Diagram: many independent trees vote/average into one prediction.", className="model-diagram-note"),
        html.P("Hundreds of decorrelated decision trees averaged together; great for quick baselines and uncertainty intuition."),
        html.Ul([
            html.Li("Robust to noisy features and less sensitive to hyperparameters."),
            html.Li("Produces smoother risk surfaces; percentile scaling still runs per year for this model."),
            html.Li("Good sanity-check against boosting models."),
        ]),
    ]),
}


def model_copy_component(model_key: str):
    return MODEL_COPY.get(model_key, MODEL_COPY.get(DEFAULT_MODEL))


ALL_MODEL_COPY = html.Div(children=[
    html.H6("LightGBM", className="fhf-section-title mt-1"),
    MODEL_COPY["lgbm"],
    html.Hr(className="my-3"),
    html.H6("Random Forest", className="fhf-section-title"),
    MODEL_COPY["rf"],
    html.Hr(className="my-3"),
    html.H6("XGBoost", className="fhf-section-title"),
    MODEL_COPY["xgb"],
])


app.layout = dbc.Container([
    dcc.Location(id="app-location", refresh=False),

    dbc.Row([
        dbc.Col(
            html.Div(className="fhf-hero", children=[
                html.H2("The Future of the Unhoused in NYC (2026 - 2029 Forecast Map)", className="fhf-hero-title mb-0"),
                html.Div(className="fhf-badges mt-3", children=[
                    dbc.Badge("Relative score (0–1)", color="primary", className="me-2", pill=True),
                    dbc.Badge("It's a percentile, not a probability", color="secondary", className="me-2", pill=True),
                    dbc.Badge("Data ingested: Dec 7, 2025", color="light", text_color="dark", pill=True),
                ]),
                html.Div(className="mt-3 fhf-links", children=[
                    html.A("Sources", id="link-sources", href="#sources", className="me-3"),
                    html.A("Method", id="link-method", href="#method", className="me-3"),
                    html.A("Limitations", id="link-limits", href="#limits", className="me-3"),
                    html.A("What am I looking at?", id="link-read-map", href="#read-map", className="me-3"),
                    html.A("For Techies", id="link-technical", href="#technical-details"),
                ]),
            ]),
            md=12,
        ),
    ], className="mt-3 mb-3"),

    dbc.Card(className="fhf-card fhf-controls mb-3", children=[
        dbc.CardBody(className="fhf-card-body", children=[
            html.H5("Controls", className="fhf-section-title mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Jump to ZIP", html_for="zip-input", className="fhf-muted mb-1"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="zip-input",
                            type="text",
                            placeholder="Enter NYC ZIP (e.g., 10027)",
                            debounce=False,
                            maxLength=10,
                            inputMode="numeric",
                            autoComplete="postal-code",
                        ),
                        dbc.Button("Go", id="zip-submit", n_clicks=0, color="primary", outline=False),
                    ]),
                    html.Div("Tip: press Enter or click Go.", className="fhf-muted mt-1"),
                ], md=5),

                dbc.Col([
                    dbc.Label("Metric", html_for="metric", className="fhf-muted mb-1"),
                    dbc.Select(
                        id="metric",
                        value="pred",
                        options=[
                            {"label": "Prediction (μ)", "value": "pred"},
                            {"label": "Lower band (lo)", "value": "lo"},
                            {"label": "Upper band (hi)", "value": "hi"},
                        ],
                    ),
                ], md=3),

                dbc.Col([
                    dbc.Label("Model", html_for="model", className="fhf-muted mb-1"),
                    dbc.Select(
                        id="model",
                        value=DEFAULT_MODEL,
                        options=MODEL_OPTIONS,
                    ),
                ], md=4),
            ], className="g-3"),

            html.Hr(className="my-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Year", html_for="year", className="fhf-muted mb-1"),
                    dcc.Slider(
                        id="year",
                        min=2026,
                        max=2029,
                        value=2026,
                        step=1,
                        marks={y: str(y) for y in [2026, 2027, 2028, 2029]},
                    ),
                ], md=12),
            ]),
        ]),
    ]),

    html.Div(id="deck-container", className="fhf-map-shell mb-3"),

    dbc.Accordion(id="info-accordion", className="mb-4", always_open=False, start_collapsed=True, children=[
        dbc.AccordionItem(
            item_id="read-map",
            title="How to read this map",
            children=html.Div(className="fhf-prose", children=[
                html.P([
                    "Algorithms already shape daily life: what we see, what we are sold, how neighborhoods are marketed, and in some cases how housing decisions are priced or prioritized. ",
                    "Those systems can reinforce inequality. This project tries to flip that logic and use forecasting tools for public good: first identify where homelessness-related pressure is likely to increase, then investigate why, and help target prevention before harm grows.",
                ]),
                html.P([
                    "The colors show a ",
                    html.B("relative risk score"),
                    " (0-1) for the selected model and year, representing the relative risk of increased homelessness pressure in that area. This is a ranking scale: darker hexes are expected to be higher than lighter hexes ",
                    "compared with other hexes in the same year.",
                ]),
                html.P([
                    html.B("Percentiles, not percentages: "),
                    "these values are percentile-style ranks on a 0-1 scale. For example, ",
                    html.B("0.80"),
                    " means a hex is ranked higher than most others that year, not \"80% chance\" of an outcome.",
                ]),
                html.P([
                    html.B("Why this is not a probability: "),
                    "a probability answers \"what is the chance this event happens here\" (for example, 70%). ",
                    "This map does not estimate that kind of chance. Instead, it rescales model outputs so the highest area in that year is near ",
                    html.B("1.0"),
                    " and lower-ranked areas are closer to ",
                    html.B("0.0"),
                    ".",
                ]),
                html.P([
                    html.B("So what does 1.0 mean? "),
                    "It means \"highest relative predicted risk in that year for this model\" - not \"100% chance\" and not \"one full event\".",
                ]),
                html.Ul([
                    html.Li("Treat the map as a hotspot ranking tool, not an absolute forecast of probability."),
                    html.Li("Use Metric to view prediction (μ) or uncertainty bands (lo/hi)."),
                    html.Li("Hover a hex for the value and year; ZIP is approximate."),
                ]),
            ]),
        ),
        dbc.AccordionItem(
            item_id="method",
            title="Model in plain English",
            children=html.Div(id="method", children=[
                html.Div(id="model-copy", children=ALL_MODEL_COPY),
            ]),
        ),
        dbc.AccordionItem(
            item_id="sources",
            title="What data feeds this right now?",
            children=html.Div(id="sources", children=[
                ANALYSIS_COPY,
            ]),
        ),
        dbc.AccordionItem(
            item_id="limits",
            title="Limitations & ethics",
            children=html.Div(id="limits", className="fhf-prose", children=[
                html.P("This is an exploratory, equity-sensitive visualization intended for discussion and learning."),
                html.Ul([
                    html.Li("Risk scores can be affected by reporting behavior (who files complaints/311), not only underlying need."),
                    html.Li("Spatial aggregation (H3) smooths local variation; do not interpret a single hex as a definitive hotspot."),
                    html.Li("Use this to prioritize questions and outreach, not to target enforcement or penalize communities."),
                    html.Li("Always pair model outputs with lived experience, service provider context, and qualitative evidence."),
                ]),
            ]),
        ),
        dbc.AccordionItem(
            item_id="technical-details",
            title="Technical details",
            children=html.Div(id="technical-details", children=[
                TECHNICAL_DETAILS_COPY,
            ]),
        ),
    ]),
], fluid=True, className="fhf-page px-3 px-md-4 pb-4")


@app.callback(
    Output("deck-container", "children"),
    Input("model", "value"),
    Input("year", "value"),
    Input("metric", "value"),
    Input("zip-submit", "n_clicks"),
    Input("zip-input", "n_submit"),
    State("zip-input", "value"),
)
def update_map(model_key, year, metric, zip_clicks, zip_enter, zip_value):  # pragma: no cover - UI wiring
    model = model_key if model_key in GJ_BY_MODEL else DEFAULT_MODEL
    gj_full = GJ_BY_MODEL[model]
    year_features = [f for f in gj_full["features"] if f["properties"]["year"] == year]
    year_gj = {"type": "FeatureCollection", "features": year_features}

    trigger_id = ctx.triggered_id if ctx.triggered_id else None
    focus_state = None
    if trigger_id in ("zip-submit", "zip-input"):
        focus_state = _zip_focus_view_state(zip_value)
    spec = make_deck_spec(year_gj, year=year, color_metric=metric, focus_view_state=focus_state)
    spec["mapboxApiAccessToken"] = settings.mapbox_token or ""
    json_spec = json.dumps(spec)
    iframe_template = """
    <!doctype html>
    <html><head>
      <meta charset='utf-8'/>
            <script src='https://unpkg.com/deck.gl@8.9.24/dist.min.js'></script>
            <script src='/assets/maplibre-gl.js'></script>
            <link href='/assets/maplibre-gl.css' rel='stylesheet'/>
      <style>html, body, #container { margin:0; height:100%; }</style>
    </head>
    <body>
      <div id='container'></div>
      <script>
                const spec = __DECK_SPEC__;
                const layerSpec = spec.layers[0];
                const { colorMetric = 'pred', colorStops = [], ...layerProps } = layerSpec;
                const focusViewState = spec.focusViewState || null;
                const defaultStops = [
                    { max: 0.0, color: [0, 0, 0, 0] },
                    { max: 0.4, color: [255, 255, 204, 120] },
                    { max: 0.6, color: [255, 237, 160, 170] },
                    { max: 0.8, color: [254, 178, 76, 210] },
                    { max: 0.9, color: [253, 141, 60, 235] },
                    { max: 1.0, color: [189, 0, 38, 255] }
                ];
                const stops = colorStops.length ? colorStops : defaultStops;
                const formatMetricValue = (val) => {
                    const num = Number(val);
                    if (!Number.isFinite(num)) { return '—'; }
                    if (Math.abs(num) >= 1000) { return num.toFixed(0); }
                    if (Math.abs(num) >= 10) { return num.toFixed(1); }
                    return num.toFixed(3);
                };

                const bandColor = (value) => {
                    if (value <= 0) { return [0, 0, 0, 0]; }
                    for (let i = 0; i < stops.length; i += 1) {
                        if (value <= stops[i].max) {
                            return stops[i].color;
                        }
                    }
                    return stops[stops.length - 1].color;
                };
                const layer = new deck.GeoJsonLayer({
                    ...layerProps,
                    getFillColor: feature => {
                        const value = feature?.properties?.[colorMetric] ?? 0;
                        const clamped = Math.max(0, Math.min(1, Number(value)));
                        return bandColor(clamped);
                    }
                });
                if (window.maplibregl) {
                    window.mapboxgl = window.maplibregl;
                }
                if (window.mapboxgl) {
                    mapboxgl.accessToken = spec.mapboxApiAccessToken;
                }
                const savedViewState = focusViewState || (window.parent && window.parent.__fhf_viewState) || null;
                const deckgl = new deck.DeckGL({
                    container: 'container',
                    mapboxApiAccessToken: spec.mapboxApiAccessToken,
                    mapStyle: spec.mapStyle,
                    initialViewState: savedViewState || spec.initialViewState,
                    controller: spec.controller,
                    layers: [layer],
                    onViewStateChange: ({ viewState }) => {
                        if (window.parent) {
                            window.parent.__fhf_viewState = viewState;
                        }
                    },
                    getTooltip: ({ object }) => {
                        if (!object) { return null; }
                        const props = object.properties || {};
                        const metricValue = formatMetricValue(props[colorMetric]);
                        const metricLabel = colorMetric.toUpperCase();
                        const zipText = props.zip_label || props.zip || '';
                        const zipLine = zipText ? `ZIP: ${zipText}<br/>` : '';
                        return {
                            html: `<div><strong>Hex ${props.hex ?? ''}</strong><br/>${zipLine}${metricLabel}: ${metricValue}<br/>Year: ${props.year ?? '—'}</div>`,
                            style: {
                                backgroundColor: 'rgba(255,255,255,0.92)',
                                color: '#0f172a',
                                fontSize: '13px',
                                padding: '6px 8px',
                                borderRadius: '6px',
                                boxShadow: '0 4px 12px rgba(15,23,42,0.18)'
                            }
                        };
                    }
                });
      </script>
    </body></html>
    """
    iframe_html = iframe_template.replace("__DECK_SPEC__", json_spec)
    iframe = html.Iframe(srcDoc=iframe_html, style={"width": "100%", "height": "720px", "border": "none"})
    return iframe


@app.callback(
    Output("info-accordion", "active_item"),
    Input("app-location", "hash"),
)
def open_section_from_hash(hash_value):
    mapping = {
        "#read-map": "read-map",
        "#method": "method",
        "#sources": "sources",
        "#limits": "limits",
        "#technical-details": "technical-details",
    }
    return mapping.get(hash_value)
