# src/app/server.py
import os, json, re
from typing import Dict
import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree

import dash
from dash import html, dcc, Output, Input, State, ctx
import dash_bootstrap_components as dbc  # <-- this import was missing

from ..config import settings
from ..utils.logging import get_logger

# Files produced by the pipeline
PRED_PATH = os.path.join(settings.processed_dir, "predictions_2026_2029.csv")
HEX_GJ_PATH = os.path.join(settings.processed_dir, "hexes.geojson")
MODZCTA_PATH = os.path.join(settings.external_dir, "modzcta.geojson")

logger = get_logger()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
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
        "zoom": 12,
        "pitch": 0,
        "bearing": 0,
        "transitionDuration": 600,
    }

def build_geojson_blob():
    if not (os.path.exists(PRED_PATH) and os.path.exists(HEX_GJ_PATH)):
        # Friendly message if user forgot to run training/bootstrap
        raise FileNotFoundError(
            "Missing predictions or hexes. Run:\n"
            "  python scripts/bootstrap_data.py\n"
            "  python scripts/train_baseline.py\n"
            "to generate data/processed/ files."
        )
    preds = pd.read_csv(PRED_PATH)
    with open(HEX_GJ_PATH, "r") as f:
        gj = json.load(f)

    props_by_hex_year = {
        (r.hex, int(r.year)): {"pred": float(r.pred), "lo": float(r.lo), "hi": float(r.hi)}
        for r in preds.itertuples()
    }
    feat_out = []
    zip_props_by_hex = {}
    for feat in gj["features"]:
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

    for feat in gj["features"]:
        hx = feat["properties"]["hex"]
        for year in sorted(preds["year"].unique()):
            p = props_by_hex_year.get((hx, int(year)))
            if p is None:
                continue
            feat_out.append({
                "type": "Feature",
                "geometry": feat["geometry"],
                "properties": {
                    "hex": hx,
                    "year": int(year),
                    "zip": zip_props_by_hex[hx].get("zip"),
                    "zip_label": zip_props_by_hex[hx].get("zip_label"),
                    **p,
                },
            })
    return {"type": "FeatureCollection", "features": feat_out}

def make_deck_spec(geojson: Dict, year: int, color_metric: str = "pred", focus_view_state: dict | None = None):
    # Define discrete bands so 0 stays transparent and quartiles deepen progressively
    brewer_stops = [
        {"max": 0.0, "color": [0, 0, 0, 0]},             # zero ⇒ transparent
        {"max": 0.4, "color": [255, 255, 204, 120]},     # pale yellow (low >0)
        {"max": 0.6, "color": [255, 237, 160, 170]},     # light yellow (medium)
        {"max": 0.8, "color": [254, 178, 76, 210]},      # orange (elevated)
        {"max": 0.9, "color": [253, 141, 60, 235]},      # orange-red (high)
        {"max": 1.0, "color": [189, 0, 38, 255]},        # dark red (extreme)
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

# Precompute GeoJSON blob and embed via data URL
GJ = build_geojson_blob()

ANALYSIS_COPY = html.Div(className="card", children=[
    html.H5("What data feeds this map right now?"),
    html.P("This deployment includes the freshest datasets we finished ingesting on December 7, 2025:"),
    html.Ul([
        html.Li("`data/raw/311.json` — service requests already geocoded to lat/lon."),
        html.Li("`data/interim/hpd_hex_counts.json` — ≈6k pre-aggregated HPD complaint totals per H3 hex, produced by `scripts/aggregate_hpd_complaints.py`."),
        html.Li("`data/raw/evictions.json` and `data/raw/filed_evictions.json` — executed and filed eviction events for `nevict`/`nfiled`."),
        html.Li("`data/processed/hexes.geojson` — the NYC H3 grid we render and join against."),
        html.Li("`data/processed/predictions_2026_2029.csv` — LightGBM outputs with conformal lower/upper bands.")
    ]),
    html.H6("Pipeline checkpoints"),
    html.P([
        html.B("Ingestion · "),
        "`scripts/bootstrap_data.py` grabs 311, eviction, and MODZCTA samples, then `scripts/aggregate_hpd_complaints.py` streams the large HPD CSV into the lightweight hex counts listed above."
    ]),
    html.P([
        html.B("Feature engineering · "),
        "`src/data/features.py` reads the aggregated HPD counts plus 311/eviction events, maps everything to the hex grid, and clones the table for each forecast year with a modest 3% growth proxy (`*_y` columns)."
    ]),
    html.P([
        html.B("Model + bands · "),
        "`src/models/baselines.py` fits LightGBM on those yearly counts (currently `n311_y`, `nhpd_y`, `nevict_y`, `nfiled_y`), while `src/models/evaluate.py` wraps the predictions with symmetric conformal intervals so each hex gets `pred`, `lo`, and `hi`."
    ]),
    html.P([
        html.B("Interpreting the color · "),
        "Scores are normalized between 0 and 1 inside each year, so a value near 1.0 simply means \"highest relative risk across the sampled hexes in that year\" rather than a literal count of future shelter placements."
    ]),
    html.H6("Source links"),
    html.Ul([
        html.Li(html.A("NYC 311 Service Requests", href="https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9", target="_blank")),
        html.Li(html.A("HPD Complaint Problems", href="https://data.cityofnewyork.us/Housing-Development/HPD-Complaint-Problems/uwyv-629c", target="_blank")),
        html.Li(html.A("Residential Evictions", href="https://data.cityofnewyork.us/City-Government/Eviction-Notices/6z8x-wfk4", target="_blank")),
        html.Li(html.A("MODZCTA Boundaries", href="https://data.cityofnewyork.us/City-Government/Modified-Zip-Code-Tabulation-Areas-MODZCTA-/pri4-ifjk", target="_blank")),
        html.Li(html.A("American Community Survey 5-year", href="https://www.census.gov/data/developers/data-sets/acs-5year.html", target="_blank")),
        html.Li(html.A("H3 Index", href="https://h3geo.org/", target="_blank"))
    ]),
    html.P("Need higher fidelity? Swap the bootstrap inputs for full NYC feeds, rerun `scripts/aggregate_hpd_complaints.py` and `scripts/train_baseline.py`, then redeploy.")
])

LIGHTGBM_COPY = html.Div(className="card", children=[
    html.H5("LightGBM in plain English"),
    html.P("LightGBM is the learning engine behind this map. You can think of it like a group project made of many tiny decision trees:"),
    html.Ul([
        html.Li("Each tree asks a few yes/no questions such as \"Was the HPD count above the city median?\" and assigns a small score."),
        html.Li("Trees are trained one after another; every new tree focuses on the mistakes the previous trees made, so the ensemble steadily improves."),
        html.Li("After about 500 trees, we add up all of their suggestions to get a final risk score for every hex.")
    ]),
    html.P("Because LightGBM only needs the engineered counts (`n311_y`, `nhpd_y`, `nevict_y`, `nfiled_y`), it stays fast enough for this project while still capturing non-linear jumps—like a sudden spike in HPD complaints—without overwhelming non-technical collaborators."),
    html.P("The conformal interval you see (lo/hi) wraps those scores with a \"give or take\" band so you can communicate uncertainty without diving into math.")
])

app.layout = dbc.Container([
    html.H3("The Future of the Unhoused — NYC Forecast Map (2026–2029)"),
    html.Div(className="small", children=[
        "Zoomable choropleth by H3 hex; hover for values. ",
        "Color encodes predicted relative risk (0–1).",
    ]),
    dbc.Row([
        dbc.Col(dbc.InputGroup([
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
        ]), md=5),
        dbc.Col(html.Small("Type a valid NYC ZIP then hit Enter or Go to fly to that area."), md=7)
    ], align="center", className="card"),
    dbc.Row([
        dbc.Col(dcc.Slider(id="year", min=2026, max=2029, value=2026, step=1,
                           marks={y: str(y) for y in [2026, 2027, 2028, 2029]}), md=8),
        dbc.Col(dbc.Select(
            id="metric",
            value="pred",
            options=[
                {"label": "Prediction (μ)", "value": "pred"},
                {"label": "Lower band (lo)", "value": "lo"},
                {"label": "Upper band (hi)", "value": "hi"},
            ],
        ), md=4)
    ], align="center", className="card"),
    html.Div(id="deck-container"),
    ANALYSIS_COPY,
    LIGHTGBM_COPY,
], fluid=True)

@app.callback(
    Output("deck-container", "children"),
    Input("year", "value"),
    Input("metric", "value"),
    Input("zip-submit", "n_clicks"),
    Input("zip-input", "n_submit"),
    State("zip-input", "value"),
)
def update_map(year, metric, zip_clicks, zip_enter, zip_value):  # pragma: no cover - UI wiring
    year_features = [f for f in GJ["features"] if f["properties"]["year"] == year]
    year_gj = {"type": "FeatureCollection", "features": year_features}

    trigger_id = ctx.triggered_id if ctx.triggered_id else None
    focus_state = None
    if trigger_id in ("zip-submit", "zip-input"):
        focus_state = _zip_focus_view_state(zip_value)
    spec = make_deck_spec(year_gj, year=year, color_metric=metric, focus_view_state=focus_state)
    spec["mapboxApiAccessToken"] = settings.mapbox_token or ""
    html_spec = json.dumps(spec)
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
                if (focusViewState && window.parent) {
                    window.parent.__fhf_viewState = focusViewState;
                }
      </script>
    </body></html>
    """
    iframe_html = iframe_template.replace("__DECK_SPEC__", html_spec)
    iframe = html.Iframe(
        srcDoc=iframe_html,
        style={"width": "100%", "height": "650px", "border": "1px solid #e5e7eb", "borderRadius": "10px"}
    )

    return iframe

if __name__ == "__main__":
    app.run_server(host=settings.app_host, port=settings.app_port, debug=True)
