# src/app/server.py
import os, json
from typing import Dict
import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree

import dash
from dash import html, dcc, Output, Input
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


def _build_zip_lookup():
    if not os.path.exists(MODZCTA_PATH):
        logger.warning("ZIP boundary file missing at %s; tooltips will omit ZIP codes.", MODZCTA_PATH)
        return None
    try:
        with open(MODZCTA_PATH, "r") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load MODZCTA boundaries (%s); tooltips will omit ZIP codes.", exc)
        return None

    geoms, attrs = [], []
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
        if not zip_code:
            continue
        geoms.append(poly)
        attrs.append({
            "zip": zip_code,
            "zip_label": (props.get("label") or "").strip(),
        })

    if not geoms:
        logger.warning("Loaded MODZCTA boundaries but none were usable; tooltips will omit ZIP codes.")
        return None

    tree = STRtree(geoms)

    def lookup(point):
        for idx in tree.query(point):
            geom = geoms[idx]
            if geom.covers(point):
                return attrs[idx]
        return None

    logger.info("Loaded %s MODZCTA polygons for ZIP enrichment", len(geoms))
    return lookup


ZIP_LOOKUP = _build_zip_lookup()

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

def make_deck_spec(geojson: Dict, year: int, color_metric: str = "pred"):
    # Define discrete bands so 0 stays transparent and quartiles deepen progressively
    brewer_stops = [
        {"max": 0.0, "color": [0, 0, 0, 0]},            # exactly zero ⇒ fully transparent
        {"max": 0.25, "color": [233, 246, 248, 80]},     # >0–0.25 lightest
        {"max": 0.5, "color": [178, 226, 226, 140]},    # 0.25–0.5 medium-light
        {"max": 0.75, "color": [102, 194, 164, 190]},   # 0.5–0.75 medium-dark
        {"max": 1.0, "color": [35, 132, 67, 235]},      # 0.75–1 darkest
    ]
    return {
        "initialViewState": {"latitude": 40.7128, "longitude": -74.0060, "zoom": 9},
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

# Precompute GeoJSON blob and embed via data URL
GJ = build_geojson_blob()

app.layout = dbc.Container([
    html.H3("The Future of the Unhoused — NYC Forecast Map (2026–2029)"),
    html.Div(className="small", children=[
        "Zoomable choropleth by H3 hex; hover for values. ",
        "Color encodes predicted relative risk (0–1).",
    ]),
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
    html.Div(className="card", children=[
        html.B("Equity scorecard (current view)"),
        html.Div(id="scorecard")
    ])
], fluid=True)

@app.callback(
    Output("deck-container", "children"),
    Output("scorecard", "children"),
    Input("year", "value"),
    Input("metric", "value")
)
def update_map(year, metric):
    year_features = [f for f in GJ["features"] if f["properties"]["year"] == year]
    year_gj = {"type": "FeatureCollection", "features": year_features}

    spec = make_deck_spec(year_gj, year=year, color_metric=metric)
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
                const defaultStops = [
                    { max: 0.0, color: [0, 0, 0, 0] },
                    { max: 0.25, color: [233, 246, 248, 80] },
                    { max: 0.5, color: [178, 226, 226, 140] },
                    { max: 0.75, color: [102, 194, 164, 190] },
                    { max: 1.0, color: [35, 132, 67, 235] }
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
                const deckgl = new deck.DeckGL({
                    container: 'container',
                    mapboxApiAccessToken: spec.mapboxApiAccessToken,
                    mapStyle: spec.mapStyle,
                    initialViewState: spec.initialViewState,
                    controller: spec.controller,
                    layers: [layer],
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
    iframe_html = iframe_template.replace("__DECK_SPEC__", html_spec)
    iframe = html.Iframe(
        srcDoc=iframe_html,
        style={"width": "100%", "height": "650px", "border": "1px solid #e5e7eb", "borderRadius": "10px"}
    )

    import numpy as np
    vals = [f["properties"][metric] for f in GJ["features"] if f["properties"]["year"] == year]
    if vals:
        v = np.array(vals)
        txt = f"Median: {np.median(v):.2f} | 80th pct: {np.percentile(v,80):.2f} | Max: {v.max():.2f}"
    else:
        txt = "No data"

    return iframe, txt

if __name__ == "__main__":
    app.run_server(host=settings.app_host, port=settings.app_port, debug=True)
