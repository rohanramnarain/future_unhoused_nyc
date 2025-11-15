# src/app/server.py
import os, json
from typing import Dict
import pandas as pd

import dash
from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc  # <-- this import was missing

from ..config import settings

# Files produced by the pipeline
PRED_PATH = os.path.join(settings.processed_dir, "predictions_2026_2029.csv")
HEX_GJ_PATH = os.path.join(settings.processed_dir, "hexes.geojson")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
server = app.server

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
    for feat in gj["features"]:
        hx = feat["properties"]["hex"]
        for year in sorted(preds["year"].unique()):
            p = props_by_hex_year.get((hx, int(year)))
            if p is None:
                continue
            feat_out.append({
                "type": "Feature",
                "geometry": feat["geometry"],
                "properties": {"hex": hx, "year": int(year), **p},
            })
    return {"type": "FeatureCollection", "features": feat_out}

def make_deck_spec(geojson: Dict, year: int, color_metric: str = "pred"):
    # Stops roughly follow ColorBrewer YlGnBu (with alpha), gives smoother gradients client-side
    brewer_stops = [
        {"value": 0.0, "color": [255, 255, 255, 20]},
        {"value": 0.2, "color": [237, 248, 251, 45]},
        {"value": 0.4, "color": [204, 235, 197, 90]},
        {"value": 0.6, "color": [168, 221, 181, 140]},
        {"value": 0.8, "color": [123, 204, 196, 185]},
        {"value": 1.0, "color": [43, 140, 190, 220]},
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
                    { value: 0, color: [255, 255, 255, 20] },
                    { value: 1, color: [43, 140, 190, 220] }
                ];
                const stops = colorStops.length ? colorStops : defaultStops;
                const formatMetricValue = (val) => {
                    const num = Number(val);
                    if (!Number.isFinite(num)) { return '—'; }
                    if (Math.abs(num) >= 1000) { return num.toFixed(0); }
                    if (Math.abs(num) >= 10) { return num.toFixed(1); }
                    return num.toFixed(3);
                };

                const interpolateColor = (value) => {
                    if (value <= stops[0].value) { return stops[0].color; }
                    if (value >= stops[stops.length - 1].value) { return stops[stops.length - 1].color; }
                    for (let i = 0; i < stops.length - 1; i += 1) {
                        const left = stops[i];
                        const right = stops[i + 1];
                        if (value >= left.value && value <= right.value) {
                            const span = right.value - left.value || 1;
                            const t = (value - left.value) / span;
                            return left.color.map((c, idx) => Math.round(c + (right.color[idx] - c) * t));
                        }
                    }
                    return stops[0].color;
                };
                const layer = new deck.GeoJsonLayer({
                    ...layerProps,
                    getFillColor: feature => {
                        const value = feature?.properties?.[colorMetric] ?? 0;
                        const clamped = Math.max(0, Math.min(1, Number(value)));
                        return interpolateColor(clamped);
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
                        return {
                            html: `<div><strong>Hex ${props.hex ?? ''}</strong><br/>${metricLabel}: ${metricValue}<br/>Year: ${props.year ?? '—'}</div>`,
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
