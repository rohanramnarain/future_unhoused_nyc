import os

app.layout = dbc.Container([
    html.H3("The Future of the Unhoused — NYC Forecast Map (2026–2029)"),
    html.Div(className="small", children=[
        "Zoomable choropleth by H3 hex; hover for values. ",
        "Color encodes predicted relative risk (0–1).",
    ]),
    dbc.Row([
        dbc.Col(dcc.Slider(id="year", min=2026, max=2029, value=2026, step=1, marks={y:str(y) for y in [2026, 2027, 2028, 2029]}), md=8),
        dbc.Col(dcc.Dropdown(id="metric", value="pred", options=[
            {"label": "Prediction (μ)", "value": "pred"},
            {"label": "Lower band (lo)", "value": "lo"},
            {"label": "Upper band (hi)", "value": "hi"},
        ]), md=4)
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
    # Embed deck.gl via Iframe using prebuilt JSON spec
    spec = make_deck_spec(GJ_DATA_URL, year=year, color_metric=metric)
    spec["mapboxApiAccessToken"] = settings.mapbox_token
    html_spec = json.dumps(spec)
    iframe = html.Iframe(srcDoc=f"""
<!doctype html>
<html><head>
<meta charset='utf-8'/>
<script src='https://unpkg.com/deck.gl@8.9.24/dist.min.js'></script>
<script src='https://api.mapbox.com/mapbox-gl-js/v2.16.1/mapbox-gl.js'></script>
<link href='https://api.mapbox.com/mapbox-gl-js/v2.16.1/mapbox-gl.css' rel='stylesheet' />
<style>html, body, #container {{ margin:0; height:100%; }}</style>
</head>
<body>
<div id='container'></div>
<script>
const spec = {html_spec};
const deckgl = new deck.DeckGL({
    container: 'container',
    mapboxApiAccessToken: spec.mapboxApiAccessToken,
    mapStyle: spec.mapStyle,
    initialViewState: spec.initialViewState,
    controller: spec.controller,
    layers: spec.layers.map(l => new deck[l['@@type']](l))
});
</script>
</body></html>
""", style={"width": "100%", "height": "650px", "border": "1px solid #e5e7eb", "borderRadius": "10px"})

    # Simple scorecard: show distribution for selected metric
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