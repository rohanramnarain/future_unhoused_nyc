import json


# Deck.gl choropleth layer factory


def make_choropleth_layer(data_url: str, year: int, color_expr: str = "pred"):
    # Using dataTransform to filter by year
    return {
        "@@type": "GeoJsonLayer",
        "id": "hex-choropleth",
        "data": data_url,
        "pickable": True,
        "stroked": False,
        "filled": True,
        "opacity": 0.6,
        "getFillColor": "[255 * properties.%s, 120, 180]" % color_expr,
        "getLineColor": [0, 0, 0],
        "lineWidthMinPixels": 0.5,
        "dataTransform": f"data.features = data.features.filter(f=> f.properties.year === {year}); return data;",
    }