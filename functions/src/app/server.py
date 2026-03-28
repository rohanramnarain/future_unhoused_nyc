# src/app/server.py
import os, json, re, glob
from typing import Dict, List
import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree

import dash
from dash import html, dcc, Output, Input, State, ctx
import dash_bootstrap_components as dbc
from flask import Response

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

COLOR_STOPS = [
    {"max": 0.0, "color": [0, 0, 0, 0]},
    {"max": 0.4, "color": [255, 255, 204, 120]},
    {"max": 0.6, "color": [255, 237, 160, 170]},
    {"max": 0.8, "color": [254, 178, 76, 210]},
    {"max": 0.9, "color": [253, 141, 60, 235]},
    {"max": 1.0, "color": [189, 0, 38, 255]},
]

METRIC_ORDER = [
    "pred",
    "lo",
    "hi",
    "n311_y",
    "nhpd_y",
    "nevict_y",
    "nfiled_y",
    "n_dcp_units",
    "n_dcp_aff_units",
    "n_dcp_expiring5yr",
    "n_dcp_expired",
    "dcp_status_median",
]

METRIC_LABELS = {
    "pred": "Prediction (mu)",
    "lo": "Lower band (lo)",
    "hi": "Upper band (hi)",
    "n311_y": "311 requests",
    "nhpd_y": "HPD complaints",
    "nevict_y": "Executed evictions",
    "nfiled_y": "Filed evictions",
    "n_dcp_units": "DCP units",
    "n_dcp_aff_units": "DCP affordable units",
    "n_dcp_expiring5yr": "DCP units expiring in 5 years",
    "n_dcp_expired": "DCP expired units",
    "dcp_status_median": "DCP status median",
}

RANK_METRICS = {"pred", "lo", "hi"}

METRIC_OPTIONS = [
    {"label": METRIC_LABELS[key], "value": key}
    for key in METRIC_ORDER
]

logger = get_logger()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LITERA],
    assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    assets_url_path="/assets",
    title="The Future of the Unhoused",
)
server = app.server
ODW_INTRO_IMAGE = "/assets/odwintro.png"
ODW_OUTRO_IMAGE = "/assets/odwoutro.png"
SITE_FAVICON = "/assets/favicon.svg"

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg" />
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

MODEL_PAGE_META = {
    "lgbm": {
        "title": "LightGBM",
        "diagram": "/assets/model_diagram_lgbm.svg?v=4",
        "summary": (
            "LightGBM learns nonlinear risk patterns from our nine engineered housing-pressure "
            "signals and builds trees sequentially to correct residual errors."
        ),
        "details": [
            "Inputs include 311 requests, HPD complaints, executed/filed evictions, and DCP housing program counts mapped to each H3 hex.",
            "The model is trained on historical outcomes and then projected across 2026-2029 feature tables.",
            "Outputs are scaled to within-year relative risk ranks and paired with conformal lower/upper bands.",
        ],
        "story_steps": [
            {
                "title": "Feature intake",
                "copy": "Each hex starts with engineered signals from 311, HPD complaints, evictions, and DCP housing attributes.",
                "x": 16,
                "y": 18,
            },
            {
                "title": "First boosted split",
                "copy": "An initial shallow tree partitions major risk structure, often on complaint and eviction intensity thresholds.",
                "x": 36,
                "y": 30,
            },
            {
                "title": "Residual correction",
                "copy": "New trees focus on remaining error pockets, refining where the first passes under- or over-estimated pressure.",
                "x": 58,
                "y": 43,
            },
            {
                "title": "Ranked output",
                "copy": "Final scores are normalized per year to relative 0-1 risk rank, then wrapped with conformal low/high bands.",
                "x": 79,
                "y": 71,
            },
        ],
    },
    "rf": {
        "title": "Random Forest",
        "diagram": "/assets/model_diagram_rf.svg?v=4",
        "summary": (
            "Random Forest averages many independently trained decision trees on the same nine "
            "engineered features to produce stable hex-level risk estimates."
        ),
        "details": [
            "Each tree sees a bootstrap sample of the training set and a randomized subset of predictors at each split.",
            "Averaging across trees reduces sensitivity to local noise in complaints or eviction spikes.",
            "Predictions are converted to relative yearly ranks and wrapped with conformal uncertainty bands.",
        ],
        "story_steps": [
            {
                "title": "Bootstrap sampling",
                "copy": "Each tree trains on a slightly different sample of hex-year records, creating diverse decision views.",
                "x": 18,
                "y": 20,
            },
            {
                "title": "Randomized feature splits",
                "copy": "At each fork, the tree tests a random subset of predictors, reducing over-reliance on any single signal.",
                "x": 43,
                "y": 36,
            },
            {
                "title": "Independent tree votes",
                "copy": "Hundreds of trees produce parallel predictions, from conservative to aggressive risk estimates.",
                "x": 64,
                "y": 52,
            },
            {
                "title": "Ensemble average",
                "copy": "The model averages tree outputs into one stable score and then converts it to annual relative rank.",
                "x": 82,
                "y": 74,
            },
        ],
    },
    "xgb": {
        "title": "XGBoost",
        "diagram": "/assets/model_diagram_xgb.svg?v=4",
        "summary": (
            "XGBoost fits regularized boosting trees level-by-level, refining error pockets in our "
            "hex-level housing instability feature space."
        ),
        "details": [
            "Training uses the same core nine predictors to keep model comparisons consistent.",
            "Regularization and step-wise boosting help balance fit quality with stability.",
            "Final outputs are transformed into annual relative risk ranks with conformal lo/hi intervals.",
        ],
        "story_steps": [
            {
                "title": "Structured baseline split",
                "copy": "The first boosted tree establishes broad separation with level-wise growth across key housing pressure signals.",
                "x": 17,
                "y": 18,
            },
            {
                "title": "Regularized expansion",
                "copy": "Subsequent trees add constrained depth and penalties, preventing unstable jumps from noisy local spikes.",
                "x": 40,
                "y": 34,
            },
            {
                "title": "Stage-wise error reduction",
                "copy": "Each round corrects residual error while keeping updates controlled through learning-rate steps.",
                "x": 61,
                "y": 50,
            },
            {
                "title": "Final risk surface",
                "copy": "Predictions are transformed into year-specific rank values and uncertainty intervals for map comparison.",
                "x": 81,
                "y": 72,
            },
        ],
    },
}


@server.route("/odwintro")
def odw_intro_page():  # pragma: no cover - simple route for static intro slide
        page = f"""
<!doctype html>
<html>
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>ODW Intro</title>
        <link rel=\"icon\" type=\"image/svg+xml\" href=\"{SITE_FAVICON}\" />
        <style>
            html, body {{
                margin: 0;
                width: 100%;
                height: 100%;
                background: #0f3f8f;
            }}
            a {{
                display: flex;
                width: 100%;
                height: 100%;
                align-items: center;
                justify-content: center;
            }}
            img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                display: block;
            }}
        </style>
    </head>
    <body>
        <a href=\"/\" aria-label=\"Open main map\">
            <img src=\"{ODW_INTRO_IMAGE}\" alt=\"Open Data Week intro slide\" />
        </a>
    </body>
</html>
"""
        return Response(page, mimetype="text/html")


@server.route("/odwoutro")
def odw_outro_page():  # pragma: no cover - simple route for static outro slide
        page = f"""
<!doctype html>
<html>
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>ODW Outro</title>
        <link rel=\"icon\" type=\"image/svg+xml\" href=\"{SITE_FAVICON}\" />
        <style>
            html, body {{
                margin: 0;
                width: 100%;
                height: 100%;
                background: #0f3f8f;
            }}
            .frame {{
                display: flex;
                width: 100%;
                height: 100%;
                align-items: center;
                justify-content: center;
            }}
            img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class=\"frame\">
            <img src=\"{ODW_OUTRO_IMAGE}\" alt=\"Open Data Week outro slide\" />
        </div>
    </body>
</html>
"""
        return Response(page, mimetype="text/html")


@server.route("/models/<model_key>")
def model_detail_page(model_key: str):  # pragma: no cover - static explainer route
    model = MODEL_PAGE_META.get(model_key)
    if not model:
        return Response("Model page not found", status=404, mimetype="text/plain")

    bullets = "".join(f"<li>{item}</li>" for item in model["details"])
    steps = model.get("story_steps", [])
    callouts = "".join(
        (
            f"<button class='model-callout' type='button' data-step='{idx}' aria-label='Jump to step {idx + 1}'>"
            f"<span>{idx + 1}</span></button>"
        )
        for idx, step in enumerate(steps)
    )
    story_cards = "".join(
        (
            f"<article class='model-step-card' id='model-step-{idx}' data-step='{idx}'>"
            f"<div class='model-step-kicker'>Step {idx + 1}</div>"
            f"<h2>{step['title']}</h2>"
            f"<p>{step['copy']}</p>"
            "</article>"
        )
        for idx, step in enumerate(steps)
    )
    page = f"""
<!doctype html>
<html>
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>{model['title']} | The Future of the Unhoused</title>
        <link rel=\"icon\" type=\"image/svg+xml\" href=\"{SITE_FAVICON}\" />
        <link href=\"/assets/styles.css\" rel=\"stylesheet\" />
        <style>
            .model-page-wrap {{
                max-width: 1800px;
                margin: 0 auto;
                padding: 1.4rem 1rem 2rem;
            }}
            .model-story-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 1rem;
                align-items: start;
            }}
            .model-page-kicker {{
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: #0f8b7a;
                margin-bottom: 0.45rem;
            }}
            .model-page-title {{
                margin: 0;
                font-family: \"Space Grotesk\", \"Avenir Next Condensed\", sans-serif;
                font-size: clamp(1.85rem, 4.1vw, 3.35rem);
                letter-spacing: -0.02em;
                line-height: 1.06;
            }}
            .model-page-copy {{
                margin-top: 0.75rem;
                max-width: 80ch;
            }}
            .model-page-copy ul {{
                margin: 0.6rem 0 0;
                padding-left: 1.2rem;
            }}
            .model-page-back {{
                display: inline-block;
                margin-bottom: 0.8rem;
                font-weight: 700;
            }}
            .model-page-diagram-shell {{
                margin-top: 1rem;
                position: sticky;
                top: 10px;
                border: 1px solid rgba(168, 143, 110, 0.32);
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.92);
                padding: 0.8rem;
                box-shadow: 0 24px 50px rgba(52, 44, 33, 0.14);
                overflow: hidden;
            }}
            .model-diagram-layer {{
                position: relative;
                width: 100%;
                height: min(78vh, 820px);
                min-height: 520px;
                border-radius: 14px;
                background: linear-gradient(180deg, #fbfbf8, #f4f3ef);
                overflow: hidden;
            }}
            .model-tree-canvas {{
                width: 100%;
                height: 100%;
                display: block;
            }}
            .model-tree-tooltip {{
                position: absolute;
                z-index: 4;
                pointer-events: none;
                max-width: min(320px, 70vw);
                padding: 0.38rem 0.52rem;
                border-radius: 8px;
                background: rgba(18, 24, 20, 0.9);
                color: #f4f7f2;
                font-size: 0.78rem;
                line-height: 1.3;
                box-shadow: 0 8px 20px rgba(12, 15, 12, 0.25);
                opacity: 0;
                transform: translateY(4px);
                transition: opacity 0.12s ease, transform 0.12s ease;
            }}
            .model-tree-tooltip.visible {{
                opacity: 1;
                transform: translateY(0);
            }}
            .model-callout-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                justify-content: flex-end;
                padding: 0.15rem 0.05rem 0.55rem;
            }}
            .model-callout {{
                position: relative;
                width: 34px;
                height: 34px;
                border-radius: 999px;
                border: 2px solid rgba(15, 139, 122, 0.94);
                background: rgba(255, 255, 255, 0.92);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: 800;
                color: #0f8b7a;
                box-shadow: 0 7px 16px rgba(24, 28, 23, 0.15);
                transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease, color 0.2s ease;
            }}
            .model-callout:hover {{
                transform: scale(1.05);
            }}
            .model-callout.active {{
                background: #0f8b7a;
                color: #ffffff;
                box-shadow: 0 0 0 6px rgba(15, 139, 122, 0.18);
            }}
            .model-story-rail {{
                margin-top: 0.2rem;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 0.8rem;
            }}
            .model-step-card {{
                border: 1px solid rgba(168, 143, 110, 0.32);
                background: rgba(255, 255, 255, 0.9);
                border-radius: 14px;
                padding: 0.9rem 1rem;
                box-shadow: 0 8px 18px rgba(36, 42, 34, 0.08);
                transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
            }}
            .model-step-card.active {{
                border-color: rgba(15, 139, 122, 0.72);
                box-shadow: 0 12px 24px rgba(15, 139, 122, 0.18);
                transform: translateY(-2px);
            }}
            .model-step-card h2 {{
                margin: 0 0 0.38rem;
                font-size: 1.02rem;
                font-weight: 800;
                color: #233028;
            }}
            .model-step-card p {{
                margin: 0;
                color: #4f5a50;
                font-size: 0.95rem;
                line-height: 1.45;
            }}
            .model-step-kicker {{
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: #0f8b7a;
                margin-bottom: 0.26rem;
            }}
            @media (max-width: 768px) {{
                .model-page-wrap {{
                    padding: 1rem 0.9rem 1.4rem;
                }}
                .model-page-diagram-shell {{
                    position: static;
                    padding: 0.35rem;
                }}
                .model-diagram-layer {{
                    height: 58vh;
                    min-height: 420px;
                }}
                .model-story-rail {{
                    grid-template-columns: 1fr;
                }}
                .model-callout {{
                    width: 30px;
                    height: 30px;
                    font-size: 0.8rem;
                }}
            }}
        </style>
    </head>
    <body>
        <main class=\"model-page-wrap\">
            <a class=\"model-page-back\" href=\"/#method\">Back to model overview</a>
            <div class=\"model-page-kicker\">Model explainer</div>
            <h1 class=\"model-page-title\">{model['title']}</h1>
            <div class=\"model-page-copy fhf-prose\">
                <p>{model['summary']}</p>
                <ul>{bullets}</ul>
            </div>
            <section class=\"model-story-grid\" aria-label=\"{model['title']} model walkthrough\">
                <div class=\"model-page-diagram-shell\">
                    <div class=\"model-callout-row\" aria-label=\"Story step shortcuts\">{callouts}</div>
                    <div class=\"model-diagram-layer\" aria-label=\"{model['title']} model diagram\">
                        <canvas id=\"tree-canvas\" class=\"model-tree-canvas\" aria-label=\"Animated decision tree\"></canvas>
                        <div id=\"tree-tooltip\" class=\"model-tree-tooltip\" role=\"status\" aria-live=\"polite\"></div>
                    </div>
                </div>
                <div class=\"model-story-rail\">{story_cards}</div>
            </section>
        </main>
        <script>
            (function () {{
                const modelKey = "{model_key}";
                const canvas = document.getElementById('tree-canvas');
                const layer = canvas ? canvas.parentElement : null;
                const cards = Array.from(document.querySelectorAll('.model-step-card'));
                const pins = Array.from(document.querySelectorAll('.model-callout'));
                if (!cards.length || !pins.length || !canvas || !layer) return;

                const palettes = {{
                    lgbm: {{ bg: '#f4f3ef', link: '#c2c3bf', blue: '#0f4f95', green: '#3f9f2f' }},
                    rf: {{ bg: '#f5f4ef', link: '#b9bbb8', blue: '#124c8d', green: '#4b9d34' }},
                    xgb: {{ bg: '#f3f2ee', link: '#b8b9b7', blue: '#174f91', green: '#4a9d35' }},
                }};
                const profiles = {{
                    lgbm: {{ depth: 7, prune: 0.18, spawnRate: 0.034, speed: 0.1, greenBias: 0.49 }},
                    rf: {{ depth: 6, prune: 0.26, spawnRate: 0.03, speed: 0.085, greenBias: 0.52 }},
                    xgb: {{ depth: 7, prune: 0.21, spawnRate: 0.036, speed: 0.095, greenBias: 0.51 }},
                }};
                const featureMap = {{
                    lgbm: ['n311_y', 'nhpd_y', 'nevict_y', 'n_dcp_aff_units', 'n_dcp_expired', 'nfiled_y'],
                    rf: ['n311_y', 'nevict_y', 'nhpd_y', 'n_dcp_units', 'dcp_status_median', 'nfiled_y'],
                    xgb: ['nfiled_y', 'nhpd_y', 'n311_y', 'n_dcp_expiring5yr', 'dcp_status_median', 'nevict_y'],
                }};

                const palette = palettes[modelKey] || palettes.lgbm;
                const profile = profiles[modelKey] || profiles.lgbm;
                const features = featureMap[modelKey] || featureMap.lgbm;
                const splitRanges = {{
                    n311_y: {{ min: 0.8, max: 7.5, d: 1 }},
                    nhpd_y: {{ min: 0.5, max: 6.0, d: 1 }},
                    nevict_y: {{ min: 0.2, max: 4.5, d: 1 }},
                    nfiled_y: {{ min: 0.5, max: 8.0, d: 1 }},
                    n_dcp_units: {{ min: 25, max: 900, d: 0 }},
                    n_dcp_aff_units: {{ min: 10, max: 500, d: 0 }},
                    n_dcp_expiring5yr: {{ min: 5, max: 250, d: 0 }},
                    n_dcp_expired: {{ min: 5, max: 220, d: 0 }},
                    dcp_status_median: {{ min: 1.0, max: 4.0, d: 1 }},
                }};

                function makeSplitRule(feature) {{
                    const spec = splitRanges[feature] || {{ min: 0.5, max: 5.0, d: 1 }};
                    const raw = spec.min + rand() * (spec.max - spec.min);
                    const threshold = Number(raw.toFixed(spec.d));
                    return {{
                        feature,
                        threshold,
                        text: feature + ' > ' + threshold.toFixed(spec.d),
                    }};
                }}

                function makeRng(seedStr) {{
                    let h = 2166136261;
                    for (let i = 0; i < seedStr.length; i += 1) {{
                        h ^= seedStr.charCodeAt(i);
                        h = Math.imul(h, 16777619);
                    }}
                    return function () {{
                        h += h << 13;
                        h ^= h >>> 7;
                        h += h << 3;
                        h ^= h >>> 17;
                        h += h << 5;
                        return (h >>> 0) / 4294967296;
                    }};
                }}

                const rand = makeRng(modelKey + '-tree');
                let nodeId = 0;
                let root = null;
                let nodes = [];
                let leaves = [];
                let links = [];
                let leafLoad = new Map();
                let particles = [];
                let lastTime = 0;
                let spawnCarry = 0;
                let ctx = null;
                let hoverTargets = [];
                const inlineDepthLimit = 2;
                const tooltip = document.getElementById('tree-tooltip');

                function pushHoverTarget(x, y, w, h, text) {{
                    hoverTargets.push({{ x, y, w, h, text }});
                }}

                function findHoverTarget(x, y) {{
                    for (let i = hoverTargets.length - 1; i >= 0; i -= 1) {{
                        const t = hoverTargets[i];
                        if (x >= t.x - t.w / 2 && x <= t.x + t.w / 2 && y >= t.y - t.h / 2 && y <= t.y + t.h / 2) {{
                            return t;
                        }}
                    }}
                    return null;
                }}

                function showTooltip(target, x, y) {{
                    if (!tooltip || !target) return;
                    tooltip.textContent = target.text;
                    tooltip.classList.add('visible');
                    const pad = 10;
                    const maxLeft = Math.max(pad, layer.clientWidth - 260);
                    const left = Math.min(maxLeft, Math.max(pad, x + 14));
                    const top = Math.max(pad, y + 12);
                    tooltip.style.left = left + 'px';
                    tooltip.style.top = top + 'px';
                }}

                function hideTooltip() {{
                    if (!tooltip) return;
                    tooltip.classList.remove('visible');
                }}

                function newNode(depth) {{
                    return {{ id: nodeId++, depth, children: [], parent: null, x: 0, y: 0, label: '' }};
                }}

                function buildTree(depth, maxDepth) {{
                    const node = newNode(depth);
                    if (depth >= maxDepth) return node;
                    const mustBranch = depth < 2;
                    const shouldBranch = mustBranch || rand() > profile.prune;
                    if (!shouldBranch) return node;
                    const makeSingle = depth > 2 && rand() < 0.2;
                    const childCount = makeSingle ? 1 : 2;
                    for (let i = 0; i < childCount; i += 1) {{
                        const child = buildTree(depth + 1, maxDepth);
                        child.parent = node;
                        node.children.push(child);
                    }}
                    return node;
                }}

                function walk(node, out) {{
                    out.push(node);
                    node.children.forEach((c) => walk(c, out));
                    return out;
                }}

                function assignLayout(width, height) {{
                    const marginX = Math.max(40, width * 0.05);
                    const marginTop = Math.max(42, height * 0.06);
                    const marginBottom = Math.max(90, height * 0.14);
                    const maxDepth = Math.max(...nodes.map((n) => n.depth), 1);
                    const levelGap = (height - marginTop - marginBottom) / maxDepth;

                    let leafIndex = 0;
                    const orderedLeaves = [];
                    function setX(node) {{
                        if (!node.children.length) {{
                            const span = Math.max(1, leaves.length - 1);
                            node.x = marginX + ((width - 2 * marginX) * leafIndex) / span;
                            leafIndex += 1;
                            orderedLeaves.push(node);
                            return node.x;
                        }}
                        const xs = node.children.map((c) => setX(c));
                        node.x = xs.reduce((a, b) => a + b, 0) / xs.length;
                        return node.x;
                    }}

                    setX(root);
                    nodes.forEach((n) => {{
                        n.y = marginTop + n.depth * levelGap + (rand() - 0.5) * 2.5;
                        if (n.children.length) {{
                            const feature = features[n.depth % features.length];
                            n.rule = makeSplitRule(feature);
                            n.label = n.rule.text;
                        }} else {{
                            n.rule = null;
                            n.label = '';
                        }}
                    }});
                }}

                function rebuildGraph() {{
                    nodeId = 0;
                    root = buildTree(0, profile.depth);
                    if (!root.children.length) {{
                        const a = newNode(1);
                        const b = newNode(1);
                        a.parent = root;
                        b.parent = root;
                        root.children = [a, b];
                    }}
                    nodes = walk(root, []);
                    leaves = nodes.filter((n) => !n.children.length);
                    links = [];
                    nodes.forEach((node) => node.children.forEach((child) => links.push({{ from: node, to: child }})));
                    leafLoad = new Map(leaves.map((l) => [l.id, 0]));
                    particles = [];
                }}

                function pickChild(node) {{
                    if (!node.children.length) return null;
                    if (node.children.length === 1) return node.children[0];
                    return rand() < 0.5 ? node.children[0] : node.children[1];
                }}

                function spawnParticle() {{
                    const first = pickChild(root);
                    if (!first) return;
                    const isGreen = rand() < profile.greenBias;
                    particles.push({{
                        node: root,
                        next: first,
                        t: 0,
                        speed: profile.speed + rand() * 0.07,
                        color: isGreen ? palette.green : palette.blue,
                        r: 3 + rand() * 1.4,
                    }});
                }}

                function advanceParticle(p, dt) {{
                    p.t += p.speed * dt;
                    while (p.t >= 1) {{
                        p.t -= 1;
                        p.node = p.next;
                        if (!p.node.children.length) {{
                            const val = (leafLoad.get(p.node.id) || 0) + 1.7;
                            leafLoad.set(p.node.id, Math.min(34, val));
                            return false;
                        }}
                        p.next = pickChild(p.node);
                        if (!p.next) return false;
                    }}
                    return true;
                }}

                function fadeLeafLoads(dt) {{
                    leaves.forEach((leaf) => {{
                        const current = leafLoad.get(leaf.id) || 0;
                        const decayed = Math.max(0, current - 0.6 * dt);
                        leafLoad.set(leaf.id, decayed);
                    }});
                }}

                function drawBranches() {{
                    ctx.strokeStyle = palette.link;
                    ctx.lineWidth = 1.4;
                    links.forEach((link) => {{
                        const midY = (link.from.y + link.to.y) * 0.48;
                        ctx.beginPath();
                        ctx.moveTo(link.from.x, link.from.y);
                        ctx.lineTo(link.from.x, midY);
                        ctx.lineTo(link.to.x, midY);
                        ctx.lineTo(link.to.x, link.to.y);
                        ctx.stroke();

                        if (link.from.rule && link.from.children.length === 2) {{
                            const isRight = link.from.children[1] === link.to;
                            const decisionTxt = (isRight ? '> ' : '<= ') + link.from.rule.threshold;
                            const tx = (link.from.x + link.to.x) * 0.5;
                            const ty = midY - 6;
                            pushHoverTarget(tx, ty, 180, 20, 'Branch rule: ' + link.from.rule.feature + ' ' + decisionTxt);
                            if (link.from.depth <= inlineDepthLimit) {{
                                ctx.fillStyle = '#6a7068';
                                ctx.font = '11px Manrope, sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText(decisionTxt, tx, ty);
                            }}
                        }}
                    }});
                }}

                function drawNodeHints() {{
                    nodes.forEach((node, idx) => {{
                        if (!node.children.length) return;
                        const color = idx % 2 ? palette.blue : palette.green;
                        ctx.fillStyle = color;
                        ctx.fillRect(node.x - 1.2, node.y + 8, 2.4, 14);
                        pushHoverTarget(node.x, node.y - 7, 210, 24, 'Split rule: ' + node.label);
                        if (node.depth <= inlineDepthLimit) {{
                            ctx.fillStyle = '#5b6158';
                            ctx.font = '12px Manrope, sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText(node.label, node.x, node.y - 7);
                        }}
                    }});
                }}

                function drawParticles() {{
                    particles.forEach((p) => {{
                        const x = p.node.x + (p.next.x - p.node.x) * p.t;
                        const y = p.node.y + (p.next.y - p.node.y) * p.t;
                        ctx.beginPath();
                        ctx.fillStyle = p.color;
                        ctx.arc(x, y, p.r, 0, Math.PI * 2);
                        ctx.fill();
                    }});
                }}

                function drawLeafBubbles(height) {{
                    const baseY = height - 34;
                    leaves.forEach((leaf, idx) => {{
                        const load = leafLoad.get(leaf.id) || 0;
                        if (load < 0.05) return;
                        const radius = 4 + Math.min(34, load * 0.72);
                        const color = idx % 2 ? palette.blue : palette.green;
                        ctx.beginPath();
                        ctx.fillStyle = color;
                        ctx.arc(leaf.x, baseY, radius, 0, Math.PI * 2);
                        ctx.fill();
                    }});
                }}

                function resizeCanvas() {{
                    const dpr = window.devicePixelRatio || 1;
                    const width = layer.clientWidth;
                    const height = layer.clientHeight;
                    canvas.width = Math.floor(width * dpr);
                    canvas.height = Math.floor(height * dpr);
                    ctx = canvas.getContext('2d');
                    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                    assignLayout(width, height);
                }}

                function render(ts) {{
                    if (!ctx) return;
                    const width = layer.clientWidth;
                    const height = layer.clientHeight;
                    const dt = Math.min(0.05, (ts - (lastTime || ts)) / 1000);
                    lastTime = ts;

                    spawnCarry += profile.spawnRate * dt * 60;
                    while (spawnCarry >= 1) {{
                        spawnParticle();
                        spawnCarry -= 1;
                    }}
                    particles = particles.filter((p) => advanceParticle(p, dt * 60));
                    fadeLeafLoads(dt * 60);
                    hoverTargets = [];

                    ctx.clearRect(0, 0, width, height);
                    drawBranches();
                    drawNodeHints();
                    drawParticles();
                    drawLeafBubbles(height);

                    window.requestAnimationFrame(render);
                }}

                const activate = (index) => {{
                    cards.forEach((el, idx) => el.classList.toggle('active', idx === index));
                    pins.forEach((el, idx) => el.classList.toggle('active', idx === index));
                }};

                canvas.addEventListener('mousemove', (event) => {{
                    const rect = canvas.getBoundingClientRect();
                    const x = event.clientX - rect.left;
                    const y = event.clientY - rect.top;
                    const target = findHoverTarget(x, y);
                    if (target) {{
                        showTooltip(target, x, y);
                    }} else {{
                        hideTooltip();
                    }}
                }});
                canvas.addEventListener('mouseleave', hideTooltip);

                pins.forEach((pin, idx) => {{
                    pin.addEventListener('click', () => {{
                        cards[idx].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        activate(idx);
                    }});
                }});

                const observer = new IntersectionObserver((entries) => {{
                    const visible = entries
                        .filter((entry) => entry.isIntersecting)
                        .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
                    if (visible.length) {{
                        const idx = Number(visible[0].target.dataset.step || 0);
                        activate(idx);
                    }}
                }}, {{ threshold: [0.35, 0.55, 0.75] }});

                cards.forEach((card) => observer.observe(card));
                rebuildGraph();
                resizeCanvas();
                window.addEventListener('resize', resizeCanvas);
                window.requestAnimationFrame(render);
                activate(0);
            }})();
        </script>
    </body>
</html>
"""
    return Response(page, mimetype="text/html")


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
        "zoom": 15,
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
    available_metrics = [m for m in METRIC_ORDER if m in preds.columns]

    props_by_hex_year = {}
    for r in preds.itertuples(index=False):
        metric_props = {}
        for metric in available_metrics:
            value = getattr(r, metric, None)
            if pd.notna(value):
                metric_props[metric] = float(value)
        props_by_hex_year[(r.hex, int(r.year))] = metric_props

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


def make_deck_spec(
    geojson: Dict,
    year: int,
    color_metric: str = "pred",
    focus_view_state: dict | None = None,
    color_enabled: bool = True,
):
    spec = {
        "initialViewState": {"latitude": 40.7128, "longitude": -74.0060, "zoom": 10},
        "controller": True,
        "mapStyle": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "metricLabels": METRIC_LABELS,
        "rankMetrics": sorted(RANK_METRICS),
        "layers": [{
            "@@type": "GeoJsonLayer",
            "id": "hex-choropleth",
            "data": geojson,
            "colorMetric": color_metric,
            "colorStops": COLOR_STOPS,
            "colorEnabled": color_enabled,
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
        html.Li([html.B("OMB"), " — NYC Office of Management and Budget (citywide economic forecast source)."]),
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
    html.H6("How we predicted 2027-2029", className="fhf-section-title"),
    html.Ul([
        html.Li("Step 1 (local baseline): we estimate ZIP-level vulnerability using ACS data (income, rent burden, unemployment, and poverty rate), then map ZIP effects into H3 hexes via MODZCTA overlap weights."),
        html.Li("Step 2 (citywide forecast): we pull NYC OMB forecast assumptions for 2027-2029 (citywide unemployment/income/rent growth)."),
        html.Li("Step 3 (downscaling): each ZIP gets the citywide shock scaled by its vulnerability profile, so higher-income/lower-poverty ZIPs are damped while lower-income/higher-poverty ZIPs are amplified."),
        html.Li("Step 4 (feature update): those ZIP-year multipliers update future feature columns (n311_y, nhpd_y, nevict_y, nfiled_y) for 2027-2029 before model scoring."),
        html.Li("Step 5 (map output): model predictions are shown as within-year relative ranks (0-1), which preserves hotspot ordering but can visually compress year-to-year magnitude differences."),
    ]),
    html.H6("What those column names mean (plain English)", className="fhf-section-title"),
    html.Ul([
        html.Li([
            html.Code("n311_y"),
            " — How many relevant 311 service requests came from that area for that year (from NYC 311 open data). In this project context, homelessness-related 311 requests are reports tied to someone experiencing homelessness, such as Homeless Person Assistance (for example, a person appears to be sleeping on a sidewalk and needs outreach/support) and Homeless Encampment (for example, a tent encampment under a bridge or in a park). Example value: 37.",
        ]),
        html.Li([
            html.Code("nhpd_y"),
            " — How many HPD housing complaints were counted in that area (from HPD Complaint Problems data). Example value: 22.",
        ]),
        html.Li([
            html.Code("nevict_y"),
            " — How many executed evictions were recorded there (from NYC Residential Evictions). Example value: 4.",
        ]),
        html.Li([
            html.Code("nfiled_y"),
            " — How many eviction cases were filed there, even if not yet executed (from filed-eviction dataset). Example value: 13.",
        ]),
        html.Li([
            html.Code("n_dcp_units"),
            " — Total housing units in DCP-tracked projects in that area (from DCP housing program data). Example value: 1,250.",
        ]),
        html.Li([
            html.Code("n_dcp_aff_units"),
            " — Affordable units among those DCP-tracked units (same DCP source). Example value: 410.",
        ]),
        html.Li([
            html.Code("n_dcp_expiring5yr"),
            " — Number of DCP-tracked units whose affordability/regulatory status is expected to expire within ~5 years. Example value: 95.",
        ]),
        html.Li([
            html.Code("n_dcp_expired"),
            " — Number of DCP-tracked units whose affordability/regulatory period is already expired. Example value: 60.",
        ]),
        html.Li([
            html.Code("dcp_status_median"),
            " — A median summary score of project status in that area (from DCP status fields, converted to numeric categories). Example value: 2.0.",
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
    html.H6("Current model scorecard", className="fhf-section-title"),
    html.P("Held-out metrics from the latest baseline training run (March 27, 2026):"),
    html.Ul([
        html.Li("Evaluation setup: target=risk_proxy, random seed=42, train/test split=80/20."),
        html.Li("Sample counts: 65,923 train rows and 16,481 test rows."),
    ]),
    html.Ul([
        html.Li("Random Forest: MAE=0.000012, RMSE=0.000247, R2=0.999979"),
        html.Li("LightGBM: MAE=0.000210, RMSE=0.002073, R2=0.998502"),
        html.Li("XGBoost: MAE=0.000284, RMSE=0.001945, R2=0.998681"),
    ]),
    html.P(
        "These are train/test-split errors on the proxy target used in this deployment, not causal impact estimates. "
        "Lower MAE/RMSE and higher R2 are better, but small differences should be interpreted cautiously because feature engineering and target construction drive much of this signal."
    ),
    html.H6("Why choose these 3 models", className="fhf-section-title"),
    html.P(
        "We use LightGBM, XGBoost, and Random Forest because this project is tabular, nonlinear, and relatively sparse at the hex-year level. LightGBM and XGBoost capture threshold effects and interactions in complaint/eviction/housing signals with strong predictive performance, while Random Forest gives a stable, low-assumption ensemble baseline that is less sensitive to local noise. Using all three lets us compare consistent feature-driven rankings across model families instead of relying on a single algorithmic view of risk."
    ),
    html.P("Need higher fidelity? Swap the bootstrap inputs for full NYC feeds, rerun scripts/aggregate_hpd_complaints.py and scripts/train_baseline.py, then redeploy."),
    html.H6("GitHub", className="fhf-section-title"),
    html.Ul([
        html.Li([
            "Repository: ",
            html.A(
                "github.com/rohanramnarain/future_unhoused_nyc",
                href="https://github.com/rohanramnarain/future_unhoused_nyc",
                target="_blank",
            ),
        ]),
        html.Li("Use this repo for source code, pipeline scripts, deployment configuration, and change history."),
    ]),
    html.P("Created by Rohan Ramnarain, Kevin Guillermo, Marilyn Echeverria, and Alice Dong."),
    html.H6("Special thanks", className="fhf-section-title"),
    html.P("Special thanks to our funders: the CUNY Graduate Center M.S. in Data Analysis and Visualization Program (MA/MS Grant support), and the Futures Initiative Equity and Social Justice Grant."),
])


MODEL_COPY = {
    "lgbm": html.Div(className="fhf-prose", children=[
        html.P("LightGBM = gradient boosting (leaf-wise trees)."),
        html.P("Light Gradient Boosting uses hundreds of tiny decision trees trained sequentially; each tree focuses on the residual mistakes of prior trees."),
        html.P("Basic tree example (this project): Tree 1 might split first on n311_y (high complaint density) and then on nevict_y; a hex with high n311_y and high nevict_y gets a higher baseline risk. Tree 2 then corrects misses by splitting on n_dcp_expiring5yr and n_dcp_aff_units, nudging risk up where affordability pressure is rising."),
        html.P([html.B("How this is different from the other two algorithms"), ": LightGBM grows trees leaf-wise, so it can quickly chase the biggest remaining error pockets." ]),
        html.P("Example: if the largest residual error is concentrated in hexes with very high n311_y but only mid-level nevict_y, LightGBM can keep splitting that branch deeper earlier than XGBoost (more level-wise growth), while Random Forest would not do sequential residual correction at all."),
        html.Ul([
            html.Li("Fast on sparse tabular data and handles nonlinear jumps in complaint/eviction patterns."),
            html.Li("Same nine engineered signals (n311_y, nhpd_y, nevict_y, nfiled_y, n_dcp_units, n_dcp_aff_units, n_dcp_expiring5yr, n_dcp_expired, dcp_status_median)."),
            html.Li("Conformal bands add a give-or-take range without extra retraining."),
        ]),
    ]),
    "xgb": html.Div(className="fhf-prose", children=[
        html.P("XGBoost = gradient boosting (level-wise trees)."),
        html.P("Another gradient-boosted tree ensemble; uses histogram splits for speed and strong performance on tabular problems."),
        html.P("Basic tree example (this project): an early XGBoost tree might split level-by-level on nfiled_y and nhpd_y to isolate hexes with both many filed evictions and many housing complaints. The next boosting tree can then focus on residual error inside that group, for example splitting on dcp_status_median to separate areas with weaker preservation pipeline signals."),
        html.P([html.B("How this is different from the other two algorithms"), ": XGBoost typically grows trees level-wise with stronger explicit regularization, which often gives more conservative, stable corrections than LightGBM." ]),
        html.P("Example: for hexes near the threshold on nfiled_y, XGBoost may apply a smaller step change after adding depth and penalty constraints, while LightGBM may make a sharper local correction and Random Forest would instead average many independent tree votes."),
        html.Ul([
            html.Li("Captures sharp thresholds (e.g., sudden eviction spikes) while staying fast enough for frequent re-trains."),
            html.Li("Same feature set and per-year percentile scaling as LightGBM so colors stay comparable within the model."),
            html.Li("Choose this to test a stronger regularized boosting baseline."),
        ]),
    ]),
    "rf": html.Div(className="fhf-prose", children=[
        html.P("Random Forest = many trees voting/averaging."),
        html.P("Hundreds of decorrelated decision trees averaged together; great for quick baselines and uncertainty intuition."),
        html.P("Basic tree example (this project): one forest tree might split on n311_y then n_dcp_expired and predict high risk for complaint-heavy hexes with many expired affordable units. Another tree, built from a different bootstrap sample, might split on nevict_y then n_dcp_units and produce a moderate score. The model output is the average of those tree-level predictions."),
        html.P([html.B("How this is different from the other two algorithms"), ": Random Forest trees are trained independently in parallel and then averaged, instead of sequentially correcting residuals like LightGBM and XGBoost." ]),
        html.P("Example: if one tree overreacts to a spike in nhpd_y for a single hex, many other trees that did not see that exact bootstrap sample can pull the final score back toward the middle; boosting models would intentionally keep fitting that residual pattern across later trees."),
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
    html.A("Open LightGBM interactive model page", href="/models/lgbm", className="btn btn-sm btn-outline-primary"),
    html.Hr(className="my-3"),
    html.H6("Random Forest", className="fhf-section-title"),
    MODEL_COPY["rf"],
    html.A("Open Random Forest interactive model page", href="/models/rf", className="btn btn-sm btn-outline-primary"),
    html.Hr(className="my-3"),
    html.H6("XGBoost", className="fhf-section-title"),
    MODEL_COPY["xgb"],
    html.A("Open XGBoost interactive model page", href="/models/xgb", className="btn btn-sm btn-outline-primary"),
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
                    html.A(
                        dbc.Badge("Thanks to Open Data Week and School of Data", color="light", text_color="dark", pill=True),
                        href="/odwoutro",
                        className="text-decoration-none",
                    ),
                ]),
                html.Div(className="mt-3 fhf-links", children=[
                    html.A("What I am looking at?", id="link-read-map", href="#read-map", className="me-3"),
                    html.A("Sources", id="link-sources", href="#sources", className="me-3"),
                    html.A("Method", id="link-method", href="#method", className="me-3"),
                    html.A("Limitations", id="link-limits", href="#limits", className="me-3"),
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
                        options=METRIC_OPTIONS,
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

            dbc.Row([
                dbc.Col([
                    dbc.Label("Map colors", html_for="show-colors", className="fhf-muted mb-1"),
                    dbc.Checklist(
                        id="show-colors",
                        options=[{"label": "Show yellow/orange/red risk colors", "value": "on"}],
                        value=["on"],
                        switch=True,
                    ),
                ], md=12),
            ], className="g-3 mt-1"),
        ]),
    ]),

    html.Div(id="map-legend", className="fhf-map-legend mb-2"),
    html.Div(id="deck-container", className="fhf-map-shell mb-3"),

    dbc.Accordion(id="info-accordion", className="mb-4", always_open=False, start_collapsed=True, children=[
        dbc.AccordionItem(
            item_id="read-map",
            title="How to read this map",
            children=html.Div(className="fhf-prose", children=[
                html.P([
                    html.B("Algorithms"),
                    " already shape ",
                    html.B("daily life"),
                    ": what we see, what we are sold, how neighborhoods are marketed, and in some cases how housing decisions are priced or prioritized. ",
                    "Those systems can ",
                    html.B("reinforce inequality"),
                    ". This project tries to ",
                    html.B("flip that logic"),
                    " and use forecasting tools for ",
                    html.B("public good"),
                    ": first ",
                    html.B("identify"),
                    " where homelessness-related pressure is likely to increase, then ",
                    html.B("investigate why"),
                    ", and help target ",
                    html.B("prevention"),
                    " before harm grows.",
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
                    html.Li("Use Metric to switch between model outputs (pred/lo/hi) and raw predictor layers."),
                    html.Li("Hover a hex for the value and year; ZIP is approximate."),
                ]),
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
            item_id="method",
            title="Model in plain English",
            children=html.Div(id="method", children=[
                html.Div(id="model-copy", children=ALL_MODEL_COPY),
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
                html.H6("Data bias", className="fhf-section-title mt-3"),
                html.P("Data bias can arise through selection and reporting effects because 311 and complaint-based features are partly shaped by trust in institutions, language access, digital access, and willingness to report. As a result, neighborhoods with lower reporting activity can appear artificially lower risk even when underlying need is high."),
                html.P("Missingness in these data is likely not at random. Gaps in filings, geocodes, or complaint records often track differences in administrative capacity and legal precarity, so simply omitting missing records can systematically underestimate risk in marginalized areas."),
                html.P("Measurement and construct bias are also important concerns. Proxy targets such as homelessness-related 311 activity capture service contact and public visibility rather than the full underlying construct of housing instability or unsheltered need."),
                html.P("Temporal drift and policy endogeneity can further weaken model reliability over time. Relationships learned from prior years may shift after policy changes, outreach efforts, weather shocks, or changes in enforcement strategy, which reduces the transportability of learned patterns."),
                html.P("Scale and denominator effects can distort comparisons across places. Hex-level aggregation may mask important within-hex heterogeneity, and using raw counts without normalizing by population at risk can over-rank denser areas relative to true per-capita burden."),
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


def _year_features_for_model(model_key: str, year: int) -> list[dict]:
    model = model_key if model_key in GJ_BY_MODEL else DEFAULT_MODEL
    gj_full = GJ_BY_MODEL[model]
    return [f for f in gj_full["features"] if f["properties"]["year"] == year]


def _legend_value_formatter(metric_key: str, value: float) -> str:
    if metric_key in RANK_METRICS:
        return f"{value:.2f}"
    av = abs(value)
    if av >= 1000:
        return f"{value:,.0f}"
    if av >= 10:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def _legend_domain(metric_key: str, year_features: list[dict]) -> tuple[float, float, str]:
    if metric_key in RANK_METRICS:
        return 0.0, 1.0, "Percentile rank scale (0 to 1)."

    vals = []
    for feat in year_features:
        v = feat.get("properties", {}).get(metric_key)
        if pd.notna(v):
            vals.append(float(v))

    if not vals:
        return 0.0, 1.0, "No data available for this layer/year."

    series = pd.Series(vals, dtype=float)
    raw_min = float(series.min())
    raw_max = float(series.max())
    p05 = float(series.quantile(0.05))
    p95 = float(series.quantile(0.95))
    dom_min = p05 if p95 > p05 else raw_min
    dom_max = p95 if p95 > p05 else raw_max

    if not dom_max > dom_min:
        dom_min = raw_min - 1.0
        dom_max = raw_max + 1.0

    note = (
        f"Color scale uses clipped range {_legend_value_formatter(metric_key, dom_min)}"
        f" to {_legend_value_formatter(metric_key, dom_max)} (5th to 95th percentile)."
    )
    return dom_min, dom_max, note


def _legend_component(metric_key: str, year_features: list[dict]):
    metric_label = METRIC_LABELS.get(metric_key, metric_key)
    dom_min, dom_max, note = _legend_domain(metric_key, year_features)

    items = []
    prev_val = dom_min
    for idx, stop in enumerate(COLOR_STOPS):
        stop_val = dom_min + stop["max"] * (dom_max - dom_min)
        rgba = stop["color"]
        swatch_style = {
            "backgroundColor": f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3] / 255.0:.3f})",
            "border": "1px solid rgba(120, 98, 72, 0.28)",
        }
        if idx == 0:
            label = f"<= {_legend_value_formatter(metric_key, stop_val)}"
        else:
            label = (
                f"{_legend_value_formatter(metric_key, prev_val)}"
                f" to {_legend_value_formatter(metric_key, stop_val)}"
            )
        prev_val = stop_val

        items.append(
            html.Div(
                className="fhf-legend-item",
                children=[
                    html.Span(className="fhf-legend-chip", style=swatch_style),
                    html.Span(label, className="fhf-legend-text"),
                ],
            )
        )

    return html.Div(
        children=[
            html.Div(
                className="fhf-legend-header",
                children=[
                    html.Span("Map Key", className="fhf-legend-kicker"),
                    html.Span(metric_label, className="fhf-legend-title"),
                ],
            ),
            html.Div(className="fhf-legend-items", children=items),
            html.Div(note, className="fhf-legend-note"),
        ]
    )


@app.callback(
    Output("map-legend", "children"),
    Input("model", "value"),
    Input("year", "value"),
    Input("metric", "value"),
)
def update_legend(model_key, year, metric):
    metric_key = metric if metric in METRIC_LABELS else "pred"
    year_features = _year_features_for_model(model_key, year)
    return _legend_component(metric_key, year_features)


@app.callback(
    Output("deck-container", "children"),
    Input("model", "value"),
    Input("year", "value"),
    Input("metric", "value"),
    Input("show-colors", "value"),
    Input("zip-submit", "n_clicks"),
    Input("zip-input", "n_submit"),
    State("zip-input", "value"),
)
def update_map(model_key, year, metric, show_colors, zip_clicks, zip_enter, zip_value):  # pragma: no cover - UI wiring
    metric_key = metric if metric in METRIC_LABELS else "pred"
    color_enabled = isinstance(show_colors, list) and "on" in show_colors
    year_features = _year_features_for_model(model_key, year)
    year_gj = {"type": "FeatureCollection", "features": year_features}

    trigger_id = ctx.triggered_id if ctx.triggered_id else None
    focus_state = None
    if trigger_id in ("zip-submit", "zip-input"):
        focus_state = _zip_focus_view_state(zip_value)
    spec = make_deck_spec(
        year_gj,
        year=year,
        color_metric=metric_key,
        focus_view_state=focus_state,
        color_enabled=color_enabled,
    )
    spec["mapboxApiAccessToken"] = settings.mapbox_token or ""
    json_spec = json.dumps(spec)
    iframe_template = """
    <!doctype html>
    <html><head>
      <meta charset='utf-8'/>
                        <link rel='icon' type='image/svg+xml' href='/assets/favicon.svg'/>
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
                const { colorMetric = 'pred', colorStops = [], colorEnabled = true, ...layerProps } = layerSpec;
                const focusViewState = spec.focusViewState || null;
                const metricLabels = spec.metricLabels || {};
                const rankMetrics = spec.rankMetrics || [];
                const defaultStops = [
                    { max: 0.0, color: [0, 0, 0, 0] },
                    { max: 0.4, color: [255, 255, 204, 120] },
                    { max: 0.6, color: [255, 237, 160, 170] },
                    { max: 0.8, color: [254, 178, 76, 210] },
                    { max: 0.9, color: [253, 141, 60, 235] },
                    { max: 1.0, color: [189, 0, 38, 255] }
                ];
                const stops = colorStops.length ? colorStops : defaultStops;
                const isRankMetric = rankMetrics.includes(colorMetric);
                const quantile = (arr, q) => {
                    if (!arr.length) { return null; }
                    const pos = (arr.length - 1) * q;
                    const base = Math.floor(pos);
                    const rest = pos - base;
                    const next = arr[base + 1] !== undefined ? arr[base + 1] : arr[base];
                    return arr[base] + rest * (next - arr[base]);
                };
                const metricValues = (layerProps?.data?.features || [])
                    .map(f => Number(f?.properties?.[colorMetric]))
                    .filter(Number.isFinite)
                    .sort((a, b) => a - b);

                let domainMin = 0;
                let domainMax = 1;
                if (!isRankMetric && metricValues.length) {
                    const p05 = quantile(metricValues, 0.05);
                    const p95 = quantile(metricValues, 0.95);
                    const rawMin = metricValues[0];
                    const rawMax = metricValues[metricValues.length - 1];
                    domainMin = Number.isFinite(p05) ? p05 : rawMin;
                    domainMax = Number.isFinite(p95) ? p95 : rawMax;
                    if (!(domainMax > domainMin)) {
                        domainMin = rawMin;
                        domainMax = rawMax;
                    }
                    if (!(domainMax > domainMin)) {
                        domainMin = rawMin - 1;
                        domainMax = rawMax + 1;
                    }
                }

                const normalizeMetricValue = (val) => {
                    const num = Number(val);
                    if (!Number.isFinite(num)) { return null; }
                    if (isRankMetric) {
                        return Math.max(0, Math.min(1, num));
                    }
                    const clipped = Math.max(domainMin, Math.min(domainMax, num));
                    return (clipped - domainMin) / (domainMax - domainMin || 1);
                };

                const formatMetricValue = (val) => {
                    const num = Number(val);
                    if (!Number.isFinite(num)) { return '—'; }
                    if (isRankMetric) { return num.toFixed(3); }
                    if (Math.abs(num) >= 1000) { return num.toLocaleString(undefined, { maximumFractionDigits: 0 }); }
                    if (Math.abs(num) >= 10) { return num.toLocaleString(undefined, { maximumFractionDigits: 1 }); }
                    return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
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
                        if (!colorEnabled) {
                            return [0, 0, 0, 0];
                        }
                        const rawValue = feature?.properties?.[colorMetric];
                        const normalized = normalizeMetricValue(rawValue);
                        if (normalized === null) {
                            return [0, 0, 0, 0];
                        }
                        return bandColor(normalized);
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
                        const metricLabel = metricLabels[colorMetric] || colorMetric;
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
