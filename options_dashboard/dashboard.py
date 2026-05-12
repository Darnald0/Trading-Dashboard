"""
dashboard.py  –  Plotly-Dash web dashboard.

Layout:  Sidebar (left)  |  Gamma  |  Charm  |  Vanna   (3 charts side by side)

Charm panel can toggle between bar chart and heatmap (time x strike)
with candlestick price overlay.
"""

import datetime as dt
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, ALL, ctx, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import SETTINGS, SIDEBAR_WIDTH, ET
import data_fetcher
import matrix_data
import cot_scraper
import market_state
import heatmap_data
from greek_calculator import (compute_exposure, compute_live_metrics,
                             classify_regime, compute_vanna_vix_signal,
                             compute_charm_clock, compute_trade_signal,
                             compute_profile_analysis)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Options Greek Dashboard",
    suppress_callback_exceptions=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL THEME  (navy + accent colors, slightly rounded cards)
# ══════════════════════════════════════════════════════════════════════════════

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      :root {
        /* base navy palette */
        --bg-main:        #181c33;
        --bg-card:        #222647;
        --bg-card-hover:  #2a2f55;
        --bg-deep:        #14172a;
        --bg-overlay:     rgba(34, 38, 71, 0.7);

        /* text */
        --text-primary:   #ffffff;
        --text-secondary: #cbd5e1;
        --text-muted:     #7986b4;
        --text-faint:     #4a527a;

        /* accents */
        --accent-blue:    #5b8dee;
        --accent-pink:    #ec4899;
        --accent-teal:    #14b8a6;
        --accent-orange:  #f97316;
        --accent-yellow:  #fbbf24;
        --accent-purple:  #a78bfa;

        /* directional */
        --bull:           #22c55e;
        --bull-soft:      #16a34a;
        --bear:           #ef4444;
        --bear-soft:      #dc2626;

        /* dividers / borders */
        --border-subtle:  rgba(255, 255, 255, 0.05);
        --border-medium:  rgba(255, 255, 255, 0.10);
        --border-accent:  rgba(91, 141, 238, 0.30);
      }

      html, body {
        background-color: var(--bg-main) !important;
        color: var(--text-secondary) !important;
        font-family: "Inter", system-ui, -apple-system, sans-serif !important;
        -webkit-font-smoothing: antialiased;
      }

      /* Override Darkly's defaults */
      .bg-dark, .navbar-dark, body.bg-dark {
        background-color: var(--bg-main) !important;
      }

      /* Headings */
      h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.01em;
      }

      /* Generic card surface */
      .form-control, .form-select {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
      }
      .form-control:focus, .form-select:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(91, 141, 238, 0.15) !important;
      }

      /* Bootstrap dropdown text */
      .Select-control, .Select-menu-outer {
        background-color: var(--bg-card) !important;
        border-color: var(--border-subtle) !important;
        border-radius: 8px !important;
      }
      .Select-value-label, .Select-input > input {
        color: var(--text-primary) !important;
      }
      .Select-menu-outer .Select-option {
        background-color: var(--bg-card) !important;
        color: var(--text-secondary) !important;
      }
      .Select-menu-outer .Select-option:hover,
      .Select-menu-outer .Select-option.is-focused {
        background-color: var(--bg-card-hover) !important;
        color: var(--text-primary) !important;
      }

      /* Radio + checkbox accent */
      .form-check-input:checked {
        background-color: var(--accent-blue) !important;
        border-color: var(--accent-blue) !important;
      }
      .form-check-input {
        background-color: var(--bg-deep) !important;
        border: 1px solid var(--border-medium) !important;
      }

      /* Form labels */
      .form-label {
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.72rem !important;
      }

      /* Slider styling (rc-slider) */
      .rc-slider-track {
        background-color: var(--accent-blue) !important;
      }
      .rc-slider-handle {
        border-color: var(--accent-blue) !important;
        background-color: var(--accent-blue) !important;
      }
      .rc-slider-rail {
        background-color: var(--bg-deep) !important;
      }
      .rc-slider-mark-text {
        color: var(--text-muted) !important;
      }
      .rc-slider-mark-text-active {
        color: var(--text-primary) !important;
      }

      /* Hr divider */
      hr {
        border-color: var(--border-subtle) !important;
        opacity: 1 !important;
        margin: 14px 0 !important;
      }

      /* Buttons */
      button, .btn {
        border-radius: 8px !important;
        font-weight: 600 !important;
      }

      /* Scrollbars */
      ::-webkit-scrollbar { width: 8px; height: 8px; }
      ::-webkit-scrollbar-track { background: var(--bg-deep); }
      ::-webkit-scrollbar-thumb {
        background: var(--bg-card-hover);
        border-radius: 4px;
      }
      ::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

      /* Generic class helpers we'll use throughout the app */
      .panel-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
      }
      .panel-card-deep {
        background-color: var(--bg-deep);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
      }
    </style>
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
'''

# ══════════════════════════════════════════════════════════════════════════════
#  WIDGET NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

NAV_WIDTH_OPEN = 180
NAV_WIDTH_CLOSED = 44

# Widget definitions: id -> (label, icon)
WIDGETS = {
    "greeks": ("Greeks Visualizer", "\U0001F4CA"),
    "matrix": ("Market Matrix", "\U0001F5FA"),
    "cot":    ("COT Board",     "\U0001F4CB"),
    "market": ("Broad Market",  "\U0001F30E"),
    "heatmap": ("Heatmap",      "\U0001F525"),
}

def _build_nav_buttons(active_id="greeks"):
    """Build the list of widget buttons for the nav sidebar."""
    buttons = []
    for wid, (label, icon) in WIDGETS.items():
        is_active = wid == active_id
        buttons.append(
            html.Div(
                html.Button(
                    [
                        # Icon takes exactly the collapsed nav width, centers itself
                        html.Span(icon, style={
                            "fontSize": "1.1rem",
                            "width": f"{NAV_WIDTH_CLOSED}px",
                            "minWidth": f"{NAV_WIDTH_CLOSED}px",
                            "textAlign": "center",
                            "flexShrink": 0,
                        }),
                        html.Span(label, className="nav-label",
                                  style={"fontSize": "0.82rem",
                                         "whiteSpace": "nowrap",
                                         "overflow": "hidden",
                                         "letterSpacing": "0.02em"}),
                    ],
                    id={"type": "nav-btn", "index": wid},
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "display": "flex",
                        "alignItems": "center",
                        "padding": "11px 0",
                        "border": "none",
                        "borderLeft": ("3px solid var(--accent-blue)" if is_active
                                        else "3px solid transparent"),
                        "borderRadius": "0",
                        "cursor": "pointer",
                        "color": "var(--text-primary)" if is_active else "var(--text-muted)",
                        "backgroundColor": ("var(--bg-card)" if is_active
                                              else "transparent"),
                        "fontWeight": "600" if is_active else "500",
                        "transition": "all 0.15s",
                    },
                ),
                style={"marginBottom": "2px"},
            )
        )
    return buttons


nav_sidebar = html.Div(
    id="nav-sidebar",
    children=[
        # Toggle button
        html.Button(
            "\u2630",
            id="nav-toggle",
            n_clicks=0,
            style={
                "width": "100%",
                "padding": "12px",
                "border": "none",
                "borderBottom": "1px solid var(--border-subtle)",
                "backgroundColor": "transparent",
                "color": "var(--text-muted)",
                "fontSize": "1.2rem",
                "cursor": "pointer",
                "textAlign": "center",
                "marginBottom": "10px",
            },
        ),
        # Widget buttons
        html.Div(
            id="nav-buttons",
            children=_build_nav_buttons("greeks"),
            style={"padding": "0"},
        ),
    ],
    style={
        "width": f"{NAV_WIDTH_OPEN}px",
        "minWidth": f"{NAV_WIDTH_OPEN}px",
        "height": "100vh",
        "overflowY": "auto",
        "overflowX": "hidden",
        "borderRight": "1px solid var(--border-subtle)",
        "backgroundColor": "var(--bg-deep)",
        "transition": "width 0.2s, min-width 0.2s",
        "flexShrink": 0,
    },
)

# Store for nav state
nav_store = dcc.Store(id="store-nav", data={"open": True, "active": "greeks"})
matrix_prev_store = dcc.Store(id="store-matrix-prev", data={})


# ══════════════════════════════════════════════════════════════════════════════
#  GREEKS VISUALIZER WIDGET
# ══════════════════════════════════════════════════════════════════════════════

settings_sidebar = html.Div(
    [
        html.H5("Greeks Visualizer", className="mb-3",
                 style={"letterSpacing": "0.05em"}),

        # Ticker
        dbc.Label("Ticker Symbol", className="fw-bold mt-1",
                  style={"fontSize": "0.85rem"}),
        dbc.Input(
            id="input-ticker", type="text", value=SETTINGS.ticker,
            placeholder="SPY, QQQ, AAPL, SPX...",
            debounce=True, className="mb-2",
            style={"textTransform": "uppercase"},
        ),

        # Expiry
        dbc.Label("Option Expiry", className="fw-bold",
                  style={"fontSize": "0.85rem"}),
        dcc.Dropdown(
            id="dropdown-expiry",
            options=[{"label": "Auto (nearest)", "value": "auto"}],
            value="auto", clearable=False, className="mb-2",
            style={"color": "#111"},
        ),

        # Refresh
        dbc.Label("Auto-Refresh", className="fw-bold",
                  style={"fontSize": "0.85rem"}),
        html.Div(id="refresh-label", className="text-info mb-1",
                 style={"fontSize": "0.80rem"}),
        dcc.Slider(
            id="slider-refresh",
            min=10, max=600, step=10,
            value=SETTINGS.refresh_seconds,
            marks={10: "10s", 60: "1m", 120: "2m", 300: "5m", 600: "10m"},
            tooltip={"placement": "bottom"},
            className="mb-3",
        ),

        # Mode
        dbc.Label("Exposure Mode", className="fw-bold",
                  style={"fontSize": "0.85rem"}),
        dbc.RadioItems(
            id="radio-mode",
            options=[
                {"label": "Open Interest",    "value": "oi"},
                {"label": "Session Volume",   "value": "volume"},
                {"label": "Combined (OI+Vol)", "value": "combined"},
                {"label": "Flow (Dealer)",    "value": "flow"},
                {"label": "OI + Flow (Live)", "value": "oi_flow"},
            ],
            value=SETTINGS.greek_mode,
            className="mb-3",
        ),

        html.Hr(),

        # -- Per-chart view modes --
        dbc.Label("Gamma View", className="fw-bold",
                  style={"fontSize": "0.80rem"}),
        dbc.RadioItems(
            id="radio-gamma-view",
            options=[
                {"label": "Bar",    "value": "bar"},
                {"label": "Values", "value": "values"},
            ],
            value="bar", inline=True, className="mb-2",
        ),

        dbc.Label("Charm View", className="fw-bold",
                  style={"fontSize": "0.80rem"}),
        dbc.RadioItems(
            id="radio-charm-view",
            options=[
                {"label": "Bar",    "value": "bar"},
                {"label": "Values", "value": "values"},
            ],
            value="bar", inline=True, className="mb-2",
        ),

        dbc.Label("Vomma View", className="fw-bold",
                  style={"fontSize": "0.80rem"}),
        dbc.RadioItems(
            id="radio-vomma-view",
            options=[
                {"label": "Bar",    "value": "bar"},
                {"label": "Values", "value": "values"},
            ],
            value="bar", inline=True, className="mb-2",
        ),

        dbc.Label("Speed View", className="fw-bold",
                  style={"fontSize": "0.80rem"}),
        dbc.RadioItems(
            id="radio-speed-view",
            options=[
                {"label": "Bar",    "value": "bar"},
                {"label": "Values", "value": "values"},
            ],
            value="bar", inline=True, className="mb-2",
        ),

        dbc.Label("Vanna View", className="fw-bold",
                  style={"fontSize": "0.80rem"}),
        dbc.RadioItems(
            id="radio-vanna-view",
            options=[
                {"label": "Bar",    "value": "bar"},
                {"label": "Values", "value": "values"},
            ],
            value="bar", inline=True, className="mb-2",
        ),

        dbc.Label("Zomma View", className="fw-bold",
                  style={"fontSize": "0.80rem"}),
        dbc.RadioItems(
            id="radio-zomma-view",
            options=[
                {"label": "Bar",    "value": "bar"},
                {"label": "Values", "value": "values"},
            ],
            value="bar", inline=True, className="mb-2",
        ),

        html.Hr(),
        html.Div(id="status-text", className="text-muted",
                 style={"fontSize": "0.75rem", "whiteSpace": "pre-line"}),

        # Hidden store for previous exposure values
        dcc.Store(id="store-prev-exposure", data={}),
    ],
    style={
        "width": f"{SIDEBAR_WIDTH}px",
        "minWidth": f"{SIDEBAR_WIDTH}px",
        "height": "100%",
        "overflowY": "auto",
        "padding": "14px",
        "borderRight": "1px solid var(--border-subtle)",
        "backgroundColor": "var(--bg-deep)",
    },
)

# Layout:  Settings | Gamma | Top row: Charm + Vomma + Speed
#                            | Bottom row: Vanna + DEX + Zomma
charts_panel = html.Div(
    [
        # Left column: Gamma (full height)
        html.Div(
            dcc.Graph(id="chart-gamma", style={"height": "100%"}),
            style={"flex": 0.7, "minWidth": 0},
        ),
        # Right column: top row (3 charts), bottom row (3 charts)
        html.Div(
            [
                # Top row: Charm + Vomma + Speed
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(id="chart-charm", style={"height": "100%"}),
                            style={"flex": 1, "minWidth": 0},
                        ),
                        html.Div(
                            dcc.Graph(id="chart-vomma", style={"height": "100%"}),
                            style={"flex": 1, "minWidth": 0},
                        ),
                        html.Div(
                            dcc.Graph(id="chart-speed", style={"height": "100%"}),
                            style={"flex": 1, "minWidth": 0},
                        ),
                    ],
                    style={
                        "flex": 1,
                        "minHeight": 0,
                        "display": "flex",
                    },
                ),
                # Bottom row: Vanna + DEX + Zomma
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(id="chart-vanna", style={"height": "100%"}),
                            style={"flex": 1, "minWidth": 0},
                        ),
                        html.Div(
                            dcc.Graph(id="chart-dex", style={"height": "100%"}),
                            style={"flex": 1, "minWidth": 0},
                        ),
                        html.Div(
                            dcc.Graph(id="chart-zomma", style={"height": "100%"}),
                            style={"flex": 1, "minWidth": 0},
                        ),
                    ],
                    style={
                        "flex": 1,
                        "minHeight": 0,
                        "display": "flex",
                    },
                ),
                # Signal panel row: rule-based (left) + Claude analyst (right)
                html.Div(
                    [
                        # Left half: rule-based signal
                        html.Div(
                            id="signal-panel",
                            children="Analyzing...",
                            style={
                                "flex": 1,
                                "minWidth": 0,
                                "padding": "10px 14px",
                                "overflowY": "auto",
                                "fontSize": "0.82rem",
                                "borderRight": "1px solid rgba(255,255,255,0.10)",
                            },
                        ),
                        # Right half: live Greek Profile Analysis
                        html.Div(
                            id="profile-panel",
                            children="Analyzing profile...",
                            style={
                                "flex": 1,
                                "minWidth": 0,
                                "padding": "10px 14px",
                                "overflowY": "auto",
                                "fontSize": "0.82rem",
                            },
                        ),
                    ],
                    style={
                        "flex": 0.7,
                        "minHeight": 0,
                        "display": "flex",
                        "backgroundColor": "var(--bg-deep)",
                        "borderTop": "1px solid var(--border-subtle)",
                    },
                ),
            ],
            style={
                "flex": 1.8,
                "minWidth": 0,
                "display": "flex",
                "flexDirection": "column",
            },
        ),
    ],
    style={
        "display": "flex",
        "flex": 1,
        "overflow": "hidden",
    },
)

# -- Metrics header bar (populated by callback) --

metrics_header = html.Div(
    id="metrics-header",
    children="Loading...",
    style={
        "display": "flex",
        "alignItems": "center",
        "padding": "6px 10px",
        "borderBottom": "1px solid var(--border-subtle)",
        "backgroundColor": "var(--bg-deep)",
        "minHeight": "44px",
        "flexShrink": 0,
        "overflowX": "auto",
        "whiteSpace": "nowrap",
        "fontSize": "0.80rem",
    },
)

# -- Greeks Visualizer widget (complete) --

greeks_widget = html.Div(
    id="widget-greeks",
    children=[
        # Header spans the full widget width (sidebar + charts)
        metrics_header,
        # Below header: sidebar on left, charts on right
        html.Div(
            [settings_sidebar, charts_panel],
            style={
                "flex": 1,
                "display": "flex",
                "minHeight": 0,
                "overflow": "hidden",
            },
        ),
    ],
    style={
        "display": "flex",
        "flexDirection": "column",
        "flex": 1,
        "height": "100vh",
        "overflow": "hidden",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
#  MARKET MATRIX WIDGET
# ══════════════════════════════════════════════════════════════════════════════

MATRIX_TICKERS = ["SPX", "SPY", "NDX", "QQQ"]

matrix_settings = html.Div(
    [
        html.H5("Market Matrix", className="mb-3",
                 style={"letterSpacing": "0.05em"}),

        dbc.Label("Greek", className="fw-bold mt-1",
                  style={"fontSize": "0.85rem"}),
        dcc.Dropdown(
            id="matrix-greek",
            options=[
                {"label": "Gamma (GEX)", "value": "gamma_exp"},
                {"label": "Charm",       "value": "charm_exp"},
                {"label": "Vanna",       "value": "vanna_exp"},
                {"label": "DEX (Delta)", "value": "dex_exp"},
                {"label": "Zomma",       "value": "zomma_exp"},
            ],
            value="gamma_exp", clearable=False, className="mb-3",
            style={"color": "#111"},
        ),

        dbc.Label("Exposure Mode", className="fw-bold",
                  style={"fontSize": "0.85rem"}),
        dbc.RadioItems(
            id="matrix-mode",
            options=[
                {"label": "Open Interest",    "value": "oi"},
                {"label": "Session Volume",   "value": "volume"},
                {"label": "Combined (OI+Vol)", "value": "combined"},
            ],
            value="oi",
            className="mb-3",
        ),

        dbc.Label("View", className="fw-bold",
                  style={"fontSize": "0.85rem"}),
        dbc.RadioItems(
            id="matrix-view",
            options=[
                {"label": "Bar",    "value": "bar"},
                {"label": "Values", "value": "values"},
            ],
            value="bar", inline=True, className="mb-3",
        ),

        html.Hr(),

        dbc.Label("Refresh Interval", className="fw-bold",
                  style={"fontSize": "0.85rem"}),
        html.Div(id="matrix-refresh-label", className="text-info mb-1",
                 style={"fontSize": "0.80rem"}),
        dcc.Slider(
            id="matrix-refresh-slider",
            min=10, max=300, step=10,
            value=30,
            marks={10: "10s", 30: "30s", 60: "1m", 120: "2m", 300: "5m"},
            tooltip={"placement": "bottom"},
            className="mb-3",
        ),

        html.Hr(),

        html.Div(id="matrix-status", className="text-muted",
                 style={"fontSize": "0.75rem", "whiteSpace": "pre-line"}),
    ],
    style={
        "width": f"{SIDEBAR_WIDTH}px",
        "minWidth": f"{SIDEBAR_WIDTH}px",
        "height": "100vh",
        "overflowY": "auto",
        "padding": "14px",
        "borderRight": "1px solid var(--border-subtle)",
        "backgroundColor": "var(--bg-deep)",
    },
)

matrix_charts = html.Div(
    [
        html.Div(
            dcc.Graph(id="matrix-chart-SPX", style={"height": "100%"}),
            style={"flex": 1, "minWidth": 0, "minHeight": 0},
        ),
        html.Div(
            dcc.Graph(id="matrix-chart-SPY", style={"height": "100%"}),
            style={"flex": 1, "minWidth": 0, "minHeight": 0},
        ),
        html.Div(
            dcc.Graph(id="matrix-chart-NDX", style={"height": "100%"}),
            style={"flex": 1, "minWidth": 0, "minHeight": 0},
        ),
        html.Div(
            dcc.Graph(id="matrix-chart-QQQ", style={"height": "100%"}),
            style={"flex": 1, "minWidth": 0, "minHeight": 0},
        ),
    ],
    style={
        "flex": 1,
        "display": "flex",
        "height": "100vh",
        "overflow": "hidden",
    },
)

matrix_widget = html.Div(
    id="widget-matrix",
    children=[matrix_settings, matrix_charts],
    style={"display": "none", "flex": 1, "height": "100vh", "overflow": "hidden"},
)

# ══════════════════════════════════════════════════════════════════════════════
#  COT BOARD WIDGET
# ══════════════════════════════════════════════════════════════════════════════

COT_COLUMNS = [
    {"name": "Symbol",          "id": "symbol"},
    {"name": "Long Contracts",  "id": "long",           "type": "numeric", "format": {"specifier": ","}},
    {"name": "Short Contracts", "id": "short",          "type": "numeric", "format": {"specifier": ","}},
    {"name": "Δ Long Contracts","id": "long_change",    "type": "numeric", "format": {"specifier": ","}},
    {"name": "Δ Short Contracts","id": "short_change",  "type": "numeric", "format": {"specifier": ","}},
    {"name": "Long %",          "id": "long_pct_str"},
    {"name": "Short %",         "id": "short_pct_str"},
    {"name": "Net % Change",    "id": "net_pct_str"},
    {"name": "Net Position",    "id": "net_position",   "type": "numeric", "format": {"specifier": ",.0f"}},
    {"name": "Open Interest",   "id": "oi_display"},
    {"name": "Δ Open Interest", "id": "oi_change",      "type": "numeric", "format": {"specifier": ","}},
]

cot_widget = html.Div(
    id="widget-cot",
    children=[
        html.Div(
            [
                html.Div(
                    [
                        html.H4("COT Board", style={"margin": 0, "color": "#fff"}),
                        html.Div(id="cot-meta",
                                 style={"fontSize": "0.80rem", "color": "#aaa",
                                        "marginTop": "2px"}),
                    ],
                    style={"marginBottom": "12px"},
                ),
                dash_table.DataTable(
                    id="cot-table",
                    columns=COT_COLUMNS,
                    data=[],
                    sort_action="native",
                    style_table={
                        "overflowX": "auto",
                        "overflowY": "auto",
                        "height": "calc(100vh - 100px)",
                        "borderRadius": "8px",
                    },
                    style_cell={
                        "backgroundColor": "#1a1e3a",
                        "color": "#cbd5e1",
                        "fontFamily": "Inter, system-ui, sans-serif",
                        "fontSize": "0.85rem",
                        "padding": "10px 14px",
                        "border": "1px solid rgba(255,255,255,0.04)",
                        "textAlign": "center",
                        "whiteSpace": "nowrap",
                    },
                    style_header={
                        "backgroundColor": "#14172a",
                        "color": "#7986b4",
                        "fontWeight": "700",
                        "fontSize": "0.74rem",
                        "borderBottom": "2px solid rgba(91,141,238,0.20)",
                        "textAlign": "center",
                        "padding": "14px 10px",
                        "lineHeight": "1.2",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.04em",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": "symbol"},
                         "textAlign": "left",
                         "fontWeight": "700",
                         "color": "#fff",
                         "minWidth": "100px",
                         "backgroundColor": "#1a1e3a"},
                    ],
                ),
            ],
            style={
                "padding": "16px",
                "width": "100%",
                "height": "100vh",
                "overflow": "hidden",
            },
        ),
    ],
    style={"display": "none", "flex": 1, "height": "100vh", "overflow": "hidden"},
)

# ══════════════════════════════════════════════════════════════════════════════
#  BROAD MARKET STATE WIDGET
# ══════════════════════════════════════════════════════════════════════════════

market_widget = html.Div(
    id="widget-market",
    children=[
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Broad Market State",
                                style={"margin": 0, "color": "#fff"}),
                        html.Div(id="market-meta",
                                 style={"fontSize": "0.80rem", "color": "#aaa",
                                        "marginTop": "2px"}),
                    ],
                    style={"marginBottom": "16px"},
                ),
                html.Div(
                    id="market-grid",
                    children="Loading...",
                    style={
                        "display": "grid",
                        "gridTemplateColumns":
                            "repeat(auto-fill, minmax(220px, 1fr))",
                        "gap": "12px",
                        "alignContent": "start",
                        "overflowY": "auto",
                        "height": "calc(100vh - 90px)",
                        "padding": "4px",
                    },
                ),
            ],
            style={
                "padding": "16px",
                "width": "100%",
                "height": "100vh",
                "overflow": "hidden",
            },
        ),
    ],
    style={"display": "none", "flex": 1, "height": "100vh", "overflow": "hidden"},
)

# ══════════════════════════════════════════════════════════════════════════════
#  HEATMAP WIDGET  (Bookmap-style L2 depth heatmap)
# ══════════════════════════════════════════════════════════════════════════════

# Sidebar settings for heatmap
heatmap_sidebar = html.Div(
    [
        html.H6("Heatmap", style={"color": "var(--text-primary)", "marginBottom": "12px"}),

        dbc.Label("Symbol", className="fw-bold", style={"fontSize": "0.85rem"}),
        dcc.Dropdown(
            id="heatmap-ticker",
            options=[{"label": label, "value": sym}
                     for label, sym, _, _, _ in heatmap_data.DEFAULT_TICKERS],
            value="ES",
            clearable=False,
            className="mb-3",
            style={"color": "#111"},
        ),

        dbc.Label("Color Scheme", className="fw-bold", style={"fontSize": "0.85rem"}),
        dbc.RadioItems(
            id="heatmap-colorscheme",
            options=[
                {"label": "Bookmap (default)", "value": "bookmap"},
                {"label": "Viridis", "value": "viridis"},
                {"label": "Hot", "value": "hot"},
            ],
            value="bookmap",
            className="mb-3",
        ),

        dbc.Label("Trade Trail", className="fw-bold", style={"fontSize": "0.85rem"}),
        dbc.RadioItems(
            id="heatmap-trail",
            options=[
                {"label": "Show", "value": "show"},
                {"label": "Hide", "value": "hide"},
            ],
            value="show",
            inline=True,
            className="mb-3",
        ),

        dbc.Label("Depth Levels", className="fw-bold", style={"fontSize": "0.85rem"}),
        dcc.Slider(
            id="heatmap-depth-slider",
            min=5, max=25, step=5, value=25,
            marks={5: "5", 10: "10", 25: "25"},
            tooltip={"placement": "bottom", "always_visible": False},
            className="mb-3",
        ),

        html.Button(
            "↻ Reset Zoom",
            id="heatmap-reset-zoom",
            n_clicks=0,
            style={
                "width": "100%",
                "padding": "8px 14px",
                "border": "1px solid var(--accent-blue)",
                "borderRadius": "8px",
                "backgroundColor": "rgba(91,141,238,0.10)",
                "color": "var(--accent-blue)",
                "cursor": "pointer",
                "fontSize": "0.82rem",
                "fontWeight": "600",
                "letterSpacing": "0.02em",
                "marginBottom": "12px",
            },
        ),

        dcc.Store(id="heatmap-uirevision", data={"key": "heatmap-v6-x-follows"}),
        dcc.Store(id="heatmap-last-applied-uirev",
                  data={"key": "", "range": None}),

        html.Hr(),
        html.Div(id="heatmap-status",
                 style={"fontSize": "0.75rem", "color": "var(--text-muted)",
                        "whiteSpace": "pre-line"}),
    ],
    style={
        "width": f"{SIDEBAR_WIDTH}px",
        "minWidth": f"{SIDEBAR_WIDTH}px",
        "height": "100vh",
        "overflowY": "auto",
        "padding": "14px",
        "borderRight": "1px solid var(--border-subtle)",
        "backgroundColor": "var(--bg-deep)",
    },
)

heatmap_widget = html.Div(
    id="widget-heatmap",
    children=[
        heatmap_sidebar,
        html.Div(
            dcc.Graph(
                id="heatmap-chart",
                style={"height": "100%"},
                config={"displayModeBar": False},
            ),
            style={"flex": 1, "minWidth": 0, "padding": "8px"},
        ),
    ],
    style={"display": "none", "flex": 1, "height": "100vh", "overflow": "hidden"},
)

# -- Widget content area (shows the active widget) --

widget_content = html.Div(
    id="widget-content",
    children=[greeks_widget, matrix_widget, cot_widget, market_widget, heatmap_widget],
    style={"flex": 1, "display": "flex", "overflow": "hidden"},
)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

poll_timer = dcc.Interval(id="interval-poll", interval=2_000, n_intervals=0)

app.layout = html.Div(
    [
        poll_timer,
        nav_store,
        matrix_prev_store,
        nav_sidebar,
        widget_content,
    ],
    style={"display": "flex", "height": "100vh", "overflow": "hidden"},
)


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# ── Nav toggle (collapse / expand) ──────────────────────────────────────────

@app.callback(
    Output("nav-sidebar", "style"),
    Output("store-nav", "data"),
    Input("nav-toggle", "n_clicks"),
    State("store-nav", "data"),
    prevent_initial_call=True,
)
def toggle_nav(n, nav_data):
    is_open = nav_data.get("open", True)
    new_open = not is_open

    if new_open:
        style = {
            "width": f"{NAV_WIDTH_OPEN}px",
            "minWidth": f"{NAV_WIDTH_OPEN}px",
            "height": "100vh",
            "overflowY": "auto",
            "overflowX": "hidden",
            "borderRight": "1px solid rgba(255,255,255,0.10)",
            "backgroundColor": "rgba(0,0,0,0.4)",
            "transition": "width 0.2s, min-width 0.2s",
            "flexShrink": 0,
        }
    else:
        style = {
            "width": f"{NAV_WIDTH_CLOSED}px",
            "minWidth": f"{NAV_WIDTH_CLOSED}px",
            "height": "100vh",
            "overflowY": "auto",
            "overflowX": "hidden",
            "borderRight": "1px solid rgba(255,255,255,0.10)",
            "backgroundColor": "rgba(0,0,0,0.4)",
            "transition": "width 0.2s, min-width 0.2s",
            "flexShrink": 0,
        }

    nav_data["open"] = new_open
    return style, nav_data


# ── Nav button click → switch active widget ─────────────────────────────────

@app.callback(
    Output("widget-greeks", "style"),
    Output("widget-matrix", "style"),
    Output("widget-cot", "style"),
    Output("widget-market", "style"),
    Output("widget-heatmap", "style"),
    Output("nav-buttons", "children"),
    Input({"type": "nav-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def switch_widget(n_clicks_list):
    """Show/hide widgets based on which nav button was clicked."""
    triggered = ctx.triggered_id
    if triggered is None:
        raise dash.exceptions.PreventUpdate

    active = triggered.get("index", "greeks") if isinstance(triggered, dict) else "greeks"

    # Greeks widget needs flexDirection: column (header on top, content below).
    # Others use the default row layout.
    greeks_style = {
        "flex": 1,
        "height": "100vh",
        "overflow": "hidden",
        "flexDirection": "column",
        "display": "flex" if active == "greeks" else "none",
    }
    matrix_style = {
        "flex": 1,
        "height": "100vh",
        "overflow": "hidden",
        "display": "flex" if active == "matrix" else "none",
    }
    cot_style = {
        "flex": 1,
        "height": "100vh",
        "overflow": "hidden",
        "display": "flex" if active == "cot" else "none",
    }
    market_style = {
        "flex": 1,
        "height": "100vh",
        "overflow": "hidden",
        "display": "flex" if active == "market" else "none",
    }
    heatmap_style = {
        "flex": 1,
        "height": "100vh",
        "overflow": "hidden",
        "display": "flex" if active == "heatmap" else "none",
    }

    # Rebuild nav buttons with updated active highlight
    nav_buttons = _build_nav_buttons(active_id=active)

    return greeks_style, matrix_style, cot_style, market_style, heatmap_style, nav_buttons


# ── Market Matrix refresh slider ─────────────────────────────────────────────

@app.callback(
    Output("matrix-refresh-label", "children"),
    Input("matrix-refresh-slider", "value"),
)
def on_matrix_refresh_change(val):
    if matrix_data.matrix_manager:
        matrix_data.matrix_manager._refresh = val
    if val >= 60:
        return f"{val // 60}m {val % 60}s" if val % 60 else f"{val // 60}m"
    return f"{val}s"


# ── Market Matrix poll callback ──────────────────────────────────────────────

MATRIX_GREEK_LABELS = {
    "gamma_exp": "Gamma (GEX)",
    "charm_exp": "Charm",
    "vanna_exp": "Vanna",
    "dex_exp":   "DEX (Delta)",
    "zomma_exp": "Zomma",
}

MATRIX_GREEK_COLORS = {
    "gamma_exp": ("#00d4aa", "#ff4d6a"),
    "charm_exp": ("#4dabf7", "#f783ac"),
    "vanna_exp": ("#a78bfa", "#fb923c"),
    "dex_exp":   ("#26c6da", "#ff7043"),
    "zomma_exp": ("#66bb6a", "#ef5350"),
}

MATRIX_Y_DTICK = {
    "SPX": 5,
    "SPY": 1,
    "NDX": 10,
    "QQQ": 1,
}

@app.callback(
    Output("matrix-chart-SPX", "figure"),
    Output("matrix-chart-SPY", "figure"),
    Output("matrix-chart-NDX", "figure"),
    Output("matrix-chart-QQQ", "figure"),
    Output("matrix-status", "children"),
    Output("store-matrix-prev", "data"),
    Input("interval-poll", "n_intervals"),
    Input("matrix-greek", "value"),
    Input("matrix-mode", "value"),
    Input("matrix-view", "value"),
    State("store-matrix-prev", "data"),
)
def poll_matrix(n, greek_col, mode, view, prev_data):
    if not matrix_data.matrix_manager:
        e = _empty_fig("Starting Matrix...")
        return e, e, e, e, "Initialising...", dash.no_update

    caches = matrix_data.matrix_manager.get_all_caches()
    figures = []
    status_lines = []

    if prev_data is None:
        prev_data = {}

    # Build new prev store structure per ticker
    new_prev = {"_greek": greek_col, "_mode": mode}

    # Reset prev tracking if greek or mode changed
    greek_or_mode_changed = (prev_data.get("_greek") != greek_col or
                              prev_data.get("_mode") != mode)

    for ticker in MATRIX_TICKERS:
        cache = caches.get(ticker, {})
        error = cache.get("error")

        if error:
            figures.append(_empty_fig(f"{ticker}: {error}"))
            status_lines.append(f"{ticker}: {error}")
            continue

        # Pick the right pre-computed exposure for the selected mode
        mode_key = {"oi": "exp_oi", "volume": "exp_vol",
                    "combined": "exp_combined"}.get(mode, "exp_oi")
        exp_df = cache.get(mode_key)
        spot = cache.get("spot", 0)
        expiry = cache.get("expiry", "")

        if exp_df is None or exp_df.empty:
            figures.append(_empty_fig(f"{ticker}: No data"))
            status_lines.append(f"{ticker}: No data")
            continue

        # Format expiry for title
        if len(expiry) == 8:
            nice_exp = f"{expiry[4:6]}/{expiry[6:]}"
        else:
            nice_exp = expiry

        label = MATRIX_GREEK_LABELS.get(greek_col, greek_col)
        colors = MATRIX_GREEK_COLORS.get(greek_col, ("#00d4aa", "#ff4d6a"))
        title = f"{ticker}  {label}  ({nice_exp})"

        # Track per-ticker previous values for delta display
        cur_snap = {str(r["strike"]): r[greek_col] for _, r in exp_df.iterrows()}
        ts = cache.get("timestamp", 0)

        prev_ticker = prev_data.get(ticker, {})
        prev_ts = prev_ticker.get("_ts", 0)

        if greek_or_mode_changed:
            dp = {}
            new_prev[ticker] = {
                "_ts":          ts,
                "display_prev": {},
                "current":      cur_snap,
            }
        elif ts != prev_ts:
            # New fetch — rotate
            dp = prev_ticker.get("current", {})
            new_prev[ticker] = {
                "_ts":          ts,
                "display_prev": dp,
                "current":      cur_snap,
            }
        else:
            dp = prev_ticker.get("display_prev", {})
            new_prev[ticker] = prev_ticker  # no change

        if view == "values":
            fig = _build_value_view(
                exp_df, "strike", greek_col,
                title, spot, dp,
                compact=True,
                ultra_compact=True,
            )
        else:
            fig = _build_chart(
                exp_df, "strike", greek_col,
                title, spot, colors[0], colors[1],
                lines=[
                    {"type": "exposure_max", "label": "Max Pos",
                     "color": colors[0], "side": "left"},
                    {"type": "exposure_min", "label": "Max Neg",
                     "color": colors[1], "side": "left"},
                ],
                compact=True,
                y_dtick=MATRIX_Y_DTICK.get(ticker),
            )
        figures.append(fig)

        t_str = dt.datetime.fromtimestamp(ts, tz=ET).strftime("%H:%M:%S") if ts else "?"
        status_lines.append(f"{ticker}: ${spot:,.1f} ({nice_exp}) @ {t_str}")

    status = "\n".join(status_lines)
    return figures[0], figures[1], figures[2], figures[3], status, new_prev



# ── Broad Market State poll callback ─────────────────────────────────────────

def _market_card(row):
    """Render one ETF card with its bullishness score visualized as a gauge bar."""
    ticker = row.get("ticker", "?")
    name = row.get("name", "")
    error = row.get("error")

    if error:
        return html.Div([
            html.Div(ticker, style={"fontSize": "1.05rem", "fontWeight": "700",
                                     "color": "#fff"}),
            html.Div(name, style={"fontSize": "0.72rem", "color": "#888",
                                   "marginBottom": "8px"}),
            html.Div(f"Error: {error}",
                     style={"fontSize": "0.72rem", "color": "#ef5350"}),
        ], style={
            "padding": "14px 16px",
            "border": "1px solid var(--border-subtle)",
            "borderRadius": "8px",
            "backgroundColor": "var(--bg-card)",
        })

    spot = row.get("spot", 0)
    pct = row.get("pct_change", 0)
    flow = row.get("flow_ratio", 0)
    score = row.get("score", 50)
    label = row.get("label", "NEUTRAL")
    vol_ratio = row.get("vol_ratio", 0)
    is_composite = row.get("is_composite", False)
    constituents = row.get("constituents", [])

    # Color gradient for the score
    if score >= 75:
        color = "#22c55e"   # strong green
    elif score >= 60:
        color = "#84cc16"   # green
    elif score >= 45:
        color = "#94a3b8"   # neutral grey
    elif score >= 30:
        color = "#f97316"   # orange
    else:
        color = "#ef4444"   # red

    pct_color = "#66bb6a" if pct >= 0 else "#ef5350"
    flow_color = "#66bb6a" if flow >= 0 else "#ef5350"

    # Vol ratio color
    if vol_ratio >= 1.3:
        vol_color = "#facc15"   # high conviction
    elif vol_ratio >= 0.8:
        vol_color = "#bbb"
    else:
        vol_color = "#666"      # low conviction

    return html.Div([
        # Header row
        html.Div([
            html.Span(ticker, style={"fontSize": "1.10rem", "fontWeight": "700",
                                      "color": "#fbbf24" if is_composite else "#fff",
                                      "letterSpacing": "0.02em"}),
            html.Span(
                "COMPOSITE" if is_composite else f"${spot:,.2f}",
                style={"marginLeft": "auto",
                       "fontSize": "0.65rem" if is_composite else "0.85rem",
                       "color": "#fbbf24" if is_composite else "#bbb",
                       "fontFamily": "monospace",
                       "letterSpacing": "0.10em" if is_composite else "0",
                       "fontWeight": "700" if is_composite else "400",
                       "padding": "2px 6px" if is_composite else "0",
                       "border": "1px solid #fbbf24" if is_composite else "none",
                       "borderRadius": "3px" if is_composite else "0"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div(name, style={"fontSize": "0.70rem", "color": "#888",
                               "marginBottom": "10px",
                               "textTransform": "uppercase",
                               "letterSpacing": "0.06em"}),

        # Score gauge
        html.Div([
            html.Div(
                style={
                    "position": "absolute",
                    "left": "0", "top": "0", "bottom": "0",
                    "width": f"{score:.0f}%",
                    "backgroundColor": color,
                    "transition": "width 0.5s ease",
                    "borderRadius": "4px",
                },
            ),
            # Center marker (50 = neutral)
            html.Div(
                style={
                    "position": "absolute",
                    "left": "50%", "top": "0", "bottom": "0",
                    "width": "1px",
                    "backgroundColor": "rgba(255,255,255,0.30)",
                },
            ),
        ], style={
            "position": "relative",
            "height": "10px",
            "backgroundColor": "rgba(255,255,255,0.06)",
            "borderRadius": "4px",
            "overflow": "hidden",
            "marginBottom": "8px",
        }),

        # Score / label row
        html.Div([
            html.Span(f"{score:.0f}",
                      style={"fontSize": "1.40rem", "fontWeight": "700",
                             "color": color, "fontFamily": "monospace"}),
            html.Span("/100", style={"fontSize": "0.72rem", "color": "#666",
                                       "marginLeft": "3px",
                                       "fontFamily": "monospace"}),
            html.Span(label,
                      style={"marginLeft": "auto",
                             "fontSize": "0.70rem", "fontWeight": "700",
                             "color": color, "letterSpacing": "0.05em"}),
        ], style={"display": "flex", "alignItems": "baseline",
                  "marginBottom": "8px"}),

        # Stats row 1: Day change / Flow / MA stack
        html.Div([
            html.Div([
                html.Div("CHG",
                         style={"fontSize": "0.62rem", "color": "#888",
                                "letterSpacing": "0.08em"}),
                html.Div(f"{pct:+.2f}%",
                         style={"fontSize": "0.85rem",
                                "color": pct_color,
                                "fontFamily": "monospace",
                                "fontWeight": "600"}),
            ], style={"flex": 1}),
            html.Div([
                html.Div("FLOW",
                         style={"fontSize": "0.62rem", "color": "#888",
                                "letterSpacing": "0.08em"}),
                html.Div(f"{flow:+.2f}",
                         style={"fontSize": "0.85rem",
                                "color": flow_color,
                                "fontFamily": "monospace",
                                "fontWeight": "600"}),
            ], style={"flex": 1, "borderLeft": "1px solid rgba(255,255,255,0.06)",
                       "paddingLeft": "10px"}),
            html.Div([
                html.Div("VOL vs 20D",
                         style={"fontSize": "0.62rem", "color": "#888",
                                "letterSpacing": "0.08em"}),
                html.Div(f"{vol_ratio:.2f}x",
                         style={"fontSize": "0.85rem",
                                "color": vol_color,
                                "fontFamily": "monospace",
                                "fontWeight": "600"}),
            ], style={"flex": 1, "borderLeft": "1px solid rgba(255,255,255,0.06)",
                       "paddingLeft": "10px"}),
        ], style={"display": "flex"}),

        # Constituent strip for composites
        html.Div(
            f"Constituents: {', '.join(constituents)}" if (is_composite and constituents) else "",
            style={
                "marginTop": "10px",
                "paddingTop": "8px",
                "borderTop": "1px solid rgba(251,191,36,0.15)",
                "fontSize": "0.62rem",
                "color": "#888",
                "letterSpacing": "0.04em",
                "display": "block" if (is_composite and constituents) else "none",
            },
        ),

    ], style={
        "padding": "14px 16px",
        "border": ("1px solid rgba(251,191,36,0.40)" if is_composite
                   else "1px solid var(--border-subtle)"),
        "borderRadius": "8px",
        "backgroundColor": ("#2a2410" if is_composite else "var(--bg-card)"),
    })


@app.callback(
    Output("market-grid", "children"),
    Output("market-meta", "children"),
    Input("interval-poll", "n_intervals"),
)
def poll_market_state(n):
    if not market_state.market_state_manager:
        return "Market state manager not initialized", ""

    cache = market_state.market_state_manager.get_cache()
    rows = cache.get("rows", [])
    error = cache.get("error")
    fetched_at = cache.get("fetched_at", 0)
    data_source = cache.get("data_source", "?")

    if not rows:
        return error or "Loading...", ""

    # Compute aggregate market state — average score
    valid = [r for r in rows if not r.get("error")]
    if valid:
        avg_score = sum(r["score"] for r in valid) / len(valid)
        bull_count = sum(1 for r in valid if r["score"] >= 60)
        bear_count = sum(1 for r in valid if r["score"] < 40)
    else:
        avg_score = 50
        bull_count = bear_count = 0

    if fetched_at:
        ts = dt.datetime.fromtimestamp(fetched_at, tz=ET).strftime("%H:%M:%S ET")
    else:
        ts = "—"

    meta_parts = []

    # Data source banner — RED if mock, green if live IB
    if data_source == "MOCK":
        meta_parts.append(
            html.Span(
                "⚠ MOCK DATA — IB CONNECTION FAILED",
                style={
                    "padding": "3px 10px",
                    "borderRadius": "6px",
                    "backgroundColor": "#7f1d1d",
                    "color": "#fecaca",
                    "fontSize": "0.72rem",
                    "fontWeight": "700",
                    "letterSpacing": "0.05em",
                    "marginRight": "12px",
                },
            )
        )
    elif data_source == "IB":
        meta_parts.append(
            html.Span(
                "● LIVE",
                style={
                    "padding": "3px 10px",
                    "borderRadius": "6px",
                    "backgroundColor": "rgba(34,197,94,0.15)",
                    "color": "#22c55e",
                    "fontSize": "0.72rem",
                    "fontWeight": "700",
                    "letterSpacing": "0.05em",
                    "marginRight": "12px",
                },
            )
        )

    meta_parts.append(
        f"Composite: {avg_score:.0f}/100  |  "
        f"Bullish: {bull_count}/{len(valid)}  |  "
        f"Bearish: {bear_count}/{len(valid)}  |  "
        f"Last fetch: {ts}"
    )

    cards = [_market_card(r) for r in rows]

    return cards, meta_parts


# ── Heatmap poll callback ───────────────────────────────────────────────────

# Bookmap-like color scheme.
# Heavily biased toward the cool half: the bottom ~95% of the gradient is
# all dark navy → blue → sky blue, so the bulk of order-book cells render
# dark. Only the top ~5% reaches yellow → orange → red for true outliers.
BOOKMAP_COLORSCALE = [
    [0.00, "rgba(12, 22, 55, 1.0)"],     # deep navy (matches background)
    [0.40, "rgba(18, 32, 80, 1.0)"],     # navy — covers most cells
    [0.65, "rgba(22, 45, 115, 1.0)"],    # darker blue — typical depth
    [0.85, "rgba(30, 75, 175, 1.0)"],    # blue — meaningfully large
    [0.95, "rgba(60, 170, 230, 1.0)"],   # sky blue — notable
    [0.97, "rgba(255, 235, 110, 1.0)"],  # yellow — large
    [0.99, "rgba(240, 140, 40, 1.0)"],   # orange — very large
    [1.00, "rgba(180, 25, 25, 1.0)"],    # dark red — extreme
]


def _build_heatmap_figure(state, color_scheme="bookmap", show_trail=True,
                            uirevision_key="heatmap-default",
                            apply_default_range=True,
                            last_applied_range=None):
    """Build the Plotly heatmap figure from manager state.

    Args:
      apply_default_range: When True, the figure uses a freshly-computed
        range fitted to recent data. When False, falls back to
        `last_applied_range` (so the range value doesn't change between
        polls, which keeps uirevision happy and preserves user zoom).
      last_applied_range: Dict {"x": (lo, hi), "y": (lo, hi)} from the
        previous render. Used when apply_default_range is False.
    """
    snapshots = state.get("snapshots", [])
    trades = state.get("trades", [])
    last_price = state.get("last_price", 0)
    best_bid = state.get("best_bid", 0)
    best_ask = state.get("best_ask", 0)

    if not snapshots:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0a1230",
            font={"color": "#cbd5e1"},
            annotations=[{
                "text": "Waiting for depth data...",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5, "showarrow": False,
                "font": {"size": 16, "color": "#7986b4"},
            }],
        )
        return empty_fig

    # ── Build a uniform price grid across the visible band ─────────
    # Find anchor spot (use latest snapshot's spot)
    latest = snapshots[-1]
    spot_now = latest.get("spot", last_price)
    if not spot_now or spot_now <= 0:
        spot_now = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 100

    # Determine tick size: use the gap between top-of-book bid prices
    # if the contract has visible depth, else fall back to 1% of spot / 100
    tick_size = 0.25  # default to 0.25 (ES tick)
    if latest["bids"] and len(latest["bids"]) >= 2:
        diffs = sorted({round(latest["bids"][i][0] - latest["bids"][i+1][0], 4)
                        for i in range(len(latest["bids"]) - 1)
                        if latest["bids"][i][0] > latest["bids"][i+1][0]})
        if diffs:
            tick_size = max(diffs[0], 0.01)

    # Build price grid: ±0.5% band centered on spot, in tick-size steps
    band_dollars = max(spot_now * 0.005, tick_size * 30)
    n_levels_each_side = int(band_dollars / tick_size)
    grid_low  = spot_now - n_levels_each_side * tick_size
    grid_high = spot_now + n_levels_each_side * tick_size
    n_rows = int(round((grid_high - grid_low) / tick_size)) + 1
    price_grid = np.linspace(grid_low, grid_high, n_rows)

    # ── Build the 2D matrix: rows=price, cols=time ─────────────────
    n_cols = len(snapshots)
    matrix = np.zeros((n_rows, n_cols), dtype=float)

    for ci, snap in enumerate(snapshots):
        # For each side of the book, accumulate sizes into nearest grid cell
        for px, sz in (snap["bids"] + snap["asks"]):
            if px <= 0 or sz <= 0:
                continue
            ri = int(round((px - grid_low) / tick_size))
            if 0 <= ri < n_rows:
                matrix[ri, ci] += sz

    # Apply log compression — order book sizes are heavily right-skewed
    matrix_disp = np.log1p(matrix)

    # Anchor the color range to a high percentile rather than the max.
    # 99th percentile = only the top 1% of cells saturate at red/orange.
    # This is the key knob: lower → more warm colors, higher → more navy.
    nonzero = matrix_disp[matrix_disp > 0]
    if len(nonzero) > 50:
        z_max = float(np.percentile(nonzero, 99))
    elif len(nonzero) > 0:
        z_max = float(nonzero.max())
    else:
        z_max = 1.0
    if z_max <= 0:
        z_max = 1.0

    # X axis: snapshot timestamps converted to local time strings
    times = [dt.datetime.fromtimestamp(s["ts"], tz=ET) for s in snapshots]

    # Pick colorscale
    if color_scheme == "bookmap":
        colorscale = BOOKMAP_COLORSCALE
    elif color_scheme == "viridis":
        colorscale = "Viridis"
    else:   # "hot"
        colorscale = "Hot"

    # ── Use subplots so the heatmap has a depth-bar sidekick on the right ──
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.92, 0.08],
        horizontal_spacing=0.005,
        shared_yaxes=True,
        specs=[[{"type": "heatmap"}, {"type": "bar"}]],
    )

    # ── Main heatmap (left, ~92% of width) ──
    fig.add_trace(go.Heatmap(
        z=matrix_disp,
        x=times,
        y=price_grid,
        colorscale=colorscale,
        showscale=False,
        zmin=0,
        zmax=z_max,
        hovertemplate=(
            "Time: %{x|%H:%M:%S}<br>"
            "Price: $%{y:,.2f}<br>"
            "Size (log): %{z:.2f}<extra></extra>"
        ),
        zsmooth=False,
    ), row=1, col=1)

    # ── Current-snapshot depth bars (right, ~8% of width) ──
    # Show the latest order book as horizontal bars per price level.
    # Bid side = green-ish, Ask side = red-ish, length proportional to size.
    latest_bids = latest.get("bids", [])
    latest_asks = latest.get("asks", [])
    bar_prices, bar_sizes, bar_colors = [], [], []
    for px, sz in latest_bids:
        if px > 0 and sz > 0:
            bar_prices.append(px)
            bar_sizes.append(sz)
            bar_colors.append("rgba(34, 197, 94, 0.75)")    # bid = green
    for px, sz in latest_asks:
        if px > 0 and sz > 0:
            bar_prices.append(px)
            bar_sizes.append(sz)
            bar_colors.append("rgba(239, 68, 68, 0.75)")    # ask = red

    if bar_prices:
        fig.add_trace(go.Bar(
            x=bar_sizes,
            y=bar_prices,
            orientation="h",
            marker={"color": bar_colors, "line": {"width": 0}},
            hovertemplate="Price: $%{y:,.2f}<br>Size: %{x:,}<extra></extra>",
            showlegend=False,
            width=tick_size * 0.9,    # bar height = ~one price tick
        ), row=1, col=2)

    # ── Trade trail overlay ─────────────────────────────────────────
    if show_trail and trades:
        trade_times = [dt.datetime.fromtimestamp(t["ts"], tz=ET) for t in trades]
        trade_prices = [t["price"] for t in trades]
        trade_sizes = [t["size"] for t in trades]
        trade_colors = ["#22c55e" if t["side"] == "buy" else "#ef4444"
                        for t in trades]

        # Marker size scaled by log of trade size (so big prints stand out)
        marker_sizes = [max(3, min(20, np.log1p(s) * 2.5)) for s in trade_sizes]

        fig.add_trace(go.Scatter(
            x=trade_times,
            y=trade_prices,
            mode="markers",
            marker={
                "size": marker_sizes,
                "color": trade_colors,
                "opacity": 0.7,
                "line": {"width": 0},
            },
            customdata=list(zip(trade_sizes,
                                 [t["side"].upper() for t in trades])),
            hovertemplate=(
                "Time: %{x|%H:%M:%S}<br>"
                "Price: $%{y:,.2f}<br>"
                "Size: %{customdata[0]}<br>"
                "Side: %{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        ), row=1, col=1)

        # Last-price line (the actual trade trail)
        last_times = [dt.datetime.fromtimestamp(s["ts"], tz=ET)
                      for s in snapshots if s.get("spot", 0) > 0]
        last_prices = [s["spot"] for s in snapshots if s.get("spot", 0) > 0]
        if last_prices:
            fig.add_trace(go.Scatter(
                x=last_times,
                y=last_prices,
                mode="lines",
                line={"color": "#fbbf24", "width": 1.2},
                hoverinfo="skip",
                showlegend=False,
            ), row=1, col=1)

    # ── Layout polish ───────────────────────────────────────────────
    # Bookmap's signature dark navy background — not pure black.
    BOOKMAP_BG = "#0a1230"

    # Default y-range: derived from the ACTUAL spread of prices in recent
    # snapshots. We look at the last 60 snapshots (~1 min of data), find the
    # full range of bid/ask prices that appeared, then pad by 15% so the band
    # has breathing room without dwarfing the action.
    recent_window = snapshots[-60:] if len(snapshots) >= 60 else snapshots
    recent_prices = []
    for snap in recent_window:
        for px, sz in (snap.get("bids", []) + snap.get("asks", [])):
            if px > 0 and sz > 0:
                recent_prices.append(px)

    if recent_prices:
        data_low = min(recent_prices)
        data_high = max(recent_prices)
        data_span = data_high - data_low
        # 15% breathing room each side, with a $2 floor so very tight books
        # still have a usable view
        pad = max(data_span * 0.15, 2.0)
        y_default_low  = data_low - pad
        y_default_high = data_high + pad
    else:
        # No data yet — fall back to ±$5 around spot
        y_default_low  = spot_now - 5.0
        y_default_high = spot_now + 5.0

    # Default x-range: last 10 minutes. The data buffer holds 30 minutes
    # of history, so the user can pan/zoom backward to see the full window.
    if times:
        x_default_high = times[-1]
        x_default_low  = x_default_high - dt.timedelta(minutes=10)
    else:
        x_default_high = dt.datetime.now(tz=ET)
        x_default_low  = x_default_high - dt.timedelta(minutes=10)

    fig.update_layout(
        paper_bgcolor=BOOKMAP_BG,
        plot_bgcolor=BOOKMAP_BG,
        font={"color": "#cbd5e1", "family": "Inter"},
        margin={"l": 50, "r": 60, "t": 8, "b": 30},
        # uirevision preserves user zoom/pan between updates.
        uirevision=uirevision_key,
        bargap=0,
    )

    # Heatmap axes (col 1)
    # X-axis: ALWAYS recompute fresh (so the chart slides forward in time
    # as new snapshots arrive). uirevision still preserves user zoom on x.
    # Y-axis: use stored range on subsequent polls so the value doesn't shift
    # underneath the user's preserved zoom.
    if not apply_default_range and last_applied_range:
        def _from_storable(v):
            if isinstance(v, str):
                try:
                    return dt.datetime.fromisoformat(v)
                except ValueError:
                    return v
            return v

        y_raw = last_applied_range.get("y", (y_default_low, y_default_high))
        y_lo, y_hi = _from_storable(y_raw[0]), _from_storable(y_raw[1])
    else:
        y_lo, y_hi = y_default_low, y_default_high

    # x range is always the freshly-computed "last 10 minutes" window
    x_lo, x_hi = x_default_low, x_default_high

    fig.update_xaxes(
        row=1, col=1,
        showgrid=False,
        color="#7986b4",
        tickformat="%H:%M:%S",
        gridcolor="rgba(255,255,255,0.04)",
        range=[x_lo, x_hi],
        autorange=False,
    )
    fig.update_yaxes(
        row=1, col=1,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.04)",
        color="#7986b4",
        tickformat=",.2f",
        range=[y_lo, y_hi],
        autorange=False,
    )

    # Depth-bar axes (col 2) — minimal, no labels, just the bars
    fig.update_xaxes(
        row=1, col=2,
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        color="#7986b4",
    )
    fig.update_yaxes(
        row=1, col=2,
        showgrid=False,
        showticklabels=True,
        color="#7986b4",
        tickformat=",.2f",
        side="right",
        zeroline=False,
        range=[y_lo, y_hi],
        autorange=False,
    )

    # Stash only the y-range we rendered. X is always recomputed so we
    # don't need to persist it.
    def _to_storable(v):
        if isinstance(v, dt.datetime):
            return v.isoformat()
        return v

    fig._applied_range = {
        "y": (_to_storable(y_lo), _to_storable(y_hi)),
    }

    return fig


@app.callback(
    Output("heatmap-uirevision", "data"),
    Input("heatmap-reset-zoom", "n_clicks"),
    Input("heatmap-ticker", "value"),
    State("heatmap-uirevision", "data"),
    prevent_initial_call=True,
)
def update_heatmap_uirevision(reset_clicks, ticker, current):
    """
    Bumps the uirevision key whenever the user clicks 'Reset Zoom' or
    switches tickers. A new key forces Plotly to reset zoom/pan state;
    keeping the same key preserves the user's current view across polls.
    The depth-level slider does NOT trigger a reset — it only changes
    resolution within the same contract.
    """
    triggered = ctx.triggered_id
    current = current or {"key": "heatmap-default"}

    if triggered == "heatmap-reset-zoom":
        prev_count = 0
        try:
            prev_count = int(current.get("key", "0").rsplit("-", 1)[-1])
        except (ValueError, AttributeError):
            prev_count = 0
        return {"key": f"heatmap-reset-{prev_count + 1}"}
    elif triggered == "heatmap-ticker":
        return {"key": f"heatmap-{ticker}"}
    return current


@app.callback(
    Output("heatmap-chart", "figure"),
    Output("heatmap-status", "children"),
    Output("heatmap-last-applied-uirev", "data"),
    Input("interval-poll", "n_intervals"),
    Input("heatmap-ticker", "value"),
    Input("heatmap-colorscheme", "value"),
    Input("heatmap-trail", "value"),
    Input("heatmap-uirevision", "data"),
    Input("heatmap-depth-slider", "value"),
    State("heatmap-last-applied-uirev", "data"),
)
def poll_heatmap(n, ticker, color_scheme, trail_mode, uirev, depth_levels,
                  last_applied):
    if not heatmap_data.heatmap_manager:
        return go.Figure(), "Heatmap manager not initialized", dash.no_update

    # Forward user controls to the manager
    if ticker:
        heatmap_data.heatmap_manager.set_ticker(ticker)
    if depth_levels:
        heatmap_data.heatmap_manager.set_depth_levels(depth_levels)

    state = heatmap_data.heatmap_manager.get_state()
    show_trail = trail_mode == "show"
    uirev_key = (uirev or {}).get("key", "heatmap-default")
    last_key = (last_applied or {}).get("key", "")
    last_range_data = (last_applied or {}).get("range")

    # Only apply the explicit default range on a FRESH render (uirevision
    # just changed — i.e. first load, Reset Zoom, or ticker change).
    # On subsequent polls, reuse the previously-applied range so Plotly's
    # uirevision can preserve user pan/zoom without our explicit range
    # values shifting underneath it.
    apply_default_range = (uirev_key != last_key)

    fig = _build_heatmap_figure(state, color_scheme=color_scheme,
                                  show_trail=show_trail,
                                  uirevision_key=uirev_key,
                                  apply_default_range=apply_default_range,
                                  last_applied_range=last_range_data)

    # Build status panel
    src = state.get("data_source", "?")
    src_badge = ("⚠ MOCK DATA" if src == "MOCK" else "● LIVE")
    src_color = ("#ef4444" if src == "MOCK" else "#22c55e")

    levels_active = state.get("depth_levels", 0)
    levels_requested = depth_levels if depth_levels else "?"
    status_lines = [
        f"Contract: {state.get('contract', '—')}",
        f"Depth:    {levels_active} active  (requested: {levels_requested})",
        f"Best bid: ${state.get('best_bid', 0):,.2f}",
        f"Best ask: ${state.get('best_ask', 0):,.2f}",
        f"Last:     ${state.get('last_price', 0):,.2f}",
        f"Snapshots: {len(state.get('snapshots', []))}",
        f"",
        f"{state.get('status', '')}",
    ]
    if state.get("error"):
        status_lines.append(f"Error: {state['error']}")

    status_text = html.Div([
        html.Div(src_badge,
                 style={"color": src_color, "fontWeight": "700",
                        "marginBottom": "8px"}),
        html.Div("\n".join(status_lines),
                 style={"whiteSpace": "pre-line"}),
    ])

    # On fresh renders, persist the range we just used. On subsequent polls,
    # leave the store untouched so the next poll continues using the same
    # range (which is what keeps uirevision honoring user zoom).
    if apply_default_range:
        applied = getattr(fig, "_applied_range", None)
        new_last_applied = {"key": uirev_key, "range": applied}
    else:
        new_last_applied = dash.no_update

    return fig, status_text, new_last_applied


# ── COT Board poll callback ──────────────────────────────────────────────────

def _format_oi_short(n):
    """Format OI like '1986K' (matches reference design)."""
    if n is None:
        return "—"
    n = int(n)
    if abs(n) >= 1000:
        return f"{n // 1000:,}K"
    return f"{n:,}"


def _heat_color(value, max_abs, positive_is_bullish=True):
    """
    Return a background color for a cell based on its value's magnitude.
    Blue for bullish, red for bearish, intensity = |value| / max_abs.
    Returns "" for ~zero values (transparent).
    """
    if max_abs <= 0 or value is None or abs(value) < 1e-9:
        return "transparent"
    intensity = min(abs(value) / max_abs, 1.0)
    # Use squared-falloff curve so only the biggest values look saturated
    alpha = 0.15 + 0.55 * (intensity ** 0.6)
    is_bullish = (value > 0) if positive_is_bullish else (value < 0)
    if is_bullish:
        return f"rgba(64, 110, 220, {alpha:.2f})"   # blue
    else:
        return f"rgba(220, 60, 80, {alpha:.2f})"     # red


def _build_cot_row_styles(rows):
    """
    Build per-cell background color rules for each row.
    Returns a list of style_data_conditional entries.

    Heatmap rules:
      - Long / Short / OI: pure magnitude (blue for high, no negatives)
      - Δ Long / Δ OI:     positive=blue, negative=red
      - Δ Short:            positive=red (bearish), negative=blue (bullish)
      - Long %:             >50%=blue, <50%=red (gradient)
      - Short %:            <50%=blue, >50%=red (gradient)
      - Net % Change:       positive=blue, negative=red
      - Net Position:       positive=blue, negative=red
    """
    if not rows:
        return []

    # Pre-compute max absolute values per column for normalization
    cols_signed = ["long_change", "short_change", "net_pct_change",
                   "net_position", "oi_change"]
    cols_pos    = ["long", "short", "open_interest"]

    max_abs = {}
    for col in cols_signed + cols_pos:
        vals = [abs(r.get(col, 0) or 0) for r in rows]
        max_abs[col] = max(vals) if vals else 1

    styles = []

    # Hover row highlight
    styles.append({
        "if": {"state": "active"},
        "backgroundColor": "rgba(255,255,255,0.04)",
        "border": "1px solid rgba(255,255,255,0.20)",
    })

    # Bold for the Net Position column
    styles.append({
        "if": {"column_id": "net_position"},
        "fontWeight": "700", "fontSize": "0.92rem",
    })
    styles.append({
        "if": {"column_id": "symbol"},
        "fontWeight": "700",
    })

    # Build per-row, per-column color
    for i, row in enumerate(rows):
        sym = row["symbol"]
        # filter_query uses {symbol} = "DOW" syntax
        fq = f'{{symbol}} = "{sym}"'

        # Long / Short / Open Interest — pure magnitude (blue, no red)
        for col in ("long", "short", "open_interest"):
            display_col = "oi_display" if col == "open_interest" else col
            v = row.get(col, 0)
            mx = max_abs[col] or 1
            intensity = min(abs(v) / mx, 1.0)
            alpha = 0.15 + 0.50 * (intensity ** 0.6)
            styles.append({
                "if": {"filter_query": fq, "column_id": display_col},
                "backgroundColor": f"rgba(64, 110, 220, {alpha:.2f})",
                "color": "#e8eef9",
            })

        # Δ Long Contracts: positive=blue, negative=red, bold negatives
        v = row.get("long_change", 0)
        bg = _heat_color(v, max_abs["long_change"] or 1)
        styles.append({
            "if": {"filter_query": fq, "column_id": "long_change"},
            "backgroundColor": bg,
            "color": "#ffd1d1" if v < 0 else "#d8e2ff",
            "fontWeight": "700" if v < 0 else "400",
        })

        # Δ Short Contracts: inverted — positive=red (bearish), negative=blue
        v = row.get("short_change", 0)
        bg = _heat_color(v, max_abs["short_change"] or 1, positive_is_bullish=False)
        styles.append({
            "if": {"filter_query": fq, "column_id": "short_change"},
            "backgroundColor": bg,
            "color": "#ffd1d1" if v > 0 else "#d8e2ff",
            "fontWeight": "700" if v > 0 else "400",
        })

        # Long % — gradient blue (always shown as bullish leaning when high)
        long_pct = row.get("long_pct", 0)
        # Center on 50%, scale ±50 → ±1
        normalized = (long_pct - 50) / 50.0
        alpha = 0.15 + 0.45 * abs(normalized)
        col_blue = normalized >= 0
        bg = (f"rgba(64, 110, 220, {alpha:.2f})" if col_blue
              else f"rgba(220, 60, 80, {alpha:.2f})")
        styles.append({
            "if": {"filter_query": fq, "column_id": "long_pct_str"},
            "backgroundColor": bg,
            "color": "#d8e2ff" if col_blue else "#ffd1d1",
            "fontWeight": "700" if not col_blue else "400",
        })

        # Short % — opposite: high short % = bearish/red
        short_pct = row.get("short_pct", 0)
        normalized = (short_pct - 50) / 50.0
        alpha = 0.15 + 0.45 * abs(normalized)
        col_red = normalized >= 0
        bg = (f"rgba(220, 60, 80, {alpha:.2f})" if col_red
              else f"rgba(64, 110, 220, {alpha:.2f})")
        styles.append({
            "if": {"filter_query": fq, "column_id": "short_pct_str"},
            "backgroundColor": bg,
            "color": "#ffd1d1" if col_red else "#d8e2ff",
            "fontWeight": "700" if col_red else "400",
        })

        # Net % Change
        v = row.get("net_pct_change", 0)
        bg = _heat_color(v, max_abs["net_pct_change"] or 1)
        styles.append({
            "if": {"filter_query": fq, "column_id": "net_pct_str"},
            "backgroundColor": bg,
            "color": "#ffd1d1" if v < 0 else "#d8e2ff",
            "fontWeight": "700" if v < 0 else "400",
        })

        # Net Position — heaviest emphasis
        v = row.get("net_position", 0)
        bg = _heat_color(v, max_abs["net_position"] or 1)
        styles.append({
            "if": {"filter_query": fq, "column_id": "net_position"},
            "backgroundColor": bg,
            "color": "#ffd1d1" if v < 0 else "#d8e2ff",
        })

        # Δ Open Interest: positive=blue, negative=red
        v = row.get("oi_change", 0)
        bg = _heat_color(v, max_abs["oi_change"] or 1)
        styles.append({
            "if": {"filter_query": fq, "column_id": "oi_change"},
            "backgroundColor": bg,
            "color": "#ffd1d1" if v < 0 else "#d8e2ff",
            "fontWeight": "700" if v < 0 else "400",
        })

    return styles


@app.callback(
    Output("cot-table", "data"),
    Output("cot-table", "style_data_conditional"),
    Output("cot-meta", "children"),
    Input("interval-poll", "n_intervals"),
)
def poll_cot(n):
    if not cot_scraper.cot_manager:
        return [], [], "COT manager not initialized"

    cache = cot_scraper.cot_manager.get_cache()
    error = cache.get("error")
    rows = cache.get("rows", [])
    report_date = cache.get("report_date", "")
    fetched_at = cache.get("fetched_at", 0)

    if error and not rows:
        return [], [], f"Error: {error}"

    # Add display-formatted fields for the styled columns
    formatted = []
    for r in rows:
        fr = dict(r)
        fr["long_pct_str"]  = f"{r.get('long_pct', 0):.2f}%"
        fr["short_pct_str"] = f"{r.get('short_pct', 0):.2f}%"
        npct = r.get("net_pct_change", 0)
        fr["net_pct_str"]   = f"{npct:+.2f}%"
        fr["oi_display"]    = _format_oi_short(r.get("open_interest", 0))
        formatted.append(fr)

    styles = _build_cot_row_styles(rows)

    if fetched_at:
        fetched_str = dt.datetime.fromtimestamp(fetched_at, tz=ET).strftime("%Y-%m-%d %H:%M:%S ET")
    else:
        fetched_str = "—"

    meta = (f"Report date: {report_date}  |  "
            f"Symbols: {len(rows)}  |  "
            f"Last fetch: {fetched_str}")

    return formatted, styles, meta


# ── Greeks Visualizer callbacks ──────────────────────────────────────────────

@app.callback(
    Output("dropdown-expiry", "options"),
    Output("dropdown-expiry", "value"),
    Input("input-ticker", "value"),
    prevent_initial_call=False,
)
def on_ticker_change(ticker):
    if not ticker:
        return [{"label": "Auto (nearest)", "value": "auto"}], "auto"
    ticker = ticker.upper().strip()
    SETTINGS.ticker = ticker
    SETTINGS.expiry = "auto"
    if data_fetcher.data_manager:
        data_fetcher.data_manager.clear_history()
        data_fetcher.data_manager.request_refresh()
    cache = data_fetcher.data_manager.get_cache() if data_fetcher.data_manager else {}
    expiries = cache.get("expiries", [])
    opts = [{"label": "Auto (nearest)", "value": "auto"}]
    for e in expiries:
        opts.append({"label": f"{e[:4]}-{e[4:6]}-{e[6:]}", "value": e})
    return opts, "auto"


@app.callback(
    Output("status-text", "children", allow_duplicate=True),
    Input("dropdown-expiry", "value"),
    prevent_initial_call=True,
)
def on_expiry_change(expiry):
    SETTINGS.expiry = expiry or "auto"
    if data_fetcher.data_manager:
        data_fetcher.data_manager.clear_history()
        data_fetcher.data_manager.request_refresh()
    return "Refreshing..."


@app.callback(
    Output("refresh-label", "children"),
    Input("slider-refresh", "value"),
)
def on_slider_change(seconds):
    SETTINGS.refresh_seconds = seconds
    if seconds < 60:
        return f"Every {seconds} seconds"
    return f"Every {seconds // 60}m {seconds % 60}s"


@app.callback(
    Output("status-text", "children", allow_duplicate=True),
    Input("radio-mode", "value"),
    prevent_initial_call=True,
)
def on_mode_change(mode):
    SETTINGS.greek_mode = mode or "oi"
    return "Mode updated..."


# ── Main poll ────────────────────────────────────────────────────────────────
@app.callback(
    Output("chart-gamma", "figure"),
    Output("chart-charm", "figure"),
    Output("chart-vomma", "figure"),
    Output("chart-speed", "figure"),
    Output("chart-vanna", "figure"),
    Output("chart-dex", "figure"),
    Output("chart-zomma", "figure"),
    Output("signal-panel", "children"),
    Output("profile-panel", "children"),
    Output("metrics-header", "children"),
    Output("status-text", "children"),
    Output("dropdown-expiry", "options", allow_duplicate=True),
    Output("store-prev-exposure", "data"),
    Input("interval-poll", "n_intervals"),
    Input("radio-gamma-view", "value"),
    Input("radio-charm-view", "value"),
    Input("radio-vomma-view", "value"),
    Input("radio-speed-view", "value"),
    Input("radio-vanna-view", "value"),
    Input("radio-zomma-view", "value"),
    State("store-prev-exposure", "data"),
    prevent_initial_call="initial_duplicate",
)
def poll_and_render(n, gamma_view, charm_view, vomma_view, speed_view,
                    vanna_view, zomma_view, prev_data):
    if not data_fetcher.data_manager:
        e = _empty_fig("Starting...")
        return e, e, e, e, e, e, e, "Analyzing...", "Analyzing profile...", "Loading...", "Initialising...", dash.no_update, dash.no_update

    cache = data_fetcher.data_manager.get_cache()

    error = cache.get("error")
    if error:
        e = _empty_fig(f"Error: {error}")
        return e, e, e, e, e, e, e, "No signal (error)", "No profile (error)", "Error", f"X  {error}", dash.no_update, dash.no_update

    chain = cache.get("chain")
    if chain is None or (hasattr(chain, "empty") and chain.empty):
        e = _empty_fig("Waiting for data...")
        return e, e, e, e, e, e, e, "Waiting for data...", "Waiting for data...", "Waiting...", "Fetching from IB...", dash.no_update, dash.no_update

    ticker   = cache["ticker"]
    spot     = cache["spot"]
    resolved = cache["expiry"]
    expiries = cache.get("expiries", [])
    mode     = SETTINGS.greek_mode

    # Flow data (available for all modes, used by flow/oi_flow)
    import pandas as pd
    flow_chain = cache.get("flow_chain", pd.DataFrame())
    oi_flow_chain = cache.get("oi_flow_chain", pd.DataFrame())
    flow_stats = cache.get("flow_stats", {})

    # Compute exposure based on selected mode
    if mode == "flow":
        if flow_chain is not None and not flow_chain.empty:
            exp_df = compute_exposure(flow_chain, spot, greek_mode="oi")
        else:
            exp_df = pd.DataFrame({
                "strike": [], "gamma_exp": [], "charm_exp": [],
                "vanna_exp": [], "zomma_exp": [],
            })
    elif mode == "oi_flow":
        if oi_flow_chain is not None and not oi_flow_chain.empty:
            exp_df = compute_exposure(oi_flow_chain, spot, greek_mode="oi")
        else:
            # Fall back to plain OI until flow data arrives
            exp_df = compute_exposure(chain, spot, greek_mode="oi")
    else:
        exp_df = compute_exposure(chain, spot, greek_mode=mode)

    # Previous day high/low from cache
    prev_hl = cache.get("prev_day_hl", {"high": 0, "low": 0})

    # Session metrics (locked at first fetch)
    session_metrics = cache.get("session_metrics", {})
    open_spot = session_metrics.get("open_spot", 0)

    # Previous exposure for computing deltas (from dcc.Store)
    # Store structure:
    #   "_ts"          = timestamp of latest fetch
    #   "display_prev" = baseline for value-view deltas (frozen between fetches)
    #   "current"      = values from latest fetch
    #   "history"      = list of up to 3 past snapshots [oldest, ..., newest]
    #                    used for dot indicators on bar charts
    if prev_data is None:
        prev_data = {}

    cache_ts = cache.get("timestamp", 0)
    prev_ts  = prev_data.get("_ts", 0)
    prev_mode = prev_data.get("_mode", "")

    # If mode changed, reset all history (dots would be from wrong mode)
    mode_changed = (mode != prev_mode and prev_mode != "")

    # Build current snapshot
    cur_snap = {
        "gamma": {str(r["strike"]): r["gamma_exp"] for _, r in exp_df.iterrows()},
        "charm": {str(r["strike"]): r["charm_exp"] for _, r in exp_df.iterrows()},
        "vomma": {str(r["strike"]): r["vomma_exp"] for _, r in exp_df.iterrows()},
        "speed": {str(r["strike"]): r["speed_exp"] for _, r in exp_df.iterrows()},
        "vanna": {str(r["strike"]): r["vanna_exp"] for _, r in exp_df.iterrows()},
        "dex":   {str(r["strike"]): r["dex_exp"]   for _, r in exp_df.iterrows()},
        "zomma": {str(r["strike"]): r["zomma_exp"] for _, r in exp_df.iterrows()},
    }

    if mode_changed:
        # Mode switched — start fresh
        new_prev = {
            "_ts":          cache_ts,
            "_mode":        mode,
            "display_prev": {},
            "current":      cur_snap,
            "history":      [],
        }
        dp = {}
    elif cache_ts != prev_ts:
        # New fetch — rotate history
        dp = prev_data.get("current", {})
        old_history = prev_data.get("history", [])
        if dp:
            new_history = (old_history + [dp])[-3:]
        else:
            new_history = old_history[-3:]
        new_prev = {
            "_ts":          cache_ts,
            "_mode":        mode,
            "display_prev": dp,
            "current":      cur_snap,
            "history":      new_history,
        }
    else:
        dp = prev_data.get("display_prev", {})
        new_prev = dash.no_update

    prev_gamma = dp.get("gamma", {})
    prev_charm = dp.get("charm", {})
    prev_vomma = dp.get("vomma", {})
    prev_speed = dp.get("speed", {})
    prev_vanna = dp.get("vanna", {})
    prev_dex   = dp.get("dex", {})
    prev_zomma = dp.get("zomma", {})

    # History for bar chart dots (list of up to 3 past snapshots)
    history = prev_data.get("history", [])

    # Common open price line for bar charts
    open_line = {"type": "price", "value": open_spot,
                 "label": "Open", "color": "#29b6f6", "side": "right"}

    # Extract history dots per greek (list of dicts for each past snapshot)
    def _hist(greek_key):
        return [snap.get(greek_key, {}) for snap in history]

    # ── Gamma regime & key levels ────────────────────────────────────
    regime = classify_regime(exp_df, spot) if not exp_df.empty else {}
    gex_flip = regime.get("gex_flip")
    call_wall = regime.get("call_wall")
    put_wall = regime.get("put_wall")

    # Build gamma-specific indicator lines
    gamma_lines = [
        {"type": "exposure_max", "label": "Max Pos GEX",
         "color": "#00ffcc", "side": "left"},
        {"type": "exposure_min", "label": "Max Neg GEX",
         "color": "#ff6b6b", "side": "left"},
        {"type": "net_max", "label": "Max Net GEX",
         "color": "#e0e0e0", "side": "left"},
        {"type": "net_min", "label": "Min Net GEX",
         "color": "#9e9e9e", "side": "left"},
        {"type": "price", "value": prev_hl["high"],
         "label": "Prev High", "color": "#80cbc4", "side": "right"},
        {"type": "price", "value": prev_hl["low"],
         "label": "Prev Low", "color": "#ef9a9a", "side": "right"},
        open_line,
    ]
    # GEX flip line
    if gex_flip is not None:
        gamma_lines.append(
            {"type": "price", "value": gex_flip,
             "label": "GEX Flip", "color": "#ff9800", "side": "left"})
    # Call wall
    if call_wall is not None:
        gamma_lines.append(
            {"type": "price", "value": call_wall,
             "label": "Call Wall", "color": "#4caf50", "side": "right"})
    # Put wall
    if put_wall is not None:
        gamma_lines.append(
            {"type": "price", "value": put_wall,
             "label": "Put Wall", "color": "#f44336", "side": "right"})

    # ── Gamma ────────────────────────────────────────────────────────
    if gamma_view == "values":
        fig_gamma = _build_value_view(exp_df, "strike", "gamma_exp",
                                       "Gamma (GEX)", spot, prev_gamma,
                                       open_price=open_spot)
    else:
        fig_gamma = _build_chart(exp_df, "strike", "gamma_exp",
                                  "Gamma (GEX)", spot, "#00d4aa", "#ff4d6a",
                                  lines=gamma_lines,
                                  history_dots=_hist("gamma"))

    # ── Charm ────────────────────────────────────────────────────────
    # ── Charm ────────────────────────────────────────────────────────
    if charm_view == "values":
        fig_charm = _build_value_view(exp_df, "strike", "charm_exp",
                                       "Charm", spot, prev_charm,
                                       compact=True, open_price=open_spot)
    else:
        fig_charm = _build_chart(exp_df, "strike", "charm_exp",
                                  "Charm", spot, "#4dabf7", "#f783ac",
                                  lines=[
                                      {"type": "exposure_max", "label": "Max Pos Charm",
                                       "color": "#4dabf7", "side": "left"},
                                      {"type": "exposure_min", "label": "Max Neg Charm",
                                       "color": "#f783ac", "side": "left"},
                                      {"type": "net_max", "label": "Max Net Charm",
                                       "color": "#e0e0e0", "side": "left"},
                                      {"type": "net_min", "label": "Min Net Charm",
                                       "color": "#9e9e9e", "side": "left"},
                                      open_line,
                                  ],
                                  compact=True,
                                  history_dots=_hist("charm"))

    # ── Vomma ────────────────────────────────────────────────────────
    if vomma_view == "values":
        fig_vomma = _build_value_view(exp_df, "strike", "vomma_exp",
                                       "Vomma", spot, prev_vomma,
                                       compact=True, open_price=open_spot)
    else:
        fig_vomma = _build_chart(exp_df, "strike", "vomma_exp",
                                  "Vomma", spot, "#ffb74d", "#7986cb",
                                  lines=[
                                      {"type": "exposure_max", "label": "Max Pos Vomma",
                                       "color": "#ffb74d", "side": "left"},
                                      {"type": "exposure_min", "label": "Max Neg Vomma",
                                       "color": "#7986cb", "side": "left"},
                                      {"type": "net_max", "label": "Max Net Vomma",
                                       "color": "#e0e0e0", "side": "left"},
                                      open_line,
                                  ],
                                  compact=True,
                                  history_dots=_hist("vomma"))

    # ── Speed ────────────────────────────────────────────────────────
    if speed_view == "values":
        fig_speed = _build_value_view(exp_df, "strike", "speed_exp",
                                       "Speed", spot, prev_speed,
                                       compact=True, open_price=open_spot)
    else:
        fig_speed = _build_chart(exp_df, "strike", "speed_exp",
                                  "Speed", spot, "#ba68c8", "#81c784",
                                  lines=[
                                      {"type": "exposure_max", "label": "Max Pos Speed",
                                       "color": "#ba68c8", "side": "left"},
                                      {"type": "exposure_min", "label": "Max Neg Speed",
                                       "color": "#81c784", "side": "left"},
                                      {"type": "net_max", "label": "Max Net Speed",
                                       "color": "#e0e0e0", "side": "left"},
                                      open_line,
                                  ],
                                  compact=True,
                                  history_dots=_hist("speed"))

    # ── Vanna ────────────────────────────────────────────────────────
    if vanna_view == "values":
        fig_vanna = _build_value_view(exp_df, "strike", "vanna_exp",
                                       "Vanna", spot, prev_vanna,
                                       compact=True, open_price=open_spot)
    else:
        fig_vanna = _build_chart(exp_df, "strike", "vanna_exp",
                                  "Vanna", spot, "#a78bfa", "#fb923c",
                                  lines=[
                                      {"type": "exposure_max", "label": "Max Pos Vanna",
                                       "color": "#a78bfa", "side": "left"},
                                      {"type": "exposure_min", "label": "Max Neg Vanna",
                                       "color": "#fb923c", "side": "left"},
                                      {"type": "net_max", "label": "Max Net Vanna",
                                       "color": "#e0e0e0", "side": "left"},
                                      {"type": "net_min", "label": "Min Net Vanna",
                                       "color": "#9e9e9e", "side": "left"},
                                      open_line,
                                  ],
                                  compact=True,
                                  history_dots=_hist("vanna"))

    # ── DEX (Delta Exposure) ─────────────────────────────────────────
    fig_dex = _build_chart(exp_df, "strike", "dex_exp",
                            "DEX (Delta)", spot, "#26c6da", "#ff7043",
                            lines=[
                                {"type": "exposure_max", "label": "Max Pos DEX",
                                 "color": "#26c6da", "side": "left"},
                                {"type": "exposure_min", "label": "Max Neg DEX",
                                 "color": "#ff7043", "side": "left"},
                                {"type": "net_max", "label": "Max Net DEX",
                                 "color": "#e0e0e0", "side": "left"},
                                open_line,
                            ],
                            compact=True,
                            history_dots=_hist("dex"))

    # ── Zomma ────────────────────────────────────────────────────────
    if zomma_view == "values":
        fig_zomma = _build_value_view(exp_df, "strike", "zomma_exp",
                                       "Zomma", spot, prev_zomma,
                                       compact=True, open_price=open_spot)
    else:
        fig_zomma = _build_chart(exp_df, "strike", "zomma_exp",
                                  "Zomma", spot, "#66bb6a", "#ef5350",
                                  lines=[
                                      {"type": "exposure_max", "label": "Max Pos Zomma",
                                       "color": "#66bb6a", "side": "left"},
                                      {"type": "exposure_min", "label": "Max Neg Zomma",
                                       "color": "#ef5350", "side": "left"},
                                      {"type": "net_max", "label": "Max Net Zomma",
                                       "color": "#e0e0e0", "side": "left"},
                                      {"type": "net_min", "label": "Min Net Zomma",
                                       "color": "#9e9e9e", "side": "left"},
                                      open_line,
                                  ],
                                  compact=True, show_spot=False,
                                  history_dots=_hist("zomma"))

    # ── Metrics header ─────────────────────────────────────────────
    # Live metrics (recomputed every refresh)
    live_metrics = compute_live_metrics(chain, spot)
    prev_hl = cache.get("prev_day_hl", {"high": 0, "low": 0})

    nice_exp = f"{resolved[:4]}-{resolved[4:6]}-{resolved[6:]}"
    exp_date = dt.date(int(resolved[:4]), int(resolved[4:6]), int(resolved[6:]))
    dte = max((exp_date - dt.date.today()).days, 0)
    mode_labels = {"oi": "Open Interest", "volume": "Volume",
                   "combined": "OI+Vol", "flow": "Flow (Dealer)",
                   "oi_flow": "OI + Flow (Live)"}
    mode_label = mode_labels.get(mode, mode)
    ts = cache.get("timestamp", 0)
    updated = dt.datetime.fromtimestamp(ts, tz=ET).strftime("%H:%M:%S ET") if ts else "-"

    history_len = len(data_fetcher.data_manager.get_charm_history()) \
        if data_fetcher.data_manager else 0

    classified = flow_stats.get("classified", 0)
    unclassified = flow_stats.get("unclassified", 0)
    total_flow = classified + unclassified
    flow_pct = f"{classified / total_flow * 100:.0f}%" if total_flow > 0 else "—"

    status = (
        f"Ticker: {ticker}\n"
        f"Spot: ${spot:,.2f}\n"
        f"Expiry: {nice_exp} ({dte} DTE)\n"
        f"Mode: {mode_label}\n"
        f"Strikes: {len(exp_df)}\n"
        f"Heatmap: {history_len} snapshots\n"
        f"Flow: {classified}/{total_flow} ({flow_pct})\n"
        f"Last fetch: {updated}"
    )

    opts = [{"label": "Auto (nearest)", "value": "auto"}]
    for e in expiries:
        opts.append({"label": f"{e[:4]}-{e[4:6]}-{e[6:]}", "value": e})

    # ── Vanna / VIX signal ──────────────────────────────────────────
    vix_data = cache.get("vix", {"current": 0, "prev_close": 0})
    vanna_vix = compute_vanna_vix_signal(
        exp_df, vix_data.get("current", 0), vix_data.get("prev_close", 0))

    # ── Charm Decay Clock ────────────────────────────────────────────
    charm_clock = compute_charm_clock(exp_df, spot)

    # ── Skew (25d put IV − 25d call IV) ──────────────────────────────
    from greek_calculator import compute_skew, compute_pinning_strength
    skew = compute_skew(chain, spot)

    # ── Term structure ───────────────────────────────────────────────
    term_raw = cache.get("term_structure", {})
    front_iv = live_metrics.get("atm_iv", 0)
    back_iv = term_raw.get("back_iv", 0)
    if front_iv > 0 and back_iv > 0:
        term_ratio = front_iv / back_iv
        if term_ratio > 1.05:
            term_state = "BACKWARDATION"
        elif term_ratio < 0.95:
            term_state = "CONTANGO"
        else:
            term_state = "FLAT"
    else:
        term_ratio = 0
        term_state = "N/A"
    term = {
        "front_iv":    front_iv,
        "back_iv":     back_iv,
        "back_dte":    term_raw.get("back_dte", 0),
        "ratio":       round(term_ratio, 3),
        "state":       term_state,
    }

    # ── IV Rank / Percentile ─────────────────────────────────────────
    import session_store
    iv_rank = session_store.get_iv_rank_percentile(ticker, front_iv)

    # ── Pinning Strength ─────────────────────────────────────────────
    dte_years_val = chain["dte_years"].values[0] if (chain is not None and not chain.empty) else None
    pinning = compute_pinning_strength(exp_df, chain, spot, dte_years=dte_years_val)

    # ── Composite Trade Signal ───────────────────────────────────────
    signal = compute_trade_signal(
        spot, regime, vanna_vix, charm_clock,
        skew, term, iv_rank, pinning, live_metrics, exp_df,
    )
    signal_panel = _build_signal_panel(signal)

    # ── Live Greek Profile Analysis ──────────────────────────────────
    profile = compute_profile_analysis(
        spot, regime, vanna_vix, charm_clock,
        skew, term, iv_rank, pinning, live_metrics, exp_df,
        prev_day_hl=prev_hl, session_metrics=session_metrics,
    )
    profile_panel = _build_profile_panel(profile)

    header = _build_metrics_header(ticker, spot, session_metrics,
                                    live_metrics, prev_hl, dte, updated,
                                    regime, vanna_vix, charm_clock,
                                    skew=skew, term=term,
                                    iv_rank=iv_rank, pinning=pinning)

    return (fig_gamma, fig_charm, fig_vomma, fig_speed,
            fig_vanna, fig_dex, fig_zomma,
            signal_panel, profile_panel, header, status, opts, new_prev)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS HEADER
# ══════════════════════════════════════════════════════════════════════════════

def _metric_cell(label, value, color="#ffffff", sub=None):
    """Build one metric cell for the header bar."""
    children = [
        html.Div(label, style={
            "fontSize": "0.60rem", "color": "#777", "lineHeight": "1",
        }),
        html.Div(value, style={
            "fontSize": "0.85rem", "fontWeight": "bold", "color": color,
            "fontFamily": "monospace", "lineHeight": "1.2",
        }),
    ]
    if sub:
        children.append(html.Div(sub, style={
            "fontSize": "0.55rem", "color": "#999", "lineHeight": "1",
        }))
    return html.Div(children, style={
        "display": "inline-block",
        "padding": "2px 10px",
        "textAlign": "center",
        "borderRight": "1px solid rgba(255,255,255,0.06)",
        "verticalAlign": "top",
    })


def _em_progress(label, spot, low, high, color_lo, color_hi):
    """
    Build an expected-move cell with a mini progress indicator
    showing where spot sits between low and high.
    """
    total = high - low
    if total <= 0:
        pct = 50
    else:
        pct = max(0, min(100, ((spot - low) / total) * 100))

    # Color based on position: green near center, yellow/red near edges
    if 30 <= pct <= 70:
        dot_color = "#66bb6a"
    elif 15 <= pct <= 85:
        dot_color = "#ffa726"
    else:
        dot_color = "#ef5350"

    bar_style = {
        "width": "100%",
        "height": "4px",
        "backgroundColor": "#333",
        "borderRadius": "2px",
        "position": "relative",
        "marginTop": "2px",
    }
    dot_style = {
        "position": "absolute",
        "left": f"{pct}%",
        "top": "-2px",
        "width": "8px",
        "height": "8px",
        "borderRadius": "50%",
        "backgroundColor": dot_color,
        "transform": "translateX(-50%)",
    }

    return html.Div([
        html.Div(label, style={
            "fontSize": "0.60rem", "color": "#777", "lineHeight": "1",
        }),
        html.Div(
            f"${low:,.1f}  —  ${high:,.1f}",
            style={
                "fontSize": "0.72rem", "fontFamily": "monospace",
                "color": "#ccc", "lineHeight": "1.2",
            },
        ),
        html.Div(
            [html.Div(style=dot_style)],
            style=bar_style,
        ),
    ], style={
        "display": "inline-block",
        "padding": "2px 10px",
        "textAlign": "center",
        "borderRight": "1px solid rgba(255,255,255,0.06)",
        "verticalAlign": "top",
        "minWidth": "120px",
    })


def _vanna_vix_cell(vanna_vix):
    """Build the Vanna/VIX signal cell for the header."""
    if not vanna_vix or vanna_vix.get("signal") == "N/A":
        return _metric_cell("Vanna/VIX", "N/A", color="#555",
                            sub="No VIX data")

    signal = vanna_vix["signal"]
    vix_cur = vanna_vix.get("vix_current", 0)
    vix_chg = vanna_vix.get("vix_change", 0)
    vix_pct = vanna_vix.get("vix_pct", 0)

    if signal == "BULLISH":
        color = "#66bb6a"
    elif signal == "BEARISH":
        color = "#ef5350"
    else:
        color = "#888"

    sub = f"VIX {vix_cur:.1f} ({vix_chg:+.1f} / {vix_pct:+.1f}%)"
    return _metric_cell("Vanna/VIX", signal, color=color, sub=sub)


def _charm_clock_cell(charm_clock):
    """Build the Charm Decay Clock cell for the header."""
    if not charm_clock or charm_clock.get("direction") == "N/A":
        return _metric_cell("Charm Clock", "N/A", color="#555", sub="No data")

    direction = charm_clock["direction"]
    hours = charm_clock.get("hours_to_close", 0)
    pressure = charm_clock.get("charm_pressure", 0)

    if direction == "SUPPORTIVE":
        color = "#66bb6a"
    elif direction == "PRESSURING":
        color = "#ef5350"
    else:
        color = "#888"

    # Format pressure with K/M suffix
    ap = abs(pressure)
    if ap >= 1e6:
        p_str = f"{pressure/1e6:+,.1f}M"
    elif ap >= 1e3:
        p_str = f"{pressure/1e3:+,.0f}K"
    else:
        p_str = f"{pressure:+,.0f}"

    sub = f"{hours:.1f}h left | {p_str}"
    return _metric_cell("Charm Clock", direction, color=color, sub=sub)


def _skew_cell(skew):
    """25-delta put IV minus 25-delta call IV."""
    if not skew:
        return _metric_cell("Skew", "N/A", color="#555", sub="No data")
    val = skew.get("skew", 0) * 100   # display in vol points
    pct = skew.get("skew_pct", 0)
    # Positive skew = puts richer than calls = bearish sentiment
    if val > 3:
        color = "#ef5350"   # bearish skew
    elif val < -1:
        color = "#66bb6a"   # call skew (bullish)
    else:
        color = "#ffd54f"   # normal
    return _metric_cell("Skew (25d)",
                         f"{val:+.2f} pts",
                         color=color,
                         sub=f"{pct:+.1f}% of ATM IV")


def _term_cell(term):
    """Term structure — front/back IV ratio."""
    if not term or term.get("state") == "N/A":
        return _metric_cell("Term", "N/A", color="#555", sub="No back-month IV")
    state = term["state"]
    ratio = term.get("ratio", 1.0)
    back_dte = term.get("back_dte", 0)
    back_iv = term.get("back_iv", 0)
    # Backwardation (front > back) = stressed / risk-off
    # Contango (front < back) = normal
    if state == "BACKWARDATION":
        color = "#ef5350"
    elif state == "CONTANGO":
        color = "#66bb6a"
    else:
        color = "#ffd54f"
    return _metric_cell("Term",
                         state,
                         color=color,
                         sub=f"Ratio {ratio:.2f} | {back_dte}d {back_iv*100:.1f}%")


def _iv_rank_cell(iv_rank):
    """IV rank / percentile from 52-week history."""
    if not iv_rank or iv_rank.get("iv_rank") is None:
        return _metric_cell("IV Rank", "N/A", color="#555",
                            sub="Building history...")
    rank = iv_rank.get("iv_rank", 0)
    pct = iv_rank.get("iv_percentile", 0)
    days = iv_rank.get("history_days", 0)
    # Low rank = cheap vol, high rank = expensive vol
    if rank >= 70:
        color = "#ef5350"   # expensive
    elif rank <= 30:
        color = "#66bb6a"   # cheap
    else:
        color = "#ffd54f"
    return _metric_cell("IV Rank",
                         f"{rank:.0f} / 100",
                         color=color,
                         sub=f"Pct {pct:.0f} | {days}d hist")


def _pin_cell(pinning):
    """Pinning strength score for 0DTE magnet strike."""
    if not pinning or pinning.get("pin_strike") is None:
        return _metric_cell("Pin", "N/A", color="#555", sub="No data")
    strike = pinning.get("pin_strike", 0)
    strength = pinning.get("pin_strength", 0)
    dist = pinning.get("pin_distance", 0)
    conf = pinning.get("confidence", "NONE")
    # Strong pin = yellow, weak = grey
    if conf == "HIGH":
        color = "#ffd54f"
    elif conf == "MEDIUM":
        color = "#ffb74d"
    elif conf == "LOW":
        color = "#888"
    else:
        color = "#555"
    return _metric_cell("Pin",
                         f"${strike:,.0f}",
                         color=color,
                         sub=f"{conf} | {strength:.0f}/100 | ±${dist:,.1f}")


def _build_signal_panel(signal):
    """Render the composite trade signal as a rich HTML panel."""
    if not signal:
        return "Analyzing..."

    direction = signal.get("direction", "NEUTRAL")
    conviction = signal.get("conviction", "NONE")
    setup = signal.get("setup", "")
    entry = signal.get("entry")
    stop = signal.get("stop_loss")
    tp1 = signal.get("take_profit_1")
    tp2 = signal.get("take_profit_2")
    rr = signal.get("risk_reward", 0)
    reasoning = signal.get("reasoning", [])
    caveats = signal.get("caveats", [])
    score = signal.get("score", 0)

    # Direction color
    if direction == "LONG":
        dir_color = "#66bb6a"
        dir_icon = "▲"
    elif direction == "SHORT":
        dir_color = "#ef5350"
        dir_icon = "▼"
    else:
        dir_color = "#888"
        dir_icon = "●"

    # Conviction color
    if conviction == "HIGH":
        conv_color = "#66bb6a" if direction != "NEUTRAL" else "#888"
    elif conviction == "MEDIUM":
        conv_color = "#ffd54f"
    elif conviction == "LOW":
        conv_color = "#ff9800"
    else:
        conv_color = "#555"

    def _level_block(label, value, color):
        if value is None:
            return html.Div([
                html.Div(label, style={"fontSize": "0.72rem", "color": "#888",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.08em"}),
                html.Div("—", style={"fontSize": "1.05rem", "color": "#555",
                                      "fontFamily": "monospace"}),
            ], style={"padding": "0 14px", "borderRight": "1px solid rgba(255,255,255,0.06)"})
        return html.Div([
            html.Div(label, style={"fontSize": "0.72rem", "color": "#888",
                                   "textTransform": "uppercase",
                                   "letterSpacing": "0.08em"}),
            html.Div(f"${value:,.2f}",
                     style={"fontSize": "1.05rem", "color": color,
                            "fontFamily": "monospace", "fontWeight": "600"}),
        ], style={"padding": "0 14px", "borderRight": "1px solid rgba(255,255,255,0.06)"})

    # ── Top row: direction / conviction / setup description ─────────
    header_row = html.Div([
        html.Div([
            html.Span(dir_icon, style={"fontSize": "1.5rem", "color": dir_color,
                                        "marginRight": "8px"}),
            html.Span(direction, style={"fontSize": "1.2rem", "color": dir_color,
                                         "fontWeight": "700", "letterSpacing": "0.05em"}),
        ], style={"display": "flex", "alignItems": "center",
                  "minWidth": "130px"}),

        html.Div([
            html.Div("Conviction", style={"fontSize": "0.72rem", "color": "#888",
                                           "textTransform": "uppercase",
                                           "letterSpacing": "0.08em"}),
            html.Div(conviction, style={"fontSize": "1.0rem", "color": conv_color,
                                         "fontWeight": "600"}),
        ], style={"padding": "0 14px", "borderLeft": "1px solid rgba(255,255,255,0.08)",
                  "borderRight": "1px solid rgba(255,255,255,0.06)"}),

        html.Div([
            html.Div("Setup", style={"fontSize": "0.72rem", "color": "#888",
                                      "textTransform": "uppercase",
                                      "letterSpacing": "0.08em"}),
            html.Div(setup if setup else "—",
                     style={"fontSize": "0.88rem", "color": "#e0e0e0"}),
        ], style={"padding": "0 14px", "flex": 1, "minWidth": 0}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"})

    # ── Levels row: entry / stop / tp1 / tp2 / R:R ──────────────────
    rr_color = "#66bb6a" if rr >= 2.0 else ("#ffd54f" if rr >= 1.0 else "#ef5350")
    levels_row = html.Div([
        _level_block("Entry", entry, "#ffd54f"),
        _level_block("Stop Loss", stop, "#ef5350"),
        _level_block("Target 1", tp1, "#66bb6a"),
        _level_block("Target 2", tp2, "#4caf50"),
        html.Div([
            html.Div("R : R", style={"fontSize": "0.72rem", "color": "#888",
                                      "textTransform": "uppercase",
                                      "letterSpacing": "0.08em"}),
            html.Div(f"{rr:.2f}" if rr > 0 else "—",
                     style={"fontSize": "1.05rem", "color": rr_color,
                            "fontFamily": "monospace", "fontWeight": "600"}),
        ], style={"padding": "0 14px"}),
    ], style={"display": "flex", "alignItems": "center",
              "marginBottom": "10px"})

    # ── Reasoning bullets ──────────────────────────────────────────
    reasoning_items = [
        html.Li(r, style={"marginBottom": "2px"}) for r in reasoning[:8]
    ] or [html.Li("No signals firing", style={"color": "#888"})]

    reasoning_block = html.Div([
        html.Div("REASONING", style={"fontSize": "0.72rem", "color": "#888",
                                      "letterSpacing": "0.08em",
                                      "marginBottom": "4px"}),
        html.Ul(reasoning_items,
                style={"margin": "0", "paddingLeft": "18px",
                       "fontSize": "0.80rem", "color": "#ccc"}),
    ], style={"flex": 1, "minWidth": 0, "paddingRight": "16px"})

    # ── Caveats bullets ─────────────────────────────────────────────
    if caveats:
        caveat_items = [
            html.Li(c, style={"marginBottom": "2px", "color": "#ffb74d"})
            for c in caveats[:6]
        ]
        caveat_block = html.Div([
            html.Div("WATCH OUT", style={"fontSize": "0.72rem", "color": "#ffb74d",
                                          "letterSpacing": "0.08em",
                                          "marginBottom": "4px"}),
            html.Ul(caveat_items,
                    style={"margin": "0", "paddingLeft": "18px",
                           "fontSize": "0.80rem"}),
        ], style={"flex": 1, "minWidth": 0,
                  "paddingLeft": "16px",
                  "borderLeft": "1px solid rgba(255,255,255,0.08)"})
    else:
        caveat_block = html.Div([
            html.Div("WATCH OUT", style={"fontSize": "0.72rem", "color": "#888",
                                          "letterSpacing": "0.08em",
                                          "marginBottom": "4px"}),
            html.Div("No major caveats", style={"fontSize": "0.80rem",
                                                  "color": "#666",
                                                  "fontStyle": "italic"}),
        ], style={"flex": 1, "minWidth": 0,
                  "paddingLeft": "16px",
                  "borderLeft": "1px solid rgba(255,255,255,0.08)"})

    return html.Div([
        header_row,
        levels_row,
        html.Div([reasoning_block, caveat_block],
                 style={"display": "flex", "alignItems": "flex-start"}),
    ])


def _build_profile_panel(profile):
    """Render the live Greek Profile Analysis as a compact info panel."""
    if not profile:
        return "Analyzing..."

    bias = profile.get("bias", "BALANCED")
    bias_strength = profile.get("bias_strength", "WEAK")
    regime_label = profile.get("regime_label", "")
    trade_type = profile.get("trade_type", "")
    summary = profile.get("summary", "")
    key_levels = profile.get("key_levels", [])
    tailwinds = profile.get("tailwinds", [])
    headwinds = profile.get("headwinds", [])
    time_notes = profile.get("time_notes", [])
    what_to_watch = profile.get("what_to_watch", [])

    # Color for bias
    if bias == "BULLISH":
        bias_color = "#22c55e" if bias_strength == "STRONG" else "#84cc16"
        bias_icon = "▲"
    elif bias == "BEARISH":
        bias_color = "#ef4444" if bias_strength == "STRONG" else "#f97316"
        bias_icon = "▼"
    else:
        bias_color = "#94a3b8"
        bias_icon = "◆"

    # Header row: bias + strength
    header_row = html.Div([
        html.Span("PROFILE ANALYSIS",
                  style={"fontSize": "0.65rem", "color": "var(--accent-purple)",
                         "fontWeight": "700", "letterSpacing": "0.10em"}),
        html.Div([
            html.Span(bias_icon, style={"fontSize": "1.1rem", "color": bias_color,
                                         "marginRight": "6px"}),
            html.Span(bias, style={"fontSize": "1.0rem", "color": bias_color,
                                    "fontWeight": "700", "letterSpacing": "0.04em"}),
            html.Span(bias_strength,
                      style={"fontSize": "0.68rem", "color": bias_color,
                             "marginLeft": "6px",
                             "padding": "1px 6px",
                             "border": f"1px solid {bias_color}",
                             "borderRadius": "4px",
                             "fontWeight": "600"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "justifyContent": "space-between",
              "alignItems": "center", "marginBottom": "8px"})

    # Regime line
    regime_block = html.Div(
        regime_label,
        style={"fontSize": "0.78rem", "color": "var(--text-secondary)",
               "marginBottom": "8px", "lineHeight": "1.4"},
    )

    # Trade type
    trade_type_block = html.Div([
        html.Span("PLAYBOOK", style={"fontSize": "0.62rem",
                                       "color": "var(--text-muted)",
                                       "letterSpacing": "0.08em",
                                       "marginRight": "6px"}),
        html.Span(trade_type, style={"fontSize": "0.78rem",
                                       "color": "#fbbf24",
                                       "fontWeight": "600"}),
    ], style={"marginBottom": "10px",
              "padding": "6px 10px",
              "backgroundColor": "rgba(251,191,36,0.08)",
              "borderRadius": "6px",
              "borderLeft": "3px solid #fbbf24"})

    # Key levels
    def _level_row(lvl):
        label, price, desc, dist = lvl
        # Color the distance: above spot = green, below = red, at = grey
        if abs(dist) < 0.05:
            dist_color = "#94a3b8"
        elif dist > 0:
            dist_color = "#66bb6a"
        else:
            dist_color = "#ef5350"
        return html.Div([
            html.Span(label, style={"flex": 1.4,
                                     "fontSize": "0.75rem",
                                     "color": "var(--text-primary)",
                                     "fontWeight": "500"}),
            html.Span(f"${price:,.2f}",
                      style={"flex": 0.8,
                             "fontSize": "0.78rem",
                             "color": "var(--text-primary)",
                             "fontFamily": "monospace",
                             "fontWeight": "600",
                             "textAlign": "right"}),
            html.Span(f"{dist:+.2f}%",
                      style={"flex": 0.6,
                             "fontSize": "0.72rem",
                             "color": dist_color,
                             "fontFamily": "monospace",
                             "textAlign": "right",
                             "marginLeft": "6px"}),
        ], style={"display": "flex", "alignItems": "center",
                  "padding": "3px 0",
                  "borderBottom": "1px solid rgba(255,255,255,0.04)"})

    levels_block = None
    if key_levels:
        levels_block = html.Div([
            html.Div("KEY LEVELS",
                     style={"fontSize": "0.62rem",
                            "color": "var(--text-muted)",
                            "letterSpacing": "0.08em",
                            "marginBottom": "4px"}),
            html.Div([_level_row(lvl) for lvl in key_levels[:7]]),
        ], style={"marginBottom": "10px"})

    # Tailwinds & Headwinds — two columns
    def _bullet_list(items, accent_color):
        if not items:
            return html.Div("None significant",
                            style={"fontSize": "0.72rem",
                                   "color": "var(--text-muted)",
                                   "fontStyle": "italic"})
        return html.Ul([html.Li(t, style={"marginBottom": "3px",
                                            "color": "var(--text-secondary)"})
                        for t in items[:5]],
                       style={"margin": 0, "paddingLeft": "16px",
                              "fontSize": "0.74rem"})

    tail_head_block = html.Div([
        html.Div([
            html.Div("TAILWINDS",
                     style={"fontSize": "0.62rem",
                            "color": "#66bb6a",
                            "letterSpacing": "0.08em",
                            "marginBottom": "4px"}),
            _bullet_list(tailwinds, "#66bb6a"),
        ], style={"flex": 1, "paddingRight": "10px"}),
        html.Div([
            html.Div("HEADWINDS",
                     style={"fontSize": "0.62rem",
                            "color": "#ef5350",
                            "letterSpacing": "0.08em",
                            "marginBottom": "4px"}),
            _bullet_list(headwinds, "#ef5350"),
        ], style={"flex": 1,
                  "paddingLeft": "10px",
                  "borderLeft": "1px solid rgba(255,255,255,0.06)"}),
    ], style={"display": "flex", "marginBottom": "10px"})

    # Time notes (only if present)
    time_block = None
    if time_notes:
        time_block = html.Div([
            html.Div("TIME OF DAY",
                     style={"fontSize": "0.62rem",
                            "color": "var(--accent-orange)",
                            "letterSpacing": "0.08em",
                            "marginBottom": "4px"}),
            html.Ul([html.Li(t, style={"marginBottom": "3px",
                                         "color": "var(--text-secondary)"})
                     for t in time_notes],
                    style={"margin": 0, "paddingLeft": "16px",
                           "fontSize": "0.74rem"}),
        ], style={"marginBottom": "10px"})

    # What to watch
    watch_block = None
    if what_to_watch:
        watch_block = html.Div([
            html.Div("WHAT WOULD CHANGE THE READ",
                     style={"fontSize": "0.62rem",
                            "color": "var(--accent-yellow)",
                            "letterSpacing": "0.08em",
                            "marginBottom": "4px"}),
            html.Ul([html.Li(t, style={"marginBottom": "3px",
                                         "color": "var(--text-secondary)"})
                     for t in what_to_watch[:5]],
                    style={"margin": 0, "paddingLeft": "16px",
                           "fontSize": "0.74rem"}),
        ])

    children = [header_row, regime_block, trade_type_block]
    if levels_block:
        children.append(levels_block)
    children.append(tail_head_block)
    if time_block:
        children.append(time_block)
    if watch_block:
        children.append(watch_block)

    return html.Div(children)


def _build_metrics_header(ticker, spot, session, live, prev_hl, dte, updated,
                          regime=None, vanna_vix=None, charm_clock=None,
                          skew=None, term=None, iv_rank=None, pinning=None):
    """
    Build the header bar cells.
    session:     locked from prev close file (EM, ranges)
    live:        updated every refresh (IV, straddle, P/C, straddle EM)
    regime:      gamma regime classification
    vanna_vix:   vanna/VIX alignment signal
    charm_clock: charm decay clock
    skew:        25d put IV − 25d call IV
    term:        term structure (front/back IV ratio)
    iv_rank:     IV rank and percentile from history
    pinning:     pinning strength for 0DTE
    """
    if not session and not live:
        return "No metrics"

    # Session metrics (from prev close file — frozen for the day)
    pc_spot    = session.get("prev_close_spot", spot)
    pc_iv      = session.get("prev_close_iv", 0)
    pc_ts      = session.get("prev_close_ts", "")
    daily_em   = session.get("daily_em", 0)
    d_hi       = session.get("daily_high", spot)
    d_lo       = session.get("daily_low", spot)

    wc_iv      = session.get("weekly_close_iv", 0)
    wc_ts      = session.get("weekly_close_ts", "")
    weekly_em  = session.get("weekly_em", 0)
    w_hi       = session.get("weekly_high", spot)
    w_lo       = session.get("weekly_low", spot)

    # Live metrics (updating)
    live_iv    = live.get("atm_iv", 0)
    straddle   = live.get("straddle", 0)
    pc_ratio   = live.get("pc_ratio", 0)

    prev_hi = prev_hl.get("high", 0)
    prev_lo = prev_hl.get("low", 0)

    # IV change from prev close
    iv_chg = ""
    if pc_iv > 0 and live_iv > 0:
        diff = (live_iv - pc_iv) * 100
        iv_chg = f"{diff:+.1f}pts"

    # Format source timestamps
    daily_src = f"@ {pc_ts}" if pc_ts else "fallback"
    weekly_src = f"@ {wc_ts}" if wc_ts else "fallback"

    # Regime
    regime = regime or {}
    gamma_sign = regime.get("gamma", "—")
    bias = regime.get("bias", "—")
    conviction = regime.get("conviction", "—")
    gex_flip = regime.get("gex_flip")
    above_flip = regime.get("above_flip")
    total_gex = regime.get("total_gex", 0)

    # Regime color
    if gamma_sign == "POSITIVE":
        regime_color = "#66bb6a"  # green
    elif gamma_sign == "NEGATIVE":
        regime_color = "#ef5350"  # red
    else:
        regime_color = "#888"

    # Flip position text
    if above_flip is True:
        flip_pos = "Above flip (+ territory)"
    elif above_flip is False:
        flip_pos = "Below flip (− territory)"
    else:
        flip_pos = "No flip detected"

    cells = [
        # Ticker + spot
        _metric_cell(ticker, f"${spot:,.2f}", color="#facc15",
                      sub=f"Prev close ${pc_spot:,.1f}"),

        # Gamma Regime
        _metric_cell("Regime", f"{gamma_sign} γ",
                      color=regime_color,
                      sub=f"{bias} | {conviction}"),

        # Vanna / VIX Signal
        _vanna_vix_cell(vanna_vix),

        # Skew (25d put IV − 25d call IV)
        _skew_cell(skew),

        # Term structure (front vs ~30d IV)
        _term_cell(term),

        # IV Rank / Percentile
        _iv_rank_cell(iv_rank),

        # Pinning strength
        _pin_cell(pinning),

        # Live ATM IV + change from prev close
        _metric_cell("ATM IV (live)", f"{live_iv * 100:.1f}%",
                      color="#4dabf7",
                      sub=f"Prev {pc_iv * 100:.1f}%  {iv_chg}"),

        # ATM Straddle — the market's priced-in expected move
        _metric_cell("ATM Straddle", f"±${straddle:,.2f}",
                      color="#ffd54f",
                      sub=f"{straddle / spot * 100:.2f}% of spot" if spot > 0 else ""),

        # Straddle EM range (live — based on current ATM straddle mid)
        _em_progress("Straddle EM",
                     spot,
                     live.get("straddle_em_low", spot),
                     live.get("straddle_em_high", spot),
                     "#ffd54f", "#ffd54f"),

        # Daily EM (from prev close IV — locked for the day)
        _metric_cell("Daily EM (1σ)", f"±${daily_em:,.1f}",
                      color="#a78bfa",
                      sub=f"{daily_em / pc_spot * 100:.2f}%  {daily_src}" if pc_spot > 0 else ""),

        # Daily range with progress
        _em_progress("Daily Range", spot, d_lo, d_hi, "#ef5350", "#66bb6a"),

        # Weekly EM (from Friday close)
        _metric_cell("Weekly EM (1σ)", f"±${weekly_em:,.1f}",
                      color="#a78bfa",
                      sub=f"{weekly_em / pc_spot * 100:.2f}%  {weekly_src}" if pc_spot > 0 else ""),

        # Weekly range with progress
        _em_progress("Weekly Range", spot, w_lo, w_hi, "#ef5350", "#66bb6a"),

        # Put/Call ratio
        _metric_cell("P/C Ratio", f"{pc_ratio:.2f}",
                      color="#66bb6a" if pc_ratio < 1.0 else "#ef5350"),

        # DTE
        _metric_cell("DTE", f"{dte}d", color="#888"),

        # Last update
        _metric_cell("Updated", updated, color="#555"),
    ]

    return cells


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_chart(df, x_col, y_col, title, spot, color_pos, color_neg,
                 lines=None, compact=False, show_spot=True,
                 history_dots=None, y_dtick=None):
    """Horizontal bar chart: Y=strike, X=exposure.
    compact=True uses tighter margins and larger dtick for small panels.
    show_spot=False hides the spot price line.
    y_dtick: override the auto Y-axis spacing (in strike units).

    history_dots: list of dicts [{strike_str: value, ...}, ...] from oldest→newest
        Up to 3 past snapshots shown as dots with fading opacity.

    lines: list of dicts, each with:
        "value":  Y-axis value (strike or price)
        "label":  text label
        "color":  line color
        "side":   "left" or "right" (annotation placement)
        "type":   "exposure_max" | "exposure_min" | "price"
            exposure_max/min: finds the strike with max/min exposure
            price: draws at the exact value given
    """
    df_sorted = df.sort_values(x_col, ascending=True).copy()
    colors = [color_pos if v >= 0 else color_neg for v in df_sorted[y_col]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_sorted[y_col],
        y=df_sorted[x_col],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        opacity=0.85,
        name=title,
    ))

    # ── History dots (past snapshot values as fading circles) ─────────
    if history_dots:
        strikes = df_sorted[x_col].values
        # Opacity: oldest=0.2, newest=0.5
        n_hist = len(history_dots)
        for i, snap in enumerate(history_dots):
            if not snap:
                continue
            opacity = 0.2 + 0.15 * i  # 0.20, 0.35, 0.50
            dot_x = []
            dot_y = []
            for s in strikes:
                val = snap.get(str(int(s)), snap.get(str(float(s))))
                if val is not None:
                    dot_x.append(val)
                    dot_y.append(s)

            if dot_x:
                fig.add_trace(go.Scatter(
                    x=dot_x, y=dot_y,
                    mode="markers",
                    marker=dict(
                        color="white",
                        size=5 if compact else 6,
                        opacity=opacity,
                        line=dict(width=0),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # Spot line
    if show_spot:
        fig.add_shape(
            type="line", x0=0, x1=1, xref="paper",
            y0=spot, y1=spot, yref="y",
            line=dict(color="#facc15", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=1, xref="paper", y=spot, yref="y",
            text=f" ${spot:,.1f}", showarrow=False,
            font=dict(color="#facc15", size=9 if compact else 10), xanchor="left",
        )

    # Indicator lines
    if lines and not df_sorted.empty:
        sz = 8 if compact else 9
        for ln in lines:
            ltype = ln.get("type", "price")
            color = ln["color"]
            label = ln["label"]
            side  = ln.get("side", "left")

            if ltype == "exposure_max":
                idx = df_sorted[y_col].idxmax()
                val = df_sorted.loc[idx, y_col]
                strike = df_sorted.loc[idx, x_col]
                if val <= 0:
                    continue
                y_pos = strike
                txt = f" {label} @ {strike:,.0f}"
            elif ltype == "exposure_min":
                idx = df_sorted[y_col].idxmin()
                val = df_sorted.loc[idx, y_col]
                strike = df_sorted.loc[idx, x_col]
                if val >= 0:
                    continue
                y_pos = strike
                txt = f" {label} @ {strike:,.0f}"
            elif ltype == "net_max":
                # Strike with the largest absolute exposure
                abs_vals = df_sorted[y_col].abs()
                idx = abs_vals.idxmax()
                strike = df_sorted.loc[idx, x_col]
                val = df_sorted.loc[idx, y_col]
                y_pos = strike
                txt = f" {label} @ {strike:,.0f}"
            elif ltype == "net_min":
                # Strike with the smallest absolute exposure (closest to zero)
                abs_vals = df_sorted[y_col].abs()
                idx = abs_vals.idxmin()
                strike = df_sorted.loc[idx, x_col]
                val = df_sorted.loc[idx, y_col]
                y_pos = strike
                txt = f" {label} @ {strike:,.0f}"
            else:  # "price"
                y_pos = ln["value"]
                if y_pos == 0:
                    continue
                txt = f" {label} ${y_pos:,.1f}"

            fig.add_shape(
                type="line", x0=0, x1=1, xref="paper",
                y0=y_pos, y1=y_pos, yref="y",
                line=dict(color=color, width=1.5, dash="dot"),
            )
            x_anchor = "left" if side == "left" else "right"
            x_pos = 0 if side == "left" else 1
            fig.add_annotation(
                x=x_pos, xref="paper", y=y_pos, yref="y",
                text=txt, showarrow=False,
                font=dict(color=color, size=sz),
                xanchor=x_anchor, yshift=10,
            )

    margins = dict(l=50, r=30, t=20, b=15) if compact else dict(l=55, r=40, t=30, b=30)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=margins,
        uirevision="keep",              # preserve zoom/pan across refreshes
        title=dict(text=title, font=dict(size=11 if compact else 13),
                   x=0.01, y=0.97),
        xaxis=dict(
            title=None if compact else "Exposure",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.12)",
            title_font=dict(size=10),
            tickfont=dict(size=9 if compact else 11),
        ),
        yaxis=dict(
            title=None if compact else "Strike",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            title_font=dict(size=10),
            dtick=y_dtick if y_dtick is not None else (10 if compact else 5),
            tickfont=dict(size=9 if compact else 11),
            fixedrange=False,           # allow vertical zoom/pan
        ),
        dragmode="zoom",
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  VALUE VIEW  —  text table with color gradient
# ══════════════════════════════════════════════════════════════════════════════

def _build_value_view(df, x_col, y_col, title, spot, prev_values,
                      compact=False, open_price=0, ultra_compact=False):
    """
    Heatmap-style table: each row is a strike, coloured by exposure value.
    Shows the value, change since last refresh, and change %.

    ultra_compact=True shortens text AND limits to ~20 ATM strikes to fit
    narrow panels like the Market Matrix.
    """
    df_sorted = df.sort_values(x_col, ascending=True).copy()

    # In ultra_compact mode, keep only the 20 strikes closest to spot
    # so the heatmap cells have enough vertical pixels to render
    if ultra_compact and len(df_sorted) > 20:
        df_sorted["_d"] = (df_sorted[x_col] - spot).abs()
        df_sorted = df_sorted.nsmallest(20, "_d").sort_values(x_col, ascending=True)
        df_sorted = df_sorted.drop(columns=["_d"])

    strikes = df_sorted[x_col].values
    values  = df_sorted[y_col].values

    n = len(strikes)
    if n == 0:
        return _empty_fig("No data")

    # Identify special rows
    max_pos_idx = int(np.argmax(values)) if np.any(values > 0) else -1
    max_neg_idx = int(np.argmin(values)) if np.any(values < 0) else -1
    max_net_idx = int(np.argmax(np.abs(values)))

    # Spot row
    spot_dists = np.abs(strikes - spot)
    spot_idx = int(np.argmin(spot_dists))

    # Build text for each cell (no badges — those go as annotations)
    cell_text = []
    for i in range(n):
        s = strikes[i]
        v = values[i]
        prev_v = prev_values.get(str(s), None)

        val_str = _fmt_value(v)

        if prev_v is not None and prev_v != 0:
            chg = v - prev_v
            chg_pct = (chg / abs(prev_v)) * 100
            sign = "+" if chg >= 0 else ""
            if ultra_compact:
                # Just value and % — skip absolute change to save horizontal space
                cell_text.append(f"${val_str}  {sign}{chg_pct:.1f}%")
            else:
                cell_text.append(
                    f"${val_str}   {sign}{_fmt_value(chg)}  ({sign}{chg_pct:.1f}%)"
                )
        else:
            cell_text.append(f"${val_str}")

    # Single column z-values
    z = values.reshape(-1, 1)
    z_max = max(abs(np.nanmin(z)), abs(np.nanmax(z)), 1)

    text_grid = [[t] for t in cell_text]

    strike_labels = [f"{s:,.0f}" for s in strikes]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=z,
        y=strike_labels,
        x=[title],
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(
            size=8 if ultra_compact else (10 if compact else 12),
            family="monospace",
            color="#ffffff",
        ),
        colorscale=[
            [0.0,  "#7f0000"],
            [0.15, "#b71c1c"],
            [0.3,  "#5a1010"],
            [0.45, "#2a0808"],
            [0.5,  "#121212"],
            [0.55, "#082a08"],
            [0.7,  "#105a10"],
            [0.85, "#1b8c1b"],
            [1.0,  "#43a047"],
        ],
        zmin=-z_max,
        zmax=z_max,
        showscale=False,
        hovertemplate="Strike: %{y}<br>%{text}<extra></extra>",
        ygap=1,
    ))

    # ── Badge annotations on the left edge ───────────────────────────
    sz = 11 if compact else 13
    badge_map = {}
    if max_pos_idx >= 0 and values[max_pos_idx] > 0:
        badge_map.setdefault(max_pos_idx, []).append("▲")
    if max_neg_idx >= 0 and values[max_neg_idx] < 0:
        badge_map.setdefault(max_neg_idx, []).append("▼")
    badge_map.setdefault(max_net_idx, []).append("◆")

    for idx, badges in badge_map.items():
        fig.add_annotation(
            x=0, xref="paper",
            y=strike_labels[idx], yref="y",
            text=" ".join(badges),
            showarrow=False,
            font=dict(color="#ffffff", size=sz),
            xanchor="left",
            xshift=4,
        )

    # ── Spot price row highlight ─────────────────────────────────────
    fig.add_shape(
        type="rect",
        x0=-0.5, x1=0.5,
        y0=spot_idx - 0.5, y1=spot_idx + 0.5,
        yref="y",
        line=dict(color="#facc15", width=3),
        fillcolor="rgba(0,0,0,0)",
    )

    # ── Open price row highlight ─────────────────────────────────────
    if open_price > 0:
        open_dists = np.abs(strikes - open_price)
        open_idx = int(np.argmin(open_dists))
        fig.add_shape(
            type="rect",
            x0=-0.5, x1=0.5,
            y0=open_idx - 0.5, y1=open_idx + 0.5,
            yref="y",
            line=dict(color="#29b6f6", width=2, dash="dash"),
            fillcolor="rgba(0,0,0,0)",
        )

    margins = dict(l=55, r=10, t=25, b=10) if compact else dict(l=60, r=15, t=30, b=15)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=margins,
        uirevision="keep",
        title=dict(text=title, font=dict(size=11 if compact else 13),
                   x=0.01, y=0.98),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
        ),
        yaxis=dict(
            title=None,
            showgrid=False,
            type="category",
            tickfont=dict(size=9 if compact else 10),
            fixedrange=False,
        ),
        showlegend=False,
    )
    return fig


def _fmt_value(v):
    """Format large numbers with K/M suffix."""
    av = abs(v)
    if av >= 1e9:
        return f"{v/1e9:,.2f}B"
    elif av >= 1e6:
        return f"{v/1e6:,.2f}M"
    elif av >= 1e3:
        return f"{v/1e3:,.1f}K"
    else:
        return f"{v:,.0f}"


# ══════════════════════════════════════════════════════════════════════════════
#  CHARM HEATMAP + CANDLESTICK
# ══════════════════════════════════════════════════════════════════════════════

def _build_charm_heatmap(history, spot, chain=None, greek_mode="oi"):
    """
    Compute charm heatmap for the full NYSE session (9:30-16:00)
    from the current chain snapshot.

    Since charm = f(S, K, T, sigma) and only T changes over the day:
    - Past (9:30 -> now): DTE was higher -> reconstructed by adding time back
    - Future (now -> 16:00): DTE shrinks -> projected by subtracting time
    """
    from scipy.ndimage import gaussian_filter, zoom
    from greek_calculator import project_charm_forward
    import pandas as pd

    if chain is None or chain.empty:
        return _empty_fig("No chain data for heatmap")

    # All market times in US Eastern
    now_et = dt.datetime.now(tz=ET)
    today_et = now_et.date()
    market_open  = dt.datetime.combine(today_et, dt.time(9, 30), tzinfo=ET)
    market_close = dt.datetime.combine(today_et, dt.time(16, 0), tzinfo=ET)

    # Clamp "now" to market hours
    now_clamped = max(now_et, market_open)
    now_clamped = min(now_clamped, market_close)

    mins_since_open = max(int((now_clamped - market_open).total_seconds() / 60), 0)
    mins_to_close   = max(int((market_close - now_clamped).total_seconds() / 60), 0)

    total_mins = mins_since_open + mins_to_close
    step = 5 if total_mins <= 180 else 10

    # Compute full-day charm grid (backward + forward)
    proj = project_charm_forward(
        chain, spot, greek_mode=greek_mode,
        minutes_ahead=mins_to_close,
        minutes_behind=mins_since_open,
        step_minutes=step,
    )

    strikes = proj["strikes"]
    offsets = proj["offsets_min"]
    z_raw   = proj["charm_grid"]
    now_idx = proj["now_index"]

    # Convert offsets to real timestamps
    times = [now_clamped + dt.timedelta(minutes=o) for o in offsets]

    # Smooth
    n_s, n_t = z_raw.shape
    if n_s >= 2 and n_t >= 2:
        up_y = max(1, min(4, 200 // n_s))
        up_x = max(1, min(3, 300 // n_t))
        z_up = zoom(z_raw, (up_y, up_x), order=1)
        z_smooth = gaussian_filter(z_up, sigma=(3, 2))
        strikes_up = np.linspace(strikes[0], strikes[-1], z_smooth.shape[0])
        times_up = pd.date_range(times[0], times[-1], periods=z_smooth.shape[1])
    else:
        z_smooth = z_raw
        strikes_up = np.array(strikes)
        times_up = times

    z_max = max(abs(np.nanmin(z_smooth)), abs(np.nanmax(z_smooth)), 1)

    # OHLC candles from recorded history
    candles = _build_ohlc(history, dt.timedelta(minutes=5))

    # Build figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Heatmap(
            x=times_up, y=strikes_up, z=z_smooth,
            zsmooth="best",
            colorscale=[
                [0.0,  "#0a1628"], [0.15, "#0d47a1"],
                [0.3,  "#1565c0"], [0.42, "#1a237e"],
                [0.5,  "#0d0d0d"],
                [0.58, "#4a3800"], [0.7,  "#f9a825"],
                [0.85, "#fdd835"], [1.0,  "#ffee58"],
            ],
            zmin=-z_max, zmax=z_max,
            colorbar=dict(title="Charm", len=0.5, thickness=8, x=1.02),
            hovertemplate="Time: %{x|%H:%M}<br>Strike: %{y:.0f}<br>Charm: %{z:.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    if candles:
        fig.add_trace(
            go.Candlestick(
                x=[c["time"] for c in candles],
                open=[c["open"] for c in candles],
                high=[c["high"] for c in candles],
                low=[c["low"] for c in candles],
                close=[c["close"] for c in candles],
                increasing_line_color="#00e676",
                decreasing_line_color="#ff1744",
                increasing_fillcolor="#00e676",
                decreasing_fillcolor="#ff1744",
                line_width=1, opacity=0.9, name="Price",
            ),
            secondary_y=True,
        )

    # Max Pos / Max Neg / Net charm lines (from the "now" column of the grid)
    now_col = z_raw[:, now_idx] if now_idx < z_raw.shape[1] else z_raw[:, -1]
    pos_idx = int(np.argmax(now_col))
    neg_idx = int(np.argmin(now_col))
    abs_col = np.abs(now_col)
    net_max_idx = int(np.argmax(abs_col))
    net_min_idx = int(np.argmin(abs_col))

    # Max Pos Charm
    if now_col[pos_idx] > 0:
        s = strikes[pos_idx]
        fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                      y0=s, y1=s, yref="y",
                      line=dict(color="#4dabf7", width=1.5, dash="dot"))
        fig.add_annotation(x=1, xref="paper", y=s, yref="y",
                           text=f" Max Pos Charm @ {s:,.0f}", showarrow=False,
                           font=dict(color="#4dabf7", size=8),
                           xanchor="left", yshift=8)

    # Max Neg Charm
    if now_col[neg_idx] < 0:
        s = strikes[neg_idx]
        fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                      y0=s, y1=s, yref="y",
                      line=dict(color="#f783ac", width=1.5, dash="dot"))
        fig.add_annotation(x=1, xref="paper", y=s, yref="y",
                           text=f" Max Neg Charm @ {s:,.0f}", showarrow=False,
                           font=dict(color="#f783ac", size=8),
                           xanchor="left", yshift=-8)

    # Max Net Charm (largest absolute)
    s = strikes[net_max_idx]
    fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                  y0=s, y1=s, yref="y",
                  line=dict(color="#e0e0e0", width=1.5, dash="dot"))
    fig.add_annotation(x=1, xref="paper", y=s, yref="y",
                       text=f" Max Net Charm @ {s:,.0f}", showarrow=False,
                       font=dict(color="#e0e0e0", size=8),
                       xanchor="left", yshift=8)

    # Min Net Charm (smallest absolute / closest to zero)
    s = strikes[net_min_idx]
    fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                  y0=s, y1=s, yref="y",
                  line=dict(color="#9e9e9e", width=1.5, dash="dot"))
    fig.add_annotation(x=1, xref="paper", y=s, yref="y",
                       text=f" Min Net Charm @ {s:,.0f}", showarrow=False,
                       font=dict(color="#9e9e9e", size=8),
                       xanchor="left", yshift=-8)

    # Layout — Y-axis is zoomable/pannable (drag to stretch vertically)
    s_min, s_max = min(strikes), max(strikes)
    pad = (s_max - s_min) * 0.05
    y_range = [s_min - pad, s_max + pad]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=60, t=30, b=30),
        uirevision="keep",
        title=dict(text="Charm Decay Heatmap", font=dict(size=13),
                   x=0.01, y=0.98),
        xaxis=dict(
            title="Time",
            gridcolor="rgba(255,255,255,0.06)",
            title_font=dict(size=10),
            range=[market_open, market_close],
            fixedrange=True,
            dtick=30 * 60 * 1000,
            tickformat="%H:%M",
        ),
        yaxis=dict(title="Strike", gridcolor="rgba(255,255,255,0.06)",
                   title_font=dict(size=10), range=y_range, dtick=5,
                   fixedrange=False),     # allow vertical drag to zoom
        yaxis2=dict(range=y_range, showgrid=False,
                    showticklabels=False, overlaying="y",
                    fixedrange=False),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        dragmode="zoom",                  # default drag = zoom (like TradingView)
    )

    return fig



def _build_ohlc(history, interval):
    """
    Bucket spot prices from history snapshots into OHLC candles.
    Returns list of {"time", "open", "high", "low", "close"}.
    """
    if not history:
        return []

    candles = []
    bucket_start = history[0]["time"]
    bucket_prices = []

    for snap in history:
        if snap["time"] - bucket_start >= interval and bucket_prices:
            candles.append({
                "time":  bucket_start + interval / 2,  # center of bucket
                "open":  bucket_prices[0],
                "high":  max(bucket_prices),
                "low":   min(bucket_prices),
                "close": bucket_prices[-1],
            })
            bucket_start = snap["time"]
            bucket_prices = []
        bucket_prices.append(snap["spot"])

    # Final partial bucket
    if bucket_prices:
        candles.append({
            "time":  bucket_start + interval / 2,
            "open":  bucket_prices[0],
            "high":  max(bucket_prices),
            "low":   min(bucket_prices),
            "close": bucket_prices[-1],
        })

    return candles


def _empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False,
                       font=dict(size=14, color="grey"))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=40, t=30, b=30),
        uirevision="keep",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_fetcher.init_data_manager(use_mock=True)
    matrix_data.init_matrix_manager(use_mock=True)
    cot_scraper.init_cot_manager()
    market_state.init_market_state_manager(use_mock=True)
    heatmap_data.init_heatmap_manager(use_mock=True)
    app.run(debug=True, host="0.0.0.0", port=8050)
