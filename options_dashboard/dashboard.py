"""
dashboard.py  –  Plotly-Dash web dashboard.

Layout:  Sidebar (left)  |  Gamma  |  Charm  |  Vanna   (3 charts side by side)

Charm panel can toggle between bar chart and heatmap (time x strike)
with candlestick price overlay.
"""

import datetime as dt
import numpy as np

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import SETTINGS, SIDEBAR_WIDTH, ET
import data_fetcher
from greek_calculator import compute_exposure, compute_live_metrics

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Options Greek Dashboard",
    suppress_callback_exceptions=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

sidebar = html.Div(
    [
        html.H5("Settings", className="mb-3",
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

        # ── Per-chart view modes ─────────────────────────────────────
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
                {"label": "Bar",     "value": "bar"},
                {"label": "Heatmap", "value": "heatmap"},
                {"label": "Values",  "value": "values"},
            ],
            value="heatmap", inline=True, className="mb-2",
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

        # Hidden store for previous exposure values (for computing deltas)
        dcc.Store(id="store-prev-exposure", data={}),
    ],
    style={
        "width": f"{SIDEBAR_WIDTH}px",
        "minWidth": f"{SIDEBAR_WIDTH}px",
        "height": "100vh",
        "overflowY": "auto",
        "padding": "12px",
        "borderRight": "1px solid rgba(255,255,255,0.08)",
    },
)

# Layout:  Sidebar | Gamma       | Charm (70%)
#                                | Vanna | Zomma (30%)
charts_panel = html.Div(
    [
        # Left column: Gamma (full height)
        html.Div(
            dcc.Graph(id="chart-gamma", style={"height": "100%"}),
            style={"flex": 0.7, "minWidth": 0},
        ),
        # Right column: Charm on top, Vanna+Zomma on bottom
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="chart-charm", style={"height": "100%"}),
                    style={"flex": 7, "minHeight": 0},
                ),
                # Bottom row: Vanna + Zomma side by side
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(id="chart-vanna", style={"height": "100%"}),
                            style={"flex": 1, "minWidth": 0, "overflowY": "auto"},
                        ),
                        html.Div(
                            dcc.Graph(id="chart-zomma", style={"height": "100%"}),
                            style={"flex": 1, "minWidth": 0, "overflowY": "auto"},
                        ),
                    ],
                    style={
                        "flex": 3,
                        "minHeight": 0,
                        "display": "flex",
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

# ── Metrics header bar (populated by callback) ──────────────────────────────

metrics_header = html.Div(
    id="metrics-header",
    children="Loading...",
    style={
        "display": "flex",
        "alignItems": "center",
        "padding": "4px 8px",
        "borderBottom": "1px solid rgba(255,255,255,0.08)",
        "backgroundColor": "rgba(0,0,0,0.3)",
        "minHeight": "42px",
        "flexShrink": 0,
        "overflowX": "auto",
        "whiteSpace": "nowrap",
        "fontSize": "0.80rem",
    },
)

poll_timer = dcc.Interval(id="interval-poll", interval=2_000, n_intervals=0)

# Main layout: Sidebar | (Header + Charts)
app.layout = html.Div(
    [
        poll_timer,
        sidebar,
        html.Div(
            [metrics_header, charts_panel],
            style={
                "flex": 1,
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
                "overflow": "hidden",
            },
        ),
    ],
    style={"display": "flex", "height": "100vh", "overflow": "hidden"},
)


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

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
    Output("chart-vanna", "figure"),
    Output("chart-zomma", "figure"),
    Output("metrics-header", "children"),
    Output("status-text", "children"),
    Output("dropdown-expiry", "options", allow_duplicate=True),
    Output("store-prev-exposure", "data"),
    Input("interval-poll", "n_intervals"),
    Input("radio-gamma-view", "value"),
    Input("radio-charm-view", "value"),
    Input("radio-vanna-view", "value"),
    Input("radio-zomma-view", "value"),
    State("store-prev-exposure", "data"),
    prevent_initial_call="initial_duplicate",
)
def poll_and_render(n, gamma_view, charm_view, vanna_view, zomma_view, prev_data):
    if not data_fetcher.data_manager:
        e = _empty_fig("Starting...")
        return e, e, e, e, "Loading...", "Initialising...", dash.no_update, dash.no_update

    cache = data_fetcher.data_manager.get_cache()

    error = cache.get("error")
    if error:
        e = _empty_fig(f"Error: {error}")
        return e, e, e, e, "Error", f"X  {error}", dash.no_update, dash.no_update

    chain = cache.get("chain")
    if chain is None or (hasattr(chain, "empty") and chain.empty):
        e = _empty_fig("Waiting for data...")
        return e, e, e, e, "Waiting...", "Fetching from IB...", dash.no_update, dash.no_update

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
    if prev_data is None:
        prev_data = {}
    prev_gamma = prev_data.get("gamma", {})
    prev_charm = prev_data.get("charm", {})
    prev_vanna = prev_data.get("vanna", {})
    prev_zomma = prev_data.get("zomma", {})

    # Save current values for next cycle
    new_prev = {
        "gamma": {str(r["strike"]): r["gamma_exp"] for _, r in exp_df.iterrows()},
        "charm": {str(r["strike"]): r["charm_exp"] for _, r in exp_df.iterrows()},
        "vanna": {str(r["strike"]): r["vanna_exp"] for _, r in exp_df.iterrows()},
        "zomma": {str(r["strike"]): r["zomma_exp"] for _, r in exp_df.iterrows()},
    }

    # Common open price line for bar charts
    open_line = {"type": "price", "value": open_spot,
                 "label": "Open", "color": "#29b6f6", "side": "right"}

    # ── Gamma ────────────────────────────────────────────────────────
    if gamma_view == "values":
        fig_gamma = _build_value_view(exp_df, "strike", "gamma_exp",
                                       "Gamma (GEX)", spot, prev_gamma,
                                       open_price=open_spot)
    else:
        fig_gamma = _build_chart(exp_df, "strike", "gamma_exp",
                                  "Gamma (GEX)", spot, "#00d4aa", "#ff4d6a",
                                  lines=[
                                      {"type": "exposure_max", "label": "Max Pos GEX",
                                       "color": "#00ffcc", "side": "left"},
                                      {"type": "exposure_min", "label": "Max Neg GEX",
                                       "color": "#ff6b6b", "side": "left"},
                                      {"type": "net_max", "label": "Max Net GEX",
                                       "color": "#e0e0e0", "side": "left"},
                                      {"type": "net_min", "label": "Min Net GEX",
                                       "color": "#9e9e9e", "side": "left"},
                                      {"type": "price", "value": prev_hl["high"],
                                       "label": "Prev High",
                                       "color": "#80cbc4", "side": "right"},
                                      {"type": "price", "value": prev_hl["low"],
                                       "label": "Prev Low",
                                       "color": "#ef9a9a", "side": "right"},
                                      open_line,
                                  ])

    # ── Charm ────────────────────────────────────────────────────────
    if charm_view == "values":
        fig_charm = _build_value_view(exp_df, "strike", "charm_exp",
                                       "Charm", spot, prev_charm,
                                       open_price=open_spot)
    elif charm_view == "heatmap":
        history = data_fetcher.data_manager.get_charm_history()
        # In flow/oi_flow mode, project charm from the appropriate chain
        if mode == "flow" and flow_chain is not None and not flow_chain.empty:
            fig_charm = _build_charm_heatmap(history, spot,
                                              chain=flow_chain, greek_mode="oi")
        elif mode == "oi_flow" and oi_flow_chain is not None and not oi_flow_chain.empty:
            fig_charm = _build_charm_heatmap(history, spot,
                                              chain=oi_flow_chain, greek_mode="oi")
        else:
            fig_charm = _build_charm_heatmap(history, spot,
                                              chain=chain, greek_mode=mode)
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
                                  show_spot=False)

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
                                  compact=True)

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
                                  compact=True, show_spot=False)

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

    header = _build_metrics_header(ticker, spot, session_metrics,
                                    live_metrics, prev_hl, dte, updated)

    return fig_gamma, fig_charm, fig_vanna, fig_zomma, header, status, opts, new_prev


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


def _build_metrics_header(ticker, spot, session, live, prev_hl, dte, updated):
    """
    Build the header bar cells.
    session: locked from prev close file (EM, ranges)
    live:    updated every refresh (IV, straddle, P/C)
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

    cells = [
        # Ticker + spot
        _metric_cell(ticker, f"${spot:,.2f}", color="#facc15",
                      sub=f"Prev close ${pc_spot:,.1f}"),

        # Live ATM IV + change from prev close
        _metric_cell("ATM IV (live)", f"{live_iv * 100:.1f}%",
                      color="#4dabf7",
                      sub=f"Prev {pc_iv * 100:.1f}%  {iv_chg}"),

        # Straddle price
        _metric_cell("ATM Straddle", f"${straddle:,.2f}",
                      color="#ffd54f",
                      sub=f"{straddle / spot * 100:.2f}% of spot" if spot > 0 else ""),

        # Daily EM (from prev close)
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

        # Previous day H/L
        _metric_cell("Prev High", f"${prev_hi:,.1f}",
                      color="#80cbc4" if prev_hi > 0 else "#555",
                      sub=f"{(spot - prev_hi) / prev_hi * 100:+.2f}%" if prev_hi > 0 else ""),
        _metric_cell("Prev Low", f"${prev_lo:,.1f}",
                      color="#ef9a9a" if prev_lo > 0 else "#555",
                      sub=f"{(spot - prev_lo) / prev_lo * 100:+.2f}%" if prev_lo > 0 else ""),

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
                 lines=None, compact=False, show_spot=True):
    """Horizontal bar chart: Y=strike, X=exposure.
    compact=True uses tighter margins and larger dtick for small panels.
    show_spot=False hides the spot price line.

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
            dtick=10 if compact else 5,
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
                      compact=False, open_price=0):
    """
    Heatmap-style table: each row is a strike, coloured by exposure value.
    Shows the value, change since last refresh, and change %.
    """
    df_sorted = df.sort_values(x_col, ascending=True).copy()

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
            size=10 if compact else 12,
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
    app.run(debug=True, host="0.0.0.0", port=8050)
