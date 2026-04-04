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
from greek_calculator import compute_exposure

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
            ],
            value=SETTINGS.greek_mode,
            className="mb-3",
        ),

        html.Hr(),

        # ── Charm view toggle ────────────────────────────────────────
        dbc.Label("Charm View", className="fw-bold",
                  style={"fontSize": "0.85rem"}),
        dbc.RadioItems(
            id="radio-charm-view",
            options=[
                {"label": "Bar Chart",  "value": "bar"},
                {"label": "Heatmap",    "value": "heatmap"},
            ],
            value="heatmap",
            inline=True,
            className="mb-3",
        ),
        html.Div(
            "Heatmap shows history + projected decay to close.",
            className="text-muted",
            style={"fontSize": "0.70rem", "fontStyle": "italic"},
        ),

        html.Hr(),
        html.Div(id="status-text", className="text-muted",
                 style={"fontSize": "0.75rem", "whiteSpace": "pre-line"}),
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

# Layout:  Sidebar | Gamma | Charm (top 70%)
#                          | Vanna (bottom 30%)
charts_panel = html.Div(
    [
        # Left column: Gamma (full height, narrower)
        html.Div(
            dcc.Graph(id="chart-gamma", style={"height": "100%"}),
            style={"flex": 0.7, "minWidth": 0, "height": "100vh"},
        ),
        # Right column: Charm stacked above Vanna (wider)
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="chart-charm", style={"height": "100%"}),
                    style={"flex": 7, "minHeight": 0},   # 70%
                ),
                html.Div(
                    dcc.Graph(id="chart-vanna", style={"height": "100%"}),
                    style={"flex": 3, "minHeight": 0},   # 30%
                ),
            ],
            style={
                "flex": 1.8,
                "minWidth": 0,
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
            },
        ),
    ],
    style={
        "display": "flex",
        "flex": 1,
        "height": "100vh",
        "overflow": "hidden",
    },
)

poll_timer = dcc.Interval(id="interval-poll", interval=2_000, n_intervals=0)

app.layout = html.Div(
    [poll_timer, sidebar, charts_panel],
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
    Output("status-text", "children"),
    Output("dropdown-expiry", "options", allow_duplicate=True),
    Input("interval-poll", "n_intervals"),
    Input("radio-charm-view", "value"),
    prevent_initial_call="initial_duplicate",
)
def poll_and_render(n, charm_view):
    if not data_fetcher.data_manager:
        e = _empty_fig("Starting...")
        return e, e, e, "Initialising...", dash.no_update

    cache = data_fetcher.data_manager.get_cache()

    error = cache.get("error")
    if error:
        e = _empty_fig(f"Error: {error}")
        return e, e, e, f"X  {error}", dash.no_update

    chain = cache.get("chain")
    if chain is None or (hasattr(chain, "empty") and chain.empty):
        e = _empty_fig("Waiting for data...")
        return e, e, e, "Fetching from IB...", dash.no_update

    ticker   = cache["ticker"]
    spot     = cache["spot"]
    resolved = cache["expiry"]
    expiries = cache.get("expiries", [])
    mode     = SETTINGS.greek_mode

    exp_df = compute_exposure(chain, spot, greek_mode=mode)

    # ── Gamma (bar) ──────────────────────────────────────────────────
    fig_gamma = _build_chart(exp_df, "strike", "gamma_exp",
                              "Gamma (GEX)", spot, "#00d4aa", "#ff4d6a",
                              max_label="Max GEX")

    # ── Charm (bar or heatmap) ───────────────────────────────────────
    if charm_view == "heatmap":
        history = data_fetcher.data_manager.get_charm_history()
        fig_charm = _build_charm_heatmap(history, spot,
                                          chain=chain, greek_mode=mode)
    else:
        fig_charm = _build_chart(exp_df, "strike", "charm_exp",
                                  "Charm", spot, "#4dabf7", "#f783ac")

    # ── Vanna (bar) ──────────────────────────────────────────────────
    fig_vanna = _build_chart(exp_df, "strike", "vanna_exp",
                              "Vanna", spot, "#a78bfa", "#fb923c",
                              compact=True)

    # ── Status ───────────────────────────────────────────────────────
    nice_exp = f"{resolved[:4]}-{resolved[4:6]}-{resolved[6:]}"
    exp_date = dt.date(int(resolved[:4]), int(resolved[4:6]), int(resolved[6:]))
    dte = max((exp_date - dt.date.today()).days, 0)
    mode_label = {"oi": "Open Interest", "volume": "Volume",
                  "combined": "OI+Vol"}[mode]
    ts = cache.get("timestamp", 0)
    updated = dt.datetime.fromtimestamp(ts, tz=ET).strftime("%H:%M:%S ET") if ts else "-"

    history_len = len(data_fetcher.data_manager.get_charm_history()) \
        if data_fetcher.data_manager else 0

    status = (
        f"Ticker: {ticker}\n"
        f"Spot: ${spot:,.2f}\n"
        f"Expiry: {nice_exp} ({dte} DTE)\n"
        f"Mode: {mode_label}\n"
        f"Strikes: {len(exp_df)}\n"
        f"Heatmap: {history_len} snapshots\n"
        f"Last fetch: {updated}"
    )

    opts = [{"label": "Auto (nearest)", "value": "auto"}]
    for e in expiries:
        opts.append({"label": f"{e[:4]}-{e[4:6]}-{e[6:]}", "value": e})

    return fig_gamma, fig_charm, fig_vanna, status, opts


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_chart(df, x_col, y_col, title, spot, color_pos, color_neg,
                 max_label=None, compact=False):
    """Horizontal bar chart: Y=strike, X=exposure.
    compact=True uses tighter margins and larger dtick for small panels."""
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

    # Max exposure line
    if max_label and not df_sorted.empty:
        max_idx = df_sorted[y_col].idxmax()
        max_strike = df_sorted.loc[max_idx, x_col]
        max_val = df_sorted.loc[max_idx, y_col]
        if max_val > 0:
            fig.add_shape(
                type="line", x0=0, x1=1, xref="paper",
                y0=max_strike, y1=max_strike, yref="y",
                line=dict(color="#00ffcc", width=1.5, dash="dot"),
            )
            fig.add_annotation(
                x=0, xref="paper", y=max_strike, yref="y",
                text=f" {max_label} @ {max_strike:,.0f}",
                showarrow=False,
                font=dict(color="#00ffcc", size=9),
                xanchor="left", yshift=10,
            )

    margins = dict(l=50, r=30, t=20, b=15) if compact else dict(l=55, r=40, t=30, b=30)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=margins,
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
        ),
        showlegend=False,
    )
    return fig


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

    # "NOW" line
    fig.add_shape(
        type="line", x0=now_clamped, x1=now_clamped,
        y0=0, y1=1, yref="paper",
        line=dict(color="#ffffff", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=now_clamped, y=1, yref="paper",
        text="  NOW  ", showarrow=False,
        font=dict(color="#ffffff", size=9, family="monospace"),
        bgcolor="rgba(255,255,255,0.15)",
        xanchor="center", yanchor="bottom",
    )

    # Spot line
    fig.add_shape(
        type="line", x0=0, x1=1, xref="paper",
        y0=spot, y1=spot, yref="y",
        line=dict(color="#facc15", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=0, xref="paper", y=spot, yref="y",
        text=f"${spot:,.1f} ", showarrow=False,
        font=dict(color="#facc15", size=9), xanchor="right",
    )

    # Layout
    s_min, s_max = min(strikes), max(strikes)
    pad = (s_max - s_min) * 0.05
    y_range = [s_min - pad, s_max + pad]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=60, t=30, b=30),
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
                   title_font=dict(size=10), range=y_range, dtick=5),
        yaxis2=dict(range=y_range, showgrid=False,
                    showticklabels=False, overlaying="y"),
        showlegend=False,
        xaxis_rangeslider_visible=False,
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
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_fetcher.init_data_manager(use_mock=True)
    app.run(debug=True, host="0.0.0.0", port=8050)
