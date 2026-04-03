"""
config.py  –  Central configuration for the Options Greek Dashboard.
"""

from dataclasses import dataclass


# ── IB Gateway / TWS connection ──────────────────────────────────────────────
IB_HOST = "127.0.0.1"
IB_PORT = 7496               # 7497 = paper trading  |  7496 = live trading
IB_CLIENT_ID = 1

# ── Risk-free rate used in Black-Scholes (annualised) ────────────────────────
RISK_FREE_RATE = 0.045

# ── Index tickers (need Index contract instead of Stock) ─────────────────────
INDEX_TICKERS = {"SPX", "NDX", "RUT", "VIX", "DJX", "XSP"}


@dataclass
class DashboardSettings:
    """Mutable settings that the sidebar controls change on the fly."""
    ticker: str = "SPX"
    expiry: str = "auto"
    refresh_seconds: int = 10
    greek_mode: str = "oi"               # "oi" | "volume" | "combined"
    resolved_expiry: str = ""
    spot_price: float = 0.0


SETTINGS = DashboardSettings()


# ── Dashboard appearance ─────────────────────────────────────────────────────
SIDEBAR_WIDTH = 190           # px
THEME = "darkly"
