"""
config.py  –  Central configuration for the Options Greek Dashboard.
"""

from dataclasses import dataclass
from datetime import timezone, timedelta


# ── IB Gateway / TWS connection ──────────────────────────────────────────────
IB_HOST = "127.0.0.1"
IB_PORT = 7496               # 7497 = paper trading  |  7496 = live trading
IB_CLIENT_ID = 1

# ── Risk-free rate used in Black-Scholes (annualised) ────────────────────────
RISK_FREE_RATE = 0.045

# ── Index tickers (need Index contract instead of Stock) ─────────────────────
INDEX_TICKERS = {"SPX", "NDX", "RUT", "VIX", "DJX", "XSP"}

# ── NYSE timezone (US Eastern) ───────────────────────────────────────────────
# EDT = UTC-4 (March–November),  EST = UTC-5 (November–March)
# Change this offset when daylight saving switches, or install `pytz`/`zoneinfo`
# and use a proper timezone.  EDT for most of the trading year:
ET = timezone(timedelta(hours=-4))     # EDT (Apr–Nov)
# ET = timezone(timedelta(hours=-5))   # EST (Nov–Mar)


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
