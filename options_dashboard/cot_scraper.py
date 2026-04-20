"""
cot_scraper.py  –  Commitments of Traders data fetcher.

Pulls data from CFTC's public API (Socrata) — no auth token needed.
Data is released every Friday at 3:30pm EST for the previous Tuesday.

Reports used:
  - Legacy Futures Only (6dca-aqww)  — broadest, most common report
  - TFF Futures Only (gpe5-46if)     — financials with Asset Mgr / Leveraged Funds

We fetch the most recent report date, and the prior week for week-over-week
change calculations.
"""

import threading
import time
import urllib.request
import urllib.parse
import json
import traceback
import datetime as dt

from config import ET


# ── API endpoints (Socrata JSON) ─────────────────────────────────────────────
API_HOST = "https://publicreporting.cftc.gov/resource"
LEGACY_FUTURES_ID = "6dca-aqww"

# Symbols we care about — mapped from CFTC contract names to display symbols
# The CFTC uses long contract names; this maps them to the short tickers users know.
TRACKED_SYMBOLS = {
    # Equity indices
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "ES (S&P 500)",
    "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE": "NQ (Nasdaq)",
    "DJIA x $5 - CHICAGO BOARD OF TRADE": "YM (Dow)",
    "E-MINI RUSSELL 2000 INDEX - CHICAGO MERCANTILE EXCHANGE": "RTY (Russell)",
    "VIX FUTURES - CBOE FUTURES EXCHANGE": "VX (VIX)",
    # Currencies
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "6E (Euro)",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "6J (Yen)",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "6B (Pound)",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6C (CAD)",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6A (AUD)",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "6S (CHF)",
    # Metals
    "GOLD - COMMODITY EXCHANGE INC.": "GC (Gold)",
    "SILVER - COMMODITY EXCHANGE INC.": "SI (Silver)",
    "COPPER- #1 - COMMODITY EXCHANGE INC.": "HG (Copper)",
    "PLATINUM - NEW YORK MERCANTILE EXCHANGE": "PL (Platinum)",
    # Energy
    "CRUDE OIL, LIGHT SWEET-WTI - NEW YORK MERCANTILE EXCHANGE": "CL (WTI)",
    "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE": "NG (NatGas)",
    # Rates
    "UST BOND - CHICAGO BOARD OF TRADE": "ZB (30Y Bond)",
    "UST 10Y NOTE - CHICAGO BOARD OF TRADE": "ZN (10Y Note)",
    "UST 5Y NOTE - CHICAGO BOARD OF TRADE": "ZF (5Y Note)",
    "UST 2Y NOTE - CHICAGO BOARD OF TRADE": "ZT (2Y Note)",
    # Ags
    "CORN - CHICAGO BOARD OF TRADE": "ZC (Corn)",
    "SOYBEANS - CHICAGO BOARD OF TRADE": "ZS (Soybeans)",
    "WHEAT-SRW - CHICAGO BOARD OF TRADE": "ZW (Wheat)",
}


# ── Low-level API fetch ──────────────────────────────────────────────────────

def _fetch_json(url: str, timeout: int = 20) -> list:
    """Fetch JSON from Socrata API, returning a list of rows."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _latest_report_date() -> str | None:
    """Get the most recent report_date from the Legacy Futures Only dataset."""
    url = (f"{API_HOST}/{LEGACY_FUTURES_ID}.json"
           f"?$select=report_date_as_yyyy_mm_dd"
           f"&$order=report_date_as_yyyy_mm_dd DESC"
           f"&$limit=1")
    rows = _fetch_json(url)
    if not rows:
        return None
    return rows[0].get("report_date_as_yyyy_mm_dd", "")[:10]


def _fetch_report_for_date(report_date: str) -> list:
    """Fetch all rows for a given report_date."""
    url = (f"{API_HOST}/{LEGACY_FUTURES_ID}.json"
           f"?report_date_as_yyyy_mm_dd={report_date}T00:00:00.000"
           f"&$limit=5000")
    return _fetch_json(url)


def _fetch_prior_week(current_date: str) -> list:
    """Fetch the report 7 days before current_date for change calculations."""
    try:
        cur = dt.datetime.strptime(current_date, "%Y-%m-%d")
    except ValueError:
        return []
    prior = (cur - dt.timedelta(days=7)).strftime("%Y-%m-%d")
    url = (f"{API_HOST}/{LEGACY_FUTURES_ID}.json"
           f"?report_date_as_yyyy_mm_dd={prior}T00:00:00.000"
           f"&$limit=5000")
    try:
        return _fetch_json(url)
    except Exception:
        return []


# ── Row processing ───────────────────────────────────────────────────────────

def _safe_int(val) -> int:
    """Parse a value to int, treating bad data as 0."""
    if val is None or val == "":
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _safe_float(val) -> float:
    if val is None or val == "":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _build_row(cur_row: dict, prior_row: dict | None, symbol: str) -> dict:
    """
    Build the display row with all 11 columns.

    Legacy Futures Only field names:
        noncomm_positions_long_all     — Non-commercial (speculator) long
        noncomm_positions_short_all    — Non-commercial short
        open_interest_all              — Total open interest
        pct_of_oi_noncomm_long_all     — % of OI held long by non-comms
        pct_of_oi_noncomm_short_all    — % of OI held short by non-comms
    """
    # Current week
    long_now  = _safe_int(cur_row.get("noncomm_positions_long_all"))
    short_now = _safe_int(cur_row.get("noncomm_positions_short_all"))
    oi_now    = _safe_int(cur_row.get("open_interest_all"))
    long_pct  = _safe_float(cur_row.get("pct_of_oi_noncomm_long_all"))
    short_pct = _safe_float(cur_row.get("pct_of_oi_noncomm_short_all"))

    # Prior week (0 if missing)
    if prior_row:
        long_prior  = _safe_int(prior_row.get("noncomm_positions_long_all"))
        short_prior = _safe_int(prior_row.get("noncomm_positions_short_all"))
        oi_prior    = _safe_int(prior_row.get("open_interest_all"))
    else:
        long_prior = short_prior = oi_prior = 0

    long_change  = long_now - long_prior
    short_change = short_now - short_prior
    oi_change    = oi_now - oi_prior

    # Net position = long - short
    net_now   = long_now - short_now
    net_prior = long_prior - short_prior
    # Net % change — relative to prior net position
    if net_prior != 0:
        net_pct_change = (net_now - net_prior) / abs(net_prior) * 100
    else:
        net_pct_change = 0.0

    return {
        "symbol":         symbol,
        "long":           long_now,
        "short":          short_now,
        "long_change":    long_change,
        "short_change":   short_change,
        "long_pct":       round(long_pct, 2),
        "short_pct":      round(short_pct, 2),
        "net_pct_change": round(net_pct_change, 2),
        "net_position":   net_now,
        "open_interest":  oi_now,
        "oi_change":      oi_change,
    }


def fetch_cot_data() -> dict:
    """
    Fetch the latest COT data and build rows for tracked symbols.

    Returns:
        {
            "report_date":  "YYYY-MM-DD",
            "rows":         [row, row, ...],
            "fetched_at":   timestamp,
            "error":        None | str,
        }
    """
    try:
        report_date = _latest_report_date()
        if not report_date:
            return {"error": "Could not get latest report date",
                    "report_date": "", "rows": [], "fetched_at": time.time()}

        cur_rows   = _fetch_report_for_date(report_date)
        prior_rows = _fetch_prior_week(report_date)

        # Index prior rows by contract name for O(1) lookup
        prior_by_name = {r.get("market_and_exchange_names", ""): r
                         for r in prior_rows}

        # Build a row per tracked symbol (preserving insertion order)
        built_rows = []
        for contract_name, symbol in TRACKED_SYMBOLS.items():
            cur = next((r for r in cur_rows
                        if r.get("market_and_exchange_names", "") == contract_name),
                       None)
            if cur is None:
                continue
            prior = prior_by_name.get(contract_name)
            built_rows.append(_build_row(cur, prior, symbol))

        return {
            "report_date": report_date,
            "rows":        built_rows,
            "fetched_at":  time.time(),
            "error":       None,
        }

    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc), "report_date": "", "rows": [],
                "fetched_at": time.time()}


# ── Background manager ───────────────────────────────────────────────────────

class CotDataManager:
    """
    Refreshes COT data periodically.
    Since CFTC publishes weekly, a 1h refresh is plenty —
    the data only actually changes on Fridays at 3:30pm EST.
    """

    def __init__(self, refresh_seconds: int = 3600):
        self._refresh = refresh_seconds
        self._lock = threading.Lock()
        self._cache = {"error": "Initializing...", "rows": [],
                       "report_date": "", "fetched_at": 0}
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_cache(self) -> dict:
        with self._lock:
            return dict(self._cache)

    def _worker(self):
        while self._running:
            try:
                data = fetch_cot_data()
                with self._lock:
                    self._cache = data
                if data.get("error"):
                    print(f"  [COT] Error: {data['error']}")
                else:
                    print(f"  [COT] Loaded {len(data['rows'])} symbols "
                          f"(report date {data['report_date']})")
            except Exception as exc:
                traceback.print_exc()
                with self._lock:
                    self._cache["error"] = str(exc)

            # Wait between cycles
            for _ in range(self._refresh * 2):
                if not self._running:
                    break
                time.sleep(0.5)


# ── module-level singleton ───────────────────────────────────────────────────
cot_manager: CotDataManager | None = None


def init_cot_manager() -> CotDataManager:
    global cot_manager
    cot_manager = CotDataManager()
    cot_manager.start()
    return cot_manager
