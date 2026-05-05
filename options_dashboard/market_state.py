"""
market_state.py  –  Broad Market State widget data manager.

Computes a per-ticker bullishness score (0-100) for a list of important ETFs.

Score formula:
    bullishness = 50% × daily_change_score + 50% × volume_flow_score

Where:
    daily_change_score:  sigmoid-like mapping of % change to 0-100
                          0%   → 50
                          ±2%  → strong bull/bear (~85/15)
                          ±5%+ → max bull/bear (~98/2)
    volume_flow_score:   based on up-volume vs down-volume ratio across
                          the session's 5-minute bars.
                          (up - down) / (up + down) ∈ [-1, +1] → mapped to 0-100

Uses its own IB connection (clientId = IB_CLIENT_ID + 20) to keep the
market data lines isolated from the main fetcher.
"""

import threading
import time
import traceback
import math
import datetime as dt

import pandas as pd

from config import IB_HOST, IB_PORT, IB_CLIENT_ID, ET


# ── Tracked tickers ──────────────────────────────────────────────────────────
# Display order matches the natural reading flow:
#   broad benchmarks → equity sectors → bonds & gold (macro)
TRACKED_ETFS = [
    # Broad benchmarks
    ("SPY",   "S&P 500"),
    ("QQQ",   "Nasdaq-100"),
    ("IWM",   "Russell 2000"),
    ("DIA",   "Dow Jones"),
    ("MAGS",  "Magnificent 7"),
    # Equity sectors
    ("SMH",   "Semiconductors"),
    ("XLK",   "Technology"),
    ("XLC",   "Communications"),
    ("XLY",   "Discretionary"),
    ("XLP",   "Staples"),
    ("XLF",   "Financials"),
    ("XLV",   "Health Care"),
    ("XLI",   "Industrials"),
    ("XLE",   "Energy"),
    ("XLB",   "Materials"),
    ("XLRE",  "Real Estate"),
    ("XLU",   "Utilities"),
    # Macro
    ("TLT",   "20Y Treasuries"),
    ("HYG",   "High Yield"),
    ("GLD",   "Gold"),
]


# ── Composite indexes ───────────────────────────────────────────────────────
# Each composite is a weighted average of underlying sector ETF scores,
# matching the actual sector weighting of the index (approx. Q1 2026 weights).
# Weights must sum to 1.0 for each composite.
COMPOSITE_INDEXES = [
    {
        "ticker": "SPX-W",
        "name":   "S&P 500 Composite",
        "weights": {
            "XLK":  0.30,   # Technology
            "XLF":  0.14,   # Financials
            "XLC":  0.10,   # Communication Services
            "XLV":  0.10,   # Health Care
            "XLY":  0.10,   # Consumer Discretionary
            "XLI":  0.09,   # Industrials
            "XLP":  0.055,  # Consumer Staples
            "XLE":  0.035,  # Energy
            "XLRE": 0.025,  # Real Estate
            "XLU":  0.025,  # Utilities
            "XLB":  0.020,  # Materials
        },
    },
    {
        "ticker": "NDX-W",
        "name":   "Nasdaq-100 Composite",
        "weights": {
            "XLK":  0.50,   # Technology dominates
            "XLY":  0.17,   # Discretionary (Amazon, Tesla)
            "XLC":  0.16,   # Communications (Meta, Google)
            "XLV":  0.06,   # Health Care
            "XLI":  0.04,   # Industrials
            "XLP":  0.04,   # Staples (Costco, Pepsi)
            "XLF":  0.03,   # Financials (very small in NDX)
        },
    },
    {
        "ticker": "DOW-W",
        "name":   "Dow Jones Composite",
        "weights": {
            # DOW is price-weighted, but sector composition approximates this:
            "XLK":  0.22,   # Tech (Microsoft, Apple, Salesforce, Cisco, IBM)
            "XLF":  0.22,   # Financials (Goldman, JPM, AmEx, Visa, Travelers)
            "XLV":  0.18,   # Health Care (UnitedHealth, J&J, Merck, Amgen)
            "XLI":  0.15,   # Industrials (Boeing, Caterpillar, 3M, Honeywell)
            "XLY":  0.13,   # Discretionary (Home Depot, Nike, McDonald's)
            "XLP":  0.05,   # Staples (P&G, Coke, Walmart)
            "XLC":  0.05,   # Communications (Verizon, Disney)
        },
    },
    {
        "ticker": "RUT-W",
        "name":   "Russell 2000 Composite",
        "weights": {
            # Russell 2000 = small caps. Very different sector mix:
            "XLF":  0.18,   # Regional banks heavy
            "XLI":  0.17,   # Industrials
            "XLV":  0.16,   # Biotechs and small healthcare
            "XLK":  0.13,   # Smaller tech (lighter than large caps)
            "XLY":  0.11,   # Discretionary
            "XLRE": 0.10,   # REITs (small caps include many)
            "XLB":  0.06,   # Materials
            "XLE":  0.05,   # Energy
            "XLU":  0.04,   # Utilities
        },
    },
]


# ── Score math ───────────────────────────────────────────────────────────────

def _daily_change_to_score(pct_change: float) -> float:
    """0% → 50, ±2% ≈ 76/24, ±5% → near saturation."""
    return 50.0 + 50.0 * math.tanh(pct_change / 2.0)


def _volume_flow_to_score(flow_ratio: float) -> float:
    """Maps [-1, +1] to [0, 100]."""
    flow = max(-1.0, min(1.0, flow_ratio))
    return 50.0 + 50.0 * flow


def _trend_alignment_score(spot: float, sma20: float, sma50: float,
                            sma200: float) -> float:
    """
    How aligned is price with its moving averages?
    Bull stack (price > 20 > 50 > 200)   → 100
    Bear stack (price < 20 < 50 < 200)   → 0
    Mixed alignments → in between, weighted by which MAs are above.
    """
    if spot <= 0:
        return 50.0
    above_20 = sma20 > 0 and spot > sma20
    above_50 = sma50 > 0 and spot > sma50
    above_200 = sma200 > 0 and spot > sma200
    # Weighted: 50 base + each MA adds/subtracts proportional to importance
    score = 50.0
    score += 12.0 if above_20 else -12.0
    score += 16.0 if above_50 else -16.0
    score += 22.0 if above_200 else -22.0   # 200MA most important
    # Bonus for full alignment (bull/bear stack)
    if sma20 > 0 and sma50 > 0 and sma200 > 0:
        if sma20 > sma50 > sma200 and above_20:
            score = min(100, score + 5)
        elif sma20 < sma50 < sma200 and not above_20:
            score = max(0, score - 5)
    return max(0.0, min(100.0, score))


def _volume_quality_score(today_vol: float, avg_vol: float,
                           pct_change: float) -> float:
    """
    Was today's move on conviction or noise?
    Volume ratio (today / 20d avg):
      Above-avg volume on a green day  → boosts bullishness
      Above-avg volume on a red day    → reduces bullishness (heavy distribution)
      Below-avg volume                 → score moves toward 50 (neutral, low conviction)
    """
    if avg_vol <= 0 or today_vol <= 0:
        return 50.0
    vol_ratio = today_vol / avg_vol
    # Cap at 3x for sanity
    vol_ratio = min(vol_ratio, 3.0)
    # Conviction = how far from 1.0 (avg)
    # Above 1.0: amplifies the day's direction
    # Below 1.0: dampens toward 50
    if vol_ratio >= 1.0:
        # Heavy volume: push direction further
        amplifier = (vol_ratio - 1.0) / 2.0   # 0 to 1 as vol goes 1x to 3x
        return 50.0 + 50.0 * math.tanh(pct_change / 1.5) * (1 + amplifier)
    else:
        # Light volume: dampen
        damper = vol_ratio   # 0 to 1
        return 50.0 + (50.0 * math.tanh(pct_change / 1.5)) * damper


def _bullishness_score(pct_change: float, flow_ratio: float,
                       trend: float, vol_qual: float) -> float:
    """
    Composite bullishness score (0-100), weighted blend:
      37.5%  daily price change         (today's price action)
      32.5%  intraday volume flow       (buy vs sell pressure)
      20%    trend alignment            (price vs MAs — context)
      10%    volume quality             (conviction of the move)
    """
    day = _daily_change_to_score(pct_change)
    flow = _volume_flow_to_score(flow_ratio)
    return (0.375 * day +
            0.325 * flow +
            0.200 * trend +
            0.100 * vol_qual)


def _classify(score: float) -> str:
    """Convert a bullishness score to a human label."""
    if score >= 75: return "STRONG BULL"
    if score >= 60: return "BULLISH"
    if score >= 45: return "NEUTRAL"
    if score >= 30: return "BEARISH"
    return "STRONG BEAR"


def _build_composite_row(composite: dict, etf_rows: list) -> dict:
    """
    Build a synthetic row for an index composite.
    Computes weighted averages from the underlying sector ETFs.
    """
    by_ticker = {r["ticker"]: r for r in etf_rows if not r.get("error")}
    weights = composite["weights"]

    # Filter to ETFs we have valid data for, then renormalize
    available = {t: w for t, w in weights.items() if t in by_ticker}
    if not available:
        return {
            "ticker": composite["ticker"],
            "name":   composite["name"],
            "error":  "No constituent ETFs available",
            "is_composite": True,
        }

    total_w = sum(available.values())
    norm_weights = {t: w / total_w for t, w in available.items()}

    pct_change = sum(by_ticker[t]["pct_change"] * w for t, w in norm_weights.items())
    flow_ratio = sum(by_ticker[t]["flow_ratio"] * w for t, w in norm_weights.items())
    score      = sum(by_ticker[t]["score"]      * w for t, w in norm_weights.items())
    trend_score    = sum(by_ticker[t].get("trend_score", 50)    * w for t, w in norm_weights.items())
    vol_qual_score = sum(by_ticker[t].get("vol_qual_score", 50) * w for t, w in norm_weights.items())
    vol_ratio      = sum(by_ticker[t].get("vol_ratio", 1.0)     * w for t, w in norm_weights.items())

    # MA stack — weighted vote: dot is "above" if >50% of weight is above its MA
    above_20  = sum(w for t, w in norm_weights.items()
                    if by_ticker[t].get("spot", 0) > by_ticker[t].get("sma20", 0))
    above_50  = sum(w for t, w in norm_weights.items()
                    if by_ticker[t].get("spot", 0) > by_ticker[t].get("sma50", 0))
    above_200 = sum(w for t, w in norm_weights.items()
                    if by_ticker[t].get("spot", 0) > by_ticker[t].get("sma200", 0))

    # Synthesize MA flags using a fake spot of 100, with smaX = 99 if mostly above
    sma20_synth  = 99 if above_20  >= 0.5 else 101
    sma50_synth  = 99 if above_50  >= 0.5 else 101
    sma200_synth = 99 if above_200 >= 0.5 else 101

    return {
        "ticker":         composite["ticker"],
        "name":           composite["name"],
        "is_composite":   True,
        "spot":           100,
        "pct_change":     round(pct_change, 2),
        "flow_ratio":     round(flow_ratio, 3),
        "sma20":          sma20_synth,
        "sma50":          sma50_synth,
        "sma200":         sma200_synth,
        "vol_ratio":      round(vol_ratio, 2),
        "trend_score":    round(trend_score, 1),
        "vol_qual_score": round(vol_qual_score, 1),
        "score":          round(score, 1),
        "label":          _classify(score),
        "error":          None,
        "constituents":   list(norm_weights.keys()),
    }


# ── Data fetcher (real IB) ───────────────────────────────────────────────────

class IBMarketStateFetcher:
    """Fetches daily change + intraday volume flow for a list of ETFs."""

    def __init__(self, host=IB_HOST, port=IB_PORT, client_id=IB_CLIENT_ID + 20):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self._connected = False

    def connect(self):
        from ib_insync import IB
        if self._connected:
            return
        self.ib = IB()
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        self._connected = True
        print(f"  Market State: connected to IB (clientId={self.client_id})")

    def disconnect(self):
        if self.ib and self._connected:
            try:
                self.ib.disconnect()
            except Exception:
                pass
            self._connected = False

    def fetch_one(self, ticker: str) -> dict:
        """
        Fetch metrics for a single ETF.
        Returns dict with all signal components.
        """
        from ib_insync import Stock

        try:
            contract = Stock(ticker, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            if contract.conId == 0:
                return {"ticker": ticker, "error": "Unknown contract"}

            self.ib.reqMarketDataType(4)

            # Fetch 1 year of daily bars (need 200 for the 200MA)
            daily_bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="1 Y",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if not daily_bars or len(daily_bars) < 21:
                return {"ticker": ticker, "error": "Not enough daily history"}

            today_bar = daily_bars[-1]
            prev_close = daily_bars[-2].close if len(daily_bars) >= 2 else today_bar.open
            spot = today_bar.close
            today_volume = today_bar.volume or 0

            if prev_close <= 0:
                return {"ticker": ticker, "error": "Bad prev close"}
            pct_change = (spot - prev_close) / prev_close * 100

            # Build closes & volumes lists (excluding today, so MAs are based on history)
            historical_closes = [b.close for b in daily_bars[:-1] if b.close > 0]
            historical_vols   = [b.volume or 0 for b in daily_bars[:-1]]

            # Moving averages
            def _sma(values, n):
                if len(values) < n:
                    return 0.0
                return sum(values[-n:]) / n

            sma20  = _sma(historical_closes, 20)
            sma50  = _sma(historical_closes, 50)
            sma200 = _sma(historical_closes, 200)

            # 20-day average volume (excluding today)
            avg_vol_20 = (sum(historical_vols[-20:]) / 20
                          if len(historical_vols) >= 20 else 0)

            # Trend alignment & volume quality scores
            trend_sc = _trend_alignment_score(spot, sma20, sma50, sma200)
            vol_qual_sc = _volume_quality_score(today_volume, avg_vol_20, pct_change)

            # Intraday 5-min bars for volume flow
            intraday_bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            up_vol = down_vol = 0
            for b in intraday_bars or []:
                v = b.volume or 0
                if b.close > b.open:
                    up_vol += v
                elif b.close < b.open:
                    down_vol += v

            total_directional = up_vol + down_vol
            flow_ratio = ((up_vol - down_vol) / total_directional
                          if total_directional > 0 else 0.0)

            # Composite score
            score = _bullishness_score(pct_change, flow_ratio, trend_sc, vol_qual_sc)
            label = _classify(score)

            return {
                "ticker":     ticker,
                "spot":       round(spot, 2),
                "prev_close": round(prev_close, 2),
                "pct_change": round(pct_change, 2),
                "up_vol":     int(up_vol),
                "down_vol":   int(down_vol),
                "flow_ratio": round(flow_ratio, 3),
                "sma20":      round(sma20, 2),
                "sma50":      round(sma50, 2),
                "sma200":     round(sma200, 2),
                "today_vol":  int(today_volume),
                "avg_vol_20": int(avg_vol_20),
                "vol_ratio":  round(today_volume / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
                "trend_score":  round(trend_sc, 1),
                "vol_qual_score": round(vol_qual_sc, 1),
                "score":      round(score, 1),
                "label":      label,
                "error":      None,
            }

        except Exception as exc:
            return {"ticker": ticker, "error": str(exc)}

    def fetch_all(self) -> list:
        """Fetch all tracked ETFs sequentially, then build composite rows."""
        if not self._connected:
            self.connect()

        results = []
        for ticker, name in TRACKED_ETFS:
            row = self.fetch_one(ticker)
            row["name"] = name
            results.append(row)
            # Small breathing room between tickers (avoid pacing violations)
            self.ib.sleep(0.3)

        # Build composite index rows from the sector ETF data we just fetched
        composites = [_build_composite_row(c, results) for c in COMPOSITE_INDEXES]
        # Composites go FIRST so they appear at the top of the grid
        return composites + results


# ── Mock fetcher (for dev / no-IB) ───────────────────────────────────────────

class MockMarketStateFetcher:
    """Generates pseudo-random plausible values for development."""

    def __init__(self):
        import random
        self._rand = random.Random(42)

    def connect(self):  pass
    def disconnect(self): pass

    def fetch_all(self) -> list:
        results = []
        for ticker, name in TRACKED_ETFS:
            pct = self._rand.uniform(-2.5, 2.5)
            flow = self._rand.uniform(-0.8, 0.8)
            spot = self._rand.uniform(50, 600)
            prev_close = spot / (1 + pct / 100)
            up_vol = int(self._rand.uniform(1e6, 2e7))
            down_vol = max(0, int(up_vol * (1 - flow) / max(1 + flow, 0.01)))
            today_vol = up_vol + down_vol
            avg_vol = int(today_vol * self._rand.uniform(0.7, 1.4))
            sma20 = spot * self._rand.uniform(0.97, 1.03)
            sma50 = spot * self._rand.uniform(0.93, 1.05)
            sma200 = spot * self._rand.uniform(0.85, 1.10)

            trend_sc = _trend_alignment_score(spot, sma20, sma50, sma200)
            vol_qual_sc = _volume_quality_score(today_vol, avg_vol, pct)
            score = _bullishness_score(pct, flow, trend_sc, vol_qual_sc)

            results.append({
                "ticker":     ticker,
                "name":       name,
                "spot":       round(spot, 2),
                "prev_close": round(prev_close, 2),
                "pct_change": round(pct, 2),
                "up_vol":     up_vol,
                "down_vol":   down_vol,
                "flow_ratio": round(flow, 3),
                "sma20":      round(sma20, 2),
                "sma50":      round(sma50, 2),
                "sma200":     round(sma200, 2),
                "today_vol":  today_vol,
                "avg_vol_20": avg_vol,
                "vol_ratio":  round(today_vol / avg_vol, 2),
                "trend_score":    round(trend_sc, 1),
                "vol_qual_score": round(vol_qual_sc, 1),
                "score":      round(score, 1),
                "label":      _classify(score),
                "error":      None,
            })

        # Build composite index rows from the sector ETF data
        composites = [_build_composite_row(c, results) for c in COMPOSITE_INDEXES]
        return composites + results


# ── Background manager ──────────────────────────────────────────────────────

class MarketStateManager:
    """Refreshes ETF state on a configurable interval."""

    def __init__(self, use_mock=False, refresh_seconds=60):
        self._refresh = refresh_seconds
        self._use_mock = use_mock
        self._lock = threading.Lock()
        self._cache = {"rows": [], "fetched_at": 0, "error": "Initializing..."}
        self._running = False
        self._thread = None
        self._fetcher = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._fetcher:
            self._fetcher.disconnect()

    def get_cache(self) -> dict:
        with self._lock:
            return dict(self._cache)

    def _worker(self):
        # Lazy-init fetcher inside the worker thread (IB needs an event loop here)
        if self._use_mock:
            self._fetcher = MockMarketStateFetcher()
        else:
            try:
                self._fetcher = IBMarketStateFetcher()
                self._fetcher.connect()
            except Exception as exc:
                print(f"  [Market State] IB connect failed: {exc}, using mock")
                traceback.print_exc()
                self._fetcher = MockMarketStateFetcher()
                self._use_mock = True

        while self._running:
            try:
                rows = self._fetcher.fetch_all()
                with self._lock:
                    self._cache = {
                        "rows":       rows,
                        "fetched_at": time.time(),
                        "error":      None,
                    }
                ok = sum(1 for r in rows if not r.get("error"))
                print(f"  [Market State] Loaded {ok}/{len(rows)} ETFs")
            except Exception as exc:
                traceback.print_exc()
                with self._lock:
                    self._cache["error"] = str(exc)

            # Sleep in small steps so we can shut down quickly
            for _ in range(self._refresh * 2):
                if not self._running:
                    break
                time.sleep(0.5)


# ── module-level singleton ───────────────────────────────────────────────────
market_state_manager: MarketStateManager | None = None


def init_market_state_manager(use_mock=False) -> MarketStateManager:
    global market_state_manager
    market_state_manager = MarketStateManager(use_mock=use_mock)
    market_state_manager.start()
    return market_state_manager
