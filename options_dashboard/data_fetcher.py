"""
data_fetcher.py  –  Interactive Brokers data layer.

All IB work runs on a dedicated background thread (see DataManager).
Dash callbacks only read from the thread-safe cache.
"""

import math
import time
import asyncio
import datetime as dt
import threading
import traceback

import numpy as np
import pandas as pd

from config import IB_HOST, IB_PORT, IB_CLIENT_ID, SETTINGS, INDEX_TICKERS, ET

USE_MOCK = False


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_underlying(ticker: str):
    """Return Stock or Index contract depending on the ticker."""
    from ib_insync import Stock, Index
    if ticker.upper() in INDEX_TICKERS:
        return Index(ticker, "CBOE", "USD")
    return Stock(ticker, "SMART", "USD")


def _safe_float(val, default=0.0):
    """Safely convert a value to float, returning default for None/NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
#  IB DATA FETCHER  (runs only on the background thread)
# ══════════════════════════════════════════════════════════════════════════════

class IBDataFetcher:

    def __init__(self):
        from ib_insync import IB
        self.ib = IB()
        self._connected = False

    def connect(self):
        if not self._connected:
            self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            self._connected = True

    def disconnect(self):
        if self._connected:
            self.ib.disconnect()
            self._connected = False

    # ── spot price ───────────────────────────────────────────────────────

    def get_spot(self, ticker: str) -> float:
        contract = _make_underlying(ticker)
        self.ib.qualifyContracts(contract)
        # Type 4 = frozen/delayed: returns live when available,
        # falls back to last known values when market is closed
        self.ib.reqMarketDataType(4)
        md = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(2)
        price = _safe_float(md.last)
        if price == 0:
            price = _safe_float(md.close)
        if price == 0:
            price = _safe_float(md.bid)
        if price == 0:
            price = _safe_float(md.marketPrice())
        self.ib.cancelMktData(contract)
        return price

    def get_prev_day_hl(self, ticker: str) -> dict:
        """Return previous trading day's high and low from IB historical data."""
        contract = _make_underlying(ticker)
        self.ib.qualifyContracts(contract)
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if len(bars) >= 2:
                prev = bars[-2]  # second-to-last = previous day
                return {"high": float(prev.high), "low": float(prev.low)}
            elif len(bars) == 1:
                return {"high": float(bars[0].high), "low": float(bars[0].low)}
        except Exception as exc:
            print(f"  Warning: could not fetch prev day H/L: {exc}")
        return {"high": 0.0, "low": 0.0}

    # ── chain definitions ────────────────────────────────────────────────

    def _get_all_chains(self, ticker: str):
        """Return raw chain definitions from IB."""
        contract = _make_underlying(ticker)
        self.ib.qualifyContracts(contract)
        return self.ib.reqSecDefOptParams(
            contract.symbol, "", contract.secType, contract.conId
        )

    def get_expiries(self, ticker: str) -> list[str]:
        """
        Collect expiries from ALL chain definitions (all exchanges,
        all trading classes).  This ensures we find 0DTE / weekly /
        daily expiries that may not appear in every chain.
        """
        chains = self._get_all_chains(ticker)
        all_expiries = set()
        for c in chains:
            all_expiries |= set(c.expirations)
        return sorted(all_expiries)

    def resolve_expiry(self, ticker: str) -> str:
        expiries = self.get_expiries(ticker)
        if not expiries:
            raise RuntimeError(f"No option expiries found for {ticker}")
        if SETTINGS.expiry.lower() == "auto":
            today = dt.date.today()
            return min(
                expiries,
                key=lambda e: abs(dt.date(int(e[:4]), int(e[4:6]), int(e[6:])) - today),
            )
        if SETTINGS.expiry in expiries:
            return SETTINGS.expiry
        raise ValueError(
            f"Expiry {SETTINGS.expiry} not found. Available: {expiries[:10]}..."
        )

    # ── full option chain ────────────────────────────────────────────────

    def fetch_chain(self, ticker: str, expiry: str, spot: float) -> pd.DataFrame:
        from ib_insync import Option

        chains = self._get_all_chains(ticker)

        # Find the chain with the MOST strikes for this expiry.
        # This avoids picking a mini-option chain with only a few strikes.
        best_chain = None
        max_strikes = 0
        for c in chains:
            if expiry in c.expirations and len(c.strikes) > max_strikes:
                max_strikes = len(c.strikes)
                best_chain = c

        if best_chain is None:
            raise RuntimeError(f"No chain found for {ticker} expiry {expiry}")

        trading_class = best_chain.tradingClass
        exchange = best_chain.exchange
        multiplier = best_chain.multiplier
        strikes = sorted(best_chain.strikes)

        print(f"  Chain: {ticker} | tradingClass={trading_class} | "
              f"exchange={exchange} | {len(strikes)} total strikes")

        # Filter to +/- 5% of spot, then cap at 45 strikes (calls+puts = 90,
        # safely under IB's 100 concurrent market-data subscription limit).
        lo, hi = spot * 0.95, spot * 1.05
        strikes = [s for s in strikes if lo <= s <= hi]

        MAX_STRIKES = 45
        if len(strikes) > MAX_STRIKES:
            # Keep the 45 strikes closest to spot
            strikes.sort(key=lambda s: abs(s - spot))
            strikes = sorted(strikes[:MAX_STRIKES])

        print(f"  Filtered to {len(strikes)} strikes in [{lo:.0f} - {hi:.0f}]"
              f" (max {MAX_STRIKES})")

        if not strikes:
            return pd.DataFrame()

        # Qualify calls first to discover which strikes actually exist
        # for this expiry (SPXW 0DTE only has $5 strikes near ATM,
        # $25 increments further out — chain definition over-reports)
        import io, contextlib

        call_contracts = []
        for s in strikes:
            call_contracts.append(
                Option(ticker, expiry, s, "C", exchange,
                       multiplier=multiplier, currency="USD",
                       tradingClass=trading_class)
            )

        # Suppress IB "Unknown contract" / "Error 200" noise during
        # qualification — these are expected for strikes that don't
        # exist on this specific expiry
        _devnull = io.StringIO()
        with contextlib.redirect_stdout(_devnull), \
             contextlib.redirect_stderr(_devnull):
            self.ib.qualifyContracts(*call_contracts)
            self.ib.sleep(0.5)  # let error messages flush

        # Only build puts for strikes where the call was valid
        valid_strikes = [c.strike for c in call_contracts if c.conId != 0]
        skipped = len(strikes) - len(valid_strikes)

        put_contracts = []
        for s in valid_strikes:
            put_contracts.append(
                Option(ticker, expiry, s, "P", exchange,
                       multiplier=multiplier, currency="USD",
                       tradingClass=trading_class)
            )

        with contextlib.redirect_stdout(_devnull), \
             contextlib.redirect_stderr(_devnull):
            self.ib.qualifyContracts(*put_contracts)
            self.ib.sleep(0.5)

        valid = [c for c in call_contracts + put_contracts if c.conId != 0]
        if skipped:
            print(f"  Qualified {len(valid)} contracts "
                  f"({skipped} strikes skipped — not listed for this expiry)")
        else:
            print(f"  Qualified {len(valid)} contracts")

        if not valid:
            return pd.DataFrame()

        # Stream market data (not snapshot — snapshot doesn't support OI/IV)
        # Type 4 = frozen/delayed: works when market is closed
        self.ib.reqMarketDataType(4)
        tickers_list = []
        for con in valid:
            t = self.ib.reqMktData(con, "100,101,104,106", False, False)
            tickers_list.append(t)

        self.ib.sleep(10)   # extra time for frozen data

        # Parse results
        data = {}
        for t in tickers_list:
            c = t.contract
            key = (c.right, c.strike)
            oi = _safe_float(t.callOpenInterest) if c.right == "C" \
                else _safe_float(t.putOpenInterest)
            data[key] = {
                "oi":     oi,
                "volume": _safe_float(t.volume),
                "iv":     _safe_float(t.impliedVolatility),
                "bid":    _safe_float(t.bid),
                "ask":    _safe_float(t.ask),
                "last":   _safe_float(t.last),
            }

        # Cancel subscriptions
        for t in tickers_list:
            try:
                self.ib.cancelMktData(t.contract)
            except Exception:
                pass

        # Build DataFrame
        exp_date = dt.date(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:]))
        dte_years = max((exp_date - dt.date.today()).days, 1) / 365.0

        seen_strikes = sorted(set(k[1] for k in data.keys()))
        rows = []
        for s in seen_strikes:
            c_data = data.get(("C", s), {"oi": 0, "volume": 0, "iv": 0.0,
                                          "bid": 0, "ask": 0, "last": 0})
            p_data = data.get(("P", s), {"oi": 0, "volume": 0, "iv": 0.0,
                                          "bid": 0, "ask": 0, "last": 0})
            rows.append({
                "strike":      s,
                "call_oi":     c_data["oi"],
                "put_oi":      p_data["oi"],
                "call_volume": c_data["volume"],
                "put_volume":  p_data["volume"],
                "call_iv":     c_data["iv"] if c_data["iv"] > 0 else 0.20,
                "put_iv":      p_data["iv"] if p_data["iv"] > 0 else 0.20,
                "call_bid":    c_data.get("bid", 0),
                "call_ask":    c_data.get("ask", 0),
                "call_last":   c_data.get("last", 0),
                "put_bid":     p_data.get("bid", 0),
                "put_ask":     p_data.get("ask", 0),
                "put_last":    p_data.get("last", 0),
                "dte_years":   dte_years,
            })

        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  MOCK FETCHER
# ══════════════════════════════════════════════════════════════════════════════

class MockDataFetcher:

    def connect(self):
        print("  MOCK MODE - using synthetic data")

    def disconnect(self):
        pass

    _base_prices = {"SPY": 545.0, "QQQ": 470.0, "AAPL": 230.0,
                     "TSLA": 250.0, "NVDA": 135.0, "AMZN": 200.0,
                     "SPX": 5450.0, "NDX": 18500.0}
    _mock_spot: dict = {}   # tracks random-walk per ticker

    def get_spot(self, ticker: str) -> float:
        base = self._base_prices.get(ticker.upper(), 100.0)
        # Random walk: drift ±0.1% each call to simulate live price movement
        import random
        prev = self._mock_spot.get(ticker.upper(), base)
        step = prev * random.gauss(0, 0.001)   # ~0.1% std dev per tick
        new_price = prev + step
        self._mock_spot[ticker.upper()] = new_price
        return round(new_price, 2)

    def get_prev_day_hl(self, ticker: str) -> dict:
        base = self._base_prices.get(ticker.upper(), 100.0)
        return {"high": base * 1.008, "low": base * 0.992}

    def get_expiries(self, ticker: str) -> list[str]:
        today = dt.date.today()
        expiries = []
        # Generate daily expiries for 2 weeks, then weekly for 2 months
        for i in range(0, 14):
            d = today + dt.timedelta(days=i)
            if d.weekday() < 5:  # weekdays only
                expiries.append(d.strftime("%Y%m%d"))
        for i in range(14, 60, 7):
            d = today + dt.timedelta(days=i)
            d += dt.timedelta(days=(4 - d.weekday()) % 7)
            expiries.append(d.strftime("%Y%m%d"))
        return sorted(set(expiries))

    def resolve_expiry(self, ticker: str) -> str:
        expiries = self.get_expiries(ticker)
        if SETTINGS.expiry.lower() == "auto":
            today = dt.date.today()
            return min(
                expiries,
                key=lambda e: abs(dt.date(int(e[:4]), int(e[4:6]), int(e[6:])) - today),
            )
        return SETTINGS.expiry if SETTINGS.expiry in expiries else expiries[0]

    def fetch_chain(self, ticker: str, expiry: str, spot: float) -> pd.DataFrame:
        rng = np.random.default_rng(42)

        # Use $5 steps for indices, $1 for equities (like real chains)
        step = 5.0 if spot > 1000 else 1.0
        lo = np.ceil(spot * 0.95 / step) * step     # first strike on grid
        hi = np.floor(spot * 1.05 / step) * step     # last strike on grid
        strikes = np.arange(lo, hi + step, step)     # guaranteed on-grid, no duplicates

        # Cap at 45 strikes (same as real fetcher)
        if len(strikes) > 45:
            center = np.argmin(np.abs(strikes - spot))
            half = 22
            start = max(0, center - half)
            strikes = strikes[start:start + 45]

        n = len(strikes)
        moneyness = (strikes - spot) / spot

        # ── Realistic OI pattern ─────────────────────────────────────
        # Calls: heavy OI above spot (OTM calls), thin below
        # Puts:  heavy OI below spot (OTM puts),  thin above
        # This creates the classic GEX profile: positive above, negative below
        base_oi = 8000 * np.exp(-0.5 * (moneyness / 0.05) ** 2)

        # Call OI peaks above spot, fades below
        call_skew = 1.0 / (1.0 + np.exp(-moneyness * 80))  # sigmoid: ~0 below, ~1 above
        call_oi = (base_oi * (0.3 + 1.5 * call_skew) * rng.uniform(0.8, 1.2, n)).astype(int)

        # Put OI peaks below spot, fades above
        put_skew = 1.0 / (1.0 + np.exp(moneyness * 80))    # sigmoid: ~1 below, ~0 above
        put_oi = (base_oi * (0.3 + 1.5 * put_skew) * rng.uniform(0.8, 1.2, n)).astype(int)

        # Round-number strikes get extra OI (like real markets)
        round_bonus = np.array([2.5 if s % 50 == 0 else 1.5 if s % 25 == 0 else 1.0
                                for s in strikes])
        call_oi = (call_oi * round_bonus).astype(int)
        put_oi  = (put_oi  * round_bonus).astype(int)

        call_vol = (call_oi * rng.uniform(0.05, 0.20, n)).astype(int)
        put_vol  = (put_oi  * rng.uniform(0.05, 0.20, n)).astype(int)

        base_iv = 0.18 + 0.12 * moneyness**2 + rng.normal(0, 0.003, n)
        call_iv = np.clip(base_iv, 0.05, 1.5)
        put_iv  = np.clip(base_iv + 0.015, 0.05, 1.5)   # put skew

        exp_date = dt.date(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:]))
        dte_years = max((exp_date - dt.date.today()).days, 1) / 365.0

        # Synthetic bid/ask/last for flow tracking compatibility
        from scipy.stats import norm as _norm
        sqrtT = np.sqrt(dte_years)
        d1 = (np.log(spot / strikes) + (0.045 + 0.5 * call_iv**2) * dte_years) / (call_iv * sqrtT)
        call_price = np.maximum(spot * _norm.cdf(d1) - strikes * np.exp(-0.045 * dte_years) * _norm.cdf(d1 - call_iv * sqrtT), 0.01)
        put_price  = np.maximum(call_price - spot + strikes * np.exp(-0.045 * dte_years), 0.01)

        call_mid = call_price
        put_mid  = put_price
        call_spread = np.clip(call_mid * 0.05, 0.05, 5.0)
        put_spread  = np.clip(put_mid * 0.05, 0.05, 5.0)

        return pd.DataFrame({
            "strike":      strikes,
            "call_oi":     call_oi,
            "put_oi":      put_oi,
            "call_volume": call_vol,
            "put_volume":  put_vol,
            "call_iv":     call_iv,
            "put_iv":      put_iv,
            "call_bid":    call_mid - call_spread / 2,
            "call_ask":    call_mid + call_spread / 2,
            "call_last":   call_mid + rng.uniform(-1, 1, n) * call_spread * 0.3,
            "put_bid":     put_mid - put_spread / 2,
            "put_ask":     put_mid + put_spread / 2,
            "put_last":    put_mid + rng.uniform(-1, 1, n) * put_spread * 0.3,
            "dte_years":   dte_years,
        })


# ══════════════════════════════════════════════════════════════════════════════
#  DATA MANAGER  –  thread-safe bridge between IB and Dash
# ══════════════════════════════════════════════════════════════════════════════

class DataManager:

    def __init__(self, use_mock: bool = False):
        self._use_mock = use_mock
        self._lock = threading.Lock()
        self._refresh_now = threading.Event()
        self._cache: dict = {"error": "Waiting for first fetch..."}
        self._running = False
        self._thread: threading.Thread | None = None
        # ── History for charm heatmap ────────────────────────────────────
        self._charm_history: list[dict] = []
        self._history_ticker: str = ""
        self._MAX_HISTORY = 500
        # ── Session metrics (locked on first fetch per ticker) ───────────
        self._session_metrics: dict = {}
        self._session_ticker: str = ""
        # ── Flow tracker ─────────────────────────────────────────────────
        from flow_tracker import FlowTracker
        self._flow_tracker = FlowTracker()
        self._flow_ticker: str = ""

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._refresh_now.set()

    def get_cache(self) -> dict:
        with self._lock:
            return dict(self._cache)

    def get_charm_history(self) -> list[dict]:
        """Return a copy of the charm history list.  Thread-safe."""
        with self._lock:
            return list(self._charm_history)

    def request_refresh(self):
        self._refresh_now.set()

    def clear_history(self):
        """Called when the user switches ticker or expiry."""
        with self._lock:
            self._charm_history.clear()
            self._session_metrics = {}
            self._session_ticker = ""
            self._flow_tracker.reset()
            self._flow_ticker = ""

    def _worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        fetcher = MockDataFetcher() if self._use_mock else IBDataFetcher()
        fetcher.connect()

        while self._running:
            try:
                self._do_fetch(fetcher)
            except Exception as exc:
                traceback.print_exc()
                with self._lock:
                    self._cache["error"] = str(exc)

            self._refresh_now.wait(timeout=SETTINGS.refresh_seconds)
            self._refresh_now.clear()

    def _do_fetch(self, fetcher):
        ticker = SETTINGS.ticker.upper().strip()
        if not ticker:
            return

        expiries = fetcher.get_expiries(ticker)
        resolved = fetcher.resolve_expiry(ticker)
        spot     = fetcher.get_spot(ticker)
        chain    = fetcher.fetch_chain(ticker, resolved, spot)

        # Fetch previous day H/L (only once per ticker)
        prev_hl = self._cache.get("prev_day_hl", {"high": 0, "low": 0})
        cached_ticker = self._cache.get("ticker", "")
        if ticker != cached_ticker or prev_hl["high"] == 0:
            prev_hl = fetcher.get_prev_day_hl(ticker)
            print(f"  Prev day: H={prev_hl['high']:.2f}  L={prev_hl['low']:.2f}")

        # ── Lock session metrics on FIRST fetch per ticker ───────────
        # Uses previous close IV from file for daily EM,
        # and last Friday close IV for weekly EM.
        if chain is not None and not chain.empty:
            from greek_calculator import compute_live_metrics
            import session_store

            live = compute_live_metrics(chain, spot)
            current_iv = live.get("atm_iv", 0)

            # Save current IV to file (for tomorrow's prev close)
            # Skip in mock mode to avoid corrupting live data
            if current_iv > 0 and not self._use_mock:
                session_store.save_current_iv(ticker, spot, current_iv)

            # Compute session metrics only once per ticker
            if ticker != self._session_ticker or not self._session_metrics:
                import numpy as _np

                prev_close = session_store.get_prev_close(ticker)
                weekly_close = session_store.get_weekly_close(ticker)

                # Daily EM from previous close IV
                if prev_close:
                    pc_spot = prev_close["spot"]
                    pc_iv   = prev_close["iv"]
                    daily_em = pc_spot * pc_iv * _np.sqrt(1 / 252)
                    daily_hi = pc_spot + daily_em
                    daily_lo = pc_spot - daily_em
                    print(f"  Daily EM from prev close ({prev_close['timestamp']}): "
                          f"IV={pc_iv*100:.1f}%  ±${daily_em:,.1f}")
                else:
                    # No history yet — use current as fallback
                    daily_em = spot * current_iv * _np.sqrt(1 / 252)
                    daily_hi = spot + daily_em
                    daily_lo = spot - daily_em
                    pc_spot = spot
                    pc_iv = current_iv
                    print(f"  Daily EM fallback (no prev close): ±${daily_em:,.1f}")

                # Weekly EM from last Friday close IV
                if weekly_close:
                    wc_spot = weekly_close["spot"]
                    wc_iv   = weekly_close["iv"]
                    weekly_em = wc_spot * wc_iv * _np.sqrt(5 / 252)
                    weekly_hi = wc_spot + weekly_em
                    weekly_lo = wc_spot - weekly_em
                    print(f"  Weekly EM from Friday close ({weekly_close['timestamp']}): "
                          f"IV={wc_iv*100:.1f}%  ±${weekly_em:,.1f}")
                else:
                    weekly_em = spot * current_iv * _np.sqrt(5 / 252)
                    weekly_hi = spot + weekly_em
                    weekly_lo = spot - weekly_em
                    wc_spot = spot
                    wc_iv = current_iv
                    print(f"  Weekly EM fallback (no Friday close): ±${weekly_em:,.1f}")

                self._session_metrics = {
                    "open_spot":       spot,       # spot at first fetch
                    "prev_close_spot": pc_spot,
                    "prev_close_iv":   pc_iv,
                    "prev_close_ts":   prev_close.get("timestamp", "now"),
                    "daily_em":        daily_em,
                    "daily_high":      daily_hi,
                    "daily_low":       daily_lo,
                    "weekly_close_spot": weekly_close.get("spot", wc_spot),
                    "weekly_close_iv":   weekly_close.get("iv", wc_iv),
                    "weekly_close_ts":   weekly_close.get("timestamp", "now"),
                    "weekly_em":       weekly_em,
                    "weekly_high":     weekly_hi,
                    "weekly_low":      weekly_lo,
                }
                self._session_ticker = ticker

        # ── Compute charm per strike for the heatmap history ─────────
        now = dt.datetime.now(tz=ET)
        charm_snapshot = {}
        if chain is not None and not chain.empty:
            from greek_calculator import compute_exposure
            exp = compute_exposure(chain, spot, greek_mode=SETTINGS.greek_mode)
            for _, row in exp.iterrows():
                charm_snapshot[row["strike"]] = row["charm_exp"]

        # ── Process flow tracker ─────────────────────────────────────
        if ticker != self._flow_ticker:
            self._flow_tracker.reset(ticker)
            self._flow_ticker = ticker
        if chain is not None and not chain.empty:
            self._flow_tracker.process(chain, spot)
        flow_chain = self._flow_tracker.get_dealer_chain(chain)
        oi_flow_chain = self._flow_tracker.get_oi_plus_flow_chain(chain)
        flow_stats = self._flow_tracker.get_stats()

        with self._lock:
            # Reset history if ticker changed
            if ticker != self._history_ticker:
                self._charm_history.clear()
                self._history_ticker = ticker

            # Append snapshot
            if charm_snapshot:
                self._charm_history.append({
                    "time": now,
                    "spot": spot,
                    "charm": charm_snapshot,
                })
                if len(self._charm_history) > self._MAX_HISTORY:
                    self._charm_history = self._charm_history[-self._MAX_HISTORY:]

            self._cache = {
                "ticker":          ticker,
                "spot":            spot,
                "expiry":          resolved,
                "expiries":        expiries,
                "chain":           chain,
                "prev_day_hl":     prev_hl,
                "session_metrics": dict(self._session_metrics),
                "flow_chain":      flow_chain,
                "oi_flow_chain":   oi_flow_chain,
                "flow_stats":      flow_stats,
                "error":           None,
                "timestamp":       time.time(),
            }


# ── module-level singleton ───────────────────────────────────────────────────
data_manager: DataManager | None = None


def init_data_manager(use_mock: bool = False) -> DataManager:
    global data_manager
    data_manager = DataManager(use_mock=use_mock)
    data_manager.start()
    time.sleep(1)
    return data_manager
