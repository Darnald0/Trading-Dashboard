"""
heatmap_data.py  –  Bookmap-style L2 depth heatmap data manager.

Subscribes to IB market depth for a chosen contract, samples the order
book every second, and maintains a rolling 60-minute grid of order sizes
indexed by (price, time).

Key design points:
  - Uses its own IB connection (clientId = IB_CLIENT_ID + 30) to keep its
    market-data lines isolated from the other widgets
  - Continuous front-month resolution for futures
  - Adaptive L2 depth: tries 100 → 50 → 20 → 10 levels, falling back if IB rejects
    (user can lower the target via the sidebar slider — that becomes the new ceiling)
  - 40-minute rolling history at 1-second resolution = 2400 columns
  - Records executed trades on a separate stream for the price trail overlay
"""

import threading
import time
import asyncio
import traceback
import datetime as dt
from collections import deque

import numpy as np

from config import IB_HOST, IB_PORT, IB_CLIENT_ID, ET


# ── Configuration ──────────────────────────────────────────────────────────
DEPTH_LEVELS_PREFERRED = [25, 20, 10, 5]      # try these in order until IB accepts one
SAMPLE_INTERVAL = 1.0                         # seconds between order-book samples
HISTORY_SECONDS = 1800                        # 30 minutes of rolling history
PRICE_BAND_PCT  = 0.005                       # ±0.5% band around spot (used as data-collection range, not display range)


# ── Default tickers offered in the sidebar ─────────────────────────────────
# Tuple format: (display_label, ticker_symbol, sec_type, exchange, currency)
DEFAULT_TICKERS = [
    # Futures (use front-month resolution at fetch time)
    ("ES — S&P 500 (front month)",   "ES",  "FUT", "CME",       "USD"),
    ("NQ — Nasdaq-100 (front month)", "NQ", "FUT", "CME",       "USD"),
    ("YM — Dow (front month)",       "YM",  "FUT", "CBOT",      "USD"),
    ("RTY — Russell 2000 (front month)", "RTY", "FUT", "CME",   "USD"),
    ("CL — Crude Oil (front month)", "CL",  "FUT", "NYMEX",     "USD"),
    ("GC — Gold (front month)",      "GC",  "FUT", "COMEX",     "USD"),
    # Equity ETFs
    ("SPY — S&P 500 ETF",            "SPY", "STK", "SMART",     "USD"),
    ("QQQ — Nasdaq-100 ETF",         "QQQ", "STK", "SMART",     "USD"),
    ("IWM — Russell 2000 ETF",       "IWM", "STK", "SMART",     "USD"),
    ("DIA — Dow ETF",                "DIA", "STK", "SMART",     "USD"),
    # Mega-cap tech (often heavily traded with deep books)
    ("AAPL — Apple",                 "AAPL","STK", "SMART",     "USD"),
    ("MSFT — Microsoft",             "MSFT","STK", "SMART",     "USD"),
    ("NVDA — Nvidia",                "NVDA","STK", "SMART",     "USD"),
    ("TSLA — Tesla",                 "TSLA","STK", "SMART",     "USD"),
]


# ── Heatmap manager ────────────────────────────────────────────────────────

class HeatmapManager:
    """
    Owns one IB connection and one streaming depth subscription.
    Maintains a rolling grid of order book snapshots.
    """

    def __init__(self, host=IB_HOST, port=IB_PORT,
                 client_id=IB_CLIENT_ID + 30, use_mock=False):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._use_mock = use_mock

        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.ib = None

        # Active subscription state
        self._current_ticker = None
        self._current_contract = None
        self._depth_ticker = None      # ib_insync Ticker for depth
        self._trade_ticker = None      # ib_insync Ticker for last sales

        # Rolling history — list of dicts:
        #   {"ts": float, "spot": float, "bids": [(price, size), ...],
        #    "asks": [(price, size), ...]}
        # Newest on the right (deque appends append-right).
        self._snapshots = deque(maxlen=int(HISTORY_SECONDS / SAMPLE_INTERVAL))

        # Trade trail — recent executions
        # {"ts": float, "price": float, "size": int, "side": "buy"|"sell"}
        self._trades = deque(maxlen=4000)

        # Current best bid / ask / last (for live readouts)
        self._best_bid = 0.0
        self._best_ask = 0.0
        self._last_price = 0.0
        self._last_volume = 0
        self._active_depth_levels = 0
        self._target_depth_levels = DEPTH_LEVELS_PREFERRED[0]   # user-controlled ceiling
        self._pending_resubscribe = False

        # Status
        self._status = "Initializing..."
        self._error = None

    # ── Public API ─────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            if self._depth_ticker and self.ib:
                self.ib.cancelMktDepth(self._current_contract)
            if self._trade_ticker and self.ib:
                self.ib.cancelMktData(self._current_contract)
        except Exception:
            pass

    def set_ticker(self, ticker_symbol: str):
        """Switch the active subscription to a different ticker."""
        with self._lock:
            self._pending_ticker = ticker_symbol

    def set_depth_levels(self, n_levels: int):
        """
        Request a different depth-level target. The manager will re-subscribe
        if it differs from the active count. IB may still cap it lower.
        """
        n_levels = max(1, min(int(n_levels), 100))
        with self._lock:
            if n_levels != self._target_depth_levels:
                self._target_depth_levels = n_levels
                # Force a resubscription next worker tick
                self._pending_resubscribe = True

    def get_state(self) -> dict:
        """Returns current state for the UI to render."""
        with self._lock:
            return {
                "ticker":      self._current_ticker,
                "contract":    str(self._current_contract) if self._current_contract else "",
                "snapshots":   list(self._snapshots),
                "trades":      list(self._trades),
                "best_bid":    self._best_bid,
                "best_ask":    self._best_ask,
                "last_price":  self._last_price,
                "status":      self._status,
                "error":       self._error,
                "data_source": "MOCK" if self._use_mock else "IB",
                "depth_levels": self._active_depth_levels,
            }

    # ── Worker thread ──────────────────────────────────────────────

    def _worker(self):
        # Each background thread needs its own asyncio event loop for ib_insync
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if self._use_mock:
            print("  [Heatmap] Starting in MOCK mode")
            self._mock_loop()
            return

        # Connect to IB
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            print(f"  [Heatmap] Connected to IB (clientId={self.client_id})")
        except Exception as exc:
            print("\n" + "=" * 70)
            print("  [Heatmap] IB CONNECTION FAILED — falling back to MOCK")
            print(f"  Error: {type(exc).__name__}: {exc}")
            print("=" * 70 + "\n")
            traceback.print_exc()
            self._use_mock = True
            self._error = str(exc)
            self._mock_loop()
            return

        # Default to ES
        self._pending_ticker = "ES"
        self._current_ticker = None

        while self._running:
            try:
                # Switch contract if requested
                pending = getattr(self, "_pending_ticker", None)
                if pending and pending != self._current_ticker:
                    self._switch_contract(pending)
                elif self._pending_resubscribe and self._current_ticker:
                    # User changed depth level — resubscribe to the same ticker
                    self._pending_resubscribe = False
                    self._resubscribe_depth()

                # Sample current order book
                self._sample_book()

                # Process any IB events queued up
                self.ib.sleep(SAMPLE_INTERVAL)

            except Exception as exc:
                self._error = str(exc)
                traceback.print_exc()
                self.ib.sleep(2.0)

    def _resubscribe_depth(self):
        """Re-request market depth at the user's current target level."""
        if self._current_contract is None or self.ib is None:
            return
        try:
            self.ib.cancelMktDepth(self._current_contract)
        except Exception:
            pass
        self._depth_ticker = None
        self._active_depth_levels = 0
        self._subscribe_depth(self._current_contract)
        print(f"  [Heatmap] Resubscribed depth at target={self._target_depth_levels} "
              f"→ active={self._active_depth_levels}")

    def _switch_contract(self, ticker_symbol: str):
        """Cancel old depth subscription and start a new one."""
        from ib_insync import Stock, ContFuture

        # Cancel previous subscriptions
        if self._current_contract:
            try:
                self.ib.cancelMktDepth(self._current_contract)
            except Exception:
                pass
            try:
                self.ib.cancelMktData(self._current_contract)
            except Exception:
                pass
            self._depth_ticker = None
            self._trade_ticker = None

        # Look up ticker in defaults to determine type
        match = None
        for label, sym, sec_type, exch, ccy in DEFAULT_TICKERS:
            if sym == ticker_symbol:
                match = (label, sym, sec_type, exch, ccy)
                break

        if not match:
            self._status = f"Unknown ticker: {ticker_symbol}"
            return

        label, sym, sec_type, exch, ccy = match

        if sec_type == "FUT":
            # Continuous front-month resolution
            contract = ContFuture(sym, exch, ccy)
        else:
            contract = Stock(sym, exch, ccy)

        try:
            self.ib.qualifyContracts(contract)
        except Exception as exc:
            self._status = f"Failed to qualify {sym}: {exc}"
            return

        if contract.conId == 0:
            self._status = f"No contract found for {sym}"
            return

        # If it's a ContFuture, IB resolves it to a real Future. We need the
        # qualified front-month contract for market data requests.
        self._current_contract = contract
        self._current_ticker = sym

        # Reset rolling state
        with self._lock:
            self._snapshots.clear()
            self._trades.clear()
            self._best_bid = 0.0
            self._best_ask = 0.0
            self._last_price = 0.0

        # Subscribe to depth + last-sale stream.
        self.ib.reqMarketDataType(1)   # real-time, falls back to delayed
        self._subscribe_depth(contract)

        # Last-sale stream for trade overlay
        if self._depth_ticker is not None:
            self._trade_ticker = self.ib.reqMktData(contract, "", False, False)
            self._status = (f"Subscribed to {sym} "
                             f"({contract.localSymbol or contract.symbol}) — "
                             f"{self._active_depth_levels} levels")
            print(f"  [Heatmap] {self._status}")

    def _subscribe_depth(self, contract):
        """
        Try to subscribe to L2 depth at the user's target level count,
        falling back through DEPTH_LEVELS_PREFERRED if IB rejects.
        Only tiers ≤ target are considered.
        """
        self._depth_ticker = None
        self._active_depth_levels = 0
        last_error = None

        tiers_to_try = [n for n in DEPTH_LEVELS_PREFERRED
                        if n <= self._target_depth_levels]
        if not tiers_to_try:
            tiers_to_try = [DEPTH_LEVELS_PREFERRED[-1]]

        for n_levels in tiers_to_try:
            try:
                self._depth_ticker = self.ib.reqMktDepth(
                    contract, numRows=n_levels, isSmartDepth=False,
                )
                self.ib.sleep(1.5)   # wait for IB to respond / reject

                domBids = getattr(self._depth_ticker, "domBids", None) or []
                domAsks = getattr(self._depth_ticker, "domAsks", None) or []

                self._active_depth_levels = n_levels
                print(f"  [Heatmap] Depth subscription OK at {n_levels} levels "
                      f"(received {len(domBids)} bids, {len(domAsks)} asks so far)")
                return
            except Exception as exc:
                last_error = exc
                print(f"  [Heatmap] Depth request rejected at {n_levels} levels: "
                      f"{type(exc).__name__}: {exc}")
                try:
                    self.ib.cancelMktDepth(contract)
                except Exception:
                    pass
                self._depth_ticker = None
                continue

        if self._depth_ticker is None:
            self._status = (f"Depth unavailable — last error: {last_error}. "
                             f"You may need an L2 subscription for this contract.")
            print(f"  [Heatmap] {self._status}")

    def _sample_book(self):
        """Sample the current order book and append to history."""
        if not self._depth_ticker:
            return

        # Pull the current bid/ask depth ladders from ib_insync's tracker
        bids = []
        asks = []

        # ib_insync exposes the depth book as `domBids` and `domAsks` on the
        # Ticker object, sorted by price (best first)
        for entry in (self._depth_ticker.domBids or []):
            if entry and entry.price > 0:
                bids.append((float(entry.price), int(entry.size)))
        for entry in (self._depth_ticker.domAsks or []):
            if entry and entry.price > 0:
                asks.append((float(entry.price), int(entry.size)))

        # Best bid / ask / last
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0

        last_price = 0.0
        if self._trade_ticker:
            lp = self._trade_ticker.last
            if lp and lp > 0 and lp == lp:   # NaN check
                last_price = float(lp)

        # If there's a fresh trade, record it
        if (self._trade_ticker and last_price > 0
                and self._trade_ticker.lastSize):
            ls = int(self._trade_ticker.lastSize)
            if ls > 0 and ls != self._last_volume:
                # Determine side: trade > midpoint = buy, else sell
                mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else last_price
                side = "buy" if last_price >= mid else "sell"
                with self._lock:
                    self._trades.append({
                        "ts":    time.time(),
                        "price": last_price,
                        "size":  ls,
                        "side":  side,
                    })
                self._last_volume = ls

        # Append snapshot
        snap = {
            "ts":    time.time(),
            "spot":  last_price if last_price else (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0,
            "bids":  bids,
            "asks":  asks,
        }
        with self._lock:
            self._snapshots.append(snap)
            self._best_bid = best_bid
            self._best_ask = best_ask
            self._last_price = last_price

    # ── Mock data loop ─────────────────────────────────────────────

    def _mock_loop(self):
        """Generate plausible synthetic depth data for development."""
        import random
        rng = random.Random(42)

        self._current_ticker = "ES"
        self._current_contract = "ES MOCK"
        self._status = "MOCK depth stream — ES synthetic"
        self._active_depth_levels = self._target_depth_levels

        spot = 5450.0
        tick_size = 0.25

        while self._running:
            n_levels = self._target_depth_levels
            self._active_depth_levels = n_levels
            # Random walk
            spot += rng.uniform(-0.5, 0.5)

            bids = []
            asks = []
            for i in range(n_levels):
                bid_price = round((spot - (i + 1) * tick_size) / tick_size) * tick_size
                ask_price = round((spot + (i + 1) * tick_size) / tick_size) * tick_size
                # Size is larger near best, with random clusters
                base_size = max(50 - i * 3, 5)
                bid_size = int(base_size * rng.uniform(0.5, 2.5))
                ask_size = int(base_size * rng.uniform(0.5, 2.5))
                # Occasionally insert a "wall" (large limit order)
                if rng.random() < 0.05:
                    bid_size *= rng.randint(5, 15)
                if rng.random() < 0.05:
                    ask_size *= rng.randint(5, 15)
                bids.append((bid_price, bid_size))
                asks.append((ask_price, ask_size))

            # Random trade
            if rng.random() < 0.4:
                side = "buy" if rng.random() > 0.5 else "sell"
                trade_price = bids[0][0] if side == "sell" else asks[0][0]
                trade_size = rng.randint(1, 50)
                with self._lock:
                    self._trades.append({
                        "ts":    time.time(),
                        "price": trade_price,
                        "size":  trade_size,
                        "side":  side,
                    })

            with self._lock:
                self._snapshots.append({
                    "ts":   time.time(),
                    "spot": spot,
                    "bids": bids,
                    "asks": asks,
                })
                self._best_bid = bids[0][0]
                self._best_ask = asks[0][0]
                self._last_price = spot

            time.sleep(SAMPLE_INTERVAL)


# ── Module-level singleton ─────────────────────────────────────────────────
heatmap_manager: HeatmapManager | None = None


def init_heatmap_manager(use_mock=False) -> HeatmapManager:
    global heatmap_manager
    mode = "MOCK" if use_mock else "IB (live)"
    print(f"  Initializing Heatmap Manager [{mode}]")
    heatmap_manager = HeatmapManager(use_mock=use_mock)
    heatmap_manager.start()
    return heatmap_manager
