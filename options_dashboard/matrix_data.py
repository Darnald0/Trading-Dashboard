"""
matrix_data.py  –  Multi-ticker data manager for Market Matrix widget.

Fetches option chains for SPXW, SPY, NDX, QQQ (0DTE or nearest)
on a separate IB connection (clientId = main + 1).
"""

import threading
import time
import asyncio
import traceback
import datetime as dt

from config import IB_HOST, IB_PORT, IB_CLIENT_ID, RISK_FREE_RATE, ET

MATRIX_TICKERS = ["SPX", "SPY", "NDX", "QQQ"]
MATRIX_CLIENT_ID = IB_CLIENT_ID + 10   # avoid collision with main fetcher


class MatrixDataManager:
    """Fetches chains for 4 tickers and stores per-ticker caches."""

    def __init__(self, use_mock: bool = False, refresh_seconds: int = 30):
        self._use_mock = use_mock
        self._refresh = refresh_seconds
        self._lock = threading.Lock()
        self._caches: dict[str, dict] = {}   # {ticker: cache_dict}
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

    def get_cache(self, ticker: str) -> dict:
        with self._lock:
            return dict(self._caches.get(ticker.upper(), {"error": "Waiting..."}))

    def get_all_caches(self) -> dict:
        with self._lock:
            return {t: dict(c) for t, c in self._caches.items()}

    def _worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if self._use_mock:
            from data_fetcher import MockDataFetcher
            fetcher = MockDataFetcher()
        else:
            from data_fetcher import IBDataFetcher
            try:
                fetcher = IBDataFetcher(client_id=MATRIX_CLIENT_ID)
                print(f"  [Matrix] Using clientId={MATRIX_CLIENT_ID}")
            except Exception as exc:
                print(f"  [Matrix] IB init failed, falling back to mock: {exc}")
                from data_fetcher import MockDataFetcher
                fetcher = MockDataFetcher()

        try:
            fetcher.connect()
        except Exception as exc:
            print(f"  [Matrix] Connection failed, falling back to mock: {exc}")
            from data_fetcher import MockDataFetcher
            fetcher = MockDataFetcher()
            fetcher.connect()

        while self._running:
            for ticker in MATRIX_TICKERS:
                if not self._running:
                    break
                try:
                    self._fetch_one(fetcher, ticker)
                except Exception as exc:
                    traceback.print_exc()
                    with self._lock:
                        self._caches[ticker] = {"error": str(exc)}

            # Wait between full cycles
            for _ in range(self._refresh * 2):
                if not self._running:
                    break
                time.sleep(0.5)

    def _fetch_one(self, fetcher, ticker: str):
        """Fetch chain for one ticker, pick 0DTE or nearest expiry."""
        from greek_calculator import compute_exposure, compute_time_to_expiry

        expiries = fetcher.get_expiries(ticker)
        spot = fetcher.get_spot(ticker)

        if not expiries or spot <= 0:
            with self._lock:
                self._caches[ticker] = {"error": f"No data for {ticker}"}
            return

        # Pick 0DTE (today) if available, otherwise nearest
        today_str = dt.datetime.now(tz=ET).strftime("%Y%m%d")
        if today_str in expiries:
            resolved = today_str
        else:
            # Nearest future expiry
            future = [e for e in expiries if e >= today_str]
            resolved = future[0] if future else expiries[0]

        chain = fetcher.fetch_chain(ticker, resolved, spot)

        if chain is None or chain.empty:
            with self._lock:
                self._caches[ticker] = {"error": f"Empty chain for {ticker}"}
            return

        # Compute all exposure modes
        exp_oi       = compute_exposure(chain, spot, greek_mode="oi")
        exp_vol      = compute_exposure(chain, spot, greek_mode="volume")
        exp_combined = compute_exposure(chain, spot, greek_mode="combined")

        with self._lock:
            self._caches[ticker] = {
                "ticker":     ticker,
                "spot":       spot,
                "expiry":     resolved,
                "chain":      chain,
                "exp_oi":     exp_oi,
                "exp_vol":    exp_vol,
                "exp_combined": exp_combined,
                "error":      None,
                "timestamp":  time.time(),
            }

        print(f"  [Matrix] {ticker}: spot=${spot:,.2f}  exp={resolved}  "
              f"strikes={len(chain)}")


# ── module-level singleton ───────────────────────────────────────────────────
matrix_manager: MatrixDataManager | None = None


def init_matrix_manager(use_mock: bool = False) -> MatrixDataManager:
    global matrix_manager
    matrix_manager = MatrixDataManager(use_mock=use_mock, refresh_seconds=60)
    matrix_manager.start()
    time.sleep(1)
    return matrix_manager
