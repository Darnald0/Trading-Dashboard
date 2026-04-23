"""
session_store.py  –  Persists close-of-day IV data to a JSON file.

On each fetch, saves the latest ATM IV + spot price with full timestamp.
On startup, loads the previous day's close data to compute fixed EM ranges.
Also tracks the last Friday close for weekly EM.

Data file: session_data.json in the project directory.
"""

import json
import datetime as dt
from pathlib import Path

DATA_FILE = Path(__file__).parent / "session_data.json"


def _load() -> dict:
    """Load persisted data from disk."""
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save(data: dict):
    """Write data to disk."""
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"  Warning: could not save session data: {e}")


def _now_str() -> str:
    """Current datetime as ISO string with seconds."""
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _date_part(timestamp_str: str) -> str:
    """Extract just the date portion (YYYY-MM-DD) from a timestamp string."""
    return timestamp_str[:10] if timestamp_str else ""


def save_current_iv(ticker: str, spot: float, atm_iv: float):
    """
    Called on every fetch.  Saves the latest IV + spot with timestamp.
    Also snapshots Friday close data for weekly EM.
    Also appends to historical IV series for rank/percentile calculations.

    SANITY GUARD: rejects IV values outside [1%, 300%] as these are almost
    always garbage from TWS (expired contracts, illiquid strikes, after-hours
    placeholder values). A legitimate ATM IV for SPX/SPY/NDX/QQQ will
    essentially never exceed 200%, and never be below 1%.
    """
    # Reject obviously bad data
    if atm_iv <= 0.01 or atm_iv > 3.0:
        print(f"  [session_store] Rejecting garbage IV for {ticker}: "
              f"{atm_iv*100:.1f}% (must be 1%–300%)")
        return
    if spot <= 0:
        return

    data = _load()
    ticker = ticker.upper()
    today = dt.date.today()
    today_date = today.isoformat()        # "2026-04-08"
    now_ts = _now_str()                    # "2026-04-08 14:32:15"
    weekday = today.weekday()             # 0=Mon ... 4=Fri

    if ticker not in data:
        data[ticker] = {}

    entry = data[ticker]

    # ── Append to IV history (for rank/percentile over 252 trading days) ──
    # Keep one entry per day (last one wins for that day)
    # History is list of {"date": "YYYY-MM-DD", "iv": float}
    history = entry.get("iv_history", [])
    # Remove any existing entry for today
    history = [h for h in history if h.get("date") != today_date]
    history.append({"date": today_date, "iv": float(atm_iv)})
    # Keep at most ~252 most recent trading days (~1 year)
    history = history[-252:]
    entry["iv_history"] = history

    # ── Always update "latest" (becomes prev_close tomorrow) ─────────
    # Save the OLD latest before overwriting (needed for promotion check)
    old_latest_date = _date_part(entry.get("latest_timestamp", ""))
    old_latest_spot = entry.get("latest_spot", 0)
    old_latest_iv = entry.get("latest_iv", 0)
    old_latest_ts = entry.get("latest_timestamp", "")

    entry["latest_timestamp"] = now_ts
    entry["latest_spot"] = spot
    entry["latest_iv"] = atm_iv

    # ── Promote yesterday's latest to prev_close ─────────────────────
    prev_close_date = _date_part(entry.get("prev_close_timestamp", ""))
    if prev_close_date != today_date:
        # Check if old latest is from a previous day
        if old_latest_date and old_latest_date < today_date:
            entry["prev_close_timestamp"] = old_latest_ts
            entry["prev_close_spot"] = old_latest_spot
            entry["prev_close_iv"] = old_latest_iv
        elif not prev_close_date:
            # First time ever — use current as fallback
            entry["prev_close_timestamp"] = now_ts
            entry["prev_close_spot"] = spot
            entry["prev_close_iv"] = atm_iv

    # ── Friday close for weekly EM ───────────────────────────────────
    if weekday == 4:
        entry["weekly_close_timestamp"] = now_ts
        entry["weekly_close_spot"] = spot
        entry["weekly_close_iv"] = atm_iv
    elif "weekly_close_timestamp" not in entry:
        # No Friday data yet — use current as fallback
        entry["weekly_close_timestamp"] = now_ts
        entry["weekly_close_spot"] = spot
        entry["weekly_close_iv"] = atm_iv

    data[ticker] = entry
    _save(data)


def get_prev_close(ticker: str) -> dict:
    """
    Return the previous day's close data for computing daily EM.
    Returns {"spot": float, "iv": float, "timestamp": str} or empty dict.
    Rejects corrupted values (IV out of 1%–300% range).
    """
    data = _load()
    entry = data.get(ticker.upper(), {})

    spot = entry.get("prev_close_spot", 0)
    iv = entry.get("prev_close_iv", 0)
    ts = entry.get("prev_close_timestamp", "")

    if spot > 0 and 0.01 < iv <= 3.0:
        return {"spot": spot, "iv": iv, "timestamp": ts}
    if iv > 3.0 or iv < 0:
        print(f"  [session_store] Ignoring corrupted prev_close IV for "
              f"{ticker}: {iv*100:.1f}% — using current IV fallback")
    return {}


def get_weekly_close(ticker: str) -> dict:
    """
    Return the last Friday's close data for computing weekly EM.
    Returns {"spot": float, "iv": float, "timestamp": str} or empty dict.
    Rejects corrupted values (IV out of 1%–300% range).
    """
    data = _load()
    entry = data.get(ticker.upper(), {})

    spot = entry.get("weekly_close_spot", 0)
    iv = entry.get("weekly_close_iv", 0)
    ts = entry.get("weekly_close_timestamp", "")

    if spot > 0 and 0.01 < iv <= 3.0:
        return {"spot": spot, "iv": iv, "timestamp": ts}
    if iv > 3.0 or iv < 0:
        print(f"  [session_store] Ignoring corrupted weekly_close IV for "
              f"{ticker}: {iv*100:.1f}% — using current IV fallback")
    return {}


def get_iv_rank_percentile(ticker: str, current_iv: float) -> dict:
    """
    Compute IV rank and IV percentile from stored history.

    IV Rank      = (current - min) / (max - min) × 100
                   Where does current IV sit in the historical high-low range?
    IV Percentile = fraction of days in history with IV < current  × 100
                   What % of the past N days had lower IV than today?

    Both are 0-100. Below 30 = low vol environment. Above 70 = high vol.

    Returns dict with:
        iv_rank, iv_percentile, iv_min_52w, iv_max_52w, history_days
    """
    data = _load()
    entry = data.get(ticker.upper(), {})
    history = entry.get("iv_history", [])

    if not history or current_iv <= 0:
        return {"iv_rank": None, "iv_percentile": None,
                "iv_min_52w": 0, "iv_max_52w": 0, "history_days": 0}

    ivs = [h.get("iv", 0) for h in history if h.get("iv", 0) > 0]
    if not ivs:
        return {"iv_rank": None, "iv_percentile": None,
                "iv_min_52w": 0, "iv_max_52w": 0, "history_days": 0}

    iv_min = min(ivs)
    iv_max = max(ivs)

    # IV Rank
    if iv_max > iv_min:
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
        iv_rank = max(0, min(100, iv_rank))   # clip to 0-100
    else:
        iv_rank = 50.0

    # IV Percentile
    count_below = sum(1 for v in ivs if v < current_iv)
    iv_percentile = count_below / len(ivs) * 100

    return {
        "iv_rank":       round(iv_rank, 1),
        "iv_percentile": round(iv_percentile, 1),
        "iv_min_52w":    round(iv_min, 4),
        "iv_max_52w":    round(iv_max, 4),
        "history_days":  len(ivs),
    }
