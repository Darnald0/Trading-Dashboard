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
    """
    data = _load()
    ticker = ticker.upper()
    today = dt.date.today()
    today_date = today.isoformat()        # "2026-04-08"
    now_ts = _now_str()                    # "2026-04-08 14:32:15"
    weekday = today.weekday()             # 0=Mon ... 4=Fri

    if ticker not in data:
        data[ticker] = {}

    entry = data[ticker]

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
    """
    data = _load()
    entry = data.get(ticker.upper(), {})

    spot = entry.get("prev_close_spot", 0)
    iv = entry.get("prev_close_iv", 0)
    ts = entry.get("prev_close_timestamp", "")

    if spot > 0 and iv > 0:
        return {"spot": spot, "iv": iv, "timestamp": ts}
    return {}


def get_weekly_close(ticker: str) -> dict:
    """
    Return the last Friday's close data for computing weekly EM.
    Returns {"spot": float, "iv": float, "timestamp": str} or empty dict.
    """
    data = _load()
    entry = data.get(ticker.upper(), {})

    spot = entry.get("weekly_close_spot", 0)
    iv = entry.get("weekly_close_iv", 0)
    ts = entry.get("weekly_close_timestamp", "")

    if spot > 0 and iv > 0:
        return {"spot": spot, "iv": iv, "timestamp": ts}
    return {}
