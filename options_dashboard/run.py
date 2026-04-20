"""
run.py  -  Entry point for the Options Greek Dashboard.

Usage
-----
    python run.py              # connect to IB TWS / Gateway
    python run.py --mock       # use fake data (no IB needed)
    python run.py --port 8051  # custom port
"""

import asyncio
import argparse

# Python 3.10+ no longer auto-creates an event loop in the main thread.
# ib_insync (via eventkit) needs one to exist at import time.
asyncio.set_event_loop(asyncio.new_event_loop())

import data_fetcher
import matrix_data
import cot_scraper
from dashboard import app


def main():
    parser = argparse.ArgumentParser(description="Options Greek Dashboard")
    parser.add_argument(
        "--mock", action="store_true",
        help="Run with synthetic data (no IB connection required)",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Web server port (default: 8050)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Dash debug / hot-reload mode",
    )
    args = parser.parse_args()

    # Start the background data-fetching threads BEFORE Dash begins serving
    use_mock = args.mock or data_fetcher.USE_MOCK
    data_fetcher.init_data_manager(use_mock=use_mock)
    matrix_data.init_matrix_manager(use_mock=use_mock)
    cot_scraper.init_cot_manager()

    print(f"\n  Dashboard starting at  http://localhost:{args.port}\n")
    app.run(debug=args.debug, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
