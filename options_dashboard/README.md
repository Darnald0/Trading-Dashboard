## TLDR

SETUP.bat — Run this once on a fresh machine. Creates the venv and installs all dependencies. You already did this manually so you can skip it.
START_DASHBOARD.bat — Double-click to launch. Activates venv, starts the server, prints the URL. Keep the window open.
START_DASHBOARD_AUTO.bat — Same thing but automatically opens http://localhost:8050 in your browser after 3 seconds.
START_MOCK.bat — Launches with synthetic data, no IB needed.

## Python

`Python 3.14.x`

## IBKR

1. Open TWS or IB Gateway and log in.
2. Go to **File → Global Configuration** (TWS) or **Configure → Settings** (Gateway).
3. Navigate to **API → Settings**.
4. Check **"Enable ActiveX and Socket Clients"**.
5. Set the **Socket port** to `7497` (paper trading) or `7496` (live trading).
6. Uncheck **"Read-Only API"** (so the app can request market data).
7. Add `127.0.0.1` to **"Trusted IPs"** if it's not already there.
8. Click **Apply** / **OK**.

## Change default ticker, refresh speed, or IB port

Open `config.py` in any text editor (Notepad, VS Code, etc.) and change the values at the top:

```python
IB_HOST = "127.0.0.1"        # usually don't change this
IB_PORT = 7497               # 7497=paper, 7496=live
IB_CLIENT_ID = 1             # change if running multiple scripts
RISK_FREE_RATE = 0.045       # update to current T-bill rate
```

And inside the `DashboardSettings` class:

```python
ticker: str = "SPY"          # default ticker on startup
refresh_seconds: int = 30    # default refresh interval
```

### Use live trading data instead of paper

In `config.py`, change:

```python
IB_PORT = 7496   # live trading port
```
