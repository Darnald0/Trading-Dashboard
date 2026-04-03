venv\Scripts\activate
python run.py

# Options Greek Dashboard — Step-by-Step Setup Guide

A real-time Gamma / Charm / Vanna exposure dashboard that connects to Interactive Brokers and refreshes as fast as every 10 seconds.

---

## What you are building

A web page (runs locally in your browser) that looks like this:

```
┌──────────────┬──────────────────────────────────────────┐
│   SETTINGS   │                                          │
│              │  ████ Gamma Exposure (GEX) ████████████  │
│  Ticker:     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ← spot price    │
│  [SPY    ]   │                                          │
│              │  ████ Charm Exposure ██████████████████   │
│  Expiry:     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │                  │
│  [Auto ▾ ]   │                                          │
│              │  ████ Vanna Exposure ██████████████████   │
│  Refresh:    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │                  │
│  ──●─── 30s  │                                          │
│              │                                          │
│  Mode:       │                                          │
│  ● OI        │                                          │
│  ○ Volume    │                                          │
│  ○ Combined  │                                          │
└──────────────┴──────────────────────────────────────────┘
```

Each chart is a bar chart of exposure by strike price, with a dashed yellow line showing the current spot price. Green/blue bars = positive exposure, red/orange bars = negative exposure.

---

## Project files overview

| File | Role |
|---|---|
| `config.py` | All settings (IB connection, defaults, appearance). This is the only file you edit to configure things. |
| `data_fetcher.py` | Talks to Interactive Brokers. Downloads the option chain (strikes, OI, volume, IV). Also contains a mock-data mode for testing. |
| `greek_calculator.py` | Pure math. Takes the chain data and calculates Gamma, Charm, and Vanna exposure per strike using Black-Scholes. |
| `dashboard.py` | The visual interface. Sidebar controls + three charts. Calls the fetcher and calculator on each refresh cycle. |
| `run.py` | The file you run to start everything. Handles command-line flags like `--mock`. |
| `requirements.txt` | List of libraries Python needs to install. |

---

## STEP 1 — Verify Python is installed

Open your **Terminal** (Mac/Linux) or **Command Prompt / PowerShell** (Windows).

Type:

```bash
python --version
```

You should see something like `Python 3.14.x`. If it says "not found", try `python3 --version` instead. On some systems the command is `python3` — in the instructions below, replace `python` with `python3` wherever needed.

---

## STEP 2 — Download the project files

Create a folder anywhere on your computer (for example on your Desktop) called `options_dashboard`. Place all six project files inside it:

```
options_dashboard/
├── config.py
├── dashboard.py
├── data_fetcher.py
├── greek_calculator.py
├── requirements.txt
└── run.py
```

---

## STEP 3 — Open a terminal in the project folder

**Windows:** Open File Explorer → navigate to the `options_dashboard` folder → click the address bar → type `cmd` → press Enter.

**Mac:** Open Terminal → type `cd ` (with a space) → drag the folder onto the Terminal window → press Enter.

**Linux:** Right-click the folder → "Open in Terminal".

You should now see something like:

```
C:\Users\you\Desktop\options_dashboard>
```
or
```
~/Desktop/options_dashboard $
```

---

## STEP 4 — Create a virtual environment (recommended)

A virtual environment keeps this project's libraries separate from the rest of your system. Run these commands one by one:

```bash
python -m venv venv
```

Then **activate** it:

| OS | Command |
|---|---|
| Windows (CMD) | `venv\Scripts\activate` |
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |
| Mac / Linux | `source venv/bin/activate` |

You'll see `(venv)` appear at the start of your prompt. This means it worked.

---

## STEP 5 — Install the required libraries

Still in the same terminal (with `(venv)` active):

```bash
pip install -r requirements.txt
```

This downloads and installs everything the dashboard needs. It may take a minute or two. You'll see a lot of text scrolling — that's normal.

If you see errors about "Microsoft Visual C++", install the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and try again.

---

## STEP 6 — Set up Interactive Brokers

The dashboard talks to IB through their **TWS (Trader Workstation)** or **IB Gateway** application, which must be running on your computer.

### 6a — Install TWS or IB Gateway

Download from: https://www.interactivebrokers.com/en/trading/tws.php

IB Gateway is lighter-weight (no GUI charts) and is better for this use case. TWS works too.

### 6b — Enable the API in TWS / IB Gateway

1. Open TWS or IB Gateway and log in.
2. Go to **File → Global Configuration** (TWS) or **Configure → Settings** (Gateway).
3. Navigate to **API → Settings**.
4. Check **"Enable ActiveX and Socket Clients"**.
5. Set the **Socket port** to `7497` (paper trading) or `7496` (live trading).
6. Uncheck **"Read-Only API"** (so the app can request market data).
7. Add `127.0.0.1` to **"Trusted IPs"** if it's not already there.
8. Click **Apply** / **OK**.

### 6c — Make sure you have market data subscriptions

IB requires a market data subscription to get real-time option chains. At minimum you need:

- **US Securities Snapshot and Futures Value Bundle** (~$10/month) for US equities/options.
- If you only have a paper account, enable **delayed data** (free) by changing line in `data_fetcher.py`: find `reqMarketDataType(1)` and change the `1` to `3`.

---

## STEP 7 — Quick test with MOCK data (no IB needed)

Before connecting to IB, verify everything works with fake data:

```bash
python run.py --mock
```

You should see:

```
🟡  MOCK MODE – using synthetic data (no IB connection)
🚀  Dashboard starting at  http://localhost:8050
```

Open your web browser and go to: **http://localhost:8050**

You should see the dashboard with three charts showing synthetic Gamma, Charm, and Vanna exposure for SPY. Try changing the ticker, the refresh slider, and the exposure mode to make sure the controls work.

Press `Ctrl+C` in the terminal to stop the dashboard.

---

## STEP 8 — Run with real IB data

Make sure TWS or IB Gateway is running and logged in, then:

```bash
python run.py
```

The dashboard will connect to IB, fetch the option chain for SPY, and display real greek exposures. Open **http://localhost:8050** in your browser.

---

## STEP 9 — Customise (optional)

### Change default ticker, refresh speed, or IB port

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

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'ib_insync'` | You forgot to activate the venv or install requirements. Run `source venv/bin/activate` then `pip install -r requirements.txt`. |
| `ConnectionRefusedError` | TWS / IB Gateway is not running, or the API port is wrong. Check Step 6. |
| `No option expiries found` | The ticker symbol might be wrong, or you don't have market data permissions for that product. |
| Charts show all zeros | IV data may be missing. Check that you have a market data subscription (Step 6c). Try switching to delayed data (`reqMarketDataType(3)`). |
| `Address already in use` | Another program is using port 8050. Run with `python run.py --port 8051` instead. |
| Dashboard is slow to refresh | IB rate-limits snapshot requests. With many strikes, 10-second refresh may be too fast. Try 30 seconds. |

---

## Understanding the charts

### Gamma Exposure (GEX)
Shows how much dealers' gamma (price sensitivity of delta) is concentrated at each strike. Large positive GEX = market tends to mean-revert (dealers hedge by selling highs, buying lows). Large negative GEX = market tends to trend (dealers amplify moves).

### Charm Exposure
Shows delta decay — how much delta the dealers will gain or lose as time passes (even if price stays flat). This tells you the direction of hedging flows the next day.

### Vanna Exposure
Shows how delta changes when implied volatility changes. When IV drops (e.g. after a calm day), vanna tells you which direction the resulting dealer hedging will push the market.

### The yellow dashed line
This is the current spot price of the underlying. It helps you see whether the stock is sitting in a zone of high or low exposure.

### The three modes
- **Open Interest**: Uses the full accumulated open interest at each strike. Best for seeing the structural "walls" that have built up over time.
- **Session Volume**: Uses only contracts traded today. Shows what fresh positioning is being added right now during the session.
- **Combined**: Adds both together. Useful to see the total picture of existing + new positioning.

---

## What to learn next

Once you're comfortable with this setup, common next steps include:

1. **Add more expiries**: Aggregate exposure across all expiries (not just one) for a full-term-structure view.
2. **Add a net GEX profile**: Show what happens to total GEX if the stock moves ±5% (the "GEX curve").
3. **Add a delta-adjusted volume tracker**: Show large block trades in real time.
4. **Deploy to a server**: Run the dashboard on a VPS so you can access it from your phone.

---

## Stopping the dashboard

Press `Ctrl+C` in the terminal where it's running. To deactivate the virtual environment afterwards, type:

```bash
deactivate
```
