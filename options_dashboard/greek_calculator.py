"""
greek_calculator.py  –  Black-Scholes greeks used for exposure calculations.

All functions are *pure* (no side effects, no IB calls).  They take
strike / spot / vol / time arrays and return the corresponding greek
values.  The dashboard multiplies these by OI or volume afterwards.

Key conventions (matching SpotGamma / UW / OD):
  - GEX = Γ × OI × 100 × S² × 0.01   (dollar gamma)
  - DTE = business_days / 252 with intraday 0DTE precision
  - IV  = Newton-Raphson from bid/ask mid, fallback to IB model IV
  - Liquidity filter: skip contracts with spread > 20% of mid
"""

import math
import numpy as np
from scipy.stats import norm
from datetime import datetime, date
from config import RISK_FREE_RATE

# ── constants ─────────────────────────────────────────────────────────────────

MAX_SPREAD_PCT = 0.20          # skip illiquid contracts (20% spread)
TRADING_MINUTES_PER_DAY = 405  # 9:30 - 16:15 ET
OPEN_MIN_OF_DAY = 9 * 60 + 30
CLOSE_MIN_OF_DAY = 16 * 60 + 15

# ── helpers ──────────────────────────────────────────────────────────────────

def _d1(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """Standard Black-Scholes d1."""
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """Standard Black-Scholes d2."""
    return _d1(S, K, T, sigma, r, q) - sigma * np.sqrt(T)


# ── time to expiry (business days / 252) ─────────────────────────────────────

def compute_time_to_expiry(expiry_str: str) -> float:
    """
    Compute time to expiry in years using business days / 252.

    Matches SpotGamma/UW convention:
    - Business days remaining / 252
    - For 0DTE: fraction of trading day remaining (intraday precision)
    - Uses ET market hours (9:30 - 16:15)
    """
    try:
        from config import ET
        now = datetime.now(tz=ET)
    except ImportError:
        now = datetime.now()

    expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()

    today = now.date()
    is_busday = bool(np.is_busday(today))

    if not is_busday:
        # Advance to next business day, treat as market open
        today = np.busday_offset(today, 0, roll="forward").astype("datetime64[D]").item()
        now_min = OPEN_MIN_OF_DAY
    else:
        now_min = max(now.hour * 60 + now.minute, OPEN_MIN_OF_DAY)

    # 0DTE: intraday fraction
    if today == expiry_date:
        remaining_min = max(CLOSE_MIN_OF_DAY - now_min, 1)
        T = (remaining_min / TRADING_MINUTES_PER_DAY) / 252
        return max(T, 1e-6)

    # Future expiries: count business days
    bdays = int(np.busday_count(today, expiry_date))
    if bdays <= 0:
        return 1e-6

    # Add partial day for current trading session
    remaining_today = max(CLOSE_MIN_OF_DAY - now_min, 0) / TRADING_MINUTES_PER_DAY
    T = (bdays + remaining_today) / 252
    return max(T, 1e-6)


# ── Newton-Raphson IV solver ─────────────────────────────────────────────────

def _bs_price_scalar(S, K, T, sigma, r, option_type):
    """Black-Scholes price for a single option (scalar inputs)."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "C" else (K - S))
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = math.exp(-r * T)
    n_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)

    if option_type == "C":
        from scipy.stats import norm as _norm
        return S * _norm.cdf(d1) - K * disc * _norm.cdf(d2)
    else:
        from scipy.stats import norm as _norm
        return K * disc * _norm.cdf(-d2) - S * _norm.cdf(-d1)


def _bs_vega_scalar(S, K, T, sigma, r):
    """Black-Scholes vega (scalar)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    n_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return S * n_d1 * sqrtT


def implied_vol_newton(S, K, T, r, market_price, option_type,
                       tol=1e-6, max_iter=50):
    """
    Newton-Raphson IV solver.
    Returns solved IV or None if convergence fails.
    """
    if T < 1 / (252 * TRADING_MINUTES_PER_DAY) or market_price <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = max(0.0, (S - K) if option_type == "C" else (K - S))
    if market_price < intrinsic - 0.01:
        return None

    # Brenner-Subrahmanyam initial guess
    sigma = math.sqrt(2 * math.pi / max(T, 1e-6)) * (market_price / S)
    sigma = max(0.01, min(sigma, 3.0))

    for _ in range(max_iter):
        price = _bs_price_scalar(S, K, T, sigma, r, option_type)
        vega = _bs_vega_scalar(S, K, T, sigma, r)
        if vega < 1e-10:
            break
        sigma -= (price - market_price) / vega
        sigma = max(0.001, min(sigma, 5.0))
        if abs(price - market_price) < tol:
            return sigma

    return None


def quality_mid(bid, ask):
    """
    Compute mid price with liquidity filter.
    Returns 0.0 if spread exceeds MAX_SPREAD_PCT (illiquid — skip this contract).
    """
    if bid <= 0 or ask <= 0:
        return 0.0
    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid
    if spread_pct > MAX_SPREAD_PCT:
        return 0.0
    return mid


# ── individual greeks ────────────────────────────────────────────────────────

def gamma(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """
    Gamma (same for calls and puts).
    Returns gamma per 1 share.
    """
    d1 = _d1(S, K, T, sigma, r, q)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def delta_call(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """Call delta = N(d1)."""
    return norm.cdf(_d1(S, K, T, sigma, r, q))


def delta_put(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """Put delta = N(d1) - 1."""
    return norm.cdf(_d1(S, K, T, sigma, r, q)) - 1.0


def charm(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """
    Charm (delta decay).
    Returns charm for a CALL.  Put charm = call charm + q·e^{-qT}
    (for q=0 they are identical in magnitude, opposite for puts via parity).
    """
    d1_val = _d1(S, K, T, sigma, r, q)
    d2_val = _d2(S, K, T, sigma, r, q)
    sqrtT  = np.sqrt(T)
    npdf   = norm.pdf(d1_val)

    # charm = -e^{-qT} * npdf * [2(r-q)T - d2*sigma*sqrtT] / (2*T*sigma*sqrtT)
    numerator   = 2.0 * (r - q) * T - d2_val * sigma * sqrtT
    denominator = 2.0 * T * sigma * sqrtT
    return -np.exp(-q * T) * npdf * numerator / denominator


def vanna(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """
    Vanna  =  ∂delta/∂sigma  =  -(N'(d1) · d2) / sigma
    Same for calls and puts.
    """
    d1_val = _d1(S, K, T, sigma, r, q)
    d2_val = _d2(S, K, T, sigma, r, q)
    return -norm.pdf(d1_val) * d2_val / sigma


def zomma(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """
    Zomma  =  ∂gamma/∂sigma  =  N'(d1) · (d1·d2 - 1) / (S · σ² · √T)
    Same for calls and puts.  Shows how gamma changes when IV moves.
    """
    d1_val = _d1(S, K, T, sigma, r, q)
    d2_val = _d2(S, K, T, sigma, r, q)
    sqrtT  = np.sqrt(T)
    return norm.pdf(d1_val) * (d1_val * d2_val - 1.0) / (S * sigma**2 * sqrtT)


# ── exposure aggregation ─────────────────────────────────────────────────────

def compute_exposure(chain_df, spot, greek_mode="oi"):
    """
    Given a DataFrame *chain_df* with columns:
        strike, call_oi, put_oi, call_volume, put_volume,
        call_iv, put_iv, dte_years
    and the current *spot* price, compute per-strike Gamma / Charm / Vanna
    exposure and return a new DataFrame with columns:
        strike, gamma_exp, charm_exp, vanna_exp

    greek_mode:
        "oi"       → weight by open interest only
        "volume"   → weight by today's traded volume only
        "combined" → OI + volume
    """
    import pandas as pd

    df = chain_df.copy()
    T = df["dte_years"].values
    S = spot

    # ── choose weighting ────────────────────────────────────────────────
    if greek_mode == "oi":
        call_w = df["call_oi"].values.astype(float)
        put_w  = df["put_oi"].values.astype(float)
    elif greek_mode == "volume":
        call_w = df["call_volume"].values.astype(float)
        put_w  = df["put_volume"].values.astype(float)
    else:  # combined
        call_w = df["call_oi"].values.astype(float) + df["call_volume"].values.astype(float)
        put_w  = df["put_oi"].values.astype(float)  + df["put_volume"].values.astype(float)

    K       = df["strike"].values.astype(float)
    call_iv = df["call_iv"].values.astype(float)
    put_iv  = df["put_iv"].values.astype(float)

    # Replace zero / NaN IVs with a small default to avoid div-by-zero
    call_iv = np.where((call_iv <= 0) | np.isnan(call_iv), 0.001, call_iv)
    put_iv  = np.where((put_iv  <= 0) | np.isnan(put_iv),  0.001, put_iv)
    T       = np.where(T <= 0, 1e-6, T)   # avoid sqrt(0)

    # ── GEX factor (SpotGamma/UW convention) ─────────────────────────
    # GEX = Γ × OI × 100 × S² × 0.01
    multiplier = 100
    gex_factor = S * S * 0.01 * multiplier

    # ── Gamma exposure ──────────────────────────────────────────────────
    # Convention:  call GEX positive (stabilizing), put GEX negative (destabilizing)
    g_call = gamma(S, K, T, call_iv) * call_w * gex_factor
    g_put  = gamma(S, K, T, put_iv)  * put_w  * gex_factor * (-1)
    gamma_exp = g_call + g_put

    # ── Charm exposure (dealer short = negate both sides) ────────────
    # Charm measures delta decay — dealer is short calls AND puts,
    # so both contribute negatively to dealer's charm exposure.
    c_call = charm(S, K, T, call_iv) * call_w * multiplier * (-1)
    c_put  = charm(S, K, T, put_iv)  * put_w  * multiplier * (-1)
    charm_exp = c_call + c_put

    # ── Vanna exposure (dealer short = negate both sides) ────────────
    # Vanna measures delta sensitivity to IV — dealer is short both,
    # so both sides negate.
    v_call = vanna(S, K, T, call_iv) * call_w * multiplier * (-1)
    v_put  = vanna(S, K, T, put_iv)  * put_w  * multiplier * (-1)
    vanna_exp = v_call + v_put

    # ── Zomma exposure (same +call/-put as gamma) ───────────────────
    # Zomma = dGamma/dSigma — same stabilizing/destabilizing asymmetry as GEX
    z_call = zomma(S, K, T, call_iv) * call_w * gex_factor
    z_put  = zomma(S, K, T, put_iv)  * put_w  * gex_factor * (-1)
    zomma_exp = z_call + z_put

    # ── DEX — Delta Exposure (dealer short = negate both) ────────────
    # Calls: dealer short calls → short delta → negate
    # Puts:  dealer short puts → long delta (put delta is negative) → negate
    d_call = delta_call(S, K, T, call_iv) * call_w * multiplier * (-1)
    d_put  = delta_put(S, K, T, put_iv)   * put_w  * multiplier * (-1)
    dex_exp = d_call + d_put

    return pd.DataFrame({
        "strike":    K,
        "gamma_exp": gamma_exp,
        "charm_exp": charm_exp,
        "vanna_exp": vanna_exp,
        "zomma_exp": zomma_exp,
        "dex_exp":   dex_exp,
    })


# ── GEX flip point ───────────────────────────────────────────────────────────

def find_gex_flip(exp_df, spot):
    """
    Find the GEX flip point — where net GEX crosses zero nearest to spot.

    Uses linear interpolation between adjacent strikes where sign changes.
    Returns the interpolated strike price, or None if no crossing found.

    The flip point is critical:
    - Above flip → positive gamma territory (mean-reverting, dealer stabilizing)
    - Below flip → negative gamma territory (trending, dealer amplifying)
    """
    if exp_df is None or exp_df.empty:
        return None

    df = exp_df.sort_values("strike").reset_index(drop=True)
    gex = df["gamma_exp"].values
    strikes = df["strike"].values

    best_flip = None
    best_dist = float("inf")

    for i in range(len(gex) - 1):
        if gex[i] * gex[i + 1] < 0:  # sign change
            # Linear interpolation
            frac = abs(gex[i]) / (abs(gex[i]) + abs(gex[i + 1]))
            flip = strikes[i] + frac * (strikes[i + 1] - strikes[i])
            dist = abs(flip - spot)
            if dist < best_dist:
                best_dist = dist
                best_flip = float(flip)

    return best_flip


# ── Gamma regime classification ──────────────────────────────────────────────

def classify_regime(exp_df, spot):
    """
    Classify the gamma environment for trading bias.

    Returns dict with:
        gamma:      "POSITIVE" or "NEGATIVE"
        bias:       "FADE" (mean-revert) or "TREND" (momentum)
        conviction: "LOW" / "MEDIUM" / "HIGH"
        above_flip: True if spot > gex_flip (in positive territory)
        gex_flip:   the flip point strike
        total_gex:  sum of all net GEX
        call_wall:  strike with highest positive (call) GEX
        put_wall:   strike with most negative (put) GEX

    Trading interpretation:
        POSITIVE + FADE → sell wings, fade moves, expect mean reversion
        NEGATIVE + TREND → buy breakouts, trend follows, expect expansion
    """
    if exp_df is None or exp_df.empty:
        return {}

    df = exp_df.sort_values("strike").reset_index(drop=True)
    total_gex = float(df["gamma_exp"].sum())
    gex_flip = find_gex_flip(exp_df, spot)

    regime = {}

    # Gamma sign
    if total_gex > 0:
        regime["gamma"] = "POSITIVE"
        regime["bias"] = "FADE"
    else:
        regime["gamma"] = "NEGATIVE"
        regime["bias"] = "TREND"

    # Position relative to flip
    regime["above_flip"] = bool(spot > gex_flip) if gex_flip is not None else None
    regime["gex_flip"] = gex_flip
    regime["total_gex"] = total_gex

    # Conviction based on magnitude
    mag = abs(total_gex)
    if mag < 500e6:
        regime["conviction"] = "LOW"
    elif mag < 2e9:
        regime["conviction"] = "MEDIUM"
    else:
        regime["conviction"] = "HIGH"

    # Call wall (highest positive GEX strike)
    pos_gex = df[df["gamma_exp"] > 0]
    if not pos_gex.empty:
        idx = pos_gex["gamma_exp"].idxmax()
        regime["call_wall"] = float(df.loc[idx, "strike"])
    else:
        regime["call_wall"] = None

    # Put wall (most negative GEX strike)
    neg_gex = df[df["gamma_exp"] < 0]
    if not neg_gex.empty:
        idx = neg_gex["gamma_exp"].idxmin()
        regime["put_wall"] = float(df.loc[idx, "strike"])
    else:
        regime["put_wall"] = None

    return regime


# ── Vanna / VIX signal ───────────────────────────────────────────────────────

def compute_vanna_vix_signal(exp_df, vix_current, vix_prev_close):
    """
    Compute Vanna/VIX alignment signal.

    Logic:
    - Net vanna > 0 AND VIX falling → BULLISH
      (falling IV + positive vanna = dealers buy stock to hedge)
    - Net vanna < 0 AND VIX rising  → BEARISH
      (rising IV + negative vanna = dealers sell stock to hedge)
    - Otherwise → MIXED

    Returns dict with signal, total_vanna, vix info.
    """
    if exp_df is None or exp_df.empty:
        return {"signal": "N/A", "total_vanna": 0, "vix_current": 0,
                "vix_change": 0, "vix_pct": 0}

    total_vanna = float(exp_df["vanna_exp"].sum())
    vix_change = vix_current - vix_prev_close
    vix_pct = (vix_change / vix_prev_close * 100) if vix_prev_close > 0 else 0

    if total_vanna > 0 and vix_change < 0:
        signal = "BULLISH"
    elif total_vanna < 0 and vix_change > 0:
        signal = "BEARISH"
    else:
        signal = "MIXED"

    return {
        "signal": signal,
        "total_vanna": total_vanna,
        "vix_current": vix_current,
        "vix_prev_close": vix_prev_close,
        "vix_change": round(vix_change, 2),
        "vix_pct": round(vix_pct, 2),
    }


# ── Charm Decay Clock ────────────────────────────────────────────────────────

def compute_charm_clock(exp_df, spot):
    """
    Compute the Charm Decay Clock — projected charm pressure until close.

    Method:
    1. Weight each strike's charm by proximity to spot (exponential decay)
       → near-ATM charm matters more than far OTM
    2. Multiply weighted charm by hours remaining until market close
       → more time left = more total charm decay to come

    Returns dict with:
        direction:      "SUPPORTIVE" (charm pushes price up) or "PRESSURING" (down)
        weighted_charm: proximity-weighted net charm at current moment
        hours_to_close: trading hours remaining
        charm_pressure: weighted_charm × hours_to_close (total projected decay)
    """
    from datetime import datetime
    from config import ET

    if exp_df is None or exp_df.empty:
        return {"direction": "N/A", "weighted_charm": 0, "hours_to_close": 0,
                "charm_pressure": 0}

    df = exp_df.copy()

    # Proximity weighting — exponential decay from spot
    # e^(-|K - S| / (S × 0.01)) → 1.0 at ATM, ~0.37 at 1% away, ~0.05 at 3%
    df["distance"] = (df["strike"] - spot).abs()
    df["prox_weight"] = np.exp(-df["distance"] / (spot * 0.01))
    weighted_charm = float((df["charm_exp"] * df["prox_weight"]).sum())

    # Hours remaining until 16:00 ET
    now_et = datetime.now(tz=ET)
    close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    hours_to_close = max((close_et - now_et).total_seconds() / 3600, 0)

    # Charm pressure = weighted charm × hours remaining
    charm_pressure = weighted_charm * hours_to_close

    # Direction: negative charm pressure = supportive (price drifts up)
    # positive charm pressure = pressuring (price drifts down)
    if charm_pressure < 0:
        direction = "SUPPORTIVE"
    elif charm_pressure > 0:
        direction = "PRESSURING"
    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "weighted_charm": round(weighted_charm, 0),
        "hours_to_close": round(hours_to_close, 2),
        "charm_pressure": round(charm_pressure, 0),
    }


# ── market metrics from option chain ─────────────────────────────────────────

def _bs_call_price(S, K, T, sigma, r=RISK_FREE_RATE):
    """Black-Scholes call price."""
    d1 = _d1(S, K, T, sigma, r)
    d2 = _d2(S, K, T, sigma, r)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S, K, T, sigma, r=RISK_FREE_RATE):
    """Black-Scholes put price."""
    d1 = _d1(S, K, T, sigma, r)
    d2 = _d2(S, K, T, sigma, r)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_live_metrics(chain_df, spot):
    """
    Compute metrics that update live every refresh:
    - Current ATM IV (Newton-solved from chain)
    - ATM straddle price (from actual bid/ask mid, fallback to BS)
    - Put/Call OI ratio
    """
    df = chain_df.copy()
    if df.empty:
        return {}

    K = df["strike"].values.astype(float)
    T = df["dte_years"].values[0] if len(df) > 0 else 1/252

    atm_idx = int(np.argmin(np.abs(K - spot)))
    atm_strike = K[atm_idx]
    atm_call_iv = float(df.iloc[atm_idx]["call_iv"])
    atm_put_iv  = float(df.iloc[atm_idx]["put_iv"])
    atm_iv = (atm_call_iv + atm_put_iv) / 2.0
    if atm_iv <= 0:
        atm_iv = 0.20

    T = max(T, 1e-6)

    # ATM straddle: prefer actual market mid (more accurate than BS)
    call_mid = 0.0
    put_mid  = 0.0
    if "call_bid" in df.columns and "call_ask" in df.columns:
        cb = float(df.iloc[atm_idx].get("call_bid", 0))
        ca = float(df.iloc[atm_idx].get("call_ask", 0))
        if cb > 0 and ca > 0:
            call_mid = (cb + ca) / 2.0
    if "put_bid" in df.columns and "put_ask" in df.columns:
        pb = float(df.iloc[atm_idx].get("put_bid", 0))
        pa = float(df.iloc[atm_idx].get("put_ask", 0))
        if pb > 0 and pa > 0:
            put_mid = (pb + pa) / 2.0

    if call_mid > 0 and put_mid > 0:
        # Use actual market straddle
        straddle = call_mid + put_mid
        call_price = call_mid
        put_price  = put_mid
    else:
        # Fallback to BS theoretical price
        call_price = float(_bs_call_price(spot, atm_strike, T, atm_call_iv))
        put_price  = float(_bs_put_price(spot, atm_strike, T, atm_put_iv))
        straddle   = call_price + put_price

    # Put/Call OI ratio
    total_call_oi = df["call_oi"].sum()
    total_put_oi  = df["put_oi"].sum()
    pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

    # Straddle-based Expected Move
    # The ATM straddle IS the market's priced-in expected move
    straddle_em = straddle
    straddle_em_pct = (straddle / spot * 100) if spot > 0 else 0
    straddle_em_high = spot + straddle
    straddle_em_low  = spot - straddle

    return {
        "atm_strike":       atm_strike,
        "atm_iv":           atm_iv,
        "straddle":         straddle,
        "call_price":       call_price,
        "put_price":        put_price,
        "pc_ratio":         pc_ratio,
        "total_call_oi":    total_call_oi,
        "total_put_oi":     total_put_oi,
        "straddle_em":      straddle_em,
        "straddle_em_pct":  round(straddle_em_pct, 2),
        "straddle_em_high": round(straddle_em_high, 2),
        "straddle_em_low":  round(straddle_em_low, 2),
    }



# ── charm projection across time ─────────────────────────────────────────────

def project_charm_forward(chain_df, spot, greek_mode="oi",
                          minutes_ahead=390, minutes_behind=0,
                          step_minutes=5):
    """
    Project charm exposure across time by adjusting DTE.

    Negative offsets = past (DTE was higher → charm was weaker).
    Positive offsets = future (DTE shrinks → charm intensifies).

    Parameters
    ----------
    chain_df       : current option chain DataFrame
    spot           : current underlying price
    greek_mode     : "oi" | "volume" | "combined"
    minutes_ahead  : how far forward to project
    minutes_behind : how far backward to reconstruct
    step_minutes   : resolution (default 5 min)

    Returns
    -------
    dict with:
        "strikes"      : list of strikes
        "offsets_min"   : list of minute offsets [-behind, ..., 0, ..., +ahead]
        "charm_grid"   : 2D numpy array (n_strikes × n_time_steps)
        "now_index"    : index of the offset=0 column (current time)
    """
    import pandas as pd

    df = chain_df.copy()
    T_base = df["dte_years"].values.astype(float)
    S = spot

    # Weighting
    if greek_mode == "oi":
        call_w = df["call_oi"].values.astype(float)
        put_w  = df["put_oi"].values.astype(float)
    elif greek_mode == "volume":
        call_w = df["call_volume"].values.astype(float)
        put_w  = df["put_volume"].values.astype(float)
    else:
        call_w = df["call_oi"].values.astype(float) + df["call_volume"].values.astype(float)
        put_w  = df["put_oi"].values.astype(float)  + df["put_volume"].values.astype(float)

    K       = df["strike"].values.astype(float)
    call_iv = df["call_iv"].values.astype(float)
    put_iv  = df["put_iv"].values.astype(float)

    call_iv = np.where((call_iv <= 0) | np.isnan(call_iv), 0.001, call_iv)
    put_iv  = np.where((put_iv  <= 0) | np.isnan(put_iv),  0.001, put_iv)

    # Offsets: negative = past, 0 = now, positive = future
    offsets = list(range(-minutes_behind, minutes_ahead + 1, step_minutes))
    if 0 not in offsets:
        offsets.append(0)
        offsets.sort()

    now_index = offsets.index(0)
    charm_grid = []

    for offset in offsets:
        # offset < 0 → past → adds time to DTE (charm was weaker)
        # offset > 0 → future → subtracts time (charm intensifies)
        # Business-day convention: 1 minute = 1/(252 × 405) years
        T = T_base - (offset / (252.0 * TRADING_MINUTES_PER_DAY))
        T = np.where(T <= 0, 1e-6, T)

        c_call = charm(S, K, T, call_iv) * call_w * 100 * (-1)
        c_put  = charm(S, K, T, put_iv)  * put_w  * 100 * (-1)
        charm_grid.append(c_call + c_put)

    charm_array = np.array(charm_grid).T  # (n_strikes, n_times)

    return {
        "strikes":     K.tolist(),
        "offsets_min":  offsets,
        "charm_grid":  charm_array,
        "now_index":   now_index,
    }
