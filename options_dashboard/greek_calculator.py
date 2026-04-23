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


def vomma(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """
    Vomma  =  ∂vega/∂sigma  =  S·√T·N'(d1) · (d1·d2 / sigma)
    Same for calls and puts.  Sensitivity of vega to IV changes —
    tells you how much vega accelerates when volatility moves.
    """
    d1_val = _d1(S, K, T, sigma, r, q)
    d2_val = _d2(S, K, T, sigma, r, q)
    sqrtT  = np.sqrt(T)
    return S * sqrtT * norm.pdf(d1_val) * (d1_val * d2_val / sigma)


def speed(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """
    Speed  =  ∂gamma/∂S  =  -Γ / S · (d1 / (σ·√T) + 1)
    Same for calls and puts.  Third derivative of price w.r.t. spot —
    tells you how fast gamma itself is changing as spot moves.
    """
    d1_val = _d1(S, K, T, sigma, r, q)
    sqrtT  = np.sqrt(T)
    gam = norm.pdf(d1_val) / (S * sigma * sqrtT)
    return -gam / S * (d1_val / (sigma * sqrtT) + 1.0)


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

    # ── Vomma exposure (dealer short = negate both) ──────────────────
    # Vomma = dVega/dSigma — sensitivity of vega to IV.
    # Dealer is short vega on both sides, so both negate.
    vom_call = vomma(S, K, T, call_iv) * call_w * multiplier * (-1)
    vom_put  = vomma(S, K, T, put_iv)  * put_w  * multiplier * (-1)
    vomma_exp = vom_call + vom_put

    # ── Speed exposure (same +call/-put asymmetry as GEX) ────────────
    # Speed = dGamma/dS — same delta-hedge stabilizing/destabilizing
    # asymmetry as gamma itself.
    sp_call = speed(S, K, T, call_iv) * call_w * gex_factor
    sp_put  = speed(S, K, T, put_iv)  * put_w  * gex_factor * (-1)
    speed_exp = sp_call + sp_put

    return pd.DataFrame({
        "strike":    K,
        "gamma_exp": gamma_exp,
        "charm_exp": charm_exp,
        "vanna_exp": vanna_exp,
        "zomma_exp": zomma_exp,
        "dex_exp":   dex_exp,
        "vomma_exp": vomma_exp,
        "speed_exp": speed_exp,
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


# ── Skew (25-delta put IV − 25-delta call IV) ────────────────────────────────

def compute_skew(chain_df, spot):
    """
    25-delta put IV minus 25-delta call IV.

    Positive skew = market pays up for downside protection (bearish).
    Rapidly rising skew often precedes drawdowns.

    Returns dict with:
        put_iv_25d:  IV at the 25-delta put strike
        call_iv_25d: IV at the 25-delta call strike
        skew:        put_iv_25d - call_iv_25d (in vol points)
        skew_pct:    skew as % of ATM IV (normalized)
    """
    if chain_df is None or chain_df.empty:
        return {"skew": 0, "skew_pct": 0, "put_iv_25d": 0, "call_iv_25d": 0}

    df = chain_df.copy()
    T = df["dte_years"].values[0] if len(df) > 0 else 1/252
    T = max(T, 1e-6)
    K = df["strike"].values.astype(float)

    # ATM IV for normalization
    atm_idx = int(np.argmin(np.abs(K - spot)))
    atm_iv = (float(df.iloc[atm_idx]["call_iv"]) + float(df.iloc[atm_idx]["put_iv"])) / 2.0
    if atm_iv <= 0:
        atm_iv = 0.20

    # Compute deltas for every strike using its own IV
    call_ivs = df["call_iv"].values.astype(float)
    put_ivs  = df["put_iv"].values.astype(float)
    call_ivs = np.where((call_ivs <= 0) | np.isnan(call_ivs), atm_iv, call_ivs)
    put_ivs  = np.where((put_ivs  <= 0) | np.isnan(put_ivs),  atm_iv, put_ivs)

    c_deltas = norm.cdf(_d1(spot, K, T, call_ivs))
    p_deltas = norm.cdf(_d1(spot, K, T, put_ivs)) - 1.0

    # Find strikes closest to ±0.25 delta
    call_idx = int(np.argmin(np.abs(c_deltas - 0.25)))
    put_idx  = int(np.argmin(np.abs(p_deltas + 0.25)))

    put_iv_25d  = float(put_ivs[put_idx])
    call_iv_25d = float(call_ivs[call_idx])
    skew = put_iv_25d - call_iv_25d
    skew_pct = (skew / atm_iv * 100) if atm_iv > 0 else 0

    return {
        "put_iv_25d":  round(put_iv_25d, 4),
        "call_iv_25d": round(call_iv_25d, 4),
        "put_strike":  float(K[put_idx]),
        "call_strike": float(K[call_idx]),
        "skew":        round(skew, 4),
        "skew_pct":    round(skew_pct, 2),
    }


# ── Pinning strength ────────────────────────────────────────────────────────

def compute_pinning_strength(exp_df, chain_df, spot, dte_years=None):
    """
    Composite score of pin-to-strike magnetism near spot.

    Method:
    1. For each near-ATM strike, compute a score combining:
       - Absolute gamma exposure (higher = stronger pull)
       - Volume × OI density (higher = more interest)
       - Distance from spot (closer = stronger pull, exponential decay)
       - Time-to-expiry penalty (pinning strongest as 0DTE approaches)
    2. Identify the dominant strike and its strength

    Returns dict:
        pin_strike:    the strike with highest pinning score
        pin_strength:  score (0-100, where 100 = maximum pin pressure)
        pin_distance:  |pin_strike - spot| in dollars
        confidence:    LOW / MEDIUM / HIGH based on absolute score
    """
    if exp_df is None or exp_df.empty or chain_df is None or chain_df.empty:
        return {"pin_strike": None, "pin_strength": 0,
                "pin_distance": 0, "confidence": "N/A"}

    # Merge exposure with raw chain to get OI+volume
    exp = exp_df.copy()
    exp["distance"] = (exp["strike"] - spot).abs()

    # Proximity weight — exponential decay, stronger than charm clock
    # (spot*0.005 = 0.5% — sharper because pinning is very local)
    exp["prox_weight"] = np.exp(-exp["distance"] / (spot * 0.005))

    # Join volume + OI from original chain
    chain_slim = chain_df[["strike", "call_oi", "put_oi",
                           "call_volume", "put_volume"]].copy()
    exp = exp.merge(chain_slim, on="strike", how="left").fillna(0)

    # Raw score per strike:
    #   |gamma_exp| × prox_weight × (log(1 + volume) * log(1 + total_oi))
    total_oi = exp["call_oi"] + exp["put_oi"]
    total_vol = exp["call_volume"] + exp["put_volume"]
    activity = np.log1p(total_vol) * np.log1p(total_oi)

    exp["pin_score"] = exp["gamma_exp"].abs() * exp["prox_weight"] * activity

    if exp["pin_score"].sum() == 0:
        return {"pin_strike": None, "pin_strength": 0,
                "pin_distance": 0, "confidence": "N/A"}

    top_idx = exp["pin_score"].idxmax()
    pin_strike = float(exp.loc[top_idx, "strike"])
    top_score = float(exp.loc[top_idx, "pin_score"])

    # Normalize score: compare top strike to average of other near-ATM strikes
    # (within 1% of spot)
    near = exp[exp["distance"] <= spot * 0.01]
    if len(near) > 1:
        avg_score = near["pin_score"].mean()
        ratio = top_score / avg_score if avg_score > 0 else 1.0
    else:
        ratio = 1.0

    # Time factor: pinning strongest in the last hour of 0DTE
    if dte_years is not None and dte_years < 1/252:
        # Intraday 0DTE — use DTE directly as a time factor
        # 1.0 at market open (full day left), rising to ~3.0 in final hour
        time_factor = 1.0 + 2.0 * (1.0 - dte_years * 252)
    else:
        # Non-0DTE pinning is much weaker
        time_factor = 0.5

    # Composite score 0-100
    strength = min(ratio * time_factor * 15, 100)

    if strength >= 70:
        confidence = "HIGH"
    elif strength >= 40:
        confidence = "MEDIUM"
    elif strength >= 20:
        confidence = "LOW"
    else:
        confidence = "NONE"

    return {
        "pin_strike":   pin_strike,
        "pin_strength": round(strength, 1),
        "pin_distance": round(abs(pin_strike - spot), 2),
        "confidence":   confidence,
    }


# ── Composite trade signal ──────────────────────────────────────────────────

def compute_trade_signal(spot, regime, vanna_vix, charm_clock, skew, term,
                          iv_rank, pinning, live_metrics, exp_df):
    """
    Combine all signals into a coherent trade setup.

    Returns dict with:
        direction:       "LONG" | "SHORT" | "NEUTRAL"
        conviction:      "LOW" | "MEDIUM" | "HIGH"
        setup:           short description of the setup
        entry:           suggested entry price/zone
        stop_loss:       invalidation level
        take_profit_1:   primary target
        take_profit_2:   secondary target
        reasoning:       list of bullet points explaining each signal
        risk_reward:     numeric R:R ratio
        caveats:         list of warnings that could change the trade
    """
    if not regime or exp_df is None or exp_df.empty:
        return _empty_signal("Insufficient data")

    # ── Extract key levels ──────────────────────────────────────────
    gex_flip  = regime.get("gex_flip")
    call_wall = regime.get("call_wall")
    put_wall  = regime.get("put_wall")
    gamma_sign = regime.get("gamma", "NEUTRAL")
    above_flip = regime.get("above_flip")
    conviction_regime = regime.get("conviction", "LOW")

    straddle_em_high = live_metrics.get("straddle_em_high", spot) if live_metrics else spot
    straddle_em_low  = live_metrics.get("straddle_em_low", spot)  if live_metrics else spot

    # ── Directional vote (weighted) ─────────────────────────────────
    # Each signal votes +1 bull / -1 bear / 0 neutral, weighted by reliability
    votes = []
    reasoning = []

    # 1. Gamma regime (heaviest weight — core framework)
    if gamma_sign == "POSITIVE":
        # Above flip → mean-reverting, drift up to call wall (slight bull)
        # Below flip → mean-reverting down, flip acts as ceiling (bear)
        if above_flip is True:
            votes.append(("regime", +1, 3.0))
            reasoning.append("Positive γ above flip — mean-reverting drift (mild bullish)")
        elif above_flip is False:
            votes.append(("regime", -1, 3.0))
            reasoning.append("Positive γ below flip — resistance overhead (bearish)")
        else:
            votes.append(("regime", 0, 1.5))
            reasoning.append("Positive γ regime — low volatility, range-bound")
    elif gamma_sign == "NEGATIVE":
        # Negative γ = trend-following
        if above_flip is True:
            votes.append(("regime", +1, 2.5))
            reasoning.append("Negative γ above flip — breakout continuation likely (bullish)")
        elif above_flip is False:
            votes.append(("regime", -1, 3.5))
            reasoning.append("Negative γ below flip — crash-like acceleration risk (bearish)")
        else:
            votes.append(("regime", 0, 1.0))
            reasoning.append("Negative γ — volatile, no clear flip reference")
    else:
        votes.append(("regime", 0, 1.0))
        reasoning.append("Neutral γ regime — no directional edge from positioning")

    # 2. Vanna/VIX (strong confirmation signal)
    if vanna_vix:
        sig = vanna_vix.get("signal", "MIXED")
        if sig == "BULLISH":
            votes.append(("vanna_vix", +1, 2.0))
            reasoning.append("Vanna/VIX BULLISH — dealer flows buying stock")
        elif sig == "BEARISH":
            votes.append(("vanna_vix", -1, 2.0))
            reasoning.append("Vanna/VIX BEARISH — dealer flows selling stock")
        else:
            reasoning.append("Vanna/VIX MIXED — no vol-flow edge")

    # 3. Charm Clock (time-of-day flow)
    if charm_clock:
        direction = charm_clock.get("direction", "NEUTRAL")
        hours = charm_clock.get("hours_to_close", 0)
        # Charm matters more as more time remains
        weight = min(hours / 6.5, 1.0) * 1.5
        if direction == "SUPPORTIVE":
            votes.append(("charm", +1, weight))
            reasoning.append(f"Charm SUPPORTIVE — decay pressure bullish ({hours:.1f}h left)")
        elif direction == "PRESSURING":
            votes.append(("charm", -1, weight))
            reasoning.append(f"Charm PRESSURING — decay pressure bearish ({hours:.1f}h left)")

    # 4. Net DEX (dealer hedge flow)
    if "dex_exp" in exp_df.columns:
        net_dex = float(exp_df["dex_exp"].sum())
        # Scale by spot for cross-ticker comparison
        dex_per_spot = net_dex / (spot if spot > 0 else 1)
        if dex_per_spot > 50:
            votes.append(("dex", +1, 1.0))
            reasoning.append(f"Net DEX positive (+${net_dex/1e6:.0f}M) — dealer long hedge bias")
        elif dex_per_spot < -50:
            votes.append(("dex", -1, 1.0))
            reasoning.append(f"Net DEX negative (-${abs(net_dex)/1e6:.0f}M) — dealer short hedge bias")

    # 5. Skew (flow sentiment)
    if skew:
        skew_val = skew.get("skew", 0) * 100   # in vol points
        if skew_val > 4:
            # Rising put demand → bearish sentiment (but can be contrarian)
            votes.append(("skew", -0.5, 0.8))
            reasoning.append(f"Skew steep (+{skew_val:.1f} pts) — elevated downside hedging")
        elif skew_val < -1:
            votes.append(("skew", +0.5, 0.8))
            reasoning.append(f"Skew inverted ({skew_val:.1f} pts) — unusual call demand (bullish)")

    # 6. Term structure (regime stability)
    if term:
        state = term.get("state", "N/A")
        if state == "BACKWARDATION":
            votes.append(("term", -0.5, 0.8))
            reasoning.append("Term BACKWARDATION — near-term stress priced in (bearish)")
        elif state == "CONTANGO":
            reasoning.append("Term CONTANGO — normal volatility term structure")

    # ── Compute direction and conviction ────────────────────────────
    score = sum(vote * weight for _, vote, weight in votes)
    max_score = sum(weight for _, _, weight in votes)
    normalized = score / max_score if max_score > 0 else 0   # -1 to +1

    if normalized >= 0.40:
        direction = "LONG"
    elif normalized <= -0.40:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    abs_norm = abs(normalized)
    if abs_norm >= 0.65:
        conviction = "HIGH"
    elif abs_norm >= 0.40:
        conviction = "MEDIUM"
    elif abs_norm >= 0.20:
        conviction = "LOW"
    else:
        conviction = "NONE"

    # ── Build the setup ─────────────────────────────────────────────
    setup = ""
    entry = None
    stop_loss = None
    tp1 = None
    tp2 = None
    caveats = []

    if direction == "LONG":
        # Entry priority: pullback to support (put wall > flip > current)
        if gamma_sign == "POSITIVE" and above_flip:
            # Mean-reverting: buy dips to put wall / flip
            if put_wall and put_wall < spot:
                entry = put_wall
                setup = f"Fade dip to put wall (${put_wall:,.1f}) in positive γ"
            elif gex_flip and gex_flip < spot:
                entry = gex_flip
                setup = f"Fade dip to GEX flip (${gex_flip:,.1f})"
            else:
                entry = spot
                setup = "Mean-reversion long at current level"
        else:
            # Breakout continuation in negative γ
            if call_wall and call_wall > spot:
                entry = call_wall
                setup = f"Breakout above call wall (${call_wall:,.1f})"
            else:
                entry = spot
                setup = "Momentum long at current level"

        # Stop: below the key support
        if put_wall and put_wall < spot:
            stop_loss = put_wall * 0.997   # just below
        elif gex_flip and gex_flip < spot:
            stop_loss = gex_flip * 0.997
        else:
            stop_loss = min(straddle_em_low, spot * 0.995)

        # Targets: EM high → next wall
        tp1 = straddle_em_high
        if call_wall and call_wall > spot and call_wall > tp1:
            tp2 = call_wall
        else:
            tp2 = spot + (spot - stop_loss) * 3.0

    elif direction == "SHORT":
        # Entry priority: rally to resistance (call wall > flip > current)
        if gamma_sign == "POSITIVE" and not above_flip:
            if call_wall and call_wall > spot:
                entry = call_wall
                setup = f"Fade rally into call wall (${call_wall:,.1f}) below flip"
            elif gex_flip and gex_flip > spot:
                entry = gex_flip
                setup = f"Fade rally into GEX flip (${gex_flip:,.1f})"
            else:
                entry = spot
                setup = "Mean-reversion short at current level"
        else:
            # Breakdown in negative γ
            if put_wall and put_wall < spot:
                entry = put_wall
                setup = f"Breakdown below put wall (${put_wall:,.1f})"
            else:
                entry = spot
                setup = "Momentum short at current level"

        if call_wall and call_wall > spot:
            stop_loss = call_wall * 1.003
        elif gex_flip and gex_flip > spot:
            stop_loss = gex_flip * 1.003
        else:
            stop_loss = max(straddle_em_high, spot * 1.005)

        tp1 = straddle_em_low
        if put_wall and put_wall < spot and put_wall < tp1:
            tp2 = put_wall
        else:
            tp2 = spot - (stop_loss - spot) * 3.0

    else:
        setup = "No clear setup — signals conflicting or weak"

    # ── Risk/reward ─────────────────────────────────────────────────
    if entry and stop_loss and tp1:
        risk = abs(entry - stop_loss)
        reward = abs(tp1 - entry)
        risk_reward = reward / risk if risk > 0 else 0
    else:
        risk_reward = 0

    # ── Caveats ─────────────────────────────────────────────────────
    if iv_rank and iv_rank.get("iv_rank") is not None:
        rank = iv_rank["iv_rank"]
        if rank >= 75:
            caveats.append(f"IV Rank {rank:.0f}/100 — premium is expensive, favor spreads")
        elif rank <= 25:
            caveats.append(f"IV Rank {rank:.0f}/100 — premium is cheap, favor long options")

    if pinning and pinning.get("confidence") in ("HIGH", "MEDIUM"):
        pin_strike = pinning.get("pin_strike")
        if pin_strike:
            caveats.append(
                f"Strong pin @ ${pin_strike:,.0f} — expect magnet effect into close"
            )

    if term and term.get("state") == "BACKWARDATION":
        caveats.append("Backwardation — watch for sudden vol regime shift")

    if gamma_sign == "NEGATIVE":
        caveats.append("Negative γ: moves accelerate, tighten stops")

    if vanna_vix and vanna_vix.get("signal") == "MIXED":
        caveats.append("Vanna/VIX mixed — conflict between direction and IV flow")

    if direction != "NEUTRAL":
        # Flip-level invalidation warning
        if gex_flip:
            if direction == "LONG" and spot > gex_flip:
                caveats.append(f"Thesis breaks if spot closes below GEX flip (${gex_flip:,.1f})")
            elif direction == "SHORT" and spot < gex_flip:
                caveats.append(f"Thesis breaks if spot closes above GEX flip (${gex_flip:,.1f})")

    return {
        "direction":     direction,
        "conviction":    conviction,
        "setup":         setup,
        "entry":         round(entry, 2) if entry else None,
        "stop_loss":     round(stop_loss, 2) if stop_loss else None,
        "take_profit_1": round(tp1, 2) if tp1 else None,
        "take_profit_2": round(tp2, 2) if tp2 else None,
        "risk_reward":   round(risk_reward, 2),
        "reasoning":     reasoning,
        "caveats":       caveats,
        "score":         round(normalized, 2),
    }


def _empty_signal(msg):
    return {
        "direction": "NEUTRAL", "conviction": "NONE", "setup": msg,
        "entry": None, "stop_loss": None, "take_profit_1": None,
        "take_profit_2": None, "risk_reward": 0,
        "reasoning": [], "caveats": [], "score": 0,
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
