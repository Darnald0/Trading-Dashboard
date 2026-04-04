"""
greek_calculator.py  –  Black-Scholes greeks used for exposure calculations.

All functions are *pure* (no side effects, no IB calls).  They take
strike / spot / vol / time arrays and return the corresponding greek
values.  The dashboard multiplies these by OI or volume afterwards.

Greek definitions (per single contract, 100 shares multiplier applied later):
    Gamma   =  N'(d1) / (S · σ · √T)
    Charm   = –N'(d1) · [2(r–q)T – d2·σ·√T] / (2T · σ · √T)
              (rate of change of delta w.r.t. time, a.k.a. "delta decay")
    Vanna   = –N'(d1) · d2 / σ
              (sensitivity of delta to implied-vol changes)
"""

import numpy as np
from scipy.stats import norm

from config import RISK_FREE_RATE

# ── helpers ──────────────────────────────────────────────────────────────────

def _d1(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """Standard Black-Scholes d1."""
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """Standard Black-Scholes d2."""
    return _d1(S, K, T, sigma, r, q) - sigma * np.sqrt(T)


# ── individual greeks ────────────────────────────────────────────────────────

def gamma(S, K, T, sigma, r=RISK_FREE_RATE, q=0.0):
    """
    Gamma (same for calls and puts).
    Returns gamma per 1 share.
    """
    d1 = _d1(S, K, T, sigma, r, q)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


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

    # ── Gamma exposure ──────────────────────────────────────────────────
    # Convention:  call GEX positive, put GEX negative (dealer perspective)
    g_call = gamma(S, K, T, call_iv) * call_w * 100 * S
    g_put  = gamma(S, K, T, put_iv)  * put_w  * 100 * S * (-1)
    gamma_exp = g_call + g_put

    # ── Charm exposure ──────────────────────────────────────────────────
    c_call = charm(S, K, T, call_iv) * call_w * 100
    c_put  = charm(S, K, T, put_iv)  * put_w  * 100 * (-1)
    charm_exp = c_call + c_put

    # ── Vanna exposure ──────────────────────────────────────────────────
    v_call = vanna(S, K, T, call_iv) * call_w * 100
    v_put  = vanna(S, K, T, put_iv)  * put_w  * 100 * (-1)
    vanna_exp = v_call + v_put

    # ── Zomma exposure ─────────────────────────────────────────────────
    z_call = zomma(S, K, T, call_iv) * call_w * 100 * S
    z_put  = zomma(S, K, T, put_iv)  * put_w  * 100 * S * (-1)
    zomma_exp = z_call + z_put

    return pd.DataFrame({
        "strike":    K,
        "gamma_exp": gamma_exp,
        "charm_exp": charm_exp,
        "vanna_exp": vanna_exp,
        "zomma_exp": zomma_exp,
    })


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
        T = T_base - (offset / (365.0 * 24 * 60))
        T = np.where(T <= 0, 1e-6, T)

        c_call = charm(S, K, T, call_iv) * call_w * 100
        c_put  = charm(S, K, T, put_iv)  * put_w  * 100 * (-1)
        charm_grid.append(c_call + c_put)

    charm_array = np.array(charm_grid).T  # (n_strikes, n_times)

    return {
        "strikes":     K.tolist(),
        "offsets_min":  offsets,
        "charm_grid":  charm_array,
        "now_index":   now_index,
    }
