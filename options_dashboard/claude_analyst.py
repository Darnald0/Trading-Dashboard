"""
claude_analyst.py  –  Sends dashboard state to Claude API for analysis.

The user presses a button in the dashboard, we package all current
data (regime, greeks, signals, levels, etc.) into a structured prompt,
send it to Claude, and return a trade recommendation.

Requires:
  - `anthropic` Python SDK installed  (pip install anthropic)
  - ANTHROPIC_API_KEY environment variable set
"""

import os
import json
import traceback
from typing import Any


DEFAULT_MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are an expert options flow analyst specializing in dealer
positioning and gamma regime analysis. You help traders interpret real-time
market data to identify high-probability short-term setups.

When given dashboard data, produce a concise, actionable trade idea with:
1. A clear directional thesis (LONG, SHORT, or STAND ASIDE)
2. Specific entry price or zone
3. A hard stop-loss level with reasoning
4. Two take-profit targets with reasoning
5. Key risks that would invalidate the trade
6. What to watch for that would change the thesis

Keep it tight and practical. Avoid hedging language like "could" or "might"
unless genuinely uncertain. If signals conflict, say so plainly and recommend
standing aside rather than forcing a trade. Your reader is an experienced
trader who wants a clear opinion, not a disclaimer-heavy analysis."""


def _format_number(n, decimals=2):
    """Helper for pretty number formatting."""
    if n is None:
        return "N/A"
    if isinstance(n, (int, float)):
        abs_n = abs(n)
        if abs_n >= 1e9:
            return f"{n/1e9:+,.{decimals}f}B"
        elif abs_n >= 1e6:
            return f"{n/1e6:+,.{decimals}f}M"
        elif abs_n >= 1e3:
            return f"{n/1e3:+,.{decimals}f}K"
        return f"{n:+,.{decimals}f}"
    return str(n)


def build_analysis_prompt(data: dict) -> str:
    """Transform the raw dashboard state into a structured markdown prompt."""
    ticker = data.get("ticker", "?")
    spot = data.get("spot", 0)
    expiry = data.get("expiry", "?")
    dte = data.get("dte", 0)
    mode = data.get("mode", "oi")

    regime = data.get("regime", {}) or {}
    vv = data.get("vanna_vix", {}) or {}
    cc = data.get("charm_clock", {}) or {}
    skew = data.get("skew", {}) or {}
    term = data.get("term", {}) or {}
    iv_rank = data.get("iv_rank", {}) or {}
    pinning = data.get("pinning", {}) or {}
    live = data.get("live_metrics", {}) or {}
    prev_hl = data.get("prev_day_hl", {}) or {}
    session = data.get("session_metrics", {}) or {}
    open_spot = session.get("open_spot", 0)
    top_strikes = data.get("top_strikes", [])

    lines = []
    lines.append(f"# {ticker} Trade Analysis Request")
    lines.append("")
    lines.append(f"**Spot:** ${spot:,.2f}")
    lines.append(f"**Session open:** ${open_spot:,.2f}  "
                 f"(change {((spot-open_spot)/open_spot*100) if open_spot else 0:+.2f}%)")
    lines.append(f"**Prev close:** ${session.get('prev_close_spot', 0):,.2f}")
    lines.append(f"**Expiry:** {expiry}  ({dte} calendar DTE)")
    lines.append(f"**Exposure mode:** {mode}")
    lines.append(f"**Prev day range:** ${prev_hl.get('low', 0):,.2f} — "
                 f"${prev_hl.get('high', 0):,.2f}")
    lines.append("")

    # ── Gamma Regime ────────────────────────────────────────────────
    lines.append("## Gamma Regime")
    lines.append(f"- **Regime:** {regime.get('gamma', 'N/A')} γ")
    lines.append(f"- **Bias:** {regime.get('bias', 'N/A')}  "
                 f"(conviction: {regime.get('conviction', 'N/A')})")
    lines.append(f"- **Total GEX:** ${regime.get('total_gex', 0):,.0f}")
    flip = regime.get("gex_flip")
    lines.append(f"- **GEX Flip:** ${flip:,.1f}" if flip else "- **GEX Flip:** N/A")
    above = regime.get("above_flip")
    lines.append(f"- **Above flip:** {above}")
    cw = regime.get("call_wall")
    pw = regime.get("put_wall")
    lines.append(f"- **Call Wall:** ${cw:,.1f}" if cw else "- **Call Wall:** N/A")
    lines.append(f"- **Put Wall:** ${pw:,.1f}" if pw else "- **Put Wall:** N/A")
    lines.append("")

    # ── Volatility ──────────────────────────────────────────────────
    lines.append("## Volatility Signals")
    atm_iv = live.get("atm_iv", 0)
    lines.append(f"- **ATM IV:** {atm_iv*100:.1f}%")
    lines.append(f"- **ATM straddle:** ${live.get('straddle', 0):,.2f}  "
                 f"(EM range: ${live.get('straddle_em_low', 0):,.2f} — "
                 f"${live.get('straddle_em_high', 0):,.2f})")
    lines.append(f"- **Daily EM (1σ from prev close):** "
                 f"${session.get('daily_em', 0):,.2f}")
    lines.append(f"- **Weekly EM (1σ from last Fri close):** "
                 f"${session.get('weekly_em', 0):,.2f}")
    vx = vv.get("vix_current", 0)
    vx_chg = vv.get("vix_change", 0)
    lines.append(f"- **VIX:** {vx:.2f} ({vx_chg:+.2f})")
    lines.append(f"- **Vanna/VIX signal:** {vv.get('signal', 'N/A')}  "
                 f"(total vanna: {_format_number(vv.get('total_vanna', 0), 0)})")
    lines.append(f"- **Skew (25d put IV − call IV):** "
                 f"{skew.get('skew', 0)*100:+.2f} vol pts  "
                 f"({skew.get('skew_pct', 0):+.1f}% of ATM)")
    lines.append(f"- **Term structure:** {term.get('state', 'N/A')}  "
                 f"(front/back ratio: {term.get('ratio', 0):.3f}, "
                 f"back IV {term.get('back_iv', 0)*100:.1f}% @ {term.get('back_dte', 0)}d)")
    rank = iv_rank.get("iv_rank")
    pct = iv_rank.get("iv_percentile")
    if rank is not None:
        lines.append(f"- **IV Rank:** {rank:.0f}/100  "
                     f"(percentile: {pct:.0f}, history: {iv_rank.get('history_days', 0)} days)")
    else:
        lines.append(f"- **IV Rank:** Insufficient history")
    lines.append("")

    # ── Flow / Time Signals ─────────────────────────────────────────
    lines.append("## Flow & Time Signals")
    lines.append(f"- **Charm Clock:** {cc.get('direction', 'N/A')}  "
                 f"({cc.get('hours_to_close', 0):.1f}h to close, "
                 f"pressure: {_format_number(cc.get('charm_pressure', 0), 0)})")
    if pinning.get("pin_strike") is not None:
        lines.append(f"- **Pinning strike:** ${pinning['pin_strike']:,.1f}  "
                     f"(strength: {pinning.get('pin_strength', 0):.0f}/100, "
                     f"confidence: {pinning.get('confidence', 'N/A')}, "
                     f"distance: ${pinning.get('pin_distance', 0):.2f})")
    else:
        lines.append(f"- **Pinning strike:** N/A")
    lines.append(f"- **P/C OI ratio:** {live.get('pc_ratio', 0):.2f}  "
                 f"(total call OI: {live.get('total_call_oi', 0):,.0f}, "
                 f"total put OI: {live.get('total_put_oi', 0):,.0f})")
    lines.append("")

    # ── Top Strikes ─────────────────────────────────────────────────
    if top_strikes:
        lines.append("## Top GEX Strikes (absolute value, near ATM)")
        lines.append("| Strike | Gamma | Charm | Vanna | DEX | Zomma |")
        lines.append("|---|---|---|---|---|---|")
        for row in top_strikes[:12]:
            lines.append(
                f"| ${row.get('strike', 0):,.0f} "
                f"| {_format_number(row.get('gamma_exp', 0), 2)} "
                f"| {_format_number(row.get('charm_exp', 0), 2)} "
                f"| {_format_number(row.get('vanna_exp', 0), 2)} "
                f"| {_format_number(row.get('dex_exp', 0), 2)} "
                f"| {_format_number(row.get('zomma_exp', 0), 2)} |"
            )
        lines.append("")

    # ── Rule-based signal from our system ────────────────────────────
    signal = data.get("rule_signal", {}) or {}
    if signal:
        lines.append("## Rule-Based System Signal (for context)")
        lines.append(f"- **Direction:** {signal.get('direction', 'N/A')}  "
                     f"(conviction: {signal.get('conviction', 'N/A')}, "
                     f"score: {signal.get('score', 0):+.2f})")
        lines.append(f"- **Setup:** {signal.get('setup', '')}")
        entry = signal.get('entry')
        stop = signal.get('stop_loss')
        tp1 = signal.get('take_profit_1')
        tp2 = signal.get('take_profit_2')
        if entry:
            lines.append(f"- **Suggested entry:** ${entry:,.2f}")
        if stop:
            lines.append(f"- **Suggested stop:** ${stop:,.2f}")
        if tp1:
            lines.append(f"- **TP1:** ${tp1:,.2f}")
        if tp2:
            lines.append(f"- **TP2:** ${tp2:,.2f}")
        lines.append(f"- **R:R:** {signal.get('risk_reward', 0):.2f}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Given all of the above, produce your own independent trade "
                 "analysis. You may agree or disagree with the rule-based signal. "
                 "Structure your response as:")
    lines.append("")
    lines.append("**THESIS:** (one sentence)")
    lines.append("")
    lines.append("**DIRECTION:** LONG | SHORT | STAND ASIDE")
    lines.append("")
    lines.append("**ENTRY:** (price or price zone, with trigger condition)")
    lines.append("")
    lines.append("**STOP LOSS:** (price, with level justification)")
    lines.append("")
    lines.append("**TAKE PROFIT 1:** (price, with reasoning)")
    lines.append("")
    lines.append("**TAKE PROFIT 2:** (price, with reasoning)")
    lines.append("")
    lines.append("**WHY THIS SETUP:** (3-5 bullet points — be specific about "
                 "which signals matter most and why)")
    lines.append("")
    lines.append("**WHAT CHANGES THE TRADE:** (specific conditions to watch; "
                 "what would flip your view?)")

    return "\n".join(lines)


def analyze(data: dict, model: str = DEFAULT_MODEL) -> dict:
    """
    Send the dashboard state to Claude and return the analysis.

    Returns dict:
        {
            "ok": bool,
            "analysis": str (the markdown response),
            "error": str (if ok is False),
            "tokens_in": int,
            "tokens_out": int,
            "model": str,
        }
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "analysis": "",
            "error": "ANTHROPIC_API_KEY environment variable not set. "
                     "Get one at https://console.anthropic.com/ and set it "
                     "in your environment before starting the dashboard.",
        }

    try:
        from anthropic import Anthropic
    except ImportError:
        return {
            "ok": False,
            "analysis": "",
            "error": "anthropic SDK not installed. "
                     "Run: pip install anthropic",
        }

    try:
        prompt = build_analysis_prompt(data)
        client = Anthropic(api_key=api_key)

        message = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        text_parts = [b.text for b in message.content if hasattr(b, "text")]
        analysis = "\n".join(text_parts).strip()

        return {
            "ok": True,
            "analysis": analysis,
            "error": None,
            "tokens_in":  message.usage.input_tokens,
            "tokens_out": message.usage.output_tokens,
            "model":      model,
        }

    except Exception as exc:
        traceback.print_exc()
        return {
            "ok": False,
            "analysis": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
