"""
flow_tracker.py  -  Infers dealer positioning from live option trade flow.

Uses a volume-delta approach:
1. Each fetch cycle, compare current volume to previous fetch
2. New trades = volume increase since last fetch
3. Classify direction: trade price near ask = buyer, near bid = seller
4. Track per-strike, per-right (call/put) buy/sell volumes
5. Build a synthetic "dealer chain" for computing all greeks

Dealer position convention:
  Customer BUYS  call -> dealer SELLS call -> dealer SHORT calls
  Customer SELLS call -> dealer BUYS  call -> dealer LONG  calls
  Customer BUYS  put  -> dealer SELLS put  -> dealer SHORT puts
  Customer SELLS put  -> dealer BUYS  put  -> dealer LONG  puts
"""

import numpy as np
import pandas as pd


class FlowTracker:

    def __init__(self):
        self._prev_vol = {}    # {(right, strike): volume}
        self._flow = {}        # {strike: {"call_buy","call_sell","put_buy","put_sell",...}}
        self._ticker = ""
        self._total_classified = 0
        self._total_unclassified = 0

    def process(self, chain_df, spot):
        """
        Process a new chain snapshot.  Detects volume deltas and
        classifies trade direction from last price vs bid/ask.
        """
        if chain_df is None or chain_df.empty:
            return

        for _, row in chain_df.iterrows():
            strike = float(row["strike"])

            for right, pfx in [("C", "call"), ("P", "put")]:
                vol  = int(row.get(f"{pfx}_volume", 0))
                last = float(row.get(f"{pfx}_last", 0))
                bid  = float(row.get(f"{pfx}_bid", 0))
                ask  = float(row.get(f"{pfx}_ask", 0))

                key = (right, strike)
                prev_vol = self._prev_vol.get(key, None)
                self._prev_vol[key] = vol

                if prev_vol is None:
                    continue
                if vol <= prev_vol or prev_vol < 0:
                    continue

                new_contracts = vol - prev_vol

                if bid <= 0 or ask <= 0 or ask <= bid or last <= 0:
                    self._total_unclassified += new_contracts
                    continue

                spread = ask - bid
                if last >= ask - spread * 0.25:
                    side = "buy"
                elif last <= bid + spread * 0.25:
                    side = "sell"
                else:
                    self._total_unclassified += new_contracts
                    continue

                self._total_classified += new_contracts

                if strike not in self._flow:
                    self._flow[strike] = {
                        "call_buy": 0, "call_sell": 0,
                        "put_buy": 0,  "put_sell": 0,
                        "trades": 0,
                    }

                if right == "C":
                    if side == "buy":
                        self._flow[strike]["call_buy"] += new_contracts
                    else:
                        self._flow[strike]["call_sell"] += new_contracts
                else:
                    if side == "buy":
                        self._flow[strike]["put_buy"] += new_contracts
                    else:
                        self._flow[strike]["put_sell"] += new_contracts

                self._flow[strike]["trades"] += 1

    # -- Build dealer chain for compute_exposure -----------------------

    def get_dealer_chain(self, real_chain):
        """
        Build a chain DataFrame where OI columns represent the dealer's
        inferred NET SHORT position from classified flow.

        Net dealer short = customer_buy - customer_sell
        Positive = dealer accumulated net short (same sign convention as OI mode)
        Negative = dealer accumulated net long

        Uses IV and DTE from the real chain so greeks use current market data.
        """
        if real_chain is None or real_chain.empty or not self._flow:
            return pd.DataFrame()

        rows = []
        for _, row in real_chain.iterrows():
            strike = float(row["strike"])
            f = self._flow.get(strike, {})

            net_call = f.get("call_buy", 0) - f.get("call_sell", 0)
            net_put  = f.get("put_buy", 0)  - f.get("put_sell", 0)

            rows.append({
                "strike":      strike,
                "call_oi":     net_call,
                "put_oi":      net_put,
                "call_volume": f.get("call_buy", 0) + f.get("call_sell", 0),
                "put_volume":  f.get("put_buy", 0) + f.get("put_sell", 0),
                "call_iv":     float(row.get("call_iv", 0.20)),
                "put_iv":      float(row.get("put_iv", 0.20)),
                "dte_years":   float(row.get("dte_years", 1/365)),
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_oi_plus_flow_chain(self, real_chain):
        """
        Build a chain where OI = original OI + flow delta.

        This shows the "live OI" — starting position plus how it's
        shifted from today's classified trade flow.

        - If flow shows customers bought 200 calls at a strike,
          that's 200 MORE dealer short calls → OI effectively +200
        - If flow shows customers sold 300 calls,
          that's dealers buying back 300 → OI effectively -300

        The result is what OI would look like if updated in real-time
        from today's trading activity.
        """
        if real_chain is None or real_chain.empty:
            return real_chain

        rows = []
        for _, row in real_chain.iterrows():
            strike = float(row["strike"])
            f = self._flow.get(strike, {})

            # Flow delta: net new dealer shorts
            flow_call = f.get("call_buy", 0) - f.get("call_sell", 0)
            flow_put  = f.get("put_buy", 0)  - f.get("put_sell", 0)

            # OI + flow (clamp to 0 — can't have negative OI)
            new_row = dict(row)
            new_row["call_oi"] = max(0, int(row.get("call_oi", 0)) + flow_call)
            new_row["put_oi"]  = max(0, int(row.get("put_oi", 0))  + flow_put)
            rows.append(new_row)

        return pd.DataFrame(rows) if rows else real_chain

    # -- Stats and access ----------------------------------------------

    def get_flow_data(self):
        return dict(self._flow)

    def get_stats(self):
        total_buy = sum(
            v.get("call_buy", 0) + v.get("put_buy", 0)
            for v in self._flow.values()
        )
        total_sell = sum(
            v.get("call_sell", 0) + v.get("put_sell", 0)
            for v in self._flow.values()
        )
        return {
            "classified":   self._total_classified,
            "unclassified": self._total_unclassified,
            "total_buy":    total_buy,
            "total_sell":   total_sell,
            "strikes":      len(self._flow),
        }

    def reset(self, ticker=""):
        self._prev_vol.clear()
        self._flow.clear()
        self._ticker = ticker
        self._total_classified = 0
        self._total_unclassified = 0
