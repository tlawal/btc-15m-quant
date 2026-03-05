"""
Railway Log Backtester — parse engine log output and compute what-if PnL.

Supports two input formats:
  1. JSON array (export from Railway)
  2. Raw text logs (copy-paste from Railway console)

Usage:
    python backtest_from_logs.py logs.json
    python backtest_from_logs.py logs.txt
"""

import re
import json
import sys
import argparse
import pandas as pd
from datetime import datetime


def parse_engine_block(text: str) -> dict:
    """
    Parse a single '=== BTC 15m Quant Engine ===' log block
    into a structured dict.
    """
    result = {}

    # Timestamps / window
    m = re.search(r'now=(\d+)', text)
    if m:
        result["now_ts"] = int(m.group(1))
        result["timestamp"] = datetime.utcfromtimestamp(result["now_ts"]).strftime("%Y-%m-%d %H:%M:%S")

    m = re.search(r'rem=([\d.]+)min', text)
    if m:
        result["minutes_remaining"] = float(m.group(1))

    # Strike & Price
    m = re.search(r'Strike=([\d.]+)', text)
    if m:
        result["strike"] = float(m.group(1))

    m = re.search(r'Price=([\d.]+)', text)
    if m:
        result["price"] = float(m.group(1))

    m = re.search(r'Dist=([-\d.]+)', text)
    if m:
        result["distance"] = float(m.group(1))

    # Bayesian
    m = re.search(r'PostUp=([\d.]+)', text)
    if m:
        result["posterior_up"] = float(m.group(1))

    m = re.search(r'PostDown=([\d.]+)', text)
    if m:
        result["posterior_down"] = float(m.group(1))

    # Edge
    m = re.search(r'Edge:\s*([\d.]+)', text)
    if m:
        result["edge"] = float(m.group(1))

    m = re.search(r'Min Req:\s*([\d.]+)', text)
    if m:
        result["min_edge"] = float(m.group(1))

    # Score
    m = re.search(r'SignedScore:\s*([-\d.]+)', text)
    if m:
        result["signed_score"] = float(m.group(1))

    m = re.search(r'Min Req:\s*(\d+)', text)
    # Already captured above for edge, skip if duplicate

    # Market prices
    m = re.search(r'YES=([\d.]+)', text)
    if m:
        result["yes_price"] = float(m.group(1))

    m = re.search(r'NO=([\d.]+)', text)
    if m:
        result["no_price"] = float(m.group(1))

    # Decision
    m = re.search(r'Decision:\s*(\S+)', text)
    if m:
        result["decision"] = m.group(1)

    # Skip gates
    m = re.search(r'SkipGates:\s*(.+?)(?:\n|$)', text)
    if m:
        result["skip_gates"] = m.group(1).strip()

    # Micro signals
    m = re.search(r'cvd(?:S)?=([-\d.]+)', text)
    if m:
        result["cvd"] = float(m.group(1))

    m = re.search(r'obi=([\d.]+)', text)
    if m:
        result["obi"] = float(m.group(1))

    # Regime
    m = re.search(r'Regime=(\w+)', text)
    if m:
        result["regime"] = m.group(1)

    # ATR
    m = re.search(r'ATR5m=([\d.]+)', text)
    if m:
        result["atr"] = float(m.group(1))

    # Balance
    m = re.search(r'Balance:\s*USDC=([\d.]+)', text)
    if m:
        result["balance"] = float(m.group(1))

    return result


def parse_log_file(filepath: str) -> list:
    """Parse a log file into a list of engine cycle dicts."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Try JSON format first
    if filepath.endswith('.json'):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Parse text format — split on engine block headers
    blocks = re.split(r'={3,}\s*BTC 15m Quant Engine\s*={3,}', content)
    results = []
    for block in blocks:
        if not block.strip():
            continue
        parsed = parse_engine_block(block)
        if parsed.get("now_ts"):
            results.append(parsed)

    return results


def simulate_whatif(cycles: list) -> pd.DataFrame:
    """
    For each cycle, determine if we would have traded and estimate PnL.
    Groups cycles by window (same strike) to determine outcome.
    """
    records = []

    # Group by window (approximate: group by strike value)
    windows = {}
    for c in cycles:
        strike = c.get("strike", 0)
        if strike == 0:
            continue
        # Round to nearest dollar for grouping
        key = round(strike, 2)
        if key not in windows:
            windows[key] = []
        windows[key].append(c)

    for strike_key, window_cycles in windows.items():
        # Use last cycle in window for outcome
        last = window_cycles[-1]
        price = last.get("price", 0)
        strike = last.get("strike", 0)
        if strike == 0 or price == 0:
            continue

        actual_up = price > strike
        actual_outcome = "UP" if actual_up else "DOWN"

        # Check each cycle: would we have traded?
        best_cycle = None
        best_score = 0
        for c in window_cycles:
            score = abs(c.get("signed_score", 0))
            if score > best_score:
                best_score = score
                best_cycle = c

        if best_cycle is None:
            continue

        c = best_cycle
        posterior_up = c.get("posterior_up", 0.5)
        direction = "UP" if posterior_up > 0.5 else "DOWN"
        edge = c.get("edge", 0)
        signed_score = c.get("signed_score", 0)
        skip_gates = c.get("skip_gates", "")
        decision = c.get("decision", "NO_TRADE")
        yes_px = c.get("yes_price", 0.50)
        no_px = c.get("no_price", 0.50)

        # Would the new dynamic edge have allowed this?
        best_posterior = max(posterior_up, 1 - posterior_up)
        minutes_rem = c.get("minutes_remaining", 15)
        market_px = yes_px if direction == "UP" else no_px

        # Dynamic edge calculation (mirrors logic.py)
        if best_posterior >= 0.95:
            dynamic_edge = 0.005 if minutes_rem < 3 else 0.012
        elif minutes_rem < 2:
            dynamic_edge = 0.010
        elif market_px < 0.10 or market_px > 0.90:
            dynamic_edge = 0.012
        else:
            dynamic_edge = 0.035

        would_trade_original = (decision != "NO_TRADE")
        would_trade_dynamic = (edge >= dynamic_edge and abs(signed_score) >= 2.5)

        # PnL if we traded
        won = (direction == actual_outcome)
        entry_px = yes_px if direction == "UP" else no_px
        entry_px = max(0.01, min(0.99, entry_px or 0.50))
        pnl_if_traded = (1.0 - entry_px) if won else (-entry_px)

        records.append({
            "timestamp": c.get("timestamp", ""),
            "strike": strike,
            "close": price,
            "distance": round(price - strike, 2),
            "direction": direction,
            "posterior": round(best_posterior, 4),
            "edge": round(edge, 4),
            "score": round(signed_score, 2),
            "regime": c.get("regime", "?"),
            "rem_min": round(minutes_rem, 1),
            "skip_gates": skip_gates[:60],
            "actual": actual_outcome,
            "won": won,
            "original_trade": would_trade_original,
            "dynamic_trade": would_trade_dynamic,
            "pnl_if_traded": round(pnl_if_traded, 4),
        })

    return pd.DataFrame(records)


def backtest_from_logs(log_file: str):
    """Main entry point."""
    print(f"═══════════════════════════════════════════════════")
    print(f"  Predi-Quant Railway Log Backtester")
    print(f"  Input: {log_file}")
    print(f"═══════════════════════════════════════════════════")

    cycles = parse_log_file(log_file)
    print(f"\n  Parsed {len(cycles)} engine cycles from logs")

    if not cycles:
        print("  No parseable engine cycles found. Check log format.")
        return

    df = simulate_whatif(cycles)
    print(f"  Analyzed {len(df)} unique windows\n")

    if len(df) == 0:
        print("  No windows with valid data found.")
        return

    # === Original Strategy Results ===
    orig_trades = df[df["original_trade"] == True]
    print(f"── ORIGINAL STRATEGY (what actually happened) ─────")
    print(f"  Trades Taken:  {len(orig_trades)}")
    if len(orig_trades) > 0:
        wins = orig_trades["won"].sum()
        print(f"  Wins/Losses:   {int(wins)} / {len(orig_trades) - int(wins)}")
        print(f"  Win Rate:      {wins/len(orig_trades):.1%}")
        print(f"  Total PnL:     ${orig_trades['pnl_if_traded'].sum():.4f}")

    # === Dynamic Edge Strategy Results ===
    dyn_trades = df[df["dynamic_trade"] == True]
    print(f"\n── DYNAMIC EDGE STRATEGY (what-if with new rules) ─")
    print(f"  Would Trade:   {len(dyn_trades)}")
    if len(dyn_trades) > 0:
        wins = dyn_trades["won"].sum()
        total_pnl = dyn_trades["pnl_if_traded"].sum()
        avg_pnl = dyn_trades["pnl_if_traded"].mean()
        print(f"  Wins/Losses:   {int(wins)} / {len(dyn_trades) - int(wins)}")
        print(f"  Win Rate:      {wins/len(dyn_trades):.1%}")
        print(f"  Total PnL:     ${total_pnl:.4f}")
        print(f"  Avg PnL/Trade: ${avg_pnl:.4f}")

    # === Missed Opportunities ===
    missed = df[(df["original_trade"] == False) & (df["won"] == True)]
    print(f"\n── MISSED WINNING TRADES ───────────────────────────")
    print(f"  Count:         {len(missed)}")
    if len(missed) > 0:
        missed_pnl = missed["pnl_if_traded"].sum()
        print(f"  Missed PnL:    ${missed_pnl:.4f}")
        print(f"\n  Top 10 missed winners:")
        top_missed = missed.nlargest(10, "pnl_if_traded")
        for _, row in top_missed.iterrows():
            print(f"    {row['timestamp']}  dir={row['direction']}  "
                  f"post={row['posterior']:.2f}  edge={row['edge']:.3f}  "
                  f"pnl=${row['pnl_if_traded']:.4f}  gates={row['skip_gates']}")

    print(f"\n═══════════════════════════════════════════════════")

    # Save
    df.to_csv("log_backtest_results.csv", index=False)
    print(f"  Full results saved to log_backtest_results.csv")

    if len(dyn_trades) > 0:
        dyn_trades.to_csv("log_backtest_trades.csv", index=False)
        print(f"  Dynamic trades saved to log_backtest_trades.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predi-Quant Railway Log Backtester")
    parser.add_argument("logfile", nargs="?", default="railway_logs.txt", help="Path to Railway log file (JSON or text)")
    args = parser.parse_args()
    backtest_from_logs(args.logfile)
