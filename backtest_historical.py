"""
Historical Backtester — replays resolved Polymarket 15m BTC markets
against Binance kline data and your signal logic.

Usage:
    python backtest_historical.py              # default 30 days
    python backtest_historical.py --days 60    # custom lookback
"""

import asyncio
import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import aiohttp
import pandas as pd

from config import Config
from data_feeds import DataFeeds, Candle
from indicators import compute_local_indicators

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("backtest")


GAMMA_API = "https://gamma-api.polymarket.com"


async def fetch_resolved_markets(session: aiohttp.ClientSession, days_back: int):
    """Fetch resolved BTC 15m up/down markets from Polymarket Gamma API."""
    markets = []
    offset = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    while True:
        url = (
            f"{GAMMA_API}/markets"
            f"?closed=true&limit=100&offset={offset}"
            f"&order=endDate&ascending=false"
        )
        try:
            async with session.get(url) as r:
                if r.status != 200:
                    log.warning(f"Gamma API returned {r.status}")
                    break
                batch = await r.json()
        except Exception as e:
            log.warning(f"Gamma API error: {e}")
            break

        if not batch:
            break

        for m in batch:
            slug = m.get("slug", "")
            if "btc" not in slug.lower() or "15m" not in slug.lower():
                continue
            if "updown" not in slug.lower() and "up-down" not in slug.lower():
                continue

            end_str = m.get("endDate") or m.get("end_date_iso")
            if not end_str:
                continue

            try:
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except Exception:
                continue

            if end_dt < cutoff:
                # Past our lookback window, stop paginating
                return markets

            # Determine resolution outcome
            outcome = m.get("outcome", "").upper()
            resolution = m.get("resolutionSource", "")

            markets.append({
                "slug": slug,
                "end_ts": int(end_dt.timestamp()),
                "start_ts": int(end_dt.timestamp()) - 900,
                "outcome": outcome,  # "YES" or "NO"
                "end_dt": end_dt,
            })

        offset += 100
        await asyncio.sleep(0.3)  # rate limit

    return markets


async def fetch_binance_klines(session: aiohttp.ClientSession, start_ts: int, interval: str = "5m", limit: int = 50):
    """Fetch Binance klines for backtesting."""
    end_ms = (start_ts + 900) * 1000  # end of 15m window
    start_ms = end_ms - (limit * 5 * 60 * 1000)  # enough history

    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol=BTCUSDT&interval={interval}&startTime={start_ms}&endTime={end_ms}&limit={limit}"
    )
    try:
        async with session.get(url) as r:
            if r.status != 200:
                return []
            raw = await r.json()
            return [
                Candle(
                    ts_ms=int(k[0]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                )
                for k in raw
            ]
    except Exception as e:
        log.warning(f"Binance klines error: {e}")
        return []


def simulate_signal(indic, strike_price: float, btc_close: float, atr: float):
    """
    Simplified signal replay: compute posterior and direction
    without requiring full engine state.
    """
    if strike_price <= 0 or btc_close <= 0:
        return {"direction": "NEUTRAL", "signed_score": 0, "posterior_up": 0.5, "edge": 0}

    distance = btc_close - strike_price
    exp_move = atr * 0.5 if atr > 0 else 100.0
    z = distance / exp_move if exp_move > 0 else 0

    # Simplified Bayesian posterior
    from math import erf, sqrt
    posterior_up = 0.5 * (1 + erf(z / sqrt(2)))
    posterior_up = max(0.001, min(0.999, posterior_up))

    direction = "UP" if posterior_up > 0.5 else "DOWN"
    chosen_posterior = posterior_up if direction == "UP" else (1 - posterior_up)

    # Simplified edge (assume market price ~ 0.50 for resolved markets)
    market_estimate = 0.50
    edge = chosen_posterior - market_estimate

    # Simplified score
    signed_score = z * 2.0  # rough proxy

    return {
        "direction": direction,
        "signed_score": signed_score,
        "posterior_up": posterior_up,
        "edge": edge,
        "distance": distance,
        "chosen_posterior": chosen_posterior,
    }


async def backtest_historical(days_back: int = 30):
    """Main backtester entry point."""
    print(f"═══════════════════════════════════════════════════")
    print(f"  Predi-Quant Historical Backtester")
    print(f"  Lookback: {days_back} days")
    print(f"═══════════════════════════════════════════════════")

    async with aiohttp.ClientSession() as session:
        # 1. Fetch resolved markets
        print("\n[1/3] Fetching resolved Polymarket 15m BTC markets...")
        markets = await fetch_resolved_markets(session, days_back)
        print(f"       Found {len(markets)} resolved markets")

        if not markets:
            print("No markets found. Exiting.")
            return

        # 2. Replay each market
        print(f"[2/3] Replaying signal logic against historical data...")
        results = []
        traded = 0
        skipped = 0

        for i, mkt in enumerate(markets):
            if i % 50 == 0 and i > 0:
                print(f"       Processed {i}/{len(markets)} markets...")
                await asyncio.sleep(0.5)  # rate limit

            # Fetch klines for this window
            k5m = await fetch_binance_klines(session, mkt["start_ts"], "5m", 50)
            await asyncio.sleep(0.15)  # rate limit

            if len(k5m) < 15:
                skipped += 1
                continue

            # Compute indicators
            indic = compute_local_indicators(k5m, [])

            # Get strike (open of the 15m window)
            # Find the candle closest to window start
            strike_price = None
            for c in k5m:
                if c.ts_ms >= mkt["start_ts"] * 1000:
                    strike_price = c.open
                    break
            if strike_price is None:
                strike_price = k5m[-1].open

            btc_close = k5m[-1].close
            atr = indic.atr14 or 150.0

            # Simulate signal
            sig = simulate_signal(indic, strike_price, btc_close, atr)

            # Determine if we would have traded
            would_trade = (
                abs(sig["signed_score"]) >= Config.MIN_SCORE_NORMAL
                and sig["edge"] >= Config.REQUIRED_EDGE_NORMAL
            )

            # Determine actual outcome
            actual_up = btc_close > strike_price
            actual_outcome = "YES" if actual_up else "NO"

            # If we would have traded, compute PnL
            pnl = 0.0
            if would_trade:
                traded += 1
                our_side = "YES" if sig["direction"] == "UP" else "NO"
                won = (our_side == actual_outcome)
                # Simplified PnL: assume entry at ~0.50 (simplified for historical)
                entry_px = 0.50 + sig["edge"] * 0.5  # rough estimate
                pnl = (1.0 - entry_px) if won else (-entry_px)

            results.append({
                "timestamp": mkt["end_dt"].strftime("%Y-%m-%d %H:%M"),
                "strike": round(strike_price, 2),
                "close": round(btc_close, 2),
                "distance": round(sig["distance"], 2),
                "direction": sig["direction"],
                "score": round(sig["signed_score"], 2),
                "posterior_up": round(sig["posterior_up"], 4),
                "edge": round(sig["edge"], 4),
                "would_trade": would_trade,
                "actual": actual_outcome,
                "won": won if would_trade else None,
                "pnl": round(pnl, 4),
            })

        # 3. Results
        print(f"[3/3] Computing results...\n")

        df = pd.DataFrame(results)
        traded_df = df[df["would_trade"] == True]

        print(f"═══════════════════════════════════════════════════")
        print(f"  BACKTEST RESULTS")
        print(f"═══════════════════════════════════════════════════")
        print(f"  Total Windows:    {len(df)}")
        print(f"  Skipped (no data): {skipped}")
        print(f"  Would Trade:      {len(traded_df)}")

        if len(traded_df) > 0:
            wins = traded_df["won"].sum()
            losses = len(traded_df) - wins
            win_rate = wins / len(traded_df)
            total_pnl = traded_df["pnl"].sum()
            avg_pnl = traded_df["pnl"].mean()
            sharpe = (traded_df["pnl"].mean() / traded_df["pnl"].std()) if traded_df["pnl"].std() > 0 else 0

            print(f"  Wins:             {int(wins)}")
            print(f"  Losses:           {int(losses)}")
            print(f"  Win Rate:         {win_rate:.1%}")
            print(f"  Total PnL:        ${total_pnl:.4f}")
            print(f"  Avg PnL/Trade:    ${avg_pnl:.4f}")
            print(f"  Sharpe Ratio:     {sharpe:.2f}")

            # Equity curve
            traded_df = traded_df.copy()
            traded_df["cum_pnl"] = traded_df["pnl"].cumsum()
            print(f"\n  Equity Curve (last 20 trades):")
            for _, row in traded_df.tail(20).iterrows():
                bar = "█" * max(1, int(abs(row["cum_pnl"]) * 20))
                color = "+" if row["cum_pnl"] >= 0 else "-"
                print(f"    {row['timestamp']}  {color}{bar}  ${row['cum_pnl']:.4f}")
        else:
            print(f"  No trades would have been taken with current thresholds.")
            print(f"  Consider lowering MIN_SCORE or REQUIRED_EDGE.")

        print(f"═══════════════════════════════════════════════════")

        # Save to CSV
        df.to_csv("backtest_results.csv", index=False)
        print(f"\n  Full results saved to backtest_results.csv")

        if len(traded_df) > 0:
            traded_df.to_csv("backtest_trades.csv", index=False)
            print(f"  Trade-only results saved to backtest_trades.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predi-Quant Historical Backtester")
    parser.add_argument("--days", type=int, default=30, help="Days to look back (default: 30)")
    args = parser.parse_args()
    asyncio.run(backtest_historical(days_back=args.days))
