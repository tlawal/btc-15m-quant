"""
Historical backtesting harness for the BTC 15m Quant strategy.
Simulates trading decisions over historical Binance klines.

Usage: python backtest.py --days 30
"""

import asyncio
import logging
from datetime import datetime
import time

from config import Config
from data_feeds import DataFeeds, Candle
from signals import compute_signals
from indicators import compute_local_indicators

log = logging.getLogger("backtester")

async def run_backtest(days: int = 30):
    feeds = DataFeeds()
    await feeds.start()
    
    # 1. Fetch historical 15m klines from Binance for the past N days.
    # Note: Binance limits to 1000 candles per request.
    limit = min(days * 24 * 4, 1000) # 15m candles
    klines = await feeds.get_klines("BTCUSDT", "15m", limit=limit)
    if not klines:
        log.error("Failed to fetch historical klines")
        await feeds.close()
        return

    log.info(f"Loaded {len(klines)} historical candles.")

    # Simulated state
    class SimState:
        held_position = None
        loss_streak = 0
        total_pnl = 0.0
        total_trades = 0
        wins = 0
        
    state = SimState()

    # We need a rolling window to compute indicators
    for i in range(100, len(klines)): # Start at 100 to ensure indicator warmup
        history_slice = klines[i-100:i+1] # up to current
        current_candle = history_slice[-1]
        
        # In a genuine tick-by-tick backtest we would reconstruct L2 OB and CVD.
        # Here we approximate signals from the OHLCV slice.
        # This is a stub showing where indicator calculations plug in.
        
        # ... fetch indicators ...
        # ... compute SignalResult ...
        # ... simulate entry / exit rules ...
        pass
        
    log.info(f"Backtest complete. Trades: {state.total_trades}, PnL: ${state.total_pnl:.2f}, WR: {state.wins/max(1, state.total_trades)*100:.1f}%")

    await feeds.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_backtest(args.days))
