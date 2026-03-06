"""
BTC 15m Quant Engine — main async loop.

Architecture:
  - 15-second inner loop (Config.LOOP_INTERVAL_SEC)
  - Each cycle: fetch data → compute signals → handle exits → handle entries
  - State persisted to SQLite after every cycle
  - Graceful shutdown on SIGINT/SIGTERM
"""

BUILD_VERSION = "v2.1-USDC-E-FIX"

import asyncio
import aiofiles
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from config import Config
from state import StateManager, EngineState, HeldPosition, TradeRecord
from data_feeds import DataFeeds
from polymarket_client import PolymarketClient
from indicators import compute_local_indicators
from logic import compute_signals, evaluate_exit, compute_position_size
from inference import InferenceEngine
from dashboard import run_dashboard
from utils import (
    setup_logging, send_telegram, Timer,
    fmt_entry, fmt_exit, fmt_halt, fmt_status, fmt_engine_block, fmt_pnl_dashboard,
    current_window_start, minutes_remaining as calc_minutes_remaining,
    window_start_iso, window_end_iso,
)

setup_logging()
log = logging.getLogger("engine")


class Engine:
    def __init__(self):
        self.state_mgr = StateManager()
        self.feeds     = DataFeeds()
        self.pm        = PolymarketClient()
        self.state: Optional[EngineState] = None
        self._running  = True
        self._status_counter = 0   # send Telegram status every N cycles
        self._last_exit_reason = "HOLD"
        self.inference = InferenceEngine()  # Phase 5: ML inference engine
        self._dashboard_task = None         # Phase 6: Dashboard task handle
        # Real-time CVD WebSocket
        from data_feeds import BinanceCVDWebsocket
        self.cvd_ws = BinanceCVDWebsocket()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _record_redemption(self, size_usd: float):
        """Find the matching OPEN trade from the previous window and mark it as a WIN."""
        # We usually redeem full sizes, meaning entry price was ~0.40 to 0.60
        # If we redeemed $10 for example, the position was 10 shares.
        # This isn't perfect tracking (since we don't return which specific order was redeemed),
        # but since we only trade 1 market per window and never hold multiple directions,
        # finding the most recent OPEN trade is robust.
        
        for tr in reversed(self.state.trade_history):
            if tr.outcome in ("OPEN", "LOSS"):
                # Always force entry price to a sane value if zero
                ep = tr.entry_price if tr.entry_price and tr.entry_price > 0 else 0.5
                tr.exit_price = 1.0  # Polymarket winning shares are always redeemed at $1
                tr.pnl = (tr.exit_price - ep) / ep
                
                # If it was prematurely marked as a LOSS, reverse the loss stats
                if tr.outcome == "LOSS":
                    self.state.total_losses -= 1
                    self.state.total_pnl_usd += size_usd # Add the size back
                
                tr.outcome = "WIN"
                
                self.state.loss_streak = 0
                if self.state.total_trades == 0:
                    self.state.total_trades += 1 # Ensure counting if lost
                self.state.total_wins += 1
                
                # Approximate PnL logic (size_usd is the *payout*)
                profit_usd = size_usd * tr.pnl if tr.pnl else 0.0
                self.state.total_pnl_usd += profit_usd
                
                log.info(f"Recorded redemption win: +{tr.pnl*100:.2f}% (approx profit: ${profit_usd:.2f})")
                
                # Immediately recalculate and store metrics inside the event loop if possible
                if self.state_mgr:
                    asyncio.create_task(self.state_mgr.calculate_performance_metrics())
                return
                
        log.warning("Redeemed position but no matching OPEN trade found in history.")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        await self.feeds.start()
        await self.pm.start()
        self.state = await self.state_mgr.load()
        if self.pm.can_trade:
            print(f"ENGINE START BUILD={BUILD_VERSION} RPC_SET={bool(Config.POLYGON_RPC_URL)} USDC_ADDR={Config.POLYGON_USDC_ADDRESS}", flush=True)
            log.info("Engine started. Ensuring Polymarket approvals...")
            await self.pm.ensure_approvals()
        else:
            # Run signals + market discovery, but do not attempt trading endpoints.
            self.state.trading_halted = True
            log.warning(
                "Polymarket trading credentials missing; running in READONLY mode (no orders, no balance, no positions)."
            )

        # Phase 6: Start Dashboard in background
        self._dashboard_task = asyncio.create_task(run_dashboard(self))
        log.info("Dashboard server started on port 8000")

        # Start real-time CVD WebSocket
        await self.cvd_ws.start()
        log.info("BinanceCVD WebSocket started")

        log.info("Ready. Starting main loop.")

    async def stop(self):
        self._running = False
        if self.state:
            await self.state_mgr.save(self.state)
        
        if self._dashboard_task:
            self._dashboard_task.cancel()
            try: await self._dashboard_task
            except asyncio.CancelledError: pass

        self.cvd_ws.stop()
        await self.feeds.close()
        await self.pm.close()
        log.info("Engine stopped.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        await self.start()
        
        # Also run once at startup
        try:
            redeemed_startup = await self.pm.redeem_winning_positions()
            if redeemed_startup > 0:
                log.info(f"✅ STARTUP CLAIM: ${redeemed_startup:.2f}")
        except Exception as e:
            log.error(f"Startup redeem failed: {e}")
            
        while self._running:
            t0 = time.monotonic()
            try:
                await self._cycle()
            except Exception as e:
                log.exception(f"Cycle error: {e}")
            elapsed = time.monotonic() - t0
            sleep   = max(0.0, Config.LOOP_INTERVAL_SEC - elapsed)
            await asyncio.sleep(sleep)

    async def _cycle(self):
        start_time_ms = int(time.time() * 1000)
        now_ts        = int(time.time())
        win_start     = current_window_start(now_ts)
        win_end      = win_start + Config.WINDOW_SEC
        min_rem      = calc_minutes_remaining(now_ts)
        win_rolled   = self.state.last_window_start_sec != win_start

        # ── Window reset ──────────────────────────────────────────────────────
        if win_rolled:
            log.info(f"New 15m window: {win_start} ({datetime.fromtimestamp(win_start, tz=timezone.utc).strftime('%H:%M:%S')} UTC)")
            # Try to redeem any winning positions
            redeemed_usd = await self.pm.redeem_winning_positions()
            if redeemed_usd > 0:
                log.info(f"✅ AUTO-CLAIMED ${redeemed_usd:.2f} at window start!")
                self._record_redemption(redeemed_usd)
            
            # Phase 4: Reset latencies on new window to clear stale metrics
            self.state.latencies             = {}
            self.state.trades_this_window   = 0
            self.state.locked_strike_price  = None
            self.state.strike_source        = None
            self.state.strike_window_start  = win_start
            self.state.cvd                  = 0.0   # reset CVD each window
            self.state.accumulated_ofi      = 0.0   # Phase 2: reset accumulated OFI
            self.cvd_ws.reset()                      # Reset real-time CVD WebSocket

        self.state.last_window_start_sec = win_start
        # Dashboard visibility: show unclaimed positions every cycle
        self.state.unclaimed_usdc = await self.pm.log_unclaimed_positions()
        
        # Dashboard visibility: recent trades from Data API
        self.api_trade_history = await self.pm.get_trade_history()

        if self.state.held_position.is_pending and self.state.held_position.order_id:
            await self._reconcile_pending_order()

        # ── Parallel Data Fetch (Phase 4: Latency track) ──────────────────────
        with Timer("data_fetch_all", self.state.latencies):
            # 1. Market discovery
            pm_task = asyncio.create_task(
                self.pm.discover_market(
                    win_start,
                    cached_slug         = self.state.last_market_slug,
                    cached_expiry       = self.state.last_market_expiry,
                    cached_condition_id = self.state.last_condition_id,
                )
            )
            # 2. Strike resolution
            strike_task = asyncio.create_task(self._resolve_strike(win_start))
            # 3. Klines (parallel sources in DataFeeds)
            k5m_task = asyncio.create_task(self.feeds.get_klines("BTCUSDT", "5m", 30))
            k1m_task = asyncio.create_task(self.feeds.get_klines("BTCUSDT", "1m", 30))
            # 4. CVD (parallel sources)
            cvd_task = asyncio.create_task(self.feeds.get_cvd_with_cb_fallback())
            # 5. Micro book
            book_task = asyncio.create_task(self._fetch_binance_book())

            # Wait for all
            (market_info, strike_info, k5m, k1m, 
             cvd_data_combined, book) = await asyncio.gather(
                pm_task, strike_task, k5m_task, k1m_task, cvd_task, book_task
            )
            
            self.state.locked_strike_price = strike_info["strike"]
            self.state.strike_source       = strike_info["source"]
            cvd_res, cvd_bin, cvd_cb = cvd_data_combined

        if self.state.locked_strike_price is None:
            log.warning(
                "Strike unresolved for window_start=%s (source=%s) — posteriors/edge will be N/A",
                win_start,
                self.state.strike_source,
            )

        if not market_info:
            log.warning("No market info available — skipping cycle")
            await self.state_mgr.save(self.state)
            return

        # ── Polymarket context (Phase 4: Latency track) ──────────────────────
        with Timer("fetch_polymarket", self.state.latencies):
            if self.pm.can_trade and not self.state.trading_halted:
                ob, margin, wallet_usdc, positions = await asyncio.gather(
                    self.pm.get_order_books(market_info.yes_token_id, market_info.no_token_id),
                    self.pm.get_margin(),
                    self.pm.get_wallet_usdc_balance(),
                    self.pm.get_positions(),
                )
            else:
                ob = await self.pm.get_order_books(market_info.yes_token_id, market_info.no_token_id)
                margin = {"balance_usdc": None, "allowance_usdc": None, "available_usdc": None}
                wallet_usdc = None
                positions = []

        # ── Reconcile Pending Orders (Phase 3) ────────────────────────────────
        # (Already done above, so we keep the order consistent)


        # Compute indicators locally
        indic = compute_local_indicators(k5m, k1m)

        # Dedicated ATR override (fixes 0.0 ATR from pandas pipeline)
        if indic.atr14 is None or indic.atr14 <= 0:
            indic.atr14 = await self.feeds.calculate_atr_binance()

        # Dedicated MACD override (fixes 0.0 MACD from pandas pipeline)
        if indic.macd_hist is None or indic.macd_hist == 0:
            indic.macd_hist = await self.feeds.calculate_macd_histogram()

        # Cache market info
        self.state.last_market_slug      = market_info.slug
        self.state.last_condition_id     = market_info.condition_id
        self.state.last_market_expiry    = market_info.expiry_ts

        # ── Fetch order book + balance (parallel) ─────────────────────────────
        # This section is now part of the "fetch_polymarket" Timer block above.
        # ob_task  = asyncio.create_task(
        #     self.pm.get_order_books(market_info.yes_token_id, market_info.no_token_id)
        # )
        # bal_task = asyncio.create_task(self.pm.get_balance())
        # pos_task = asyncio.create_task(self.pm.get_positions())
        # ob, balance, positions = await asyncio.gather(ob_task, bal_task, pos_task)

        if self.pm.can_trade and not self.state.trading_halted:
            # Priority: direct on-chain wallet > CLOB margin available > CLOB margin balance
            balance = wallet_usdc
            if balance is None or balance <= 1e-6:
                balance = (margin or {}).get("available_usdc")
            if balance is None or balance <= 1e-6:
                balance = (margin or {}).get("balance_usdc")
            if balance is None or balance <= 1e-6:
                log.warning(
                    "BALANCE_ZERO: wallet_usdc=%s margin=%s — check POLYGON_RPC_URL and USDC.e on Polygon",
                    wallet_usdc, margin,
                )
        else:
            balance = None

        balance = balance or 0.0
        if ob.yes_mid:  self.state.last_pm_px_yes = ob.yes_mid
        if ob.no_mid:   self.state.last_pm_px_no  = ob.no_mid

        # ── Reconcile held position from live API ──────────────────────────────
        for pos in positions:
            cid = pos.get("conditionId") or pos.get("condition_id")
            if cid != market_info.condition_id:
                continue
            outcome = (pos.get("outcome") or "").lower()
            bal_pos = float(pos.get("balance") or 0)
            if bal_pos <= 0:
                continue
            side = "YES" if "yes" in outcome or "up" in outcome else "NO"
            if self.state.held_position.side is None:
                self.state.held_position.side      = side
                self.state.held_position.token_id  = (
                    market_info.yes_token_id if side == "YES" else market_info.no_token_id
                )
                self.state.held_position.size      = bal_pos
                self.state.held_position.condition_id = market_info.condition_id

        # ── Current BTC price ─────────────────────────────────────────────────
        btc_price = indic.close or book.mid
        if btc_price == 0.0 or btc_price is None:
            btc_price = await self.feeds.get_btc_price()
        
        btc_price = btc_price or self.state.prev_price or 0.0
        if btc_price == 0.0:
            log.warning("No BTC price available — skipping cycle")
            await self.state_mgr.save(self.state)
            return

        # ── Real CVD: prefer WebSocket, fall back to REST ─────────────────────
        ws_cvd, ws_buy, ws_sell = self.cvd_ws.get_volumes()
        ws_total = ws_buy + ws_sell
        if ws_total > 0:
            # WebSocket has data — use it as the primary source
            self.state.cvd = ws_cvd
            self.state.prev_cvd_slope = self.cvd_ws.get_cvd_slope()
            cvd_total_vol = ws_total
            cvd_delta_for_signals = self.state.prev_cvd_slope
            log.debug(f"CVD_WS: cvd={ws_cvd:.2f} slope={self.state.prev_cvd_slope:.2f} buy={ws_buy:.2f} sell={ws_sell:.2f}")
        else:
            # Fallback to REST-based CVD
            self.state.cvd         += cvd_res.cvd_delta
            self.state.prev_cvd_slope = cvd_res.cvd_delta
            cvd_total_vol = cvd_res.buy_vol + cvd_res.sell_vol
            cvd_delta_for_signals = cvd_res.cvd_delta

        # ── Phase 2: Cross-exchange CVD agreement ────────────────────────────
        bin_dir = 1 if cvd_delta_for_signals > 0 else (-1 if cvd_delta_for_signals < 0 else 0)
        cb_dir  = 1 if cvd_cb.cvd_delta > 0 else (-1 if cvd_cb.cvd_delta < 0 else 0)
        self.state.cross_cvd_agree = (bin_dir == cb_dir) or bin_dir == 0 or cb_dir == 0

        # ── Phase 2: Accumulate OFI within window ────────────────────────────
        if not book.is_stale:
            self.state.accumulated_ofi += book.deep_ofi

        # ── Compute signals ───────────────────────────────────────────────────
        is_stale = book.is_stale

        sig = compute_signals(
            indic             = indic,
            btc_price         = btc_price,
            minutes_remaining = min_rem,
            now_ts            = now_ts,
            state             = self.state,
            strike            = self.state.locked_strike_price,
            strike_source     = self.state.strike_source or "none",
            bid_depth20       = book.bid_depth20,
            ask_depth20       = book.ask_depth20,
            deep_imbalance    = book.deep_imbalance,
            vpin_proxy        = book.vpin_proxy,
            deep_ofi          = book.deep_ofi,
            microprice        = book.microprice,
            is_stale_micro    = is_stale,
            cvd_delta         = cvd_delta_for_signals,
            true_cvd          = self.state.cvd,
            accumulated_ofi   = self.state.accumulated_ofi,
            cross_cvd_agree   = self.state.cross_cvd_agree,
            cvd_total_vol     = cvd_total_vol,
            prev_cvd_total_vol = self.state.prev_cvd_total_vol,
            yes_mid           = ob.yes_mid,
            no_mid            = ob.no_mid,
            yes_ask           = ob.yes_ask,
            no_ask            = ob.no_ask,
            total_bid_size    = ob.total_bid_size,
            total_ask_size    = ob.total_ask_size,
            inference_engine  = self.inference,
        )

        # Store CVD total vol for next cycle's volume-weighted scoring
        self.state.prev_cvd_total_vol = cvd_total_vol

        # Update score/posterior memory
        if not is_stale:
            self.state.last_cvd_score        = sig.cvd_score
            self.state.last_ofi_score        = sig.ofi_score
            self.state.last_imbalance_score  = sig.imbalance_score
            self.state.last_flow_accel_score = sig.flow_accel_score
            self.state.last_posterior_up     = sig.posterior_final_up
            self.state.last_posterior_down   = sig.posterior_final_down

        # Update price/indicator memory (shift back)
        self.state.prev_price3 = self.state.prev_price2
        self.state.prev_price2 = self.state.prev_price
        self.state.prev_price  = btc_price
        self.state.prev_mfi    = indic.mfi14
        self.state.prev_obv4   = self.state.prev_obv3
        self.state.prev_obv3   = self.state.prev_obv2
        self.state.prev_obv2   = self.state.prev_obv
        self.state.prev_obv    = indic.obv
        self.state.prev_ofi_recent = book.deep_ofi

        # ── Exit handling ─────────────────────────────────────────────────────
        # Phase 3: pass CVD and posteriors for new exit types
        exit_executed = await self._handle_exits(
            sig, ob, market_info, min_rem, btc_price, balance, cvd_res.cvd_delta
        )

        # ── Entry handling ─────────────────────────────────────────────────────
        if not exit_executed and not self.state.trading_halted:
            await self._handle_entry(
                sig, ob, market_info, min_rem, btc_price, balance
            )

        # ── Halt check ────────────────────────────────────────────────────────
        if self.state.loss_streak >= Config.STREAK_HALT_AT and not self.state.trading_halted:
            self.state.trading_halted = True
            log.error(f"TRADING HALTED: {self.state.loss_streak} consecutive losses")
            await send_telegram(
                self.feeds.session,
                fmt_halt(self.state.loss_streak, balance),
            )

        # ── Periodic status message ────────────────────────────────────────────
        self._status_counter += 1
        if self._status_counter % 20 == 0:   # every ~5 min
            pos_str = (
                f"{self.state.held_position.side} @ {self.state.held_position.avg_entry_price:.3f}"
                if self.state.held_position.side else "FLAT"
            )
            await send_telegram(
                self.feeds.session,
                fmt_status(
                    pos_str, sig.signed_score,
                    sig.posterior_final_up or 0.0,
                    ob.yes_mid or 0.0, ob.no_mid or 0.0,
                    balance, min_rem, sig.skip_gates,
                    question=market_info.question,
                    url=market_info.url,
                )
            )

        # ── Dashboard Logging (Phase 4) ───────────────────────────────────────
        runtime_ms = int(time.time() * 1000) - start_time_ms
        self.state.latencies["total_cycle"] = runtime_ms
        
        # Phase 5: Log features for ML training every cycle
        await self._log_cycle_features(sig, btc_price, min_rem, now_ts)

        # PnL Dashboard (console only)
        if self._status_counter % 4 == 0: # Print every ~1 min
            print(fmt_pnl_dashboard(self.state.trade_history, balance))

        # Heartbeat file
        await self._write_heartbeat(now_ts, balance, runtime_ms, margin=margin, wallet_usdc=wallet_usdc, sig=sig)

        # ── Outcome logging for previous window ───────────────────────────────
        if win_rolled:
            # We just moved into a new window; log the outcome of the one that just finished
            # Using prev_win_start which we should track
            prev_win = win_start - Config.WINDOW_SEC
            asyncio.create_task(self._log_window_outcome(prev_win))

        # Calculate sizing for reporting if flat
        if not self.state.held_position.side:
            p_up = sig.posterior_final_up or 0.5
            p_dn = sig.posterior_final_down or 0.5
            px_up = ob.yes_ask or 0.99
            px_dn = ob.no_ask or 0.99
            
            s_up = compute_position_size(posterior=p_up, entry_price=px_up, balance=balance, loss_streak=self.state.loss_streak, monster_signal=sig.monster_signal) or 0.0
            s_down = compute_position_size(posterior=p_dn, entry_price=px_dn, balance=balance, loss_streak=self.state.loss_streak, monster_signal=sig.monster_signal) or 0.0
            sig.sizing = max(s_up, s_down)
        else:
            sig.sizing = self.state.held_position.size_usd or 0.0

        dashboard = fmt_engine_block(
            res=sig, state=self.state, btc_price=btc_price, 
            min_rem=min_rem, balance=balance, runtime_ms=runtime_ms,
            decision="NO_TRADE" if not exit_executed else "EXIT",
            exec_bool=exit_executed,
            mode="none" if not self.state.held_position.side else self.state.held_position.side,
            exit_reason=str(getattr(self, "_last_exit_reason", "HOLD"))
        )
        log.info(dashboard)

        # Update cycle memory for deltas
        self.state.prev_cycle_score = sig.signed_score
        self.state.prev_cycle_price = btc_price

        # ── Save state ────────────────────────────────────────────────────────
        await self.state_mgr.save(self.state)

    # ── Strike resolution (FIX #2) ────────────────────────────────────────────

    async def _resolve_strike(self, win_start: int) -> dict:
        win_start_ms = win_start * 1000

        # Priority 1: Binance 15m kline open
        try:
            p = await self.feeds.get_binance_15m_open(win_start_ms)
            if p:
                return {"strike": p, "source": "binance_15m_open"}
        except Exception as e:
            log.debug(f"Strike P1 (Binance 15m) failed: {e}")

        # Priority 2: Coinbase 15m candle open
        try:
            start_iso = window_start_iso(win_start)
            end_iso   = window_end_iso(win_start)
            p = await self.feeds.get_coinbase_15m_open(start_iso, end_iso)
            if p:
                return {"strike": p, "source": "coinbase_15m_open"}
        except Exception as e:
            log.debug(f"Strike P2 (Coinbase 15m) failed: {e}")

        # Priority 3: First 5m kline that overlaps the window start
        #   We already fetch 5m klines every cycle — use the one closest to window open
        try:
            klines_5m = await self.feeds.get_klines("BTCUSDT", "5m", 5)
            if klines_5m:
                for c in klines_5m:
                    if c.ts_ms >= win_start_ms:
                        return {"strike": c.open, "source": "binance_5m_open"}
                # If no kline exactly at window start, use the latest kline's close
                return {"strike": klines_5m[-1].close, "source": "binance_5m_close"}
        except Exception as e:
            log.debug(f"Strike P3 (Binance 5m) failed: {e}")

        # Priority 4: Previous book mid price
        if self.state.prev_hl_mid:
            return {"strike": self.state.prev_hl_mid, "source": "binance_mid_prev"}

        # Priority 5: Live BTC price (last resort)
        try:
            p = await self.feeds.get_btc_price()
            if p:
                log.warning(f"Strike fallback to live BTC price: ${p:.2f}")
                return {"strike": p, "source": "live_price_fallback"}
        except Exception as e:
            log.debug(f"Strike P5 (live price) failed: {e}")

        return {"strike": None, "source": "none"}

    # ── Binance book fetch ─────────────────────────────────────────────────────

    async def _fetch_binance_book(self):
        """
        Fetch Binance L2 book. We no longer use a strict cooldown like HL 
        since Binance limits are much more generous for depth polling.
        """
        book = await self.feeds.get_binance_book(
            symbol           = "BTCUSDT",
            prev_bid_depth20 = self.state.prev_bid_depth20,
            prev_ask_depth20 = self.state.prev_ask_depth20,
        )

        if not book.is_stale:
            self.state.prev_bid_depth20    = book.bid_depth20
            self.state.prev_ask_depth20    = book.ask_depth20
            self.state.prev_deep_imbalance = book.deep_imbalance
            self.state.last_vpin_proxy     = book.vpin_proxy
            self.state.prev_hl_mid         = book.mid
            self.state.prev_bid_px         = book.best_bid_px
            self.state.prev_ask_px         = book.best_ask_px
            self.state.prev_bid_sz         = book.best_bid_sz
            self.state.prev_ask_sz         = book.best_ask_sz
            self.state.last_hl_fetch_ts    = int(time.time())

        return book

    async def _reconcile_pending_order(self):
        """Phase 3: Verify if a pending order has filled."""
        pos = self.state.held_position
        if not pos.order_id:
            return

        status = await self.pm.get_order_status(pos.order_id)
        if not status:
            return

        st = status.get("status")
        matched = status.get("size_matched", 0)

        if st == "FILLED":
            log.info(f"PENDING ORDER FILLED: {pos.order_id} ({matched} shares)")
            pos.is_pending = False
        elif st == "CANCELED" or st == "EXPIRED":
            if matched > 0:
                log.info(f"PENDING ORDER {st} PARTIALLY: {matched}/{pos.size} shares filled")
                pos.size = matched
                pos.is_pending = False
            else:
                log.info(f"PENDING ORDER {st}: clearing position state")
                self.state.held_position = HeldPosition()
        elif matched > 0 and matched >= pos.size:
            # Fallback for status strings that might differ
            log.info(f"PENDING ORDER MATCHED: {matched} shares")
            pos.is_pending = False

    async def _write_heartbeat(self, ts: int, balance: float, runtime_ms: int, margin: dict | None = None, wallet_usdc: float | None = None, sig = None):
        """Phase 4: Structured heartbeat for external health monitoring."""
        hb = {
            "ts": ts,
            "status": "HEALTHY",
            "balance": balance,
            "wallet_usdc": wallet_usdc,
            "pm_collateral_usdc": (margin or {}).get("balance_usdc") if isinstance(margin, dict) else None,
            "pm_available_usdc": (margin or {}).get("available_usdc") if isinstance(margin, dict) else None,
            "pm_allowance_usdc": (margin or {}).get("allowance_usdc") if isinstance(margin, dict) else None,
            "position": self.state.held_position.side or "FLAT",
            "runtime_ms": runtime_ms,
            "latencies": self.state.latencies,
            "trades_15m": self.state.trades_this_window,
            "last_market_slug": self.state.last_market_slug,
            "last_condition_id": self.state.last_condition_id,
            "unclaimed_usdc": self.state.unclaimed_usdc,
            "api_trade_history": getattr(self, "api_trade_history", []),
        }
        if sig:
            hb["signal"] = sig.to_feature_dict()
            hb["signal"]["regime"] = sig.regime
            hb["signal"]["direction"] = sig.direction
            hb["signal"]["edge_up"] = sig.edge_up
            hb["signal"]["edge_down"] = sig.edge_down
            hb["signal"]["target_edge"] = sig.target_edge
            hb["signal"]["target_side"] = sig.target_side
            hb["signal"]["required_edge"] = sig.required_edge
            hb["signal"]["min_score"] = sig.min_score
            hb["signal"]["skip_gates"] = sig.skip_gates
            hb["signal"]["strike_price"] = sig.strike_price
            hb["signal"]["distance"] = sig.distance
            # Raw micro values for dashboard (continuous, not discretized)
            hb["signal"]["raw_deep_ofi"] = sig.deep_ofi
            hb["signal"]["raw_cvd"] = sig.cvd
            hb["signal"]["raw_vpin"] = sig.vpin_proxy
            hb["signal"]["raw_deep_imbalance"] = sig.deep_imbalance
            hb["signal"]["raw_obi"] = sig.obi
            hb["signal"]["monster_signal"] = sig.monster_signal
            hb["signal"]["posterior_final_up"] = sig.posterior_final_up
            hb["signal"]["posterior_final_down"] = sig.posterior_final_down

        # ── Real performance metrics from trade history ──────────────
        trades = self.state.trade_history
        total_trades = len(trades)
        closed = [t for t in trades if t.outcome in ("WIN", "LOSS")]
        wins = [t for t in closed if t.outcome == "WIN"]
        losses = [t for t in closed if t.outcome == "LOSS"]
        win_rate = len(wins) / len(closed) if closed else 0.0
        pnl_list = [t.pnl for t in closed if t.pnl is not None]
        avg_pnl = sum(pnl_list) / len(pnl_list) if pnl_list else 0.0
        total_pnl = sum(pnl_list)
        # Sharpe: mean(pnl) / std(pnl) * sqrt(N)  (annualization not meaningful for 15m)
        import statistics
        sharpe = 0.0
        if len(pnl_list) >= 2:
            try:
                sharpe = (statistics.mean(pnl_list) / statistics.stdev(pnl_list)) * (len(pnl_list) ** 0.5)
            except (ZeroDivisionError, statistics.StatisticsError):
                sharpe = 0.0
        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        hb["performance"] = {
            "total_trades": total_trades,
            "closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "avg_trade": avg_pnl if closed else 0.0,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "loss_streak": self.state.loss_streak,
        }
        
        try:
            hb["perf_db"] = await self.state_mgr.calculate_performance_metrics()
        except Exception as e:
            log.error(f"Failed to calc metrics: {e}")
            hb["perf_db"] = {
                "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "sharpe_ratio": 0.0, "avg_pnl_per_trade": 0.0
            }

        # ── Recent events for the event stream ──────────────────────
        recent_events = []
        for t in reversed(trades[-10:]):
            from datetime import datetime, timezone
            ts_str = datetime.fromtimestamp(t.ts, tz=timezone.utc).strftime("%H:%M:%S")
            if t.outcome:
                pnl_str = f"{(t.pnl or 0)*100:.2f}%"
                exit_str = f"{t.exit_price:.3f}" if t.exit_price is not None else "N/A"
                entry_str = f"{t.entry_price:.3f}" if t.entry_price is not None else "N/A"
                recent_events.append({
                    "ts": ts_str,
                    "type": "trade",
                    "msg": f"Trade {t.side} closed: {t.outcome} | Entry={entry_str} Exit={exit_str} PnL={pnl_str}"
                })
            else:
                recent_events.append({
                    "ts": ts_str,
                    "type": "trade",
                    "msg": f"Trade {t.side} OPEN | Entry={t.entry_price:.3f}" if t.entry_price else f"Trade {t.side} OPEN"
                })
        hb["events"] = recent_events

        try:
            # Write to /data if on Railway, otherwise local
            hb_path = "/data/heartbeat.json" if os.path.exists("/data") else "heartbeat.json"
            async with aiofiles.open(hb_path, mode='w') as f:
                await f.write(json.dumps(hb, indent=2))
        except Exception as e:
            log.debug(f"Heartbeat write aborted: {e}")

    # ── Exit handler ──────────────────────────────────────────────────────────

    async def _handle_exits(self, sig, ob, market_info, min_rem, btc_price, balance, cvd_delta=0.0) -> bool:
        pos = self.state.held_position
        if not pos.side or not pos.token_id or not pos.size or pos.size <= 0:
            return False

        # Phase 3: Don't exit while pending (wait for fill/cancel)
        if pos.is_pending:
            return False

        current_px = ob.yes_mid if pos.side == "YES" else ob.no_mid
        entry_px   = pos.avg_entry_price or pos.entry_price or (current_px or 0)

        # Get previous posterior for decay detection
        prev_post = self.state.last_posterior_up if pos.side == "YES" else self.state.last_posterior_down
        curr_post = sig.posterior_final_up if pos.side == "YES" else sig.posterior_final_down

        reason = evaluate_exit(
            held_side         = pos.side,
            entry_price       = entry_px,
            current_price     = current_px,
            minutes_remaining = min_rem,
            signed_score      = sig.signed_score,
            entry_score       = pos.entry_signed_score or 0.0,
            distance          = sig.distance,
            cvd_delta         = cvd_delta,
            posterior         = curr_post,
            prev_posterior    = prev_post,
        )
        if not reason:
            return False

        # Cancel open limit orders first
        if self.state.last_condition_id:
            for oid in await self.pm.get_open_orders(self.state.last_condition_id):
                await self.pm.cancel_order(oid)

        # Execute exit — use bid price for FOK exits (ensures fill on thin books)
        # For GTC exits, mid is acceptable as a limit price
        exit_bid = (ob.yes_bid if pos.side == "YES" else ob.no_bid) or current_px or entry_px
        exit_px_gtc = current_px or entry_px
        # Clamp to Polymarket valid price range [0.01, 0.99]
        exit_bid = max(0.01, min(0.99, exit_bid))
        exit_px_gtc = max(0.01, min(0.99, exit_px_gtc))
        if reason in ("FORCED_DRAWDOWN", "ALPHA_DECAY", "FORCED_LATE_EXIT"):
            order_id = await self.pm.limit_sell(pos.token_id, exit_bid, pos.size, order_type="FOK")
        else:
            order_id = await self.pm.limit_sell(pos.token_id, exit_px_gtc, pos.size, order_type="GTC")
        exit_px = exit_bid if reason in ("FORCED_DRAWDOWN", "ALPHA_DECAY", "FORCED_LATE_EXIT") else exit_px_gtc

        if not order_id:
            log.error(f"Exit order failed ({reason})")
            return False

        pnl_pct = (exit_px - entry_px) / entry_px if entry_px > 0 else 0.0

        # Loss tracking
        if pnl_pct < 0:
            self.state.loss_streak += 1
        else:
            self.state.loss_streak = 0

        # Update trade history
        outcome = "WIN" if pnl_pct >= 0 else "LOSS"
        for tr in reversed(self.state.trade_history):
            if tr.side == pos.side and tr.outcome == "OPEN":
                tr.exit_price = exit_px
                tr.pnl        = pnl_pct
                tr.outcome    = outcome
                break

        await send_telegram(
            self.feeds.session,
            fmt_exit(pos.side, exit_px, entry_px, pnl_pct, reason, balance),
        )
        log.info(f"EXIT: {pos.side} {reason} pnl={pnl_pct*100:.2f}%")

        # Clear position
        self.state.held_position = HeldPosition()
        self.state.total_trades += 1
        if outcome == "WIN":
            self.state.total_wins += 1
        else:
            self.state.total_losses += 1
            
        return True

    # ── Entry handler ─────────────────────────────────────────────────────────

    async def _handle_entry(self, sig, ob, market_info, min_rem, btc_price, balance):
        if self.state.held_position.side is not None:
            return   # already holding

        # ── Hard capital protections ──────────────────────────────────────────
        if Config.KILL_SWITCH:
            log.warning("KILL_SWITCH is active — no new entries")
            return

        if self.state.trades_this_window >= Config.MAX_TRADES_PER_WINDOW and not sig.monster_signal:
            return

        # Daily loss limit: sum realized losses from today's trades
        today_start = int(time.time()) - 86400
        daily_loss = sum(
            abs(t.pnl or 0) * (t.entry_price or 0) * (getattr(t, 'size', 0) or 1)
            for t in self.state.trade_history
            if t.ts >= today_start and t.outcome == "LOSS" and t.pnl is not None
        )
        if daily_loss >= Config.DAILY_LOSS_LIMIT_USD:
            log.warning("Daily loss limit reached ($%.2f >= $%.2f) — no new entries", daily_loss, Config.DAILY_LOSS_LIMIT_USD)
            return

        if sig.skip_gates:
            log.debug(f"Entry blocked: {sig.skip_gates}")
            return

        if sig.direction == "NEUTRAL":
            return

        # Determine token + price (Phase 3: Smart Pricing)
        if sig.direction == "UP":
            token_id  = market_info.yes_token_id
            side_name = "YES"
            entry_px  = self.pm.smart_entry_price(ob.yes_bid, ob.yes_ask) or 0.0
            posterior = sig.posterior_final_up or 0.5
        else:
            token_id  = market_info.no_token_id
            side_name = "NO"
            entry_px  = self.pm.smart_entry_price(ob.no_bid, ob.no_ask) or 0.0
            posterior = sig.posterior_final_down or 0.5

        if entry_px <= 0 or entry_px >= 1.0:
            return

        # Size (FIX #8: Kelly with riskPct floor)
        position_usd = compute_position_size(
            posterior   = posterior,
            entry_price = entry_px,
            balance     = balance,
            loss_streak = self.state.loss_streak,
            monster_signal = sig.monster_signal,
        )
        if not position_usd:
            log.info(f"Position size below minimum (bal={balance:.2f})")
            return

        shares = round(position_usd / entry_px, 2)
        if shares < 1:
            return

        # Place order
        if sig.monster_signal:
            order_id = await self.pm.limit_buy(token_id, entry_px, shares, order_type="FOK")
        else:
            order_id = await self.pm.limit_buy(token_id, entry_px, shares, order_type="GTC")

        if not order_id:
            log.error("Entry order failed")
            return

        # Phase 3: record as PENDING and track order_id
        self.state.held_position = HeldPosition(
            side               = side_name,
            token_id           = token_id,
            entry_price        = entry_px,
            avg_entry_price    = entry_px,
            size               = shares,
            size_usd           = position_usd,
            entry_signed_score = sig.signed_score,
            condition_id       = market_info.condition_id,
            order_id           = order_id,
            is_pending         = True,
            placed_at_ts       = int(time.time()),
        )
        self.state.trades_this_window += 1

        log.info(f"Order placed: {order_id} — waiting for fill confirmation.")

        # Record trade history
        self.state.trade_history.append(TradeRecord(
            ts          = int(time.time()),
            side        = side_name,
            entry_price = entry_px,
            exit_price  = None,
            pnl         = None,
            outcome     = "OPEN",
            score       = sig.signed_score,
            window      = self.state.last_window_start_sec or 0,
            size        = shares, # store the size for PnL calcs
        ))
        if len(self.state.trade_history) > 100:
            self.state.trade_history = self.state.trade_history[-100:]

        await send_telegram(
            self.feeds.session,
            fmt_entry(
                side_name, entry_px, shares, sig.signed_score,
                sig.target_edge or 0.0, posterior,
                self.state.last_window_start_sec or 0, balance,
            )
        )
    async def _log_cycle_features(self, sig, btc_price: float, min_rem: float, ts: int):
        """Phase 5: Log all signals/features to JSONL every cycle."""
        try:
            feats = sig.to_feature_dict()
            feats.update({
                "ts":         ts,
                "btc_price":  round(btc_price, 2),
                "min_rem":    round(min_rem, 2),
                "window":     self.state.last_window_start_sec,
                "strike":     self.state.locked_strike_price,
            })
            log_dir = "/data" if os.path.exists("/data") else "./logs"
            os.makedirs(log_dir, exist_ok=True)
            async with aiofiles.open(f"{log_dir}/features.jsonl", mode='a') as f:
                await f.write(json.dumps(feats) + "\n")
        except Exception as e:
            log.debug(f"Feature logging error: {e}")

    async def _log_window_outcome(self, win_start: int):
        """Phase 5: Record the final BTC price at window end for labeling."""
        try:
            # Wait 30s for the candle to definitely settle
            await asyncio.sleep(30)
            
            # Fetch the candle that just closed (starting at win_start, ending +15m)
            # Binance klines are inclusive of start
            cand = await self.feeds.get_binance_15m_open(win_start * 1000)
            if not cand:
                # Fallback to 1m kline to find close at win_start + 15m
                k1m = await self.feeds.get_klines("BTCUSDT", "1m", 1)
                if k1m: cand = k1m[0].close
            
            if cand:
                outcome_data = {
                    "window_start": win_start,
                    "btc_close":    round(cand, 2),
                    "ts_logged":    int(time.time()),
                }
                log_dir = "/data" if os.path.exists("/data") else "./logs"
                async with aiofiles.open(f"{log_dir}/outcomes.jsonl", mode='a') as f:
                    await f.write(json.dumps(outcome_data) + "\n")
                log.info(f"Outcome logged for window {win_start}: BTC closed at {cand:.2f}")

                # Record any expired out-of-the-money trades as losses in the DB
                for tr in reversed(self.state.trade_history):
                    if tr.window == win_start and tr.outcome == "OPEN":
                        tr.outcome = "LOSS"
                        tr.exit_price = 0.0
                        tr.pnl = -1.0
                        try:
                            size = self.state.last_sizing or 0.0
                            await self.state_mgr.record_closed_trade(
                                ts=int(time.time()),
                                market_slug=self.state.last_market_slug or "unknown",
                                size=size,
                                entry_price=tr.entry_price,
                                exit_price=0.0,
                                pnl_usd=-size,
                                outcome_win=0
                            )
                        except Exception as e:
                            log.error(f"Failed to record expired loss to DB: {e}")
        except Exception as e:
            log.debug(f"Outcome logging error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main():
    engine = Engine()

    # Handle --reset flag to clear state
    if "--reset" in sys.argv:
        log.info("Resetting state database...")
        import os
        db_file = "state.db"
        if os.path.exists(db_file):
            os.remove(db_file)
            log.info(f"Deleted {db_file}")

    def _shutdown(signum, frame):
        log.info(f"Received signal {signum} — shutting down...")
        engine._running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    await engine.run()
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(_main())
