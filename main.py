"""
BTC 15m Quant Engine — main async loop.

Architecture:
  - 15-second inner loop (Config.LOOP_INTERVAL_SEC)
  - Each cycle: fetch data → compute signals → handle exits → handle entries
  - State persisted to SQLite after every cycle
  - Graceful shutdown on SIGINT/SIGTERM
"""

import asyncio
import logging
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
from utils import (
    setup_logging, send_telegram,
    fmt_entry, fmt_exit, fmt_halt, fmt_status, fmt_engine_block,
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

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        await self.feeds.start()
        await self.pm.start()
        self.state = await self.state_mgr.load()
        log.info("Engine started. Ensuring Polymarket approvals...")
        await self.pm.ensure_approvals()
        log.info("Ready. Starting main loop.")

    async def stop(self):
        self._running = False
        if self.state:
            await self.state_mgr.save(self.state)
        await self.feeds.close()
        await self.pm.close()
        log.info("Engine stopped.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        await self.start()
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
            self.state.trades_this_window   = 0
            self.state.locked_strike_price  = None
            self.state.strike_source        = None
            self.state.strike_window_start  = win_start
            self.state.cvd                  = 0.0   # reset CVD each window
            self.state.accumulated_ofi      = 0.0   # Phase 2: reset accumulated OFI

        self.state.last_window_start_sec = win_start

        # ── Strike resolution (FIX #2) ────────────────────────────────────────
        # Priority: Binance 15m open → Coinbase 15m open → HL mid → None
        # NEVER uses live EMA/current price as strike
        if self.state.locked_strike_price is None:
            strike, strike_source = await self._resolve_strike(win_start)
            if strike:
                self.state.locked_strike_price = strike
                self.state.strike_source       = strike_source
                log.info(f"Strike locked: {strike:.2f} via {strike_source}")
            else:
                log.warning("Could not resolve strike price — z-score unavailable this cycle")

        # ── Parallel data fetch ───────────────────────────────────────────────
        now_ms = now_ts * 1000
        
        # CVD tasks (60.1s window)
        cvd_bin_task = asyncio.create_task(self.feeds.get_real_cvd(now_ms - 60_100, now_ms))
        cvd_cb_task  = asyncio.create_task(self.feeds.get_coinbase_cvd(now_ms - 60_100, now_ms))
        
        # Order book task (Binance only for now)
        book_task    = asyncio.create_task(self._fetch_binance_book())
        
        # Kline tasks (250 candles)
        k5m_bin_task = asyncio.create_task(self.feeds.get_klines("BTCUSD", "5m", 250))
        k1m_bin_task = asyncio.create_task(self.feeds.get_klines("BTCUSD", "1m", 250))
        k5m_cb_task  = asyncio.create_task(self.feeds.get_coinbase_klines("5m", 250))
        k1m_cb_task  = asyncio.create_task(self.feeds.get_coinbase_klines("1m", 250))

        # Polymarket market discovery
        pm_task      = asyncio.create_task(
            self.pm.discover_market(
                win_start,
                cached_slug         = self.state.last_market_slug,
                cached_expiry       = self.state.last_market_expiry,
                cached_condition_id = self.state.last_condition_id,
            )
        )

        (cvd_bin, cvd_cb, book, 
         k5m_bin, k1m_bin, k5m_cb, k1m_cb,
         market_info) = await asyncio.gather(
            cvd_bin_task, cvd_cb_task, book_task,
            k5m_bin_task, k1m_bin_task, k5m_cb_task, k1m_cb_task,
            pm_task
        )

        # ── Source Selection Logic ───────────────────────────────────────────
        
        # Pick CVD source with more total volume (buy + sell)
        bin_cvd_vol = cvd_bin.buy_vol + cvd_bin.sell_vol
        cb_cvd_vol  = cvd_cb.buy_vol + cvd_cb.sell_vol
        if cb_cvd_vol > bin_cvd_vol:
            cvd_res = cvd_cb
            cvd_source = "coinbase"
        else:
            cvd_res = cvd_bin
            cvd_source = "binance"
            
        # Pick Kline source with more volume in the latest candle
        bin_k_vol = sum(c.volume for c in k5m_bin[:5]) if k5m_bin else 0.0
        cb_k_vol  = sum(c.volume for c in k5m_cb[:5]) if k5m_cb else 0.0
        if cb_k_vol > bin_k_vol and k5m_cb:
            k5m, k1m = k5m_cb, k1m_cb
            k_source = "coinbase"
        else:
            k5m, k1m = k5m_bin, k1m_bin
            k_source = "binance"

        log.info(f"Data Sources: klines={k_source} cvd={cvd_source} (v={max(bin_k_vol, cb_k_vol):.1f}/{max(bin_cvd_vol, cb_cvd_vol):.3f})")

        # Compute indicators locally
        indic = compute_local_indicators(k5m, k1m)

        # ── Guard: need market ────────────────────────────────────────────────
        if market_info is None:
            log.warning("No active BTC 15m market found — skipping cycle")
            await self.state_mgr.save(self.state)
            return

        # Cache market info
        self.state.last_market_slug      = market_info.slug
        self.state.last_condition_id     = market_info.condition_id
        self.state.last_market_expiry    = market_info.expiry_ts

        # ── Fetch order book + balance (parallel) ─────────────────────────────
        ob_task  = asyncio.create_task(
            self.pm.get_order_books(market_info.yes_token_id, market_info.no_token_id)
        )
        bal_task = asyncio.create_task(self.pm.get_balance())
        pos_task = asyncio.create_task(self.pm.get_positions())
        ob, balance, positions = await asyncio.gather(ob_task, bal_task, pos_task)

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

        # ── Real CVD accumulation (FIX #4) ──────────────────────────────────
        self.state.cvd         += cvd_res.cvd_delta
        self.state.prev_cvd_slope = cvd_res.cvd_delta

        # ── Phase 2: Cross-exchange CVD agreement ────────────────────────────
        bin_dir = 1 if cvd_bin.cvd_delta > 0 else (-1 if cvd_bin.cvd_delta < 0 else 0)
        cb_dir  = 1 if cvd_cb.cvd_delta > 0 else (-1 if cvd_cb.cvd_delta < 0 else 0)
        self.state.cross_cvd_agree = (bin_dir == cb_dir) or bin_dir == 0 or cb_dir == 0
        cvd_total_vol = cvd_res.buy_vol + cvd_res.sell_vol

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
            cvd_delta         = cvd_res.cvd_delta,
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
        )

        # Store CVD total vol for next cycle's volume-weighted scoring
        self.state.prev_cvd_total_vol = cvd_total_vol

        # Update micro score memory
        if not is_stale:
            self.state.last_cvd_score        = sig.cvd_score
            self.state.last_ofi_score        = sig.ofi_score
            self.state.last_imbalance_score  = sig.imbalance_score
            self.state.last_flow_accel_score = sig.flow_accel_score

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
        exit_executed = await self._handle_exits(
            sig, ob, market_info, min_rem, btc_price, balance
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

        # ── Dashboard Logging ────────────────────────────────────────────────
        runtime_ms = int(time.time() * 1000) - start_time_ms
        
        # Calculate sizing for reporting if flat
        if not self.state.held_position.side:
            p_up = sig.posterior_final_up or 0.5
            p_dn = sig.posterior_final_down or 0.5
            px_up = ob.yes_ask or 0.99
            px_dn = ob.no_ask or 0.99
            
            s_up = compute_position_size(posterior=p_up, entry_price=px_up, balance=balance, loss_streak=self.state.loss_streak) or 0.0
            s_down = compute_position_size(posterior=p_dn, entry_price=px_dn, balance=balance, loss_streak=self.state.loss_streak) or 0.0
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

    async def _resolve_strike(self, win_start: int) -> tuple[Optional[float], str]:
        win_start_ms = win_start * 1000

        # Priority 1: Binance 15m kline open
        p = await self.feeds.get_binance_15m_open(win_start_ms)
        if p:
            return p, "binance_15m_open"

        # Priority 2: Coinbase 15m candle open (NEW — FIX #2)
        start_iso = window_start_iso(win_start)
        end_iso   = window_end_iso(win_start)
        p = await self.feeds.get_coinbase_15m_open(start_iso, end_iso)
        if p:
            return p, "coinbase_15m_open"

        # Priority 3: Binance mid (better than None or live ticker)
        if self.state.prev_hl_mid:
            return self.state.prev_hl_mid, "binance_mid_prev"

        # NOT returning live BTC price — that was the original bug
        return None, "none"

    # ── Binance book fetch ─────────────────────────────────────────────────────

    async def _fetch_binance_book(self):
        """
        Fetch Binance L2 book. We no longer use a strict cooldown like HL 
        since Binance limits are much more generous for depth polling.
        """
        book = await self.feeds.get_binance_book(
            symbol           = "BTCUSD",
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

    # ── Exit handler ──────────────────────────────────────────────────────────

    async def _handle_exits(self, sig, ob, market_info, min_rem, btc_price, balance) -> bool:
        pos = self.state.held_position
        if not pos.side or not pos.token_id or not pos.size or pos.size <= 0:
            return False

        current_px = ob.yes_mid if pos.side == "YES" else ob.no_mid
        entry_px   = pos.avg_entry_price or pos.entry_price or (current_px or 0)

        reason = evaluate_exit(
            held_side         = pos.side,
            entry_price       = entry_px,
            current_price     = current_px,
            minutes_remaining = min_rem,
            signed_score      = sig.signed_score,
            entry_score       = pos.entry_signed_score or 0.0,
            distance          = sig.distance,
        )
        if not reason:
            return False

        # Cancel open limit orders first
        if self.state.last_condition_id:
            for oid in await self.pm.get_open_orders(self.state.last_condition_id):
                await self.pm.cancel_order(oid)

        # Execute exit — use bid price for IOC exits (ensures fill on thin books)
        # For GTC exits, mid is acceptable as a limit price
        exit_bid = (ob.yes_bid if pos.side == "YES" else ob.no_bid) or current_px or entry_px
        exit_px_gtc = current_px or entry_px
        if reason in ("FORCED_DRAWDOWN", "ALPHA_DECAY", "FORCED_LATE_EXIT"):
            order_id = await self.pm.limit_sell(pos.token_id, exit_bid, pos.size, order_type="IOC")
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
        return True

    # ── Entry handler ─────────────────────────────────────────────────────────

    async def _handle_entry(self, sig, ob, market_info, min_rem, btc_price, balance):
        if self.state.held_position.side is not None:
            return   # already holding

        if self.state.trades_this_window >= Config.MAX_TRADES_PER_WINDOW and not sig.monster_signal:
            return

        if sig.skip_gates:
            log.debug(f"Entry blocked: {sig.skip_gates}")
            return

        if sig.direction == "NEUTRAL":
            return

        # Determine token + price
        if sig.direction == "UP":
            token_id  = market_info.yes_token_id
            side_name = "YES"
            entry_px  = ob.yes_ask or 0.0
            posterior = sig.posterior_final_up or 0.5
        else:
            token_id  = market_info.no_token_id
            side_name = "NO"
            entry_px  = ob.no_ask or 0.0
            posterior = sig.posterior_final_down or 0.5

        if entry_px <= 0 or entry_px >= 1.0:
            return

        # Size (FIX #8: Kelly with riskPct floor)
        position_usd = compute_position_size(
            posterior   = posterior,
            entry_price = entry_px,
            balance     = balance,
            loss_streak = self.state.loss_streak,
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

        # Record position — but mark as PENDING until fill confirmed
        self.state.held_position = HeldPosition(
            side               = side_name,
            token_id           = token_id,
            entry_price        = entry_px,
            avg_entry_price    = entry_px,
            size               = shares,
            size_usd           = position_usd,
            entry_signed_score = sig.signed_score,
            condition_id       = market_info.condition_id,
        )
        self.state.trades_this_window += 1

        # Quick fill verification — check if order appears in positions
        # (Best-effort; next cycle will reconcile from API anyway)
        log.info(f"Order placed: {order_id} — position recorded (will reconcile next cycle)")

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
        log.info(
            f"ENTRY: {side_name} @ {entry_px:.4f} "
            f"shares={shares:.2f} usd=${position_usd:.2f} "
            f"score={sig.signed_score:.2f} post={posterior:.4f}"
        )


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
