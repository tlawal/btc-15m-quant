"""
BTC 15m Quant Engine — main async loop.

Architecture:
  - 3-second inner loop (Config.LOOP_INTERVAL_SEC)
  - Each cycle: fetch data → compute signals → handle exits → handle entries
  - State persisted to SQLite after every cycle
  - Graceful shutdown on SIGINT/SIGTERM
"""

BUILD_VERSION = "optimized"

import asyncio
import aiofiles
import json
import logging
import math
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
from signals import SignalResult, compute_signals
from exit_policy import evaluate_exit
from sizing import compute_position_size
from inference import InferenceEngine
from dashboard import run_dashboard
from utils import (
    setup_logging, send_telegram, Timer,
    fmt_entry, fmt_exit, fmt_halt, fmt_status, fmt_engine_block, fmt_pnl_dashboard,
    fmt_performance_summary, fmt_signal_decay_alert,
    current_window_start, minutes_remaining as calc_minutes_remaining,
    window_start_iso, window_end_iso, AlertTier, json_logger
)
from optimizer import SignalOptimizer
from attribution import FeatureAttributor
from reviewer import run_nightly_review
import metrics_exporter

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
        self._consecutive_skips = 0  # no-trade alert counter
        self._last_exit_reason = "HOLD"
        self.inference = InferenceEngine()  # Phase 5: ML inference engine
        self._dashboard_task = None         # Phase 6: Dashboard task handle
        self._reconcile_task = None         # Phase 5: Hourly Position sync
        # Real-time CVD WebSocket
        from data_feeds import BinanceCVDWebsocket
        self.cvd_ws = BinanceCVDWebsocket()
        self.optimizer = SignalOptimizer(self.state)
        self.attributor = FeatureAttributor(self.state_mgr)
        self._last_sig = None # Phase 6: Store signal for trade logging
        self._last_early_claim_window = 0  # throttle: one early-claim attempt per window

        self._manual_cmd_q: asyncio.Queue = asyncio.Queue()

    # ── Dashboard manual exit — direct execution (no queue delay) ────────
    async def enqueue_manual_exit_limit(self, *, price: float, order_type: str = "GTC") -> dict:
        """Called directly from dashboard API — executes immediately."""
        try:
            result = await self._manual_exit_limit(price=float(price), order_type=str(order_type))
            return result or {"status": "ok", "message": "Limit exit placed"}
        except Exception as e:
            log.error(f"enqueue_manual_exit_limit failed: {e}")
            return {"status": "error", "message": str(e)}

    async def enqueue_manual_exit_replace(self, *, price: float) -> dict:
        try:
            result = await self._manual_exit_replace(price=float(price))
            return result or {"status": "ok", "message": "Exit order replaced"}
        except Exception as e:
            log.error(f"enqueue_manual_exit_replace failed: {e}")
            return {"status": "error", "message": str(e)}

    async def enqueue_manual_exit_cancel(self) -> dict:
        try:
            result = await self._manual_exit_cancel()
            return result or {"status": "ok", "message": "Exit order cancelled"}
        except Exception as e:
            log.error(f"enqueue_manual_exit_cancel failed: {e}")
            return {"status": "error", "message": str(e)}

    async def enqueue_manual_exit_now(self) -> dict:
        try:
            result = await self._manual_exit_now()
            return result or {"status": "ok", "message": "Market sell placed"}
        except Exception as e:
            log.error(f"enqueue_manual_exit_now failed: {e}")
            return {"status": "error", "message": str(e)}

    async def enqueue_manual_entry(self, *, side: str, price: float, size_usd: float, order_type: str = "GTC") -> dict:
        """Manual entry from dashboard — bypasses signal gates but respects risk limits."""
        try:
            pos = self.state.held_position
            if pos.side:
                return {"status": "error", "message": f"Already holding {pos.side} position — exit first"}
            # Basic risk check
            balance = await self.pm.get_wallet_usdc_balance() or 0.0
            if size_usd > balance:
                return {"status": "error", "message": f"Insufficient balance: ${balance:.2f} < ${size_usd:.2f}"}
            if size_usd > Config.MAX_EXPOSURE_USD:
                return {"status": "error", "message": f"Exceeds max exposure ${Config.MAX_EXPOSURE_USD}"}
            # Discover current market
            now = int(time.time())
            win_start = now - (now % Config.WINDOW_SEC)
            mkt = await self.pm.discover_market(
                win_start,
                cached_slug=self.state.last_market_slug,
                cached_expiry=self.state.last_market_expiry,
                cached_condition_id=self.state.last_condition_id,
            )
            if not mkt:
                return {"status": "error", "message": "Could not discover current market"}
            token_id = mkt.yes_token_id if side == "YES" else mkt.no_token_id
            shares = int(size_usd / price)
            if shares < 1:
                return {"status": "error", "message": "Size too small for price"}
            log.info(f"MANUAL_ENTRY: side={side} price={price} shares={shares} size_usd={size_usd} order_type={order_type}")
            if order_type == "FOK":
                order_id = await self.pm.market_buy(token_id=token_id, amount_usd=size_usd)
            else:
                order_id = await self.pm.limit_buy(token_id=token_id, price=price, size=size_usd)
            if not order_id:
                return {"status": "error", "message": "Order placement failed"}
            # Update held position
            pos.side = side
            pos.token_id = token_id
            pos.entry_price = price
            pos.avg_entry_price = price
            pos.size = float(shares)
            pos.size_usd = size_usd
            pos.condition_id = mkt.condition_id
            pos.order_id = order_id
            pos.is_pending = True
            pos.market_expiry = mkt.expiry_ts
            await self._save_state()
            return {"status": "ok", "message": f"Manual {side} buy placed: {shares} shares @ ${price:.2f} (order {str(order_id)[:10]}...)"}
        except Exception as e:
            log.error(f"enqueue_manual_entry failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _process_manual_commands(self):
        # Process any queued manual commands (fallback path).
        while True:
            try:
                cmd = self._manual_cmd_q.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                op = (cmd or {}).get("op")
                if op == "exit_limit":
                    await self._manual_exit_limit(price=float(cmd.get("price")), order_type=str(cmd.get("order_type") or "GTC"))
                elif op == "exit_replace":
                    await self._manual_exit_replace(price=float(cmd.get("price")))
                elif op == "exit_cancel":
                    await self._manual_exit_cancel()
                elif op == "exit_now":
                    await self._manual_exit_now()
            except Exception as e:
                log.warning(f"manual_cmd_failed: op={cmd.get('op') if isinstance(cmd, dict) else 'unknown'} err={e}")

    async def _manual_exit_limit(self, *, price: float, order_type: str = "GTC"):
        pos = self.state.held_position
        if not pos.side or not pos.token_id or not pos.size or pos.size <= 0:
            log.warning("MANUAL UI EXIT: no active position — cannot place limit exit")
            return {"status": "error", "message": "No active position"}

        requested_size = float(pos.size)
        available_size = None
        sell_size = requested_size
        full_close = False
        try:
            live_positions = await self.pm.get_positions()
            for p in live_positions or []:
                if str((p or {}).get("asset")) == str(pos.token_id):
                    available_size = float((p or {}).get("size") or 0.0)
                    break
            if available_size is not None and available_size > 0:
                sell_size = min(sell_size, available_size)
            # Full-close sizing: if user is effectively closing, sell the full available size
            # (rounded down), without a safety buffer that can create leftovers.
            if available_size is not None and requested_size >= (available_size - 0.001):
                full_close = True
                sell_size = available_size
            if not full_close:
                # Safety buffer + round down to avoid overselling due to decimals.
                sell_size = max(0.0, sell_size - 0.001)
            sell_size = math.floor(sell_size * 10000.0) / 10000.0
        except Exception as e:
            log.warning(f"MANUAL UI EXIT: failed to fetch live position size; using state size. err={e}")
        # If there's an existing pending order, cancel it first before placing new one
        if pos.is_pending and pos.order_id:
            log.warning(f"MANUAL UI EXIT: cancelling existing pending order {pos.order_id} before placing new limit exit")
            try:
                await self.pm.cancel_order(str(pos.order_id))
            except Exception as e:
                log.warning(f"MANUAL UI EXIT: cancel of existing order failed: {e} — proceeding anyway")
            pos.is_pending = False
            pos.order_id = None
        px = max(0.01, min(0.99, round(float(price), 2)))
        oid = await self.pm.limit_sell(str(pos.token_id), px, float(sell_size), order_type=order_type)
        if not oid:
            log.warning("MANUAL UI EXIT: limit sell order placement failed")
            return {
                "status": "error",
                "message": "Order placement failed on CLOB",
                "requested_size": requested_size,
                "available_size": available_size,
                "sell_size_used": sell_size,
            }
        # IMPORTANT: set position size to the intended exit size so reconciliation can
        # correctly close the position when matched >= pos.size.
        pos.size = float(sell_size)
        pos.exit_reason = "MANUAL_LIMIT_EXIT"
        pos.intended_exit_price = px
        pos.order_id = oid
        pos.is_pending = True
        pos.placed_at_ts = int(time.time())
        log.warning(
            "MANUAL UI EXIT TRIGGERED: limit exit placed — side=%s price=%.3f size=%.4f (requested=%.4f available=%s) type=%s oid=%s",
            pos.side,
            float(px),
            float(sell_size),
            float(requested_size),
            str(available_size),
            str(order_type),
            str(oid),
        )
        return {
            "status": "ok",
            "message": f"Limit exit placed at {px:.3f} ({order_type})",
            "requested_size": requested_size,
            "available_size": available_size,
            "sell_size_used": sell_size,
        }

    async def _manual_exit_replace(self, *, price: float):
        pos = self.state.held_position
        if not pos.side or not pos.token_id or not pos.size or pos.size <= 0:
            log.warning("MANUAL UI EXIT: no active position — cannot replace")
            return {"status": "error", "message": "No active position"}
        if not pos.order_id:
            log.warning("MANUAL UI EXIT: no existing order_id to replace")
            return {"status": "error", "message": "No existing order to replace"}

        requested_size = float(pos.size)
        available_size = None
        sell_size = requested_size
        full_close = False
        try:
            live_positions = await self.pm.get_positions()
            for p in live_positions or []:
                if str((p or {}).get("asset")) == str(pos.token_id):
                    available_size = float((p or {}).get("size") or 0.0)
                    break
            if available_size is not None and available_size > 0:
                sell_size = min(sell_size, available_size)
            if available_size is not None and requested_size >= (available_size - 0.001):
                full_close = True
                sell_size = available_size
            if not full_close:
                sell_size = max(0.0, sell_size - 0.001)
            sell_size = math.floor(sell_size * 10000.0) / 10000.0
        except Exception as e:
            log.warning(f"MANUAL UI EXIT: failed to fetch live position size for replace; using state size. err={e}")

        px = max(0.01, min(0.99, round(float(price), 2)))
        new_oid = await self.pm.replace_order(
            old_order_id=str(pos.order_id),
            token_id=str(pos.token_id),
            new_price=px,
            size=float(sell_size),
            side="SELL",
            order_type="GTC",
        )
        if not new_oid:
            log.warning("MANUAL UI EXIT: replace failed")
            return {
                "status": "error",
                "message": "Order replace failed on CLOB",
                "requested_size": requested_size,
                "available_size": available_size,
                "sell_size_used": sell_size,
            }
        pos.size = float(sell_size)
        pos.exit_reason = "MANUAL_LIMIT_EXIT"
        pos.intended_exit_price = px
        pos.order_id = new_oid
        pos.is_pending = True
        pos.placed_at_ts = int(time.time())
        log.warning(
            "MANUAL UI EXIT TRIGGERED: order replaced — side=%s price=%.3f size=%.4f (requested=%.4f available=%s) oid=%s",
            pos.side,
            float(px),
            float(sell_size),
            float(requested_size),
            str(available_size),
            str(new_oid),
        )
        return {
            "status": "ok",
            "message": f"Exit order replaced at {px:.3f}",
            "requested_size": requested_size,
            "available_size": available_size,
            "sell_size_used": sell_size,
        }

    async def _manual_exit_cancel(self):
        pos = self.state.held_position
        if not pos.side or not pos.token_id or not pos.size or pos.size <= 0:
            log.warning("MANUAL UI EXIT: no active position — cannot cancel")
            return {"status": "error", "message": "No active position"}
        if not pos.order_id:
            log.warning("MANUAL UI EXIT: no order_id to cancel")
            return {"status": "error", "message": "No existing order to cancel"}
        try:
            await self.pm.cancel_order(str(pos.order_id))
        except Exception as e:
            log.warning(f"MANUAL UI EXIT: cancel failed: {e}")
            return {"status": "error", "message": f"Cancel failed: {e}"}
        pos.is_pending = False
        pos.exit_reason = None
        pos.intended_exit_price = None
        pos.order_id = None
        log.warning("MANUAL UI EXIT TRIGGERED: exit order cancelled")
        return {"status": "ok", "message": "Exit order cancelled"}

    async def _manual_exit_now(self):
        pos = self.state.held_position
        if not pos.side or not pos.token_id or not pos.size or pos.size <= 0:
            log.warning("MANUAL UI EXIT: no active position — cannot force-exit")
            return {"status": "error", "message": "No active position"}

        requested_size = float(pos.size)
        available_size = None
        sell_size = requested_size
        full_close = False
        try:
            live_positions = await self.pm.get_positions()
            for p in live_positions or []:
                if str((p or {}).get("asset")) == str(pos.token_id):
                    available_size = float((p or {}).get("size") or 0.0)
                    break
            if available_size is not None and available_size > 0:
                sell_size = min(sell_size, available_size)
            if available_size is not None and requested_size >= (available_size - 0.001):
                full_close = True
                sell_size = available_size
            if not full_close:
                sell_size = max(0.0, sell_size - 0.001)
            sell_size = math.floor(sell_size * 10000.0) / 10000.0
        except Exception as e:
            log.warning(f"MANUAL UI EXIT: failed to fetch live position size for market exit; using state size. err={e}")
        # If there's a pending order, cancel it first — user wants immediate exit
        if pos.is_pending and pos.order_id:
            log.warning(f"MANUAL UI EXIT: cancelling existing pending order {pos.order_id} before force-exit")
            try:
                await self.pm.cancel_order(str(pos.order_id))
            except Exception as e:
                log.warning(f"MANUAL UI EXIT: cancel of existing order failed: {e} — proceeding with market sell anyway")
            pos.is_pending = False
            pos.order_id = None
        oid = await self.pm.market_sell(str(pos.token_id), float(sell_size))
        if not oid:
            log.warning("MANUAL UI EXIT: market sell order placement failed")
            return {
                "status": "error",
                "message": "Market sell failed on CLOB",
                "requested_size": requested_size,
                "available_size": available_size,
                "sell_size_used": sell_size,
            }
        pos.size = float(sell_size)
        pos.exit_reason = "MANUAL_EXIT_NOW"
        pos.intended_exit_price = None
        pos.order_id = oid
        pos.is_pending = True
        pos.placed_at_ts = int(time.time())
        log.warning(
            "MANUAL UI EXIT TRIGGERED: force-closing position — side=%s size=%.4f (requested=%.4f available=%s) oid=%s",
            pos.side,
            float(sell_size),
            float(requested_size),
            str(available_size),
            str(oid),
        )
        return {
            "status": "ok",
            "message": f"Market sell placed for {sell_size} shares",
            "requested_size": requested_size,
            "available_size": available_size,
            "sell_size_used": sell_size,
        }

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
                
                # Fix stats depending on prior outcome
                if tr.outcome == "LOSS":
                    # Was already counted as a trade+loss — flip to win
                    self.state.total_losses = max(0, self.state.total_losses - 1)
                    self.state.total_wins += 1
                    self.state.total_pnl_usd += size_usd  # reverse the loss PnL
                else:
                    # OPEN → WIN: not yet counted — add as new trade + win
                    self.state.total_trades += 1
                    self.state.total_wins += 1

                tr.outcome = "WIN"

                self.state.loss_streak = 0
                self.state.session_start_balance = None  # force re-record on next cycle so drawdown resets
                
                # Approximate PnL logic (size_usd is the *payout*)
                profit_usd = size_usd * tr.pnl if tr.pnl else 0.0
                self.state.total_pnl_usd += profit_usd
                
                log.info(f"Recorded redemption win: +{tr.pnl*100:.2f}% (approx profit: ${profit_usd:.2f})")

                # Record to closed_trades DB so performance metrics stay in sync
                if self.state_mgr:
                    asyncio.create_task(self.state_mgr.record_closed_trade(
                        ts=int(time.time()),
                        market_slug=self.state.last_market_slug or "unknown",
                        size=tr.size or size_usd,
                        entry_price=ep,
                        exit_price=1.0,
                        pnl_usd=profit_usd,
                        outcome_win=1,
                        regime=getattr(self.state, "entry_regime", None),
                    ))

                # Clear held position — the old market has settled
                self.state.held_position = HeldPosition()

                # Phase 6: Log trade for self-learning
                if self._last_sig:
                    self.optimizer.log_trade(self._last_sig, "WIN")

                # Immediately recalculate and store metrics inside the event loop if possible
                if self.state_mgr:
                    asyncio.create_task(self.state_mgr.calculate_performance_metrics())
                return
                
        log.warning("Redeemed position but no matching OPEN trade found in history.")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _reconcile_loop(self):
        """Phase 5: Hourly position reconciliation job."""
        while self._running:
            await asyncio.sleep(60) # check shortly after start, then sleep
            try:
                if not self.pm.can_trade:
                    await asyncio.sleep(3600)
                    continue
                
                log.info("Running hourly position reconciliation...")
                open_pos = await self.pm.get_open_positions()
                has_active = any(p.get("size", 0) > 0.01 for p in open_pos)
                db_side = self.state.held_position.side

                if db_side and not has_active:
                    msg = f"⚠️ Reconciliation Mismatch: DB shows {db_side} open, but Polymarket shows FLAT."
                    log.error(msg)
                    from utils import AlertTier
                    await send_telegram(self.feeds.session, msg, tier=AlertTier.CRITICAL)
                    
                elif not db_side and has_active:
                    msg = "⚠️ Reconciliation Mismatch: DB is FLAT, but Polymarket shows active positions."
                    log.warning(msg)
                    from utils import AlertTier
                    await send_telegram(self.feeds.session, msg, tier=AlertTier.WARN)
                
            except Exception as e:
                log.error(f"Reconciliation error: {e}")
            
            # Sleep 1 hour
            for _ in range(60):
                if not self._running: break
                await asyncio.sleep(60)


    async def start(self):
        # Phase 0: Start Dashboard IMMEDIATELLY
        # This ensures /api/debug and other checks are available even if next steps hang.
        self._dashboard_task = asyncio.create_task(run_dashboard(self))
        log.info("Dashboard server started on port 8000 (pre-init)")

        # Phase 1 Checks: DB existence and write access
        db_url = Config.DATABASE_URL
        log.info(f"Using database: {db_url}")
        db_path = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
        
        if db_url.startswith("sqlite"):
            db_dir = os.path.dirname(db_path) or "."
            if not os.path.exists(db_path):
                log.warning(f"🚨 STARTUP ALERT: Database file {db_path} is missing. A new one will be created.")
            
            if not os.access(db_dir, os.W_OK):
                log.error(f"❌ CRITICAL: Database directory {db_dir} is NOT WRITABLE. Check Railway Volume permissions.")
            elif os.path.exists(db_path) and not os.access(db_path, os.W_OK):
                log.error(f"❌ CRITICAL: Database file {db_path} is NOT WRITABLE.")

        # Phase 2: Data Feeds
        try:
            async with asyncio.timeout(15):
                await self.feeds.start()
        except asyncio.TimeoutError:
            log.warning("Data feeds startup timed out; proceeding anyway.")

        # Phase 3: Trade Manager (Polymarket)
        await self.pm.start()

        # Phase 4: State & Optimizer
        try:
            async with asyncio.timeout(20):
                self.state = await self.state_mgr.load()
                self.optimizer.state = self.state
        except asyncio.TimeoutError:
            log.error("CRITICAL: Database load/migration timed out!")
            # Fallback to empty state if possible, though this is risky
            from state import EngineState
            self.state = EngineState()
            self.optimizer.state = self.state

        # Reset prev_cycle_score if restarted across a window boundary.
        # Without this, a stale negative score from a previous window suppresses the
        # first post-restart signal via EMA smoothing (e.g. 0.6×5.0 + 0.4×(-3.0) = 1.8).
        if self.state.last_window_start_sec is not None:
            _current_win = current_window_start(int(time.time()))
            if _current_win != self.state.last_window_start_sec:
                self.state.prev_cycle_score = None
                log.info("prev_cycle_score cleared on cross-window restart")

        # Startup position reconciliation — sync held position with Polymarket API
        if self.pm.can_trade and not self.state.held_position.side:
            try:
                async with asyncio.timeout(10):
                    api_positions = await self.pm.get_open_positions()
                    for pos in (api_positions or []):
                        token_id = pos.get("asset_id") or pos.get("token_id")
                        size = float(pos.get("size") or pos.get("balance") or 0)
                        price = float(pos.get("price") or 0.5)
                        if token_id and size > 0:
                            outcome = (pos.get("outcome") or "").lower()
                            side = "YES" if ("yes" in outcome or "up" in outcome) else "NO"
                            log.warning(f"STARTUP RECONCILE: API position found — {side} size={size:.2f} @ {price:.3f}")
                            self.state.held_position.side = side
                            self.state.held_position.token_id = token_id
                            self.state.held_position.size = size
                            self.state.held_position.entry_price = price
                            self.state.held_position.avg_entry_price = price
                            self.state.held_position.is_pending = False
                            break
            except Exception as e:
                log.warning(f"Startup position reconciliation failed: {e}")

        if self.pm.can_trade:
            print(f"ENGINE START BUILD={BUILD_VERSION} RPC_SET={bool(Config.POLYGON_RPC_URL)} USDC_ADDR={Config.POLYGON_USDC_ADDRESS}", flush=True)
            log.info("Engine started. Ensuring Polymarket approvals...")
            try:
                async with asyncio.timeout(10):
                    await self.pm.ensure_approvals()
            except Exception as e:
                log.warning(f"Polymarket approvals check failed or timed out: {e}")

        # Record initial balance for 20% loss limit
        # Wrap in timeout to prevent startup hang if network/API is slow
        try:
            async with asyncio.timeout(10):
                wallet_usdc = await self.pm.get_wallet_usdc_balance()
                margin_usdc = await self.pm.get_margin()
                positions = await self.pm.get_positions()
                balance = wallet_usdc or 0.0
                if balance <= 1e-6:
                    balance = (margin_usdc or {}).get("available_usdc", 0.0)
                if balance <= 1e-6:
                    balance = (margin_usdc or {}).get("balance_usdc", 0.0)

                equity_usd = balance
                try:
                    exposure = self.pm.summarize_exposure(positions or []) if hasattr(self.pm, "summarize_exposure") else None
                    exposure_usd = float((exposure or {}).get("exposure_usd") or 0.0)
                    wallet_component = wallet_usdc if (wallet_usdc is not None and wallet_usdc > 0) else balance
                    equity_usd = float(wallet_component or 0.0) + exposure_usd
                except Exception:
                    equity_usd = balance
                    
                if getattr(self.state, "session_start_balance", None) is None or self.state.session_start_balance == 0:
                    self.state.session_start_balance = equity_usd
                    log.info(f"Session start balance recorded: ${equity_usd:.2f} (20% limit = ${equity_usd * 0.20:.2f})")
        except asyncio.TimeoutError:
            log.warning("Timeout while fetching initial balance; dashboard will show 0 until first successful cycle.")
        except Exception as e:
            log.warning(f"Failed to fetch initial balance: {e}")

        # Startup reconciliation: check if a pending order survived a restart
        if self.pm.can_trade and self.state.held_position.is_pending and self.state.held_position.order_id:
            log.info(f"STARTUP RECONCILE: pending order {self.state.held_position.order_id} found — checking fill status")
            try:
                async with asyncio.timeout(10):
                    await self._reconcile_pending_order()
            except Exception as e:
                log.warning(f"Startup reconciliation failed: {e}")

        # Start real-time CVD WebSocket
        await self.cvd_ws.start()
        log.info("BinanceCVD WebSocket started")

        # Phase 5: Start Hourly Reconciliation
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        log.info("Hourly position reconciliation started")

        log.info("Ready. Starting main loop.")

    async def stop(self):
        self._running = False
        if self.state:
            await self.state_mgr.save(self.state)
        
        if self._dashboard_task:
            self._dashboard_task.cancel()
            try: await self._dashboard_task
            except asyncio.CancelledError: pass

        if getattr(self, "_reconcile_task", None):
            self._reconcile_task.cancel()
            try: await self._reconcile_task
            except asyncio.CancelledError: pass

        self.cvd_ws.stop()
        await self.feeds.close()
        await self.pm.close()
        log.info("Engine stopped.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        await self.start()
        
        # Also run once at startup - non-blocking to prevent UI hang
        try:
            # Wrap in timeout to prevent startup hang
            async with asyncio.timeout(15):
                redeemed_startup = await self.pm.redeem_winning_positions()
                if redeemed_startup > 0:
                    log.info(f"✅ STARTUP CLAIM: ${redeemed_startup:.2f}")
        except asyncio.TimeoutError:
            log.warning("Startup redeem timed out; will retry in background cycle.")
        except Exception as e:
            log.error(f"Startup redeem failed: {e}")
            
        _crash_streak = 0
        while self._running:
            t0 = time.monotonic()
            try:
                await self._cycle()
                _crash_streak = 0
            except Exception as e:
                import traceback
                self.last_cycle_error = traceback.format_exc()
                log.exception(f"Cycle error: {e}")
                _crash_streak += 1
                if _crash_streak >= 5:
                    log.critical(f"CRASH_LOOP: {_crash_streak} consecutive cycle failures — last: {e}")
            elapsed = time.monotonic() - t0
            sleep   = max(0.0, Config.LOOP_INTERVAL_SEC - elapsed)
            await asyncio.sleep(sleep)
        log.info(f"Main loop exited (_running=False). Crash streak at exit: {_crash_streak}")

    async def _cycle(self):
        start_time_ms = int(time.time() * 1000)
        now_ts        = int(time.time())
        win_start     = current_window_start(now_ts)
        win_end      = win_start + Config.WINDOW_SEC
        min_rem      = calc_minutes_remaining(now_ts)
        win_rolled   = self.state.last_window_start_sec != win_start

        outside_preferred_hours = not Config.is_preferred_trading_time()
        if outside_preferred_hours:
            log.info("Outside preferred trading hours: exits/monitoring enabled, entries disabled")

        # ── Window reset ──────────────────────────────────────────────────────
        if win_rolled:
            log.info(f"New 15m window: {win_start} ({datetime.fromtimestamp(win_start, tz=timezone.utc).strftime('%H:%M:%S')} UTC)")
            # Try to redeem any winning positions
            redeemed_usd = await self.pm.redeem_winning_positions()
            if redeemed_usd > 0:
                log.info(f"✅ AUTO-CLAIMED ${redeemed_usd:.2f} at window start!")
                self._record_redemption(redeemed_usd)

            # Phase A: fill settlement outcome for previous window's exit records
            # prev_win_start is the window that just expired
            prev_win_start = win_start - Config.WINDOW_SEC
            # Determine if previous window's exits landed ITM:
            # redeemed_usd > 0 means we had a winning position — exit was ITM for that side.
            # For positions exited early, settlement_itm reflects whether holding would have won.
            # Approximation: if we redeemed anything on this window roll, the side we held was correct.
            prev_window_won = redeemed_usd > 0
            try:
                self.optimizer.fill_exit_settlement(prev_win_start, prev_window_won)
            except Exception as _fes_err:
                log.warning(f"fill_exit_settlement error: {_fes_err}")
            
            # Phase 4: Reset latencies on new window to clear stale metrics
            self.state.latencies             = {}
            self.state.trades_this_window   = 0
            self.state.window_trade_count  = 0
            self.state.locked_strike_price  = None
            self.state.strike_source        = None
            self.state.strike_window_start  = win_start
            self.state.cvd                  = 0.0   # reset CVD each window
            self.state.accumulated_ofi      = 0.0   # Phase 2: reset accumulated OFI
            self.state.prev_cycle_score     = None  # reset EMA score — no contamination from prior window
            self.cvd_ws.reset(win_start * 1000)      # Reset real-time CVD WebSocket

        self.state.last_window_start_sec = win_start
        # Dashboard visibility: show unclaimed positions every cycle
        self.state.unclaimed_usdc = await self.pm.log_unclaimed_positions()

        # ── Early claim: if unclaimed gains exist, attempt once 2–4 min into window ──
        # Avoids waiting a full 15 min. Throttled to one attempt per window.
        mins_into_window = (now_ts - win_start) / 60.0
        if (
            self.state.unclaimed_usdc > 0
            and 2.0 <= mins_into_window <= 4.0
            and self._last_early_claim_window != win_start
        ):
            self._last_early_claim_window = win_start
            log.info(f"EARLY_CLAIM: ${self.state.unclaimed_usdc:.2f} unclaimed, {mins_into_window:.1f}min into window — attempting claim")
            try:
                claimed = await self.pm.redeem_winning_positions()
                if claimed > 0:
                    log.info(f"✅ EARLY_CLAIM SUCCESS: ${claimed:.2f}")
                    self._record_redemption(claimed)
                    self.state.unclaimed_usdc = 0.0
            except Exception as _eclaim_err:
                log.warning(f"EARLY_CLAIM failed: {_eclaim_err}")
        
        # Dashboard visibility: recent trades from Data API
        self.api_trade_history = await self.pm.get_trade_history()

        if self.state.held_position.is_pending and self.state.held_position.order_id:
            await self._reconcile_pending_order()

        await self._process_manual_commands()

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
            # 4. CVD (parallel sources) - exact window boundary
            cvd_task = asyncio.create_task(self.feeds.get_cvd_with_cb_fallback(win_start * 1000, now_ts * 1000))
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

        await self.pm.set_active_market_assets(market_info.yes_token_id, market_info.no_token_id)

        # ── Polymarket context (Phase 4: Latency track) ──────────────────────
        # Always fetch balance/margin even when halted so session_drawdown is
        # computed from real wallet data and the resume check can fire correctly.
        with Timer("fetch_polymarket", self.state.latencies):
            if self.pm.can_trade:
                ob, margin, wallet_usdc, positions = await asyncio.gather(
                    self.pm.get_order_books(market_info.yes_token_id, market_info.no_token_id),
                    self.pm.get_margin(),
                    self.pm.get_wallet_usdc_balance(),
                    self.pm.get_positions(),
                )
                self._last_ob = ob
            else:
                # Cannot trade: still compute a partial order book using REST for display.
                ob = await self.pm.get_order_books(market_info.yes_token_id, market_info.no_token_id)
                self._last_ob = ob
                margin = None
                wallet_usdc = None
                positions = []

        # Fast unrealized PnL: update current price + PnL each cycle using the latest Polymarket mid.
        try:
            pos = self.state.held_position
            if pos.side and pos.token_id and pos.size and pos.size > 0:
                current_px = None
                if str(pos.token_id) == str(market_info.yes_token_id):
                    current_px = ob.yes_mid
                elif str(pos.token_id) == str(market_info.no_token_id):
                    current_px = ob.no_mid
                else:
                    # Cross-window / orphan holdings: price from positions API (already fetched this cycle)
                    for p in positions or []:
                        if str((p or {}).get("asset")) == str(pos.token_id):
                            _cur = (p or {}).get("curPrice")
                            if _cur is None:
                                try:
                                    _cv = float((p or {}).get("currentValue") or 0.0)
                                    _sz = float((p or {}).get("size") or 0.0)
                                    _cur = (_cv / _sz) if _sz > 0 else None
                                except Exception:
                                    _cur = None
                            try:
                                current_px = float(_cur) if _cur is not None else None
                            except Exception:
                                current_px = None
                            break
                entry_px = pos.avg_entry_price or pos.entry_price
                self.state.pos_current_price = current_px
                if current_px is not None and entry_px is not None and entry_px > 0:
                    self.state.pos_unrealized_pnl_pct = (current_px - entry_px) / entry_px
                    self.state.pos_unrealized_pnl_usd = (current_px - entry_px) * float(pos.size)
                else:
                    self.state.pos_unrealized_pnl_pct = None
                    self.state.pos_unrealized_pnl_usd = None
            else:
                self.state.pos_unrealized_pnl_pct = None
                self.state.pos_unrealized_pnl_usd = None
        except Exception:
            pass



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

        if self.pm.can_trade:
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

        # Total equity (wallet + open position notional). This prevents false session drawdown
        # halts when cash is converted into an open position.
        equity_usd = balance
        try:
            # Preferred: mark-to-market the currently held position using mid.
            mtm_usd = 0.0
            if self.state.held_position and self.state.held_position.side and self.state.held_position.size:
                if self.state.held_position.side == "YES":
                    mid = ob.yes_mid
                else:
                    mid = ob.no_mid
                if mid is not None and mid > 0:
                    mtm_usd = float(self.state.held_position.size or 0.0) * float(mid)

            exposure_usd = 0.0
            if mtm_usd <= 1e-9:
                exposure = self.pm.summarize_exposure(positions or []) if hasattr(self.pm, "summarize_exposure") else None
                exposure_usd = float((exposure or {}).get("exposure_usd") or 0.0)
            else:
                exposure_usd = mtm_usd

            # Use wallet_usdc when available since it reflects true on-chain free collateral.
            wallet_component = wallet_usdc if (wallet_usdc is not None and wallet_usdc > 0) else balance
            equity_usd = float(wallet_component or 0.0) + float(exposure_usd or 0.0)
        except Exception:
            equity_usd = balance
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

        # ── BTC price sanity check ────────────────────────────────────────────
        if btc_price < 10000 or btc_price > 500000:
            log.error(f"BTC price sanity check failed: {btc_price:.2f} — skipping cycle")
            await self.state_mgr.save(self.state)
            return
        if indic.close and abs(indic.close - btc_price) / btc_price > 0.05:
            log.error(f"Indicator close ({indic.close:.2f}) diverged >5% from BTC price ({btc_price:.2f}) — skipping cycle")
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
            # Fallback to REST-based CVD (exact boundary accumulated within get_cvd)
            self.state.cvd         = cvd_res.cvd_delta
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

        # ── Phase 2: Polymarket Book Freshness Check ─────────────────────────
        is_stale = book.is_stale
        ob_age_ms = now_ts * 1000 - ob.fetch_ms if ob.fetch_ms else 0
        if ob_age_ms > 5000:
            log.warning(f"Polymarket orderbook stale ({ob_age_ms}ms) — blocking entries")
            is_stale_micro = True
        else:
            is_stale_micro = is_stale

        # ── Phase 2: Chainlink Oracle Tracking ─────────────────────────────────
        oracle_px = self.feeds.oracle.get_price()
        oracle_update_ts = self.feeds.oracle.get_last_update()
        if oracle_px > 0:
            log.debug(f"Chainlink Oracle: ${oracle_px:.2f} (age {now_ts - oracle_update_ts}s)")

        # ── Phase 2: Perpetual Funding Rate ─────────────────────────────────────
        funding_rate = self.feeds.funding_ws.get_rate()
        if funding_rate is not None:
            log.debug(f"Perps Funding Rate: {funding_rate:.5f}")
        # Store previous funding rate for delta signal, then update
        # (last_funding_rate is set at end of cycle, below)

        # ── Phase 3: Polymarket trade flow ──────────────────────────────────────
        pm_net_flow = 0.0
        whale_flow  = 0.0
        if self.state.last_condition_id and self.pm.can_trade:
            try:
                pm_trades = await self.pm.get_market_trades(
                    self.state.last_condition_id, since_ts=win_start
                )
                pm_net_flow = pm_trades.get("net_flow", 0.0)
                whale_flow  = pm_trades.get("whale_flow", 0.0)
            except Exception as e:
                log.debug(f"PM trade flow fetch: {e}")

        # ── Tier 2: Liquidation cascade (async, non-blocking) ─────────────────
        liq_cascade = 0.0
        try:
            liq_cascade = await self.feeds.get_liquidation_cascade(lookback_s=60)
        except Exception as e:
            log.debug(f"liq_cascade fetch: {e}")

        # ── Market Quality Filter ──────────────────────────────────────────────
        spread_yes = (ob.yes_ask - ob.yes_bid) if (ob.yes_ask and ob.yes_bid) else 1.0
        spread_no  = (ob.no_ask - ob.no_bid)   if (ob.no_ask  and ob.no_bid)  else 1.0
        book_depth = (ob.total_bid_size or 0) + (ob.total_ask_size or 0)
        if spread_yes > 0.08 and spread_no > 0.08:
            log.debug(f"Market quality skip: spreads too wide (YES={spread_yes:.3f} NO={spread_no:.3f})")
            await self.state_mgr.save(self.state)
            return
        if book_depth < 20:
            log.debug(f"Market quality skip: book too thin (depth={book_depth:.1f})")
            await self.state_mgr.save(self.state)
            return

        # Kline staleness check
        if k5m and len(k5m) > 0:
            last_candle_age_sec = (int(time.time() * 1000) - k5m[-1].ts_ms) / 1000
            if last_candle_age_sec > 300:
                log.warning(f"Stale kline data: last candle is {last_candle_age_sec:.0f}s old — skipping cycle")
                await self.state_mgr.save(self.state)
                return

        # Phase 6: Institutional Signal Optimization
        score_offset, edge_offset = self.optimizer.get_adjusted_thresholds(0.0, 0.0)

        # ── Phase 4: Compute full signal suite ───────────────────────────────
        perp_basis_pct = await self.feeds.get_binance_perp_basis_pct("BTCUSDT")

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
            bid_levels        = getattr(book, "bids", None),
            ask_levels        = getattr(book, "asks", None),
            deep_imbalance    = book.deep_imbalance,
            vpin_proxy        = book.vpin_proxy,
            deep_ofi          = book.deep_ofi,
            microprice        = book.microprice,
            is_stale_micro    = is_stale_micro,
            tob_imbalance     = book.tob_imbalance,
            cvd_velocity      = self.cvd_ws.get_cvd_slope(),
            arrival_ts_ms     = getattr(self.cvd_ws, "cvd_timestamps", None),
            pm_net_flow       = pm_net_flow,
            cvd_delta         = cvd_delta_for_signals,
            true_cvd          = self.state.cvd,
            accumulated_ofi   = self.state.accumulated_ofi,
            cross_cvd_agree   = self.state.cross_cvd_agree,
            cvd_total_vol     = cvd_total_vol,
            prev_cvd_total_vol = self.state.prev_cvd_total_vol,
            oracle_px         = oracle_px,
            oracle_update_ts  = oracle_update_ts,
            funding_rate      = funding_rate,
            perp_basis_pct    = perp_basis_pct,
            yes_mid           = ob.yes_mid,
            no_mid            = ob.no_mid,
            yes_bid           = ob.yes_bid,
            yes_ask           = ob.yes_ask,
            no_bid            = ob.no_bid,
            no_ask            = ob.no_ask,
            total_bid_size    = ob.total_bid_size,
            total_ask_size    = ob.total_ask_size,
            whale_flow        = whale_flow,
            window_outcomes   = list(self.state.window_outcomes),
            liq_cascade       = liq_cascade,
            funding_rate_prev = self.state.last_funding_rate,
            inference_engine  = self.inference,
            score_offset      = score_offset,
            edge_offset       = edge_offset,
            balance           = balance,
        )
        self._last_sig = sig # Phase 6: Store for trade logging feedback

        # Store CVD total vol for next cycle's volume-weighted scoring
        self.state.prev_cvd_total_vol = cvd_total_vol

        # Update funding rate memory for delta signal
        if funding_rate is not None:
            self.state.last_funding_rate = funding_rate

        # Update score/posterior memory
        if not is_stale_micro:
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

        # ── Position API sync (for dashboard) ──────────────────────────────────
        open_positions = await self.pm.get_open_positions()
        self.state.open_positions_api = open_positions
        # NOTE: monitor_and_exit_open_positions removed — all exits consolidated
        # through exit_policy.evaluate_exit() via _handle_exits below.

        # ── Exit handling ─────────────────────────────────────────────────────
        # Phase 3: pass CVD and posteriors for new exit types
        exit_executed = await self._handle_exits(
            sig, ob, market_info, min_rem, btc_price, balance, cvd_res.cvd_delta
        )

        # ── Entry handling ─────────────────────────────────────────────────────
        if not exit_executed and not self.state.trading_halted and not outside_preferred_hours:
            if self.state.window_trade_count >= 1:
                # Skip entry to prevent simultaneous entries in window
                pass
            else:
                prev_side = self.state.held_position.side
                await self._handle_entry(
                    sig,
                    ob,
                    market_info,
                    min_rem,
                    btc_price,
                    balance,
                    equity_usd,
                    win_start=win_start,
                )
                if self.state.held_position.side and not prev_side:  # entry made
                    await self.state_mgr.save(self.state)
                    return

        # ── No-trade alert ────────────────────────────────────────────────────
        if self.state.held_position.side:
            self._consecutive_skips = 0
        elif not exit_executed:
            self._consecutive_skips += 1
            if self._consecutive_skips == 100:
                log.error(f"NO-TRADE ALERT: {self._consecutive_skips} consecutive cycles without entry")
                await send_telegram(
                    self.feeds.session,
                    f"⚠️ CRITICAL: Bot has skipped {self._consecutive_skips} cycles (~{self._consecutive_skips * 15 // 60}min) without placing a trade. Check gates/balance/signals.",
                    tier=AlertTier.CRITICAL
                )

        # ── Halt check ────────────────────────────────────────────────────────
        session_drawdown = 0.0
        if self.state.session_start_balance and self.state.session_start_balance > 0:
            session_drawdown = (self.state.session_start_balance - equity_usd) / self.state.session_start_balance
        
        # Resume if conditions cleared — but NOT if kill switch is active (manual override)
        if self.state.trading_halted and not Config.KILL_SWITCH and not (self.state.loss_streak >= Config.STREAK_HALT_AT or session_drawdown > 0.30):
            self.state.trading_halted = False
            log.info("Trading resumed: halt conditions no longer met")
            await send_telegram(
                self.feeds.session,
                "✅ Trading resumed: loss_streak <5 and session_drawdown <=30%.",
                tier=AlertTier.INFO
            )

        if (self.state.loss_streak >= Config.STREAK_HALT_AT or session_drawdown > 0.30) and not self.state.trading_halted:
            # Pre-halt alert
            await send_telegram(
                self.feeds.session,
                f"⚠️ PRE-HALT WARNING: loss_streak={self.state.loss_streak}, session_drawdown={session_drawdown*100:.1f}% — halting after this cycle.",
                tier=AlertTier.CRITICAL
            )
            
            self.state.trading_halted = True
            log.error(f"TRADING HALTED: loss_streak={self.state.loss_streak}, session_drawdown={session_drawdown*100:.1f}%")
            await send_telegram(
                self.feeds.session,
                fmt_halt(self.state.loss_streak, balance),
                tier=AlertTier.CRITICAL
            )

        # ── Periodic status message ────────────────────────────────────────────
        self._status_counter += 1
        if self._status_counter % 4 == 0:   # every ~1 min
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
        await self._log_cycle_features(sig, btc_price, min_rem, now_ts, getattr(self, "_last_ob", None))

        # PnL Dashboard (console only)
        if self._status_counter % 4 == 0: # Print every ~1 min
            print(fmt_pnl_dashboard(self.state.trade_history, balance))

        # Phase 5 #30: Hourly Telegram performance summary (~240 cycles = 1hr)
        if self._status_counter % 240 == 0 and self._status_counter > 0:
            try:
                closed = [t for t in self.state.trade_history if t.outcome in ("WIN", "LOSS")]
                wins = sum(1 for t in closed if t.outcome == "WIN")
                wr = wins / len(closed) if closed else 0.0
                pnls = [t.pnl for t in closed if t.pnl is not None]
                total_pnl = sum(pnls)
                await send_telegram(
                    self.feeds.session,
                    fmt_performance_summary(
                        win_rate=wr, total_trades=len(closed),
                        total_pnl=total_pnl, balance=balance,
                        sharpe=0.0, kelly_mult=self.optimizer.get_kelly_multiplier(),
                    )
                )
                # Alert on signal decay
                decayed = list(self.optimizer.get_disabled_signals())
                if decayed:
                    accuracies = self.optimizer.get_signal_accuracies()
                    await send_telegram(
                        self.feeds.session,
                        fmt_signal_decay_alert(decayed, accuracies),
                        tier=AlertTier.WARN
                    )
            except Exception as e:
                log.debug(f"Hourly summary telegram: {e}")

        # Phase 6: Periodic Institutional Retraining (every hour)
        if self._status_counter % 240 == 0 and self._status_counter > 0:
            self.optimizer.retrain_and_adjust()
            self.optimizer.try_fit_platt()
            if self.inference:
                self.inference.reload_model()

        # Heartbeat file
        await self._write_heartbeat(
            now_ts,
            balance,
            runtime_ms,
            margin=margin,
            wallet_usdc=wallet_usdc,
            equity_usd=equity_usd,
            sig=sig,
        )

        # ── Signal snapshot → structured_logs (feeds Plotly charts) ──────────
        json_logger.log("signal", {
            "signed_score":       sig.signed_score,
            "abs_score":          sig.abs_score,
            "posterior_final_up": sig.posterior_final_up,
            "posterior_final_down": sig.posterior_final_down,
            "cvd_score":          sig.cvd_score,
            "ofi_score":          sig.ofi_score,
            "regime":             sig.regime,
            "direction":          sig.direction,
            "monster":            sig.monster_signal,
            "gates":              sig.skip_gates,
            "balance":            balance,
            "btc_price":          btc_price,
        })

        # ── Calibration point logging (every cycle) ────────────────────────────
        if sig.posterior_final_up is not None and self.state.locked_strike_price:
            try:
                self.optimizer.log_calibration_point(
                    posterior=sig.posterior_final_up,
                    market_price=ob.yes_mid or 0.5,
                    strike=self.state.locked_strike_price,
                    btc_price=btc_price,
                    window_id=win_start,
                    timestamp=now_ts,
                )
            except Exception as _cal_err:
                log.debug(f"calibration log error: {_cal_err}")

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
            
            s_up = compute_position_size(posterior=p_up, entry_price=px_up, balance=balance, loss_streak=self.state.loss_streak, monster_signal=sig.monster_signal, book=ob, edge=sig.edge_up) or 0.0
            s_down = compute_position_size(posterior=p_dn, entry_price=px_dn, balance=balance, loss_streak=self.state.loss_streak, monster_signal=sig.monster_signal, book=ob, edge=sig.edge_down) or 0.0
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
        self.state.prev_cycle_score = max(-8.0, min(8.0, sig.signed_score))
        self.state.prev_cycle_price = btc_price

        # ── Update Prometheus Metrics ─────────────────────────────────────────
        metrics_exporter.update_metrics(self.state, asdict(self.state), sig)

        # ── Phase 5: WebSocket push to dashboard clients ─────────────────────
        try:
            from dashboard import broadcast_cycle_update
            await broadcast_cycle_update({
                "ts": now_ts,
                "btc_price": btc_price,
                "balance": balance,
                "score": sig.signed_score,
                "direction": sig.direction,
                "posterior_up": sig.posterior_final_up,
                "edge": sig.target_edge,
                "regime": sig.regime,
                "monster": sig.monster_signal,
                "gates": sig.skip_gates,
                "min_rem": min_rem,
                "position": self.state.held_position.side,
            })
        except Exception:
            pass  # WS broadcast is best-effort

        # Heartbeat is written by _write_heartbeat() earlier in _cycle()

        # ── Nightly AI review (00:05 UTC) ─────────────────────────────────────
        utc_now = datetime.now(timezone.utc)
        today_str = utc_now.strftime("%Y-%m-%d")
        if utc_now.hour == 0 and utc_now.minute == 5 and self.state.last_review_date != today_str:
            self.state.last_review_date = today_str
            asyncio.create_task(run_nightly_review(session=self.feeds.session))
            log.info(f"Nightly AI review triggered for {today_str}")

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
                # Use the open of the kline nearest to (but before) window start
                best = min(klines_5m, key=lambda c: abs(c.ts_ms - win_start_ms))
                return {"strike": best.open, "source": "binance_5m_nearest_open"}
        except Exception as e:
            log.debug(f"Strike P3 (Binance 5m) failed: {e}")

        # Priority 4: Previous book mid price
        if self.state.prev_hl_mid:
            return {"strike": self.state.prev_hl_mid, "source": "binance_mid_prev"}

        # Priority 5: REFUSED — never use live/current price as strike
        # Live price contaminates the signal by removing the distance-from-open
        # that the entire Bayesian framework depends on.
        log.error("All strike sources failed — cannot determine window open price")
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
            limit            = 100,
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
        if not pos.order_id or not pos.is_pending:
            return

        status = await self.pm.get_order_status(pos.order_id)
        age_s = int(time.time()) - (pos.placed_at_ts or 0)

        if not status:
            # API failure — force-resolve via position check after timeout
            if pos.exit_reason and age_s > 15:
                # EXIT order: check if shares are gone (exit filled)
                log.warning(f"RECONCILE: get_order_status returned None for exit order (age={age_s}s) — checking positions API")
                actual = await self.pm.get_positions()
                still_held = any(
                    p.get("asset") == pos.token_id and float(p.get("size", 0)) > 0
                    for p in (actual or [])
                )
                if not still_held:
                    log.info("RECONCILE: shares no longer held — exit order was filled (status API missed it)")
                    pos.is_pending = False
                    entry_px = pos.avg_entry_price or pos.entry_price or 0.5
                    actual_px = pos.intended_exit_price or entry_px
                    pnl_pct = (actual_px - entry_px) / entry_px if entry_px > 0 else 0.0
                    outcome = "WIN" if pnl_pct >= 0 else "LOSS"
                    if pnl_pct < 0:
                        self.state.loss_streak += 1
                    else:
                        self.state.loss_streak = 0
                    for tr in reversed(self.state.trade_history):
                        if tr.side == pos.side and tr.outcome == "OPEN":
                            tr.exit_price = actual_px
                            tr.pnl = pnl_pct
                            tr.outcome = outcome
                            break
                    self.state.total_trades += 1
                    if outcome == "WIN":
                        self.state.total_wins += 1
                    else:
                        self.state.total_losses += 1
                    self.state.total_pnl_usd += (actual_px - entry_px) * (pos.size or 0)
                    await self.state_mgr.record_closed_trade(
                        ts=int(time.time()),
                        market_slug=self.state.last_market_slug or "unknown",
                        size=pos.size or 0,
                        entry_price=entry_px,
                        exit_price=actual_px,
                        pnl_usd=(actual_px - entry_px) * (pos.size or 0),
                        outcome_win=1 if outcome == "WIN" else 0,
                    )
                    log.info(f"RECONCILE FORCE EXIT: {outcome} pnl={pnl_pct*100:.1f}%")
                    self.state.held_position = HeldPosition()
                elif age_s > 30:
                    log.warning(f"RECONCILE: shares still held, exit order status unknown for {age_s}s — clearing pending flag")
                    try:
                        await self.pm.cancel_order(str(pos.order_id))
                    except Exception:
                        pass
                    pos.is_pending = False
                    pos.exit_reason = None
                    pos.order_id = None
            elif not pos.exit_reason and age_s > 10:
                # ENTRY order: check if shares appeared (entry filled but status API missed it)
                log.warning(f"RECONCILE: get_order_status returned None for ENTRY order (age={age_s}s) — checking positions API")
                actual = await self.pm.get_positions()
                held_size = 0.0
                held_value = 0.0
                for p in (actual or []):
                    if str((p or {}).get("asset")) == str(pos.token_id):
                        held_size = float((p or {}).get("size", 0))
                        held_value = float((p or {}).get("entryValue", 0))
                        break
                if held_size > 0:
                    log.info(f"RECONCILE: entry fill confirmed via positions API — size={held_size} value={held_value:.4f}")
                    pos.is_pending = False
                    pos.size = held_size
                    if held_value > 0 and held_size > 0:
                        pos.avg_entry_price = round(held_value / held_size, 4)
                    # Update trade record with actual fill
                    for tr in reversed(self.state.trade_history):
                        if tr.side == pos.side and tr.outcome == "OPEN":
                            if held_value > 0 and held_size > 0:
                                tr.entry_price = pos.avg_entry_price
                            tr.size = held_size
                            break
                elif age_s > 30:
                    # No position found — attempt cancel. Only clear state if cancel succeeds
                    # (order still open = entry didn't fill). If cancel fails (order not found),
                    # the entry may have filled; keep position alive to avoid orphaned real shares.
                    canceled = False
                    try:
                        await self.pm.cancel_order(str(pos.order_id))
                        canceled = True
                    except Exception:
                        pass
                    if canceled:
                        log.warning(f"RECONCILE: entry order canceled after {age_s}s with no fill — clearing position")
                        if self.state.trade_history and self.state.trade_history[-1].outcome == "OPEN":
                            self.state.trade_history.pop()
                        self.state.held_position = HeldPosition()
                    else:
                        # Cancel failed → order may already have filled; just clear pending flag
                        log.warning(f"RECONCILE: cancel failed for entry order — assuming filled, clearing is_pending only")
                        pos.is_pending = False
                        pos.order_id = None
            return

        st = status.get("status")
        matched = status.get("size_matched", 0)
        fill_price = status.get("price", 0)

        if st == "FILLED" or (matched > 0 and matched >= pos.size):
            log.info(f"PENDING ORDER FILLED: {pos.order_id} ({matched} shares)")
            
            # Use fill price, fallback to intended
            actual_px = fill_price or pos.intended_entry_price or pos.intended_exit_price or 0
            
            if pos.exit_reason:
                # EXIT FILL handle
                intended_px = pos.intended_exit_price or actual_px
                # Sell: slippage = actual - intended (positive is better price)
                slippage = (actual_px - intended_px) / intended_px if intended_px > 0 else 0.0
                
                entry_px = pos.avg_entry_price or pos.entry_price or actual_px
                pnl_pct = (actual_px - entry_px) / entry_px if entry_px > 0 else 0.0
                
                # Update stats
                if pnl_pct < 0:
                    self.state.loss_streak += 1
                else:
                    self.state.loss_streak = 0
                
                outcome = "WIN" if pnl_pct >= 0 else "LOSS"
                # Reset session drawdown baseline on wins
                if outcome == "WIN":
                    margin = await self.pm.get_margin()
                    balance = margin.get("balance_usdc", 0.0)
                    self.state.session_start_balance = balance
                # Determine actual sell size (matched shares, not full position)
                sell_size = matched if matched > 0 else pos.size

                # Handle partial vs full exits
                _partial_reasons = ("TP1", "TP2", "TP3")
                remaining = pos.size - sell_size
                is_partial = pos.exit_reason in _partial_reasons and remaining > 0.5

                if is_partial:
                    # Partial fill — record the partial exit, keep TradeRecord as OPEN
                    pos.partial_exits.append({
                        "size": sell_size,
                        "price": actual_px,
                        "reason": pos.exit_reason,
                        "ts": int(time.time()),
                        "pnl_pct": pnl_pct,
                    })
                    log.info(f"PARTIAL EXIT: sold {sell_size}@{actual_px:.3f} ({pos.exit_reason}), remaining {remaining} shares — position stays open")

                    # Record partial to DB for tracking
                    entry_feats = getattr(self.state, "entry_features", None)
                    await self.state_mgr.record_closed_trade(
                        ts=int(time.time()),
                        market_slug=self.state.last_market_slug or "unknown",
                        size=sell_size,
                        entry_price=entry_px,
                        exit_price=actual_px,
                        pnl_usd=(actual_px - entry_px) * sell_size,
                        outcome_win=1 if outcome == "WIN" else 0,
                        regime=getattr(self.state, "entry_regime", None),
                        features=entry_feats,
                        kelly_fraction=getattr(self.state, "last_kelly_fraction", None)
                    )

                    pos.size = remaining
                    pos.is_pending = False
                    pos.exit_reason = None
                    pos.order_id = None
                else:
                    # Full exit — finalize TradeRecord with weighted average if partial exits exist
                    all_exits = list(pos.partial_exits) + [{"size": sell_size, "price": actual_px}]
                    total_sold = sum(e["size"] for e in all_exits)
                    wavg_exit = sum(e["size"] * e["price"] for e in all_exits) / total_sold if total_sold > 0 else actual_px
                    final_pnl = (wavg_exit - entry_px) / entry_px if entry_px > 0 else 0.0
                    final_outcome = "WIN" if final_pnl > 0 else "LOSS"

                    for tr in reversed(self.state.trade_history):
                        if tr.side == pos.side and tr.outcome == "OPEN":
                            tr.exit_price = wavg_exit
                            tr.pnl        = final_pnl
                            tr.slippage   = slippage
                            tr.outcome    = final_outcome
                            break

                    # Record final exit portion to DB
                    entry_feats = getattr(self.state, "entry_features", None)
                    await self.state_mgr.record_closed_trade(
                        ts=int(time.time()),
                        market_slug=self.state.last_market_slug or "unknown",
                        size=sell_size,
                        entry_price=entry_px,
                        exit_price=actual_px,
                        pnl_usd=(actual_px - entry_px) * sell_size,
                        outcome_win=1 if final_outcome == "WIN" else 0,
                        regime=getattr(self.state, "entry_regime", None),
                        features=entry_feats,
                        kelly_fraction=getattr(self.state, "last_kelly_fraction", None)
                    )
                    # Feed optimizer
                    if entry_feats:
                        if self._last_sig:
                            self.optimizer.log_trade(self._last_sig, final_outcome)
                    self.optimizer.record_trade_pnl(final_pnl)

                    # Notify
                    margin = await self.pm.get_margin()
                    balance = margin.get("balance_usdc", 0.0)
                    await send_telegram(
                        self.feeds.session,
                        fmt_exit(pos.side, wavg_exit, entry_px, final_pnl, pos.exit_reason, balance),
                    )
                    log.info(f"EXIT CONFIRMED: {pos.side} {pos.exit_reason} pnl={final_pnl*100:.2f}% wavg_exit={wavg_exit:.3f} partials={len(pos.partial_exits)} slippage={slippage*100:.4f}%")

                    self.state.total_trades += 1
                    if final_outcome == "WIN":
                        self.state.total_wins += 1
                        self.state.loss_streak = 0
                    else:
                        self.state.total_losses += 1
                        self.state.loss_streak += 1

                    # Full exit — clear position
                    self.state.held_position = HeldPosition()
            else:
                # ENTRY FILL handle
                intended_px = pos.intended_entry_price or actual_px
                # Buy: slippage = intended - actual (positive is better price)
                slippage = (intended_px - actual_px) / intended_px if intended_px > 0 else 0.0
                
                pos.is_pending = False
                pos.avg_entry_price = actual_px
                # Update size from actual fill (CLOB may partially fill)
                if matched > 0 and matched != pos.size:
                    log.info(f"ENTRY SIZE ADJUSTED: requested={pos.size} filled={matched}")
                    pos.size = matched

                # Set tx_hash for link
                pos.tx_hash = status.get("tx_hash")

                # Update trade_history with tx_hash and actual fill size
                for tr in reversed(self.state.trade_history):
                    if tr.side == pos.side and tr.outcome == "OPEN":
                        tr.tx_hash = status.get("tx_hash")
                        tr.entry_price = actual_px
                        tr.slippage = slippage
                        if matched > 0:
                            tr.size = matched
                        break

                # Update TradeRecord in history
                if abs(slippage) > 0.01:
                    log.warning(f"HIGH SLIPPAGE: {slippage*100:.2f}% (intended={intended_px:.3f} actual={actual_px:.3f})")

                log.info(f"ENTRY CONFIRMED: {pos.side} @ {actual_px} size={matched or pos.size} slippage={slippage*100:.4f}%")

        elif st in ("CANCELED", "EXPIRED"):
            if matched > 0:
                log.info(f"PENDING ORDER {st} PARTIALLY: {matched}/{pos.size} shares filled")
                pos.size = matched
                pos.is_pending = False
                if pos.exit_reason:
                    # Partial exit handled as full exit for state clearing, but log real size
                    pos.is_pending = False 
            else:
                log.info(f"PENDING ORDER {st}: clearing position state")
                if not pos.exit_reason:
                    if self.state.trade_history and self.state.trade_history[-1].outcome == "OPEN":
                        self.state.trade_history.pop()
                    self.state.held_position = HeldPosition()
                else:
                    # Exit failed, keep position but unmark pending
                    pos.is_pending = False
                    pos.order_id = None

        elif st in ("OPEN", "LIVE"):
            # Stale order replacement: if >12s old, cancel and re-place at current price
            if age_s > 30 and pos.exit_reason:
                # Exit order stale >60s — force FOK at bid-1tick for guaranteed fill
                log.warning(f"EXIT TIMEOUT ({age_s}s): re-placing {pos.order_id} as FOK")
                try:
                    ob = await self.pm.get_order_books(pos.token_id, pos.token_id)
                    if ob and pos.side == "YES":
                        force_px = max((ob.yes_bid or 0.01) - 0.01, 0.01)
                    elif ob and pos.side == "NO":
                        force_px = max((ob.no_bid or 0.01) - 0.01, 0.01)
                    else:
                        force_px = max((pos.intended_exit_price or 0.50) - 0.01, 0.01)
                    # Verify actual position size to avoid overselling
                    replace_size = pos.size
                    try:
                        live_pos = await self.pm.get_positions()
                        for p in (live_pos or []):
                            if str((p or {}).get("asset")) == str(pos.token_id):
                                replace_size = float((p or {}).get("size") or 0.0)
                                break
                    except Exception:
                        pass
                    if replace_size <= 0:
                        log.info("RECONCILE: no shares held — exit already filled, clearing position")
                        pos.is_pending = False
                        pos.order_id = None
                        # Let _handle_exits auto-settle or clear on next cycle
                        return
                    new_oid = await self.pm.replace_order(
                        old_order_id=pos.order_id,
                        token_id=pos.token_id,
                        new_price=force_px,
                        size=replace_size,
                        order_type="FOK",
                    )
                    if new_oid:
                        pos.order_id = new_oid
                        pos.placed_at_ts = int(time.time())
                        log.info(f"EXIT FOK placed @ {force_px} -> {new_oid}")
                    else:
                        log.warning("Exit FOK replace failed — clearing pending to allow re-evaluation")
                        try:
                            await self.pm.cancel_order(str(pos.order_id))
                        except Exception:
                            pass
                        pos.is_pending = False
                        pos.order_id = None
                except Exception as e:
                    log.warning(f"Exit timeout replace error: {e}")
                    # Clear pending on failure to prevent permanent stuck state
                    pos.is_pending = False
                    pos.order_id = None

            elif age_s > 12 and not pos.exit_reason:
                log.info(f"STALE ORDER ({age_s}s): replacing {pos.order_id}")
                # Fetch current orderbook for fresh price
                try:
                    ob = await self.pm.get_order_books(pos.token_id, pos.token_id)
                    if ob and pos.side == "YES":
                        new_px = self.pm.smart_entry_price(ob.yes_bid, ob.yes_ask) or pos.intended_entry_price
                    elif ob and pos.side == "NO":
                        new_px = self.pm.smart_entry_price(ob.no_bid, ob.no_ask) or pos.intended_entry_price
                    else:
                        new_px = pos.intended_entry_price

                    new_oid = await self.pm.replace_order(
                        old_order_id=pos.order_id,
                        token_id=pos.token_id,
                        new_price=new_px,
                        size=pos.size,
                    )
                    if new_oid:
                        pos.order_id = new_oid
                        pos.placed_at_ts = int(time.time())
                        pos.intended_entry_price = new_px
                        log.info(f"REPLACED -> {new_oid} @ {new_px}")
                    else:
                        # Cancel succeeded but re-place failed — clear position
                        log.warning("Replace failed: clearing pending position")
                        if self.state.trade_history and self.state.trade_history[-1].outcome == "OPEN":
                            self.state.trade_history.pop()
                        self.state.held_position = HeldPosition()
                except Exception as e:
                    log.warning(f"Stale order replace error: {e}")

    async def _write_heartbeat(
        self,
        ts: int,
        balance: float,
        runtime_ms: int,
        margin: dict | None = None,
        wallet_usdc: float | None = None,
        equity_usd: float | None = None,
        sig = None,
    ):
        """Phase 4: Structured heartbeat for external health monitoring."""
        hb = {
            "ts": ts,
            "status": "HEALTHY",
            "balance": balance,
            "runtime_ms": runtime_ms,
            "latencies": self.state.latencies,
            "trades_15m": self.state.trades_this_window,
            "last_market_slug": self.state.last_market_slug,
            "last_market_expiry": self.state.last_market_expiry,
            "last_condition_id": self.state.last_condition_id,
            "unclaimed_usdc": self.state.unclaimed_usdc,
            "api_trade_history": getattr(self, "api_trade_history", []),
            "pos_current_price": getattr(self.state, "pos_current_price", None),
            "pos_unrealized_pnl_pct": getattr(self.state, "pos_unrealized_pnl_pct", None),
            "pos_unrealized_pnl_usd": getattr(self.state, "pos_unrealized_pnl_usd", None),
            "held_position": {
                "side":        self.state.held_position.side,
                "size":        self.state.held_position.size,
                "entry_price": self.state.held_position.avg_entry_price or self.state.held_position.entry_price,
                "tx_hash":     self.state.held_position.tx_hash,
                "is_pending":  self.state.held_position.is_pending,
                "order_id":    self.state.held_position.order_id,
                "exit_reason": self.state.held_position.exit_reason,
                "intended_exit_price": self.state.held_position.intended_exit_price,
            },
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
        pnl_list = [max(-1.0, min(5.0, t.pnl)) for t in closed if t.pnl is not None]
        avg_pnl = sum(pnl_list) / len(pnl_list) if pnl_list else 0.0
        total_pnl = sum(pnl_list)
        # USD-based total: more intuitive than fractional sum
        total_pnl_usd = sum(
            t.pnl * (t.entry_price or 0.0) * (t.size or 0.0)
            for t in closed
            if t.pnl is not None and t.entry_price and t.size
        )
        # Sharpe: mean(pnl) / std(pnl) * sqrt(N)  (annualization not meaningful for 15m)
        import statistics
        sharpe = 0.0
        if len(pnl_list) >= 2:
            try:
                sharpe = statistics.mean(pnl_list) / statistics.stdev(pnl_list)
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
            "total_pnl_usd": total_pnl_usd,
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
                "sharpe_ratio": 0.0, "avg_pnl_per_trade": 0.0, "regimes": {}
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
            hb_path = "/data/heartbeat.json" if os.path.isdir("/data") else "heartbeat.json"
            async with aiofiles.open(hb_path, mode='w') as f:
                await f.write(json.dumps(hb, indent=2, default=str))
        except Exception as e:
            log.error(f"Heartbeat write aborted: {e}", exc_info=True)

    # ── Exit handler ──────────────────────────────────────────────────────────

    async def _handle_exits(self, sig, ob, market_info, min_rem, btc_price, balance, cvd_delta=0.0) -> bool:
        pos = self.state.held_position
        if not pos.side or not pos.token_id or not pos.size or pos.size <= 0:
            return False

        # Phase 3: Don't exit while pending (wait for fill/cancel)
        # EXCEPTION: if on an expired old market, force-clear pending so auto-settle can run.
        # This prevents positions stuck forever when exit orders fail on expired markets.
        if pos.is_pending:
            pos_expired = (
                pos.condition_id is not None
                and market_info is not None
                and pos.condition_id != market_info.condition_id
                and pos.market_expiry
                and (pos.market_expiry - time.time()) <= 0
            )
            if pos_expired:
                log.warning(f"EXIT: is_pending on EXPIRED market — force-clearing pending to allow auto-settle")
                try:
                    if pos.order_id:
                        await self.pm.cancel_order(str(pos.order_id))
                except Exception:
                    pass
                pos.is_pending = False
                pos.order_id = None
                # Fall through to normal exit handling (auto-settle will fire below)
            else:
                # HARD_STOP bypass: check if position breached unconditional stop while pending.
                # This prevents hard stops from being blocked by stale pending orders.
                _bypass = False
                try:
                    _pending_ob = await self.pm.get_order_books(pos.token_id, pos.token_id)
                    if _pending_ob and pos.entry_price and pos.entry_price > 0:
                        _px = _pending_ob.yes_mid if pos.side == "YES" else _pending_ob.no_mid
                        _unreal = (_px - pos.entry_price) / pos.entry_price if _px else None
                        if _unreal is not None and _unreal < -Config.HARD_STOP_PCT:
                            log.warning(
                                "HARD_STOP_BYPASS: unrealized=%.1f%% breached -%.1f%% while is_pending — cancelling pending order",
                                _unreal * 100, Config.HARD_STOP_PCT * 100,
                            )
                            try:
                                if pos.order_id:
                                    await self.pm.cancel_order(str(pos.order_id))
                            except Exception:
                                pass
                            pos.is_pending = False
                            pos.order_id = None
                            _bypass = True
                except Exception as e:
                    log.warning(f"HARD_STOP_BYPASS check failed: {e}")
                if not _bypass:
                    return False

        # ALWAYS fetch the order book for the position's own token_id.
        # The passed-in `ob` may be from a different market after window roll.
        pos_ob = await self.pm.get_order_books(pos.token_id, pos.token_id)
        if pos_ob:
            ob = pos_ob

        # Detect if position is on an old (rolled) market
        pos_on_old_market = (
            pos.condition_id is not None
            and market_info is not None
            and pos.condition_id != market_info.condition_id
        )
        if pos_on_old_market:
            log.info(f"EXIT: Position on old market {pos.condition_id[:16]}...")
            # After window roll, use remaining time from the OLD market's expiry
            old_expiry = pos.market_expiry
            if old_expiry:
                min_rem = max(0.0, (old_expiry - time.time()) / 60.0)
            # Auto-settle when old market is near/past expiry (<= 2 min).
            # Polymarket endDate includes a ~2min settlement buffer after the nominal
            # window end, which causes stale OB prices and phantom negative PnL.
            # Clear immediately once the trading window is effectively over.
            if min_rem <= 2.0:
                # Determine if position won or lost by checking current value
                settle_px = None
                # Method 1: order book mid price
                if ob:
                    settle_px = ob.yes_mid if pos.side == "YES" else ob.no_mid
                # Method 2: positions API as fallback
                if settle_px is None or settle_px == 0:
                    try:
                        live_pos = await self.pm.get_positions()
                        for p in (live_pos or []):
                            if str((p or {}).get("asset")) == str(pos.token_id):
                                pv = float((p or {}).get("currentValue", 0))
                                ps = float((p or {}).get("size", 0))
                                if ps > 0:
                                    settle_px = pv / ps
                                break
                    except Exception:
                        pass

                position_won = settle_px is not None and settle_px >= 0.50

                for tr in reversed(self.state.trade_history):
                    if tr.side == pos.side and tr.outcome == "OPEN":
                        ep = tr.entry_price or pos.entry_price or pos.avg_entry_price or 0.5
                        if position_won:
                            tr.exit_price = 1.0
                            tr.pnl = (1.0 - ep) / ep if ep > 0 else 0.0
                            tr.outcome = "WIN"
                            self.state.total_wins += 1
                            self.state.loss_streak = 0
                            profit_usd = (1.0 - ep) * (pos.size or 0)
                            self.state.total_pnl_usd += profit_usd
                            log.info(f"AUTO_SETTLE_WIN: {pos.side} entry={ep:.3f} -> $1.00 pnl={tr.pnl*100:.1f}% settle_px={settle_px}")
                            await self.state_mgr.record_closed_trade(
                                ts=int(time.time()),
                                market_slug=self.state.last_market_slug or "unknown",
                                size=tr.size or pos.size or 0,
                                entry_price=ep,
                                exit_price=1.0,
                                pnl_usd=profit_usd,
                                outcome_win=1,
                                regime=getattr(self.state, "entry_regime", None),
                            )
                        else:
                            tr.exit_price = 0.0
                            tr.pnl = -1.0
                            tr.outcome = "LOSS"
                            self.state.total_losses += 1
                            self.state.loss_streak += 1
                            loss_usd = ep * (pos.size or 0)
                            self.state.total_pnl_usd -= loss_usd
                            log.warning(f"AUTO_SETTLE_LOSS: {pos.side} entry={ep:.3f} -> $0.00 settle_px={settle_px} — position lost")
                            await self.state_mgr.record_closed_trade(
                                ts=int(time.time()),
                                market_slug=self.state.last_market_slug or "unknown",
                                size=tr.size or pos.size or 0,
                                entry_price=ep,
                                exit_price=0.0,
                                pnl_usd=-loss_usd,
                                outcome_win=0,
                                regime=getattr(self.state, "entry_regime", None),
                            )
                        self.state.total_trades += 1
                        break
                self.state.held_position = HeldPosition()
                return False

        current_px = ob.yes_mid if pos.side == "YES" else ob.no_mid
        entry_px   = pos.avg_entry_price or pos.entry_price or (current_px or 0)

        # Store position's correct current price for dashboard display
        self.state.pos_current_price = current_px

        # ── MAE/MFE tracking ────────────────────────────────────────────────
        if current_px is not None and entry_px > 0:
            if pos.lowest_price_since_entry is None or current_px < pos.lowest_price_since_entry:
                pos.lowest_price_since_entry = current_px
            if pos.highest_price_since_entry is None or current_px > pos.highest_price_since_entry:
                pos.highest_price_since_entry = current_px
            pos.mae_pct = (entry_px - pos.lowest_price_since_entry) / entry_px
            pos.mfe_pct = (pos.highest_price_since_entry - entry_px) / entry_px
            # Record when deep drawdown first occurred
            if pos.mae_pct >= Config.MAE_RECOVERY_EXIT_THRESHOLD and pos.deep_drawdown_ts is None:
                pos.deep_drawdown_ts = int(time.time())

        # Get previous posterior for decay detection
        prev_post = self.state.last_posterior_up if pos.side == "YES" else self.state.last_posterior_down
        curr_post = sig.posterior_final_up if pos.side == "YES" else sig.posterior_final_down

        # Track peak posterior for trailing exits
        if curr_post is not None:
            prev_peak = getattr(pos, "peak_posterior", None)
            if prev_peak is None or curr_post > prev_peak:
                setattr(pos, "peak_posterior", curr_post)

        # Track adverse-selection / book-flip persistence
        # A "flip" is meaningful if OBI crosses sign against the held side with enough magnitude.
        # Config may not define book-flip params if this exit is disabled.
        flip = False
        _book_flip_imb = getattr(Config, "BOOK_FLIP_IMB_THRESH", None)
        if _book_flip_imb is not None and hasattr(sig, "obi"):
            if pos.side == "YES" and sig.obi <= -_book_flip_imb:
                flip = True
            if pos.side == "NO" and sig.obi >= _book_flip_imb:
                flip = True
        if flip:
            setattr(pos, "book_flip_count", int(getattr(pos, "book_flip_count", 0)) + 1)
        else:
            setattr(pos, "book_flip_count", 0)

        hold_secs = float(int(time.time()) - (pos.placed_at_ts or int(time.time())))
        # Resolve bid/ask for spread-aware exits
        _exit_bid = (ob.yes_bid if pos.side == "YES" else ob.no_bid) or current_px or entry_px
        _exit_ask = (ob.yes_ask if pos.side == "YES" else ob.no_ask) or current_px or entry_px
        exit_result = evaluate_exit(
            held_side         = pos.side,
            entry_price       = entry_px,
            current_price     = current_px,
            minutes_remaining = min_rem,
            signed_score      = sig.signed_score,
            entry_score       = pos.entry_signed_score or 0.0,
            distance          = sig.distance,
            cvd_delta         = cvd_delta,
            cvd_velocity      = self.cvd_ws.get_cvd_slope(),
            deep_ofi          = getattr(sig, "deep_ofi", 0.0) or 0.0,
            obi               = getattr(sig, "obi", 0.0) or 0.0,
            atr14             = getattr(sig, "atr14", None),
            vpin              = getattr(sig, "vpin_proxy", 0.0) or 0.0,
            posterior         = curr_post,
            prev_posterior    = prev_post,
            entry_posterior   = pos.entry_posterior,
            peak_posterior    = getattr(pos, "peak_posterior", None),
            book_flip_count   = int(getattr(pos, "book_flip_count", 0)),
            hold_seconds      = hold_secs,
            entry_min_rem     = getattr(pos, "entry_min_rem", None),
            yes_mid           = getattr(sig, "yes_mid", 0.5),
            bid_price         = _exit_bid,
            ask_price         = _exit_ask,
            tp1_hit           = getattr(pos, "tp1_hit", False),
            tp2_hit           = getattr(pos, "tp2_hit", False),
            tp3_hit           = getattr(pos, "tp3_hit", False),
            mae_pct           = getattr(pos, "mae_pct", 0.0),
            mfe_pct           = getattr(pos, "mfe_pct", 0.0),
            deep_drawdown_ts  = getattr(pos, "deep_drawdown_ts", None),
        )
        # Unpack structured exit result (dict or None)
        if not exit_result:
            return False
        reason      = exit_result["reason"]
        partial_pct = exit_result.get("partial_pct", 1.0)
        use_maker   = exit_result.get("use_maker", False)

        if reason == "FORCED_LATE_EXIT" and getattr(pos, "entry_min_rem", None) is not None:
            return False

        # Phase A: log exit attempt for counterfactual analysis
        try:
            self.optimizer.log_exit_attempt(
                exit_reason=reason,
                held_side=pos.side,
                entry_price=entry_px,
                current_price=current_px or entry_px,
                entry_posterior=pos.entry_posterior,
                current_posterior=curr_post,
                minutes_remaining=min_rem,
                hold_seconds=hold_secs,
                signed_score=sig.signed_score,
                entry_score=pos.entry_signed_score or 0.0,
                market_slug=getattr(self.state, "last_market_slug", "") or "",
                window_ts=int(pos.placed_at_ts or 0),
            )
        except Exception as _ex_log_err:
            log.warning(f"exit log error: {_ex_log_err}")

        # Compute sell size — verify against actual Polymarket position to avoid overselling
        actual_size = pos.size
        try:
            live_positions = await self.pm.get_positions()
            for p in (live_positions or []):
                if str((p or {}).get("asset")) == str(pos.token_id):
                    actual_size = float((p or {}).get("size") or 0.0)
                    if actual_size != pos.size:
                        log.warning(f"EXIT SIZE MISMATCH: state={pos.size} actual={actual_size} — using actual")
                        pos.size = actual_size
                    break
        except Exception as _sz_err:
            log.warning(f"Failed to verify position size: {_sz_err}")

        if actual_size <= 0:
            # API returned no position — fall back to state size rather than clearing.
            # Clearing here would allow re-entry while real shares still exist on-chain.
            log.warning("EXIT: positions API returned 0 — using state size as fallback")
            actual_size = pos.size

        sell_size = actual_size if partial_pct >= 1.0 else max(1, int(actual_size * partial_pct))
        # Clamp to actual available
        sell_size = min(sell_size, actual_size)

        # Paper-trade mode: apply exits locally without touching Polymarket.
        if Config.PAPER_TRADE_ENABLED:
            pnl_pct = (current_px - entry_px) / entry_px if entry_px > 0 else 0.0

            if pnl_pct < 0:
                self.state.loss_streak += 1
            else:
                self.state.loss_streak = 0

            outcome = "WIN" if pnl_pct >= 0 else "LOSS"
            for tr in reversed(self.state.trade_history):
                if tr.side == pos.side and tr.outcome == "OPEN":
                    tr.exit_price = current_px
                    tr.pnl        = pnl_pct
                    tr.outcome    = outcome
                    break

            # Record to DB for performance stats
            try:
                await self.state_mgr.record_closed_trade(
                    ts=int(time.time()),
                    market_slug=self.state.last_market_slug or "unknown",
                    size=sell_size,
                    entry_price=entry_px,
                    exit_price=current_px,
                    pnl_usd=(current_px - entry_px) * sell_size if entry_px > 0 else 0.0,
                    outcome_win=1 if outcome == "WIN" else 0,
                    regime=getattr(self.state, "entry_regime", None),
                    features=getattr(self.state, "entry_features", None),
                    kelly_fraction=getattr(self.state, "last_kelly_fraction", None)
                )
            except Exception as e:
                log.error(f"Failed to record paper exit to DB: {e}")

            self.state.total_trades += 1
            if outcome == "WIN":
                self.state.total_wins += 1
                self.state.session_start_balance = balance  # reset drawdown baseline on win
            else:
                self.state.total_losses += 1

            self.state.total_pnl_usd += (current_px - entry_px) * sell_size if entry_px > 0 else 0.0

            # Update TP tier state and handle partial vs full exits
            if reason in ("TP1", "TP2", "TP3", "TP_FULL", "TP_LATE_ENTRY") and partial_pct < 1.0:
                if reason == "TP1":
                    pos.tp1_hit = True
                elif reason == "TP2":
                    pos.tp2_hit = True
                elif reason == "TP3":
                    pos.tp3_hit = True
                pos.size = max(0, pos.size - sell_size)
                if pos.size <= 0:
                    self.state.held_position = HeldPosition()
            else:
                self.state.held_position = HeldPosition()

            self._last_exit_reason = reason
            log.info(f"PAPER EXIT: {pos.side} reason={reason} pnl={pnl_pct*100:.2f}% sell_size={sell_size}")
            return True

        # Cancel open limit orders first
        if self.state.last_condition_id:
            for oid in await self.pm.get_open_orders(self.state.last_condition_id):
                await self.pm.cancel_order(oid)

        # Execute exit — spread-aware order placement
        exit_bid = _exit_bid
        exit_px_gtc = current_px or entry_px
        # Clamp to Polymarket valid price range [0.01, 0.99]
        exit_bid = max(0.01, min(0.99, exit_bid))
        exit_px_gtc = max(0.01, min(0.99, exit_px_gtc))

        # Reasons that must use FOK at bid for guaranteed fill:
        # Emergency stops + take-profits + near-certain locks (we want the money NOW)
        _fok_reasons = ("HARD_STOP", "FORCED_DRAWDOWN", "ALPHA_DECAY", "FORCED_LATE_EXIT", "VOL_HARD_STOP",
                        "FORCED_ADVERSE_OFI", "FORCED_DISTANCE_LATE",
                        "MAE_RECOVERY_EXIT", "MAE_LATE_RECOVERY",
                        "TP1", "TP2", "TP3", "TP_FULL", "TP_LATE_ENTRY",
                        "NEAR_CERTAIN_LOCK", "PROB_CONVERGENCE")
        if reason in _fok_reasons:
            order_id = await self.pm.limit_sell(pos.token_id, exit_bid, sell_size, order_type="FOK")
            exit_px = exit_bid
        elif use_maker:
            # Spread-aware: place maker limit at best_bid + tick to capture spread
            maker_px = max(0.01, min(0.99, exit_bid + Config.SPREAD_MAKER_TICK))
            order_id = await self.pm.limit_sell(pos.token_id, maker_px, sell_size, order_type="GTC")
            exit_px = maker_px
            log.info(f"SPREAD_AWARE_EXIT: maker limit at {maker_px:.2f} (bid={exit_bid:.2f}, spread tick={Config.SPREAD_MAKER_TICK})")
        else:
            order_id = await self.pm.limit_sell(pos.token_id, exit_px_gtc, sell_size, order_type="GTC")
            exit_px = exit_px_gtc

        if not order_id:
            pos.consecutive_sell_failures += 1
            log.error(f"Exit order failed ({reason}) — consecutive_sell_failures={pos.consecutive_sell_failures}")

            # Escalation: after 3 consecutive failures, try half position size
            if pos.consecutive_sell_failures >= 3 and sell_size > 1:
                half_size = max(1, int(sell_size / 2))
                log.warning(f"SELL_ESCALATION: trying smaller size {half_size} (was {sell_size}) at bid {exit_bid:.2f} FOK")
                order_id = await self.pm.limit_sell(pos.token_id, exit_bid, half_size, order_type="FOK")
                if order_id:
                    sell_size = half_size
                    pos.consecutive_sell_failures = 0

            # Escalation: after 10 consecutive failures, try market_sell as last resort
            if not order_id and pos.consecutive_sell_failures >= 10:
                log.critical(f"SELL_ESCALATION_MARKET: {pos.consecutive_sell_failures} consecutive failures — trying market_sell")
                order_id = await self.pm.market_sell(pos.token_id, sell_size)
                if order_id:
                    pos.consecutive_sell_failures = 0

            if not order_id:
                return False

        # Reset failure counter on success
        pos.consecutive_sell_failures = 0

        # Update TP tier flags for partial exits
        if reason == "TP1":
            pos.tp1_hit = True
        elif reason == "TP2":
            pos.tp2_hit = True
        elif reason == "TP3":
            pos.tp3_hit = True

        # Phase 3: Wait for _reconcile_pending_order to confirm fill
        pos.is_pending = True
        pos.order_id = order_id
        pos.exit_reason = reason
        pos.intended_exit_price = exit_px
        pos.placed_at_ts = int(time.time())
        # NOTE: Do NOT set pos.size = sell_size here. pos.size must reflect the FULL
        # position so partial exit reconciliation can compute remaining = pos.size - matched.
        # Setting pos.size = sell_size makes remaining=0 and clears the entire position.
        log.info(f"Exit order placed ({order_id}) for {reason} — sell_size={sell_size} pos_size={pos.size} use_maker={use_maker} — waiting for fill")
        return True



    # ── Entry handler ─────────────────────────────────────────────────────────

    async def _handle_entry(self, sig, ob, market_info, min_rem, btc_price, balance, equity_usd, win_start: int = 0):
        if self.state.held_position.side is not None:
            return   # already holding

        is_monster = bool(getattr(sig, "monster_signal", False))

        def _entry_skipped(reason: str, **extra):
            try:
                import json as _json

                payload = {
                    "reason": reason,
                    "market_slug": getattr(market_info, "slug", None),
                    "minutes_remaining": float(min_rem) if min_rem is not None else None,
                    "monster": bool(is_monster),
                    "direction": getattr(sig, "direction", None),
                    "signed_score": getattr(sig, "signed_score", None),
                    "edge": getattr(sig, "target_edge", None),
                    "posterior_up": getattr(sig, "posterior_final_up", None),
                    "posterior_down": getattr(sig, "posterior_final_down", None),
                    "gates": list(getattr(sig, "skip_gates", None) or []),
                }
                payload.update(extra)
                log.info("ENTRY_SKIPPED %s", _json.dumps(payload, separators=(",", ":"), default=str))
            except Exception:
                log.info(f"ENTRY_SKIPPED reason={reason}")

        # Hard early window no-trade zone: block entries for the first Config.EARLY_WINDOW_GUARD_MIN minutes.
        # min_rem starts near window_len and decreases toward 0.
        window_min = Config.WINDOW_SEC / 60.0
        early_block_cutoff_min_rem = max(0.0, window_min - float(getattr(Config, "EARLY_WINDOW_GUARD_MIN", 0.0) or 0.0))
        if min_rem is not None and min_rem > early_block_cutoff_min_rem and not is_monster:
            log.info(
                "HARD_EARLY_WINDOW: no trades in first %.1f min for normal signals (min_rem=%.1f > cutoff=%.1f)",
                float(getattr(Config, "EARLY_WINDOW_GUARD_MIN", 0.0) or 0.0),
                float(min_rem),
                float(early_block_cutoff_min_rem),
            )
            _entry_skipped(
                "hard_early_window",
                min_rem=float(min_rem),
                cutoff_min_rem=float(early_block_cutoff_min_rem),
            )
            return

        # Preferred trading hours: 7 AM - 6 PM ET (11:00-22:00 UTC) on weekdays
        if not Config.is_preferred_trading_time():
            log.info("Outside preferred trading window: skip entries.")
            _entry_skipped("outside_preferred_hours")
            return

        # Daily reset of session_start_balance (tracks equity, not just wallet)
        now = int(time.time())
        today = now // 86400
        last_reset_day = getattr(self.state, "last_session_reset_day", 0)
        if today > last_reset_day:
            self.state.session_start_balance = equity_usd
            self.state.last_session_reset_day = today
            log.info(f"Daily session start reset: new session_start_equity={equity_usd:.2f}")

        # Trailing drawdown: reset session_start_balance on wins (equity highs)
        session_start = getattr(self.state, "session_start_balance", 0.0) or 0.0
        if equity_usd > session_start:
            self.state.session_start_balance = equity_usd
            log.info(f"Trailing drawdown reset: new session_start_equity={equity_usd:.2f}")
            session_start = equity_usd  # update local var

        # Calculate and log session drawdown (equity-based)
        session_drawdown_pct = ((session_start - equity_usd) / session_start) * 100 if session_start > 0 else 0.0
        log.info(
            f"Session drawdown: {session_drawdown_pct:.1f}% "
            f"(start_equity={session_start:.2f}, current_equity={equity_usd:.2f}, wallet={balance:.2f})"
        )

        # Session drawdown halt with hysteresis
        session_halted = getattr(self.state, "session_drawdown_halted", False)
        if not session_halted and session_drawdown_pct > Config.MAX_SESSION_DRAWDOWN_PCT:
            self.state.session_drawdown_halted = True
            log.error(f"TRADING HALTED: session_drawdown={session_drawdown_pct:.1f}% > {Config.MAX_SESSION_DRAWDOWN_PCT:.1f}%")
            _entry_skipped("session_drawdown_halt", session_drawdown_pct=session_drawdown_pct)
            return
        elif session_halted and session_drawdown_pct < Config.SESSION_DRAWDOWN_RESUME_PCT:
            self.state.session_drawdown_halted = False
            log.info(f"Session drawdown halt cleared: {session_drawdown_pct:.1f}% < {Config.SESSION_DRAWDOWN_RESUME_PCT:.1f}%, resuming trading")

        if getattr(self.state, "session_drawdown_halted", False):
            log.warning(f"Trading halted due to session drawdown ({session_drawdown_pct:.1f}%)")
            _entry_skipped("session_drawdown_halted", session_drawdown_pct=session_drawdown_pct)
            return

        # Log balance for drawdown monitoring
        log.info(f"Equity check: current={equity_usd:.2f}, session_start={session_start:.2f} (wallet={balance:.2f})")

        # ── Hard capital protections ──────────────────────────────────────────
        if Config.KILL_SWITCH:
            log.warning("KILL_SWITCH is active — no new entries")
            _entry_skipped("kill_switch")
            return

        if self.state.trades_this_window >= Config.MAX_TRADES_PER_WINDOW and not sig.monster_signal:
            _entry_skipped("max_trades_per_window")
            return

        # ── Stale kline data gate ──────────────────────────────────────────────
        if self.feeds.is_kline_stale(30):
            log.warning("STALE_KLINE: last kline fetch > 30s ago — blocking entry")
            _entry_skipped("stale_kline_data")
            return

        # ── Phase 3: Time-to-Expiry Gate ────────────────────────────────────────
        if min_rem < 2.0 and not sig.monster_signal:
            log.debug(f"Entry blocked: Time to expiry ({min_rem:.1f}m) < 2.0m")
            _entry_skipped("time_to_expiry", min_rem=float(min_rem) if min_rem is not None else None)
            return

        # ── Phase 3: Rolling 1-Hour Trade Limit ─────────────────────────────────
        now_sec = int(time.time())
        hour_ago = now_sec - 3600
        recent_trades = [
            t for t in self.state.trade_history
            if t.ts >= hour_ago  # Count all entries placed in last hour
        ]
        if len(recent_trades) >= getattr(Config, "MAX_TRADES_PER_HOUR", 14):
            log.warning(f"Hourly trade limit reached ({len(recent_trades)}/hour) — no new entries")
            _entry_skipped("max_trades_per_hour", recent_trades=len(recent_trades))
            return

        # ── Daily trade counter (absolute, reset at midnight UTC) ──────────────
        today_day = now_sec // 86400
        if self.state.daily_trade_reset_day != today_day:
            self.state.daily_trade_count = 0
            self.state.daily_trade_reset_day = today_day
        if self.state.daily_trade_count >= Config.MAX_DAILY_TRADES:
            log.warning(f"MAX_DAILY_TRADES reached ({self.state.daily_trade_count}/{Config.MAX_DAILY_TRADES}) — no new entries")
            _entry_skipped("max_daily_trades", count=self.state.daily_trade_count)
            return

        # Daily loss limit: sum realized losses from today's trades
        today_start = now_sec - 86400  # rolling 24h window (not calendar day reset)
        daily_loss = sum(
            abs(t.pnl or 0) * (t.entry_price or 0) * (getattr(t, 'size', 0) or 0)
            for t in self.state.trade_history
            if t.ts >= today_start and t.outcome == "LOSS" and t.pnl is not None
        )
        session_start = getattr(self.state, "session_start_balance", 0.0) or 0.0
        max_daily_loss = session_start * Config.DAILY_LOSS_LIMIT_PCT if session_start > 0 else Config.DAILY_LOSS_LIMIT_USD
        self.state.daily_loss_limit_usd = max_daily_loss
        self.state.current_drawdown_usd = daily_loss
        self.state.daily_loss_soft_scale = 1.0
        self.state.trading_halted_reason = None
        if daily_loss >= max_daily_loss:
            pct = (daily_loss / session_start) if session_start > 0 else None
            if getattr(Config, "DAILY_LOSS_HARD_HALT", False):
                if pct is not None:
                    log.warning(
                        "Daily loss limit reached ($%.2f >= $%.2f, %.0f%% of session start) — no new entries",
                        daily_loss,
                        max_daily_loss,
                        pct * 100.0,
                    )
                else:
                    log.warning(
                        "Daily loss limit reached ($%.2f >= $%.2f) — no new entries",
                        daily_loss,
                        max_daily_loss,
                    )
                return

            base_scale = float(getattr(Config, "DAILY_LOSS_SOFT_SCALE", 0.25) or 0.25)
            mult = (daily_loss / max_daily_loss) if max_daily_loss > 0 else 1.0
            if mult >= 3.0:
                self.state.daily_loss_soft_scale = min(base_scale, 0.10)
            elif mult >= 2.0:
                self.state.daily_loss_soft_scale = min(base_scale, 0.15)
            else:
                self.state.daily_loss_soft_scale = base_scale

            self.state.trading_halted_reason = "daily_loss_limit"
            if pct is not None:
                log.warning(
                    "Daily loss limit reached ($%.2f >= $%.2f, %.0f%% of session start) — continuing with size_scale=%.2f",
                    daily_loss,
                    max_daily_loss,
                    pct * 100.0,
                    self.state.daily_loss_soft_scale,
                )
            else:
                log.warning(
                    "Daily loss limit reached ($%.2f >= $%.2f) — continuing with size_scale=%.2f",
                    daily_loss,
                    max_daily_loss,
                    self.state.daily_loss_soft_scale,
                )

            if not sig.monster_signal:
                log.warning(
                    "Daily loss limit active — monster-only mode (scale=%.2f): skipping non-monster entry",
                    self.state.daily_loss_soft_scale,
                )
                _entry_skipped("daily_loss_monster_only", daily_loss=daily_loss, daily_loss_limit=max_daily_loss)
                return

        # ── Exposure cap ──────────────────────────────────────────────────────
        current_exposure = self.state.held_position.size_usd or 0.0
        # position_usd not computed yet — check against conservative max
        if current_exposure >= Config.MAX_EXPOSURE_USD:
            log.warning(f"Exposure limit reached (${current_exposure:.2f} >= ${Config.MAX_EXPOSURE_USD:.2f}) — no new entries")
            _entry_skipped("exposure_cap", current_exposure=current_exposure)
            return

        # Low-balance near-certain override: bypass non-fatal gates when posterior is
        # near-certain (≥97%) and we have a small but real edge. This breaks the
        # zero-trade deadlock at sub-$20 balance where edge gates are too conservative.
        _best_posterior = max(
            sig.posterior_final_up or 0.0,
            sig.posterior_final_down or 0.0,
        )
        skip_gates = list(sig.skip_gates or [])
        if is_monster:
            skip_gates = [g for g in skip_gates if "monster_too_early" not in g]

        _fatal_gates = {"neutral_direction"}
        _has_fatal = any(any(f in g for f in _fatal_gates) for g in skip_gates)
        _override_active = (
            balance < Config.LOW_BALANCE_THRESHOLD_USD
            and _best_posterior >= 0.97
            and sig.target_edge is not None and sig.target_edge > 0.005
            and min_rem >= 2.0 and min_rem <= 12.0
            and sig.direction != "NEUTRAL"
            and not _has_fatal
        )
        if _override_active:
            log.info(
                f"LOW_BAL_NEAR_CERTAIN_OVERRIDE: posterior={_best_posterior:.4f} "
                f"edge={sig.target_edge:.4f} gates_bypassed={sig.skip_gates}"
            )
        elif skip_gates:
            primary_gate = skip_gates[0] if skip_gates else "unknown"
            log.info(f"Entry blocked by gate: {primary_gate} (all={skip_gates})")
            _entry_skipped("gated", primary_gate=primary_gate, gates=list(skip_gates))
            return

        try:
            import json as _json
            self.state.last_entry_telemetry = {
                "ts": int(time.time()),
                "market_slug": getattr(market_info, "slug", None),
                "condition_id": getattr(market_info, "condition_id", None),
                "window_start": self.state.last_window_start_sec,
                "minutes_remaining": float(min_rem) if min_rem is not None else None,
                "direction": sig.direction,
                "posterior_up": sig.posterior_final_up,
                "posterior_down": sig.posterior_final_down,
                "posterior_chosen": _best_posterior,
                "abs_score": getattr(sig, "abs_score", None),
                "signed_score": sig.signed_score,
                "edge": sig.target_edge,
                "monster": bool(sig.monster_signal),
                "override_active": bool(_override_active),
                "gates": list(skip_gates),
                "daily_loss": daily_loss,
                "daily_loss_limit": max_daily_loss,
                "daily_loss_scale": getattr(self.state, "daily_loss_soft_scale", 1.0),
                "daily_loss_reason": getattr(self.state, "trading_halted_reason", None),
            }
            log.info("ENTRY_TELEMETRY %s", _json.dumps(self.state.last_entry_telemetry, separators=(",", ":"), default=str))
        except Exception:
            pass

        if sig.direction == "NEUTRAL":
            _entry_skipped("neutral_direction")
            return

        # Determine token + price (Phase 3: Smart Pricing)
        if sig.direction == "UP":
            token_id  = market_info.yes_token_id
            side_name = "YES"
            entry_px  = self.pm.smart_entry_price(ob.yes_bid, ob.yes_ask, aggressive=sig.monster_signal)
            if entry_px is None and ob.yes_mid:
                entry_px = round(min(ob.yes_mid + 0.01, 0.99), 2)
            entry_px  = round(entry_px or 0.0, 2)
            posterior = sig.posterior_final_up or 0.5
        else:
            token_id  = market_info.no_token_id
            side_name = "NO"
            entry_px  = self.pm.smart_entry_price(ob.no_bid, ob.no_ask, aggressive=sig.monster_signal)
            if entry_px is None and ob.no_mid:
                entry_px = round(min(ob.no_mid + 0.01, 0.99), 2)
            entry_px  = round(entry_px or 0.0, 2)
            posterior = sig.posterior_final_down or 0.5

        if entry_px <= 0.0 or entry_px >= 1.0:
            _entry_skipped("invalid_entry_price", entry_px=entry_px, side=side_name)
            return

        if entry_px >= 0.98:
            _entry_skipped("entry_price_too_high", entry_px=entry_px, side=side_name)
            return

        # Size (FIX #8: Kelly with riskPct floor)
        # Phase 4: Use recalibrated Kelly if available
        metrics = self.state.performance_metrics or {}
        win_rate = metrics.get("win_rate")
        profit_factor = metrics.get("profit_factor")

        position_usd = compute_position_size(
            posterior   = posterior,
            entry_price = entry_px,
            balance     = balance,
            loss_streak = self.state.loss_streak,
            monster_signal = sig.monster_signal,
            win_rate    = win_rate,
            profit_factor = profit_factor,
            kelly_multiplier = self.optimizer.get_kelly_multiplier(),
            book = ob,
            edge = sig.edge_up if sig.direction == "UP" else sig.edge_down,
        )
        if getattr(self.state, "daily_loss_soft_scale", 1.0) < 1.0:
            position_usd *= self.state.daily_loss_soft_scale
        if not position_usd:
            log.info(f"Position size below minimum (bal={balance:.2f})")
            _entry_skipped("position_size_below_min", balance=balance, entry_px=entry_px)
            return

        # Integer shares for CLOB precision (maker_amount must have ≤2 decimals)
        # Use ceil when sizing is at floor levels to avoid dropping below CLOB minimum
        raw_shares = position_usd / entry_px
        shares = math.ceil(raw_shares) if raw_shares < 10 else math.floor(raw_shares)
        if shares < 1:
            _entry_skipped("shares_below_one", shares=shares, entry_px=entry_px)
            return
        # Cap shares so we don't exceed balance
        max_shares = math.floor(balance / entry_px) if entry_px > 0 else shares
        shares = min(shares, max_shares)

        # Store kelly fraction used for logging
        self.state.last_kelly_fraction = position_usd / balance if balance > 0 else 0

        # Paper-trade mode: record hypothetical entry without sending an order.
        if Config.PAPER_TRADE_ENABLED:
            self.state.held_position = HeldPosition(
                side               = side_name,
                token_id           = token_id,
                entry_price        = entry_px,
                avg_entry_price    = entry_px,
                size               = shares,
                size_usd           = position_usd,
                entry_signed_score = sig.signed_score,
                entry_posterior    = posterior,
                condition_id       = market_info.condition_id,
                order_id           = None,
                is_pending         = False,
                placed_at_ts       = int(time.time()),
                intended_entry_price = entry_px,
                market_expiry      = market_info.expiry_ts,
                entry_min_rem      = min_rem,
            )
            self.state.entry_features = sig.to_feature_dict()
            self.state.entry_regime = sig.regime
            self.state.trades_this_window += 1
            self.state.window_trade_count += 1

            self.state.trade_history.append(TradeRecord(
                ts          = int(time.time()),
                side        = side_name,
                entry_price = entry_px,
                exit_price  = None,
                pnl         = None,
                outcome     = "OPEN",
                score       = sig.signed_score,
                window      = self.state.last_window_start_sec or 0,
                size        = shares,
            ))
            if len(self.state.trade_history) > 100:
                self.state.trade_history = self.state.trade_history[-100:]

            log.info(f"PAPER ENTRY: {side_name} @ {entry_px:.3f} size={shares} score={sig.signed_score:.2f}")
            return

        # Checkpoint state before placing order (crash recovery)
        await self.state_mgr.save(self.state)

        # Place order (nonce=0 lets the CLOB API auto-assign)
        if sig.monster_signal:
            order_id = await self.pm.limit_buy(token_id, entry_px, shares, order_type="FOK")
        else:
            order_id = await self.pm.limit_buy(token_id, entry_px, shares, order_type="GTC")

        if not order_id:
            log.error("Entry order failed")
            _entry_skipped("entry_order_failed", entry_px=entry_px, shares=shares)
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
            entry_posterior    = posterior,
            condition_id       = market_info.condition_id,
            order_id           = order_id,
            is_pending         = True,
            placed_at_ts       = int(time.time()),
            intended_entry_price = entry_px,
            market_expiry      = market_info.expiry_ts,
        )
        # Store entry features for matching when closed
        self.state.entry_features = sig.to_feature_dict()
        self.state.entry_regime = sig.regime
        self.state.trades_this_window += 1
        self.state.window_trade_count += 1
        self.state.daily_trade_count += 1

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
    async def _log_cycle_features(self, sig, btc_price: float, min_rem: float, ts: int, ob=None):
        """Phase 5: Log all signals/features to JSONL every cycle."""
        try:
            feats = sig.to_feature_dict()
            feats.update({
                "ts":         ts,
                "btc_price":  round(btc_price, 2),
                "min_rem":    round(min_rem, 2),
                "window":     self.state.last_window_start_sec,
                "strike":     self.state.locked_strike_price,
                "direction":  sig.direction,
                "monster":    sig.monster_signal,
                "regime":     sig.regime,
                "gates":      sig.skip_gates,
                "kelly_mult": self.optimizer.get_kelly_multiplier(),
            })

            if ob is not None:
                feats.update({
                    "pm_ob_source": getattr(ob, "source", None),
                    "pm_ob_yes_age_ms": getattr(ob, "yes_age_ms", None),
                    "pm_ob_no_age_ms": getattr(ob, "no_age_ms", None),
                    "pm_ws_connected": getattr(ob, "ws_connected", None),
                    "pm_ws_last_msg_age_ms": getattr(ob, "ws_last_msg_age_ms", None),
                    "pm_ws_fallback_to_rest": getattr(ob, "ws_fallback_to_rest", None),
                })
            
            # Phase 5: Structured JSON logging
            json_logger.log("feature_snapshot", feats)
            
        except Exception as e:
            log.debug(f"Failed to log features: {e}")

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
                # Determine UP/DOWN outcome vs strike for cross-window momentum
                strike = self.state.locked_strike_price
                if strike and strike > 0:
                    direction = "UP" if cand > strike else "DOWN"
                    self.state.window_outcomes.append(direction)
                    if len(self.state.window_outcomes) > 10:  # keep last 10
                        self.state.window_outcomes = self.state.window_outcomes[-10:]
                    log.info(f"Window outcome: BTC {cand:.2f} vs strike {strike:.2f} → {direction}")

                    # Fill calibration outcome for this window's records
                    try:
                        self.optimizer.fill_calibration_outcome(win_start, cand, strike)
                    except Exception as _co_err:
                        log.debug(f"fill_calibration_outcome error: {_co_err}")

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
                        # Skip if there's a pending exit for this position —
                        # let reconciliation handle it instead of wrongly marking LOSS
                        pos = self.state.held_position
                        if pos.is_pending and pos.exit_reason and pos.side == tr.side:
                            log.info(f"OUTCOME_LOG: skipping LOSS mark for {tr.side} — pending exit ({pos.exit_reason})")
                            continue
                        tr.outcome = "LOSS"
                        tr.exit_price = 0.0
                        tr.pnl = -1.0
                        try:
                            # Use stored entry features/regime
                            regime = getattr(self.state, "entry_regime", "unknown")
                            feats = getattr(self.state, "entry_features", {})
                            kelly = getattr(self.state, "last_kelly_fraction", None)
                            
                            size = tr.size or 0.0
                            await self.state_mgr.record_closed_trade(
                                ts=int(time.time()),
                                market_slug=self.state.last_market_slug or "unknown",
                                size=size,
                                entry_price=tr.entry_price,
                                exit_price=0.0,
                                pnl_usd=-size * tr.entry_price, # Roughly
                                outcome_win=0,
                                regime=regime,
                                features=feats,
                                kelly_fraction=kelly
                            )
                            # Feed optimizer for expired loss
                            if self._last_sig:
                                self.optimizer.log_trade(self._last_sig, "LOSS")
                                self.optimizer.record_trade_pnl(-1.0)
                        except Exception as e:
                            log.error(f"Failed to record expired loss to DB: {e}")
        except Exception as e:
            log.debug(f"Outcome logging error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main():
    metrics_exporter.start_exporter(port=9090)
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
