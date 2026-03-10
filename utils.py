"""
Utilities: structured logging setup, Telegram alerts, and small helpers.
"""

import logging
import sys
import time
import os
from typing import Optional
import aiohttp

from config import Config

log = logging.getLogger(__name__)


class Timer:
    """Helper for Phase 4 latency tracking."""
    def __init__(self, name: str, state_latencies: dict):
        self.name = name
        self.latencies = state_latencies
        self.start_t = None

    def __enter__(self):
        self.start_t = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_t:
            dt = (time.monotonic() - self.start_t) * 1000
            self.latencies[self.name] = round(dt, 1)


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging():
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    
    # Force sys.stdout to be unbuffered so logs appear immediately in Docker/Railway
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    # Quiet noisy libs
    for lib in ("aiohttp", "asyncio", "urllib3", "websockets"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ── Telegram ──────────────────────────────────────────────────────────────────

from enum import Enum
import json

class AlertTier(Enum):
    INFO = "🟢 [INFO]"
    WARN = "🟡 [WARN]"
    CRITICAL = "🔴 [CRITICAL]"

async def send_telegram(
    session: aiohttp.ClientSession,
    message: str,
    tier: AlertTier = AlertTier.INFO,
    parse_mode: str = "HTML",
):
    if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
    
    # Prefix message with tier exception if strictly INFO
    full_message = f"{tier.value} {message}" if tier != AlertTier.INFO else message

    try:
        async with session.post(url, json={
            "chat_id":    Config.TELEGRAM_CHAT_ID,
            "text":       full_message,
            "parse_mode": parse_mode,
        }, timeout=aiohttp.ClientTimeout(total=5)) as r:
            if r.status != 200:
                log.warning(f"Telegram status {r.status}")
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")

class StructuredJSONLogger:
    """Phase 5: Structured JSON logging to persistent storage on /data."""
    def __init__(self, log_dir="/data"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.filepath = os.path.join(self.log_dir, "structured_logs.json")

    def _rotate_if_needed(self, max_bytes: int = 10 * 1024 * 1024):
        """Rotate log file if it exceeds max_bytes (default 10MB)."""
        try:
            if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > max_bytes:
                rotated = self.filepath + ".1"
                os.replace(self.filepath, rotated)
                log.info(f"Log rotated: {self.filepath} -> {rotated}")
        except Exception as e:
            log.warning(f"Log rotation failed: {e}")

    def log(self, event_type: str, data: dict):
        self._rotate_if_needed()
        entry = {
            "ts": int(time.time()),
            "type": event_type,
            "data": data
        }
        try:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            log.error(f"Failed to write structured log: {e}")

# Create global instance
json_logger = StructuredJSONLogger()


def fmt_entry(
    side: str, price: float, shares: float, score: float,
    edge: float, posterior: float, window: int, balance: float
) -> str:
    return (
        f"🟢 <b>ENTRY</b> | BTC 15m\n"
        f"Side: <b>{side}</b> @ {price:.3f}\n"
        f"Shares: {shares:.1f} | Score: {score:.2f} | Edge: {edge:.3f}\n"
        f"Posterior: {posterior*100:.1f}% | Balance: ${balance:.2f}\n"
        f"Window: {window}"
    )


def fmt_exit(
    side: str, exit_px: float, entry_px: float,
    pnl_pct: float, reason: str, balance: float
) -> str:
    icon = "🔵" if pnl_pct >= 0 else "🔴"
    
    # Enhanced reason details for new logic
    reason_details = {
        "FORCED_LATE_EXIT": " (<1 min remaining, posterior hold if >0.70)",
        "TAKE_SMALL_PROFIT": " (2% unrealized gain)",
        "FORCED_DRAWDOWN": " (posterior drop, tightened tolerances)",
        "TRAIL_POSTERIOR": " (confidence drop, tightened guard)",
        "HARD_STOP": " (-25% loss limit)",
        "VPIN_TOXIC_TIME": " (high VPIN toxicity)",
    }
    detail = reason_details.get(reason, "")
    
    return (
        f"{icon} <b>EXIT</b> | {side} | {reason}{detail}\n"
        f"Entry: {entry_px:.3f} → Exit: {exit_px:.3f} | "
        f"PnL: {pnl_pct*100:.1f}%\n"
        f"Balance: ${balance:.2f}"
    )


def fmt_halt(streak: int, balance: float) -> str:
    return (
        f"🛑 <b>TRADING HALTED</b>\n"
        f"Loss streak: {streak} consecutive losses\n"
        f"Balance: ${balance:.2f}\n"
        f"Manual reset required (delete state.db or run with --reset)."
    )


def fmt_status(
    position: str, score: float, posterior_up: float,
    yes_px: float, no_px: float, balance: float,
    minutes_rem: float, skip_gates: list,
    question: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    gates_str = ", ".join(skip_gates) if skip_gates else "CLEAR"
    title_line = f"📊 <b>STATUS</b>\n"
    if question:
        title_line = f"📊 <b>{question}</b>\n"

    url_line = f"<a href='{url}'>View on Polymarket</a>\n" if url else ""

    return (
        f"{title_line}"
        f"{url_line}"
        f"Position: {position} | Rem: {minutes_rem:.1f}min\n"
        f"Score: {score:.2f} | PostUp: {posterior_up*100:.1f}%\n"
        f"YES: {yes_px:.3f} | NO: {no_px:.3f}\n"
        f"Balance: ${balance:.2f}\n"
        f"Gates: {gates_str}"
    )


def fmt_performance_summary(
    win_rate: float, total_trades: int, total_pnl: float,
    balance: float, sharpe: float, kelly_mult: float,
) -> str:
    return (
        f"📈 <b>PERFORMANCE SUMMARY</b>\n"
        f"Trades: {total_trades} | Win Rate: {win_rate*100:.1f}%\n"
        f"Total PnL: ${total_pnl:.2f} | Sharpe: {sharpe:.2f}\n"
        f"Kelly Mult: {kelly_mult:.2f}x | Balance: ${balance:.2f}"
    )


def fmt_signal_decay_alert(decayed_signals: list[str], accuracies: dict) -> str:
    lines = [f"⚠️ <b>SIGNAL DECAY DETECTED</b>"]
    for sig in decayed_signals[:5]:
        stats = accuracies.get(sig, {})
        acc = stats.get("accuracy", 0)
        n = stats.get("n_samples", 0)
        lines.append(f"  {sig}: {acc*100:.1f}% accuracy (n={n})")
    return "\n".join(lines)


# ── Window helpers ────────────────────────────────────────────────────────────

def current_window_start(now_ts: Optional[int] = None) -> int:
    t = now_ts or int(time.time())
    return (t // Config.WINDOW_SEC) * Config.WINDOW_SEC


def minutes_remaining(now_ts: Optional[int] = None) -> float:
    t = now_ts or int(time.time())
    ws = current_window_start(t)
    we = ws + Config.WINDOW_SEC
    return max(0.0, (we - t) / 60.0)


def fmt_engine_block(
    res, state, btc_price: float, min_rem: float, 
    balance: float, runtime_ms: int,
    decision: str = "NO_TRADE", exec_bool: bool = False, 
    mode: str = "none", exit_reason: str = "HOLD"
) -> str:
    """Produces the highly detailed ASCII engine log block."""
    from datetime import datetime
    now_ts = int(time.time())
    ws     = current_window_start(now_ts)
    we     = ws + Config.WINDOW_SEC
    
    # Delta symbols
    s_delta = f" (+{res.score_delta:.1f})" if res.score_delta and res.score_delta > 0 else (f" ({res.score_delta:.1f})" if res.score_delta else "")
    p_delta = f" (+{res.price_delta:.1f})" if res.price_delta and res.price_delta > 0 else (f" ({res.price_delta:.1f})" if res.price_delta else "")
    
    gates = ", ".join(res.skip_gates) if res.skip_gates else "none"
    
    # Micro components formatted
    micro_line = (
        f"Micro: cvdS={res.cvd_score:.1f} | cvd={res.cvd:.0f} | "
        f"obi={getattr(res, 'obi', 0.0):.4f} | "
        f"vpinProxy={res.vpin_proxy:.4f} | deepImb={res.deep_imbalance:.4f} | "
        f"macd={res.macd_score:.1f}"
    )

    return (
        f"\n=== BTC 15m Quant Engine ===\n"
        f"Position: {state.held_position.side or 'FLAT'}\n"
        f"now={now_ts} | window={ws}-{we} | rem={min_rem:.1f}min\n"
        f"Strike={res.strike_price or 0:.2f} | Price={btc_price:.2f} | Dist={res.distance or 0:.1f}\n"
        f"StrikeSource: {res.strike_source} (locked={'true' if state.locked_strike_price else 'false'})\n"
        f"Polymarket: https://polymarket.com/event/{state.last_market_slug}\n"
        f"Vol: ATR5m={res.atr14 or 0:.1f} Regime={res.regime} ExpMove={res.expected_move or 0:.1f} MinRem={min_rem:.1f}\n"
        f"Bayesian: z={res.z_score or 0:.4f} PostUp={res.posterior_final_up or 0:.4f} PostDown={res.posterior_final_down or 0:.4f}\n"
        f"Market: YES={res.yes_mid or 0:.2f} NO={res.no_mid or 0:.2f}\n"
        f"Final: Up={res.posterior_final_up or 0:.4f} Down={res.posterior_final_down or 0:.4f}\n"
        f"Edge: {res.target_edge or 0:.4f} (Min Req: {res.required_edge:.3f})\n"
        f"SignedScore: {res.signed_score:.1f}{s_delta} (Min Req: {res.min_score:.0f}).\n"
        f"ScoreDelta: {res.score_delta or 'N/A'}\n"
        f"PriceDelta: {res.price_delta or 'N/A'}\n"
        f"Balance: USDC={balance:.2f}\n"
        f"TP: price={Config.TAKE_PROFIT_PRICE} timeRiskLow=0.85 maxLoss={Config.FORCED_LATE_LOSS_PCT}\n"
        f"{micro_line}\n"
        f"Sizing: {res.sizing or 0} (Min Notional: {Config.MIN_TRADE_USD}).\n"
        f"Decision: {decision} | Exec={str(exec_bool).lower()} | Mode={mode} | Exit={exit_reason}\n"
        f"SkipGates: {gates}\n"
        f"REDEEM_READY_PREVIEW: 0\n"
        f"RUNTIME: {runtime_ms}ms\n"
    )

def window_start_iso(window_start: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(window_start, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def window_end_iso(window_start: int) -> str:
    return window_start_iso(window_start + Config.WINDOW_SEC)


def fmt_pnl_dashboard(trade_history: list, balance: float) -> str:
    """Phase 4: Visual PnL summary for the console."""
    if not trade_history:
        return f"\n--- PnL Dashboard ---\nBalance: ${balance:.2f}\nTrades: 0\nWin Rate: 0%\n"

    wins = [t for t in trade_history if t.outcome == "WIN"]
    losses = [t for t in trade_history if t.outcome == "LOSS"]
    total = len(wins) + len(losses)
    wr = (len(wins) / total * 100) if total > 0 else 0

    history_str = ""
    for t in trade_history[-10:]:
        icon = "🟢" if t.outcome == "WIN" else "🔴" if t.outcome == "LOSS" else "⏳"
        pnl = f"{t.pnl*100:+.1f}%" if t.pnl is not None else "OPEN"
        history_str += f"{icon} {pnl} | "

    return (
        f"\n{'='*50}\n"
        f"  📊 DASHBOARD\n"
        f"  Balance:  ${balance:,.2f}\n"
        f"  Trades:   {total}  (W: {len(wins)} / L: {len(losses)})\n"
        f"  Win Rate: {total > 0 and wr:.1f}%\n"
        f"  History:  {history_str[:-3]}\n"
        f"{'='*50}\n"
    )
