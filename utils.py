"""
Utilities: structured logging setup, Telegram alerts, and small helpers.
"""

import logging
import sys
import time
from typing import Optional
import aiohttp

from config import Config

log = logging.getLogger(__name__)


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging():
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Quiet noisy libs
    for lib in ("aiohttp", "asyncio", "urllib3", "websockets"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ── Telegram ──────────────────────────────────────────────────────────────────

async def send_telegram(
    session: aiohttp.ClientSession,
    message: str,
    parse_mode: str = "HTML",
):
    if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={
            "chat_id":    Config.TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": parse_mode,
        }, timeout=aiohttp.ClientTimeout(total=5)) as r:
            if r.status != 200:
                log.warning(f"Telegram status {r.status}")
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


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
    return (
        f"{icon} <b>EXIT</b> | {side} | {reason}\n"
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
        f"Micro: conv={res.cvd_score:.1f} | cvd={res.cvd:.0f} | "
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
