"""
Phase 7: Nightly Trade Journal — uses OpenRouter (OpenAI-compatible API) to analyze
24h trading performance and generate a structured trade journal with reflections,
saved to /data/nightly_review_{date}.md.
"""
import asyncio
import json
import os
import logging
import time
import math
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional

from openai import OpenAI

from config import Config

log = logging.getLogger("reviewer")

DATA_DIR = "/data" if os.path.isdir("/data") else "."


def _load_structured_logs(hours: int = 24) -> list[dict]:
    path = os.path.join(DATA_DIR, "structured_logs.json")
    if not os.path.exists(path):
        return []
    cutoff = time.time() - hours * 3600
    entries = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("ts", 0) >= cutoff:
                        entries.append(e)
                except Exception:
                    continue
    except Exception as ex:
        log.warning(f"Failed to load structured logs: {ex}")
    return entries


def _load_trade_features(hours: int = 24) -> list[dict]:
    path = os.path.join(DATA_DIR, "trade_features.jsonl")
    if not os.path.exists(path):
        return []
    cutoff = time.time() - hours * 3600
    entries = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("ts", 0) >= cutoff:
                        entries.append(e)
                except Exception:
                    continue
    except Exception as ex:
        log.warning(f"Failed to load trade features: {ex}")
    return entries


def _compute_metrics(trades: list[dict], logs: list[dict]) -> dict:
    """Compute summary statistics from the last 24h of data."""
    outcomes = [t["outcome"] for t in trades]
    wins = outcomes.count(1)
    losses = outcomes.count(0)
    total = len(outcomes)
    win_rate = wins / total if total > 0 else None

    # Gate frequency from signal logs
    gate_counts: dict[str, int] = defaultdict(int)
    signal_logs = [e for e in logs if e.get("type") == "signal"]
    for e in signal_logs:
        for gate in (e.get("data") or {}).get("skip_gates", []):
            gate_counts[gate] += 1

    # Signal correlations: for each feature, compute avg value on wins vs losses
    feature_keys = ["cvd_score", "ofi_score", "flow_accel_score", "imbalance_score", "signed_score"]
    correlations: dict[str, dict] = {}
    for k in feature_keys:
        win_vals = [t["features"].get(k) for t in trades if t["outcome"] == 1 and t["features"].get(k) is not None]
        loss_vals = [t["features"].get(k) for t in trades if t["outcome"] == 0 and t["features"].get(k) is not None]
        avg_win = sum(win_vals) / len(win_vals) if win_vals else None
        avg_loss = sum(loss_vals) / len(loss_vals) if loss_vals else None
        correlations[k] = {"avg_on_win": avg_win, "avg_on_loss": avg_loss}

    # Sharpe: need per-trade pnl from structured logs
    pnl_from_logs = []
    for e in logs:
        if e.get("type") == "exit":
            pnl = (e.get("data") or {}).get("pnl_pct")
            if pnl is not None:
                pnl_from_logs.append(float(pnl))

    sharpe = None
    if len(pnl_from_logs) >= 3:
        mean_pnl = sum(pnl_from_logs) / len(pnl_from_logs)
        variance = sum((x - mean_pnl) ** 2 for x in pnl_from_logs) / len(pnl_from_logs)
        std = math.sqrt(variance) if variance > 0 else None
        if std:
            sharpe = mean_pnl / std

    # Exit reason breakdown from structured logs
    exit_reasons: dict[str, int] = defaultdict(int)
    exit_pnls: dict[str, list[float]] = defaultdict(list)
    for e in logs:
        if e.get("type") == "exit":
            reason = (e.get("data") or {}).get("reason", "UNKNOWN")
            exit_reasons[reason] += 1
            pnl = (e.get("data") or {}).get("pnl_pct")
            if pnl is not None:
                exit_pnls[reason].append(float(pnl))

    # BTC price context from signal logs (last known price, ATR regime)
    btc_last = None
    atr_regime = None
    for e in reversed(signal_logs):
        sig = e.get("data") or {}
        if sig.get("btc_price") and btc_last is None:
            btc_last = sig["btc_price"]
        if sig.get("regime") and atr_regime is None:
            atr_regime = sig["regime"]
        if btc_last and atr_regime:
            break

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_pnl": sum(pnl_from_logs) / len(pnl_from_logs) if pnl_from_logs else None,
        "sharpe": sharpe,
        "signal_correlations": correlations,
        "gate_frequency": dict(sorted(gate_counts.items(), key=lambda x: -x[1])),
        "total_signal_cycles": len(signal_logs),
        "exit_reasons": dict(sorted(exit_reasons.items(), key=lambda x: -x[1])),
        "exit_pnls": {k: {"count": len(v), "avg_pnl": sum(v)/len(v)} for k, v in exit_pnls.items() if v},
        "btc_last": btc_last,
        "atr_regime": atr_regime,
    }


def _build_prompt(metrics: dict, date_str: str) -> str:
    """Build the 9-section NIGHTLY_REVIEW_PROMPT for the Trade Journal."""
    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def fmt_pct(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.2%}"
        return str(v)

    corr_lines = ""
    for sig, vals in metrics.get("signal_correlations", {}).items():
        corr_lines += f"  {sig}: avg_on_win={fmt(vals['avg_on_win'])}, avg_on_loss={fmt(vals['avg_on_loss'])}\n"

    gate_lines = ""
    for gate, count in list(metrics.get("gate_frequency", {}).items())[:8]:
        gate_lines += f"  {gate}: {count} times\n"

    exit_lines = ""
    for reason, count in list(metrics.get("exit_reasons", {}).items())[:10]:
        pnl_info = metrics.get("exit_pnls", {}).get(reason, {})
        avg_pnl_str = fmt_pct(pnl_info.get("avg_pnl")) if pnl_info else "N/A"
        exit_lines += f"  {reason}: {count} exits (avg PnL: {avg_pnl_str})\n"

    return f"""You are the senior risk manager and performance analyst for a quantitative BTC 15-minute binary prediction market trading bot (Polymarket). You are writing the **Nightly Trade Journal** — a daily institutional-grade review.

Date: {date_str}
BTC Last Price: {fmt(metrics.get('btc_last'))}
ATR Regime: {fmt(metrics.get('atr_regime'))}

## Last 24h Performance Summary

- Total trades: {metrics['total_trades']}
- Wins: {metrics['wins']} | Losses: {metrics['losses']}
- Win rate: {fmt_pct(metrics['win_rate'])}
- Average PnL per trade: {fmt_pct(metrics['avg_pnl'])}
- Sharpe ratio: {fmt(metrics['sharpe'])}
- Total signal evaluation cycles: {metrics['total_signal_cycles']}

## Signal Feature Correlations (avg feature value on wins vs losses)

{corr_lines or "  No trade data available"}

## Gate Fire Frequency (top blocking gates)

{gate_lines or "  No gate data available"}

## Exit Reason Breakdown

{exit_lines or "  No exit data available"}

---

## Your Task

Produce a **Nightly Trade Journal** with exactly these 9 sections. Be concise, quantitative, and direct. Format as structured markdown.

### Section 1: Performance Assessment
Is the bot trading profitably? Are there concerning patterns (e.g., clustering of losses, degrading win rate, unusual PnL distribution)? Compare today's Sharpe to a 1.0 baseline. If win rate < 55% or Sharpe < 0.5, flag explicitly.

### Section 2: Signal Quality Analysis
Which signals (CVD, OFI, flow_accel, imbalance, signed_score) are most predictive today? Which may be noisy? Cite the avg_on_win vs avg_on_loss differentials. If a signal's win/loss spread is < 0.3, flag as potentially uninformative.

### Section 3: Gate Optimization
Are gates firing too often (blocking good trades) or too rarely (letting bad trades through)? What threshold adjustments could help? Focus on the top 3–5 gates by frequency.

### Section 4: Exit Strategy Audit
Audit the exit reason breakdown. Specifically evaluate:
- **Monster early-hold grace**: Are monster-signal entries (score >= 8.0, posterior >= 0.90) being paper-handed by MODEL_REVERSAL or microstructure exits within 60s of entry? The bot has a 45s MONSTER_STRIKE_GRACE_SEC and MODEL_REVERSAL suppression for < 5 min remaining — are these working?
- **TP1 adaptive sizing**: Is TP1 exiting too much (full exit when conviction is high) or too little (letting winners reverse)?
- **Hard stop vs trailing**: Is HARD_STOP firing more than TRAIL_PRICE_STOP? That indicates trailing is too loose or hard stop is too tight.
- **Dust/write-off frequency**: Are DUST_WRITEOFF exits occurring? These indicate fractional share rounding issues.

### Section 5: Risk Flags & Drawdown Analysis
Any patterns suggesting over-fitting, excessive risk, or degraded signal quality? Check for:
- Loss streaks >= 3
- Session drawdown approaching 30%
- Single-trade losses exceeding -15% (should have been stopped earlier)
- Gate frequency anomalies (e.g., score_low dominating = model is consistently weak)

### Section 6: BTC Cross-Validation Audit
For each trade, cross-reference the bot's directional bet with what BTC actually did:
- Did BTC price action **corroborate** the trade direction (e.g., bot bought YES/UP and BTC moved above strike)?
- Did BTC price action **contradict** the trade (e.g., bot bought YES/UP but BTC dropped below strike)?
- Were there **whipsaw** scenarios where BTC crossed the strike multiple times, making direction uncertain?
- If the bot lost on a trade where BTC confirmed the direction, the exit strategy is the problem (not the signal).
- If the bot lost on a trade where BTC contradicted the direction, the signal/entry is the problem.

### Section 7: Actionable Recommendations
3–5 specific, concrete parameter changes or logic improvements ranked by expected impact. Each recommendation must include:
- The specific config parameter or code change
- The current value and proposed value
- Expected impact (quantitative if possible)
- Risk of the change

### Section 8: Quant Projections & Next-Day Monitoring Plan
Based on today's data and current BTC price/regime, provide:
- Expected trade count for next 24h (based on signal cycle frequency and gate pass rate)
- Key levels to watch (BTC price levels where the bot's edge is strongest/weakest)
- Regime forecast: is the current ATR regime likely to persist or shift?
- Specific monitoring instructions: what pattern to watch for, and what action to consider if it occurs

### Section 9: Trade Journal Reflections
Write a first-person narrative journal entry reflecting on today's trading session. Include:
- **Key lessons** from each trade (what worked, what didn't)
- **Psychological / risk-management notes** (was I overconfident after a streak? Did I panic on a drawdown?)
- **"What I would do differently"** for any loss or early exit — be specific about which exit reason was premature and what the counterfactual outcome was
- **Forward-looking thesis** for the next 24h: what market regime am I expecting, what signals am I trusting, and what am I cautious about?

Write this section in a reflective, first-person style as if you are the bot's operator reviewing your own decisions. Be honest about mistakes and clear about convictions.
"""


async def run_nightly_review(session=None) -> Optional[str]:
    """Main entry point: load data, compute metrics, call OpenRouter LLM, save + send."""
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    log.info(f"Starting nightly trade journal for {date_str}")

    api_key = Config.OPENROUTER_API_KEY
    if not api_key:
        log.warning("OPENROUTER_API_KEY not set — skipping nightly trade journal")
        return None

    logs = _load_structured_logs(hours=24)
    trades = _load_trade_features(hours=24)
    metrics = _compute_metrics(trades, logs)

    prompt = _build_prompt(metrics, date_str)
    model_name = Config.NIGHTLY_JOURNAL_MODEL

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://btc-15-quant.up.railway.app",
                "X-OpenRouter-Title": "BTC 15m Quant Bot",
            },
        )
        # OpenAI client is synchronous; run in thread to avoid blocking event loop
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
        )
        review_text = response.choices[0].message.content
    except Exception as e:
        log.error(f"OpenRouter API call failed ({model_name}): {e}")
        return None

    # Save to file
    out_path = os.path.join(DATA_DIR, f"nightly_review_{date_str}.md")
    try:
        with open(out_path, "w") as f:
            f.write(f"# Nightly Trade Journal — {date_str}\n\n")
            f.write(f"_Generated by {model_name} via OpenRouter_\n\n")
            f.write(f"## Input Metrics\n\n")
            if metrics['win_rate'] is not None:
                f.write(f"- Trades: {metrics['total_trades']} | Win rate: {metrics['win_rate']:.2%} | Sharpe: {metrics['sharpe']}\n\n")
            else:
                f.write(f"- Trades: {metrics['total_trades']} (insufficient data for win rate)\n\n")
            f.write(f"## Analysis\n\n")
            f.write(review_text)
        log.info(f"Nightly trade journal saved to {out_path}")
    except Exception as e:
        log.error(f"Failed to save nightly trade journal: {e}")

    # Send summary to Telegram (first 800 chars)
    if session:
        from utils import send_telegram, AlertTier
        summary = review_text[:800] + ("..." if len(review_text) > 800 else "")
        await send_telegram(
            session,
            f"� <b>Nightly Trade Journal — {date_str}</b>\n\n{summary}",
            tier=AlertTier.INFO
        )

    return review_text
