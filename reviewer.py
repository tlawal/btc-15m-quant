"""
Phase 7: Nightly AI Reviewer — uses Claude API to analyze 24h trading performance
and generate actionable recommendations saved to /data/nightly_review_{date}.md.
"""
import json
import os
import logging
import time
import math
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional

import anthropic

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

    # Average PnL (if features contain pnl)
    pnls = []
    for t in trades:
        feats = t.get("features", {})
        # pnl might be logged separately via structured logs
        pass

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
    }


def _build_prompt(metrics: dict, date_str: str) -> str:
    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    corr_lines = ""
    for sig, vals in metrics.get("signal_correlations", {}).items():
        corr_lines += f"  {sig}: avg_on_win={fmt(vals['avg_on_win'])}, avg_on_loss={fmt(vals['avg_on_loss'])}\n"

    gate_lines = ""
    for gate, count in list(metrics.get("gate_frequency", {}).items())[:8]:
        gate_lines += f"  {gate}: {count} times\n"

    return f"""You are reviewing a quantitative BTC 15-minute binary prediction market trading bot.

Date: {date_str}

## Last 24h Performance Summary

- Total trades: {metrics['total_trades']}
- Wins: {metrics['wins']} | Losses: {metrics['losses']}
- Win rate: {fmt(metrics['win_rate'])}
- Average PnL per trade: {fmt(metrics['avg_pnl'])}
- Sharpe ratio: {fmt(metrics['sharpe'])}
- Total signal evaluation cycles: {metrics['total_signal_cycles']}

## Signal Feature Correlations (avg feature value on wins vs losses)

{corr_lines or "  No trade data available"}

## Gate Fire Frequency (top blocking gates)

{gate_lines or "  No gate data available"}

## Your Task

Analyze this data and provide:

1. **Performance Assessment** — Is the bot trading profitably? Are there concerning patterns?
2. **Signal Quality Analysis** — Which signals (CVD, OFI, flow_accel, imbalance) are most predictive? Which may be noisy?
3. **Gate Optimization** — Are gates firing too often or too rarely? What threshold adjustments could help?
4. **Actionable Recommendations** — 3–5 specific, concrete parameter changes or logic improvements ranked by expected impact.
5. **Risk Flags** — Any patterns that suggest the bot is over-fit, taking on too much risk, or has degraded signal quality?

Be concise, quantitative, and direct. Format your response as structured markdown.
"""


async def run_nightly_review(session=None) -> Optional[str]:
    """Main entry point: load data, compute metrics, call Claude, save + send."""
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    log.info(f"Starting nightly review for {date_str}")

    api_key = Config.ANTHROPIC_API_KEY
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — skipping nightly review")
        return None

    logs = _load_structured_logs(hours=24)
    trades = _load_trade_features(hours=24)
    metrics = _compute_metrics(trades, logs)

    prompt = _build_prompt(metrics, date_str)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        review_text = message.content[0].text
    except Exception as e:
        log.error(f"Claude API call failed: {e}")
        return None

    # Save to file
    out_path = os.path.join(DATA_DIR, f"nightly_review_{date_str}.md")
    try:
        with open(out_path, "w") as f:
            f.write(f"# Nightly AI Review — {date_str}\n\n")
            f.write(f"_Generated by claude-sonnet-4-6_\n\n")
            f.write(f"## Input Metrics\n\n")
            f.write(f"- Trades: {metrics['total_trades']} | Win rate: {metrics['win_rate']:.2%} | Sharpe: {metrics['sharpe']}\n\n" if metrics['win_rate'] else f"- Trades: {metrics['total_trades']} (insufficient data for win rate)\n\n")
            f.write(f"## Analysis\n\n")
            f.write(review_text)
        log.info(f"Nightly review saved to {out_path}")
    except Exception as e:
        log.error(f"Failed to save nightly review: {e}")

    # Send summary to Telegram (first 800 chars)
    if session:
        from utils import send_telegram, AlertTier
        summary = review_text[:800] + ("..." if len(review_text) > 800 else "")
        await send_telegram(
            session,
            f"📊 <b>Nightly AI Review {date_str}</b>\n\n{summary}",
            tier=AlertTier.INFO
        )

    return review_text
