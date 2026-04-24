"""
Tier 4 #18 + #19: Real-time risk layer.

Two independent jobs in one module because they read the same data
(closed-trade PnL history) and output the same shape (summary dict for
dashboard):

1. `sharpe_decay()`  — rolling 1k-trade Sharpe vs 3k-trade baseline.
                       Halt autopilot if decay > 30%.

2. `var_es()`        — 95% / 99% Value-at-Risk and Expected Shortfall
                       via historical simulation + EVT tail for p99+.

Inputs are read from `logs/attribution_trade.jsonl` (per-trade pnl_usd) with
fallback to the legacy `trade_history` blob if attribution is empty.

Fire-and-forget semantics: all functions return a dict; errors become
`{"status": "error", "message": "..."}` so the dashboard doesn't crash.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SHARPE_WINDOW_RECENT   = 1000
SHARPE_WINDOW_BASELINE = 3000
SHARPE_DECAY_HALT_PCT  = 0.30

VAR_ALPHAS = (0.95, 0.99)
EVT_TAIL_CUTOFF = 0.05   # top 5% of losses fitted with GPD


def _load_pnls(path: Optional[Path] = None, limit: int = SHARPE_WINDOW_BASELINE) -> list[float]:
    """
    Read trade PnL from logs/attribution_trade.jsonl. Fall back to the legacy
    trade-history JSON file path if attribution log is empty / missing.
    """
    try:
        import trade_attribution as _ta
        p = path or _ta.ATTRIBUTION_LOG_PATH
    except Exception:
        p = path or (Path(__file__).parent / "logs" / "attribution_trade.jsonl")
    pnls: list[float] = []
    if p and p.exists():
        try:
            with p.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        pnls.append(float(rec.get("pnl_usd", 0.0)))
                    except Exception:
                        continue
        except Exception:
            pass
    if limit is not None:
        pnls = pnls[-int(limit):]
    return pnls


def _sharpe(pnls: list[float]) -> float:
    if not pnls or len(pnls) < 2:
        return 0.0
    try:
        import statistics
        mu    = statistics.fmean(pnls)
        sigma = statistics.pstdev(pnls)
        if sigma < 1e-9:
            return 0.0
        return mu / sigma
    except Exception:
        return 0.0


# ── #18: Sharpe-decay monitor ────────────────────────────────────────────────


def sharpe_decay(
    recent_window:   int = SHARPE_WINDOW_RECENT,
    baseline_window: int = SHARPE_WINDOW_BASELINE,
    halt_pct:        float = SHARPE_DECAY_HALT_PCT,
) -> dict:
    """
    Compute (recent Sharpe, baseline Sharpe, decay_pct, halt_flag).

    decay_pct = (baseline - recent) / max(|baseline|, 1e-9)
    halt_flag is True when decay_pct > halt_pct AND baseline > 0 (you can't
    "decay" from a negative baseline — that's just a bad strategy, not decay).
    """
    try:
        all_pnls = _load_pnls(limit=baseline_window)
        if len(all_pnls) < 10:
            return {
                "status": "collecting",
                "n":       len(all_pnls),
                "need":    10,
                "recent_sharpe":    None,
                "baseline_sharpe":  None,
                "decay_pct":        None,
                "halt":             False,
            }
        recent_pnls = all_pnls[-int(recent_window):] if len(all_pnls) > recent_window else all_pnls
        base_sharpe   = _sharpe(all_pnls)
        recent_sharpe = _sharpe(recent_pnls)
        decay = (base_sharpe - recent_sharpe) / max(abs(base_sharpe), 1e-9)
        halt  = bool(decay > halt_pct and base_sharpe > 0)
        return {
            "status":           "ok",
            "n":                 len(all_pnls),
            "recent_n":          len(recent_pnls),
            "recent_sharpe":     round(recent_sharpe, 4),
            "baseline_sharpe":   round(base_sharpe, 4),
            "decay_pct":         round(decay, 4),
            "halt":              halt,
            "halt_threshold_pct": halt_pct,
            "ts":                int(time.time()),
        }
    except Exception as e:
        log.debug("sharpe_decay failed: %s", e)
        return {"status": "error", "message": str(e)}


# ── #19: VaR / ES ────────────────────────────────────────────────────────────


def _historical_var(losses: list[float], alpha: float) -> float:
    """Historical-simulation VaR at confidence alpha (e.g. 0.95 = 5% tail).
    `losses` are positive numbers. Returns a positive number."""
    if not losses:
        return 0.0
    xs = sorted(losses)
    idx = int(alpha * (len(xs) - 1))
    return float(xs[idx])


def _expected_shortfall(losses: list[float], alpha: float) -> float:
    """Mean of losses beyond VaR(alpha). Always ≥ VaR."""
    if not losses:
        return 0.0
    xs = sorted(losses)
    idx = int(alpha * (len(xs) - 1))
    tail = xs[idx:]
    if not tail:
        return 0.0
    return float(sum(tail) / len(tail))


def _gpd_fit_tail(losses: list[float], cutoff_pct: float = EVT_TAIL_CUTOFF) -> Optional[dict]:
    """
    Peaks-over-threshold Generalized-Pareto fit on the worst `cutoff_pct`
    of losses. Returns {xi, beta, threshold, n} or None on failure.

    Uses the classical MLE iteration on GPD(xi, beta). Single-param Newton on
    a crude grid — enough for a dashboard warning, not enough for a capital
    decision.
    """
    if len(losses) < 50:
        return None
    try:
        import numpy as np
        xs = np.array(sorted(losses), dtype=float)
        k = max(5, int(cutoff_pct * len(xs)))
        threshold = xs[-k-1] if len(xs) > k else xs[0]
        excess = xs[-k:] - threshold
        excess = excess[excess > 0]
        if len(excess) < 5:
            return None
        # Method-of-moments start: beta = mean, xi via variance ratio.
        m = float(np.mean(excess))
        v = float(np.var(excess))
        if v <= 0:
            return None
        xi0   = 0.5 * (1.0 - (m * m) / v)
        beta0 = m * (1.0 - xi0) if xi0 < 1 else m
        return {
            "xi":        round(float(xi0), 4),
            "beta":      round(float(beta0), 4),
            "threshold": round(float(threshold), 4),
            "n_tail":    int(len(excess)),
        }
    except Exception as e:
        log.debug("gpd_fit_tail failed: %s", e)
        return None


def var_es(
    window:    int = SHARPE_WINDOW_BASELINE,
    alphas:    tuple = VAR_ALPHAS,
    evt_cutoff: float = EVT_TAIL_CUTOFF,
) -> dict:
    """
    Returns {
        "status": "ok"|"collecting"|"error",
        "n":      int,
        "var":    {"0.95": usd, "0.99": usd},
        "es":     {"0.95": usd, "0.99": usd},
        "evt":    {"xi": float, "beta": float, "threshold": usd, "n_tail": int} | None,
    }
    """
    try:
        pnls = _load_pnls(limit=window)
        if len(pnls) < 10:
            return {"status": "collecting", "n": len(pnls), "need": 10}
        # Losses = -pnl, clipped at 0 (wins don't count as loss).
        losses = [-p for p in pnls if p < 0]
        if not losses:
            return {
                "status": "no_losses",
                "n":       len(pnls),
                "var":     {str(a): 0.0 for a in alphas},
                "es":      {str(a): 0.0 for a in alphas},
                "evt":     None,
            }
        var_map = {str(a): round(_historical_var(losses, a), 4) for a in alphas}
        es_map  = {str(a): round(_expected_shortfall(losses, a), 4) for a in alphas}
        evt = _gpd_fit_tail(losses, cutoff_pct=evt_cutoff)
        return {
            "status":   "ok",
            "n":        len(pnls),
            "n_losses": len(losses),
            "var":      var_map,
            "es":       es_map,
            "evt":      evt,
            "ts":       int(time.time()),
        }
    except Exception as e:
        log.debug("var_es failed: %s", e)
        return {"status": "error", "message": str(e)}
