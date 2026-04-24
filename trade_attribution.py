"""
Tier 2 #16: Per-trade alpha decomposition.

Transforms a closed trade into a 4-bucket PnL breakdown so we can answer
"which piece lost how much" instead of "it lost." Citadel / DE Shaw level
decomposition is the baseline for any strategy that wants to outlive a
quarter of bad performance.

    trade_pnl_usd = signal_pnl + prior_pnl + execution_pnl + residual

Bucket definitions (all in USD):

  signal_pnl:    (posterior − market_prior) × direction × |realized_move|
                 — how much we made because our *model* picked a different
                   side than the market, and that side was right.

  prior_pnl:     (market_prior − 0.5) × direction × |realized_move|
                 — how much we made because the *market* was already leaning
                   the right way at entry. This is the piece a passive
                   "buy-whatever's-cheap" strategy captures.

  execution_pnl: −(entry_slippage + exit_slippage + fees)
                 — Polymarket taker fees, spread-cross, partial-fill leakage.
                   Always ≤ 0.

  residual:      trade_pnl_usd − (signal_pnl + prior_pnl + execution_pnl)
                 — everything the model can't explain: vol collapse, timing,
                   MM inventory moves, post-decision posterior drift.

Writes each decomposition to `logs/attribution_trade.jsonl`.

Not confused with `attribution.py` (the existing feature-importance logistic-
regression attributor over the closed-trades DB). That file answers
"which signals matter"; this file answers "for this trade, where did the
PnL go?"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

ATTRIBUTION_LOG_PATH = Path(__file__).parent / "logs" / "attribution_trade.jsonl"


@dataclass
class Decomposition:
    trade_id:      str
    side:          str
    size:          float
    entry_price:   float
    exit_price:    float
    pnl_usd:       float
    signal_pnl:    float
    prior_pnl:     float
    execution_pnl: float
    residual:      float
    meta:          dict

    def to_dict(self) -> dict:
        return {
            "trade_id":      self.trade_id,
            "side":          self.side,
            "size":          self.size,
            "entry_price":   self.entry_price,
            "exit_price":    self.exit_price,
            "pnl_usd":       self.pnl_usd,
            "signal_pnl":    self.signal_pnl,
            "prior_pnl":     self.prior_pnl,
            "execution_pnl": self.execution_pnl,
            "residual":      self.residual,
            "meta":          self.meta,
        }


def decompose(
    *,
    side: str,                 # "YES" | "NO"
    size: float,               # shares
    entry_price: float,        # cost basis per share, ∈ (0,1)
    exit_price: float,         # realized exit per share, ∈ (0,1)
    posterior: Optional[float],
    mkt_prior: Optional[float],
    entry_slippage_usd: float = 0.0,
    exit_slippage_usd:  float = 0.0,
    fees_usd:           float = 0.0,
    trade_id: str = "",
    meta: Optional[dict] = None,
) -> Decomposition:
    """
    Decompose a single closed trade's PnL into the 4 buckets.

    Convention: posterior and mkt_prior are both expressed as P(the side we
    took wins). So for a YES position, posterior = P(YES wins); for a NO
    position, posterior = P(NO wins).
    """
    direction  = +1.0 if str(side).upper() == "YES" else -1.0
    _size      = max(0.0, float(size))
    _entry     = float(entry_price)
    _exit      = float(exit_price)

    # For YES: (exit - entry) positive = win. For NO: (exit - entry) positive =
    # YES price rose, NO position lost → direction=−1 flips sign correctly.
    realized_move_per_share = (_exit - _entry) * direction
    raw_pnl_usd = realized_move_per_share * _size                # excludes exec
    pnl_usd     = raw_pnl_usd - entry_slippage_usd - exit_slippage_usd - fees_usd

    p_sig = None if posterior is None else max(0.0, min(1.0, float(posterior)))
    p_mkt = None if mkt_prior is None else max(0.0, min(1.0, float(mkt_prior)))

    # Edge decomposition. Both components in the same direction (posterior and
    # mkt_prior are P(side we took wins), so 0.5 is neutral).
    if p_sig is not None and p_mkt is not None:
        signal_edge = p_sig - p_mkt          # incremental model over market
        prior_edge  = p_mkt - 0.5            # market lean from neutral
    elif p_sig is not None:
        signal_edge = p_sig - 0.5
        prior_edge  = 0.0
    elif p_mkt is not None:
        signal_edge = 0.0
        prior_edge  = p_mkt - 0.5
    else:
        signal_edge = 0.0
        prior_edge  = 0.0

    # Attribute raw_pnl proportionally to each edge's magnitude.
    total_edge_abs = abs(signal_edge) + abs(prior_edge)
    if total_edge_abs > 1e-9:
        signal_share = abs(signal_edge) / total_edge_abs
        prior_share  = abs(prior_edge)  / total_edge_abs
    else:
        signal_share = 0.0
        prior_share  = 0.0

    signal_pnl    = raw_pnl_usd * signal_share * (1.0 if signal_edge >= 0 else -1.0)
    prior_pnl     = raw_pnl_usd * prior_share  * (1.0 if prior_edge  >= 0 else -1.0)
    execution_pnl = -(float(entry_slippage_usd) + float(exit_slippage_usd) + float(fees_usd))
    residual      = pnl_usd - (signal_pnl + prior_pnl + execution_pnl)

    return Decomposition(
        trade_id=trade_id or f"{side}-{int(time.time())}",
        side=side,
        size=_size,
        entry_price=_entry,
        exit_price=_exit,
        pnl_usd=pnl_usd,
        signal_pnl=signal_pnl,
        prior_pnl=prior_pnl,
        execution_pnl=execution_pnl,
        residual=residual,
        meta=meta or {},
    )


def record_closed_trade(
    trade: Any,
    features_snapshot: Optional[dict] = None,
) -> Optional[Decomposition]:
    """
    Pull fields off a TradeRecord-like object and decompose. Persists to
    logs/attribution_trade.jsonl. Fire-and-forget: logs and swallows errors.
    """
    try:
        side  = getattr(trade, "side", None) or "YES"
        size  = float(getattr(trade, "size", 0.0) or 0.0)
        ep    = float(getattr(trade, "entry_price", 0.0) or 0.0)
        xp    = float(getattr(trade, "exit_price", 0.0) or 0.0)
        ts    = int(getattr(trade, "ts", int(time.time())))
        trade_id = str(ts)

        feats = features_snapshot or (getattr(trade, "features", None) or {})
        # Convert market/posterior to "side-relative" P(side wins).
        if side == "YES":
            posterior_side = feats.get("posterior_final_up")
            mkt_prior_side = feats.get("yes_mid")
        else:
            pp = feats.get("posterior_final_up")
            ym = feats.get("yes_mid")
            posterior_side = None if pp is None else 1.0 - float(pp)
            mkt_prior_side = None if ym is None else 1.0 - float(ym)

        entry_slip_usd = _safe_slippage_usd(feats.get("entry_slippage_pct"), ep, size)
        exit_slip_usd  = _safe_slippage_usd(feats.get("exit_slippage_pct"),  xp, size)
        fees_usd       = float(feats.get("fees_usd", 0.0) or 0.0)

        dec = decompose(
            side=side, size=size,
            entry_price=ep, exit_price=xp,
            posterior=posterior_side, mkt_prior=mkt_prior_side,
            entry_slippage_usd=entry_slip_usd,
            exit_slippage_usd=exit_slip_usd,
            fees_usd=fees_usd,
            trade_id=trade_id,
            meta={"ts": ts, "exit_reason": getattr(trade, "exit_reason", None)},
        )

        try:
            ATTRIBUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with ATTRIBUTION_LOG_PATH.open("a") as f:
                f.write(json.dumps(dec.to_dict(), separators=(",", ":")) + "\n")
        except Exception as e:
            log.debug("trade_attribution write failed: %s", e)

        return dec
    except Exception as e:
        log.debug("trade_attribution.record_closed_trade error: %s", e)
        return None


def iter_records(limit: Optional[int] = None):
    """Iterate persisted rows (oldest first). Optional last-N cap."""
    if not ATTRIBUTION_LOG_PATH.exists():
        return
    try:
        with ATTRIBUTION_LOG_PATH.open("r") as f:
            lines = f.readlines()
    except Exception:
        return
    if limit is not None:
        lines = lines[-int(limit):]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def summary(last_n: int = 500) -> dict:
    """Aggregate decomposition across recent closed trades."""
    rows = list(iter_records(limit=last_n))
    if not rows:
        return {
            "n": 0, "total_pnl": 0.0,
            "signal_pnl": 0.0, "prior_pnl": 0.0,
            "execution_pnl": 0.0, "residual": 0.0,
            "win_rate": 0.0,
        }
    total     = sum(float(r.get("pnl_usd", 0.0))       for r in rows)
    signal    = sum(float(r.get("signal_pnl", 0.0))    for r in rows)
    prior     = sum(float(r.get("prior_pnl", 0.0))     for r in rows)
    execution = sum(float(r.get("execution_pnl", 0.0)) for r in rows)
    residual  = sum(float(r.get("residual", 0.0))      for r in rows)
    wins      = sum(1 for r in rows if float(r.get("pnl_usd", 0.0)) > 0)
    return {
        "n": len(rows),
        "total_pnl":    total,
        "signal_pnl":   signal,
        "prior_pnl":    prior,
        "execution_pnl": execution,
        "residual":     residual,
        "win_rate":     wins / len(rows),
    }


# ── Helpers ────────────────────────────────────────────────────────────────


def _safe_slippage_usd(slip_pct: Any, px: float, size: float) -> float:
    try:
        return max(0.0, float(slip_pct or 0.0)) * float(px) * float(size)
    except Exception:
        return 0.0
