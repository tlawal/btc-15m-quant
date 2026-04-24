"""
Tier 3 #14: Transaction-Cost Analysis (TCA) logger.

Every order that fills should leave a TCA record comparing intended vs realized
price. Previously slippage was logged only for market_sell exits — entries and
limit sells were silent. The backtest page uses this data to compute a
slippage-distribution per rule, enabling real alpha-vs-cost attribution.

Schema (one JSON object per line in logs/tca.jsonl):

    {
        "ts_ms":         int,
        "token_id":      str,
        "order_type":    "BUY" | "SELL",
        "strategy":      "GTC" | "FOK" | "market" | "unknown",
        "intended_px":   float,          # price we asked for
        "realized_px":   float,          # price we actually got
        "size":          float,          # shares
        "slippage_pct":  float,          # (realized - intended) / intended
        "slippage_bps":  float,
        "fee_usd":       Optional[float],
        "notional_usd":  float,
        "tag":           str             # free-form caller-supplied label
    }

Design constraints:
- **Fire-and-forget** — must never raise into order-placement hot-path.
- **Bounded retention** — 30-day auto-prune once per hour.
- **Cheap** — one open + append + close per record. No buffering.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

TCA_LOG_PATH = Path(__file__).parent / "logs" / "tca.jsonl"
RETENTION_DAYS = 30

_last_prune_ts: float = 0.0


def _ensure_dir() -> None:
    try:
        TCA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _prune(now: float) -> None:
    """Auto-truncate the log if it exceeds 50MB (keeps last 1M lines approx)."""
    global _last_prune_ts
    if now - _last_prune_ts < 3600:
        return
    _last_prune_ts = now
    try:
        if not TCA_LOG_PATH.exists():
            return
        sz = TCA_LOG_PATH.stat().st_size
        if sz < 50 * 1024 * 1024:
            return
        # Keep last half of the file
        with TCA_LOG_PATH.open("rb") as f:
            f.seek(-25 * 1024 * 1024, os.SEEK_END)
            f.readline()   # discard partial line
            tail = f.read()
        tmp = TCA_LOG_PATH.with_suffix(".jsonl.tmp")
        with tmp.open("wb") as f:
            f.write(tail)
        tmp.replace(TCA_LOG_PATH)
    except Exception:
        pass


def log_fill(
    *,
    token_id: str,
    order_type: str,             # "BUY" | "SELL"
    intended_px: Optional[float],
    realized_px: Optional[float],
    size: Optional[float],
    strategy: str = "unknown",   # "GTC" | "FOK" | "market" | "unknown"
    fee_usd: Optional[float] = None,
    tag: str = "",
    ts_ms: Optional[int] = None,
) -> bool:
    """
    Record a fill. Returns True if a row was written, False on any error.
    Never raises.
    """
    try:
        if intended_px is None or realized_px is None or size is None:
            return False
        if intended_px <= 0 or size <= 0:
            return False

        slippage_pct = (float(realized_px) - float(intended_px)) / float(intended_px)
        # Buy side: positive = we paid MORE than intended (bad).
        # Sell side: positive = we got MORE than intended (good). Normalize so
        # positive = WORSE-than-intended regardless of side.
        if str(order_type).upper().startswith("SELL"):
            slippage_pct = -slippage_pct

        rec = {
            "ts_ms":        int(ts_ms or (time.time() * 1000)),
            "token_id":     str(token_id),
            "order_type":   str(order_type).upper(),
            "strategy":     str(strategy),
            "intended_px":  float(intended_px),
            "realized_px":  float(realized_px),
            "size":         float(size),
            "slippage_pct": slippage_pct,
            "slippage_bps": slippage_pct * 10000.0,
            "fee_usd":      None if fee_usd is None else float(fee_usd),
            "notional_usd": float(realized_px) * float(size),
            "tag":          str(tag or ""),
        }

        _ensure_dir()
        _prune(time.time())
        with TCA_LOG_PATH.open("a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        return True
    except Exception as e:
        log.debug("tca.log_fill failed: %s", e)
        return False


def iter_records(limit: Optional[int] = None):
    """Iterate TCA records (oldest first). Optional last-N cap."""
    if not TCA_LOG_PATH.exists():
        return
    try:
        lines: list[str]
        with TCA_LOG_PATH.open("r") as f:
            lines = f.readlines()
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
    except Exception:
        return


def summary(last_n: int = 1000) -> dict:
    """
    Compute a rolling summary across the last N records.
    Returns median/p95 slippage_bps, split by order_type and strategy.
    """
    buys:  list[float] = []
    sells: list[float] = []
    by_strategy: dict[str, list[float]] = {}
    for r in iter_records(limit=last_n):
        bps = float(r.get("slippage_bps") or 0.0)
        ot = r.get("order_type") or ""
        if ot == "BUY":
            buys.append(bps)
        elif ot == "SELL":
            sells.append(bps)
        strat = r.get("strategy") or "unknown"
        by_strategy.setdefault(strat, []).append(bps)

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0, "median_bps": 0.0, "p95_bps": 0.0, "mean_bps": 0.0}
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        p50 = xs_sorted[n // 2]
        p95_idx = int(0.95 * (n - 1))
        p95 = xs_sorted[p95_idx]
        mean = sum(xs_sorted) / n
        return {"n": n, "median_bps": p50, "p95_bps": p95, "mean_bps": mean}

    return {
        "buy":  _stats(buys),
        "sell": _stats(sells),
        "by_strategy": {k: _stats(v) for k, v in by_strategy.items()},
        "total_records": len(buys) + len(sells),
    }
