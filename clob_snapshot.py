"""
Phase 2 P0.5: CLOB snapshot recorder.

Persists Polymarket Level-2 book snapshots each engine cycle so the replay
backtester can simulate real fills (walk the recorded book with spread-cross,
queue position, taker fees, 500ms latency) instead of assuming ideal fills.

Writes one JSONL file per 15m window: ``data/clob_snapshots/{window_ts}.jsonl``.

Each line:

    {
        "ts_ms": 1766000123456,
        "window_ts": 1765999999,
        "yes_token": "0x…",
        "no_token":  "0x…",
        "yes": {
            "bids": [[price, size], …]  // descending by price
            "asks": [[price, size], …]  // ascending by price
            "src":  "ws" | "rest",
            "age_ms": int,
        },
        "no":  { …same schema… }
    }

Design constraints:
- **Fire-and-forget**: never raise into the hot-loop. All I/O wrapped in try/except.
- **Bounded retention**: background prune keeps only the last 30 days (configurable).
- **Top-N cap**: stores up to TOP_N_LEVELS per side (default 20) to bound file size.
- **Sparse writes**: skip the append if the book hasn't changed since the last write.

Lazily imports nothing heavier than ``json`` and ``os``; safe to use from async.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

# Tunables (intentionally module constants, not Config — keep the recorder
# self-contained so it can't be broken by config reloads)
SNAPSHOT_DIR      = Path(__file__).parent / "data" / "clob_snapshots"
TOP_N_LEVELS      = 20
RETENTION_DAYS    = 30
MIN_CHANGE_TICK   = 0.0001  # skip writes if price change is < this


# ── Internal state (module-level, single-process) ──────────────────────────
_last_snapshot: dict[str, Any] = {}       # keyed by window_ts
_last_prune_ts: float = 0.0               # last time we pruned old files


def _ensure_dir() -> None:
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _top_levels(book: dict[float, float] | None, reverse: bool, n: int) -> list[list[float]]:
    """Return top-n levels as [[price, size], …] sorted by price."""
    if not book:
        return []
    items = sorted(book.items(), key=lambda kv: kv[0], reverse=reverse)[:n]
    return [[float(p), float(s)] for p, s in items if s > 0]


def _l2book_to_dict(l2book: Any, n: int = TOP_N_LEVELS) -> Optional[dict[str, Any]]:
    """Serialise a polymarket_ws.L2Book (or duck-typed object) to a dict."""
    if l2book is None:
        return None
    try:
        bids = _top_levels(getattr(l2book, "bids", None), reverse=True,  n=n)
        asks = _top_levels(getattr(l2book, "asks", None), reverse=False, n=n)
        last_ts_ms = int(getattr(l2book, "last_ts_ms", 0) or 0)
        age_ms = int(time.time() * 1000) - last_ts_ms if last_ts_ms else None
        return {"bids": bids, "asks": asks, "src": "ws", "age_ms": age_ms}
    except Exception:
        return None


def _top_of_book_dict(
    bid: Optional[float], ask: Optional[float], size_bid: float = 0.0, size_ask: float = 0.0,
) -> Optional[dict[str, Any]]:
    """Fallback: build a 1-level snapshot from top-of-book values (REST source)."""
    if bid is None and ask is None:
        return None
    bids = [[float(bid), float(size_bid)]] if bid is not None else []
    asks = [[float(ask), float(size_ask)]] if ask is not None else []
    return {"bids": bids, "asks": asks, "src": "rest", "age_ms": None}


def _should_skip(window_ts: int, new_snap: dict[str, Any]) -> bool:
    """Skip if the top-of-book has not moved since last write."""
    prev = _last_snapshot.get(str(window_ts))
    if prev is None:
        return False
    for side in ("yes", "no"):
        prev_side = prev.get(side) or {}
        new_side = new_snap.get(side) or {}
        for level_key in ("bids", "asks"):
            prev_top = (prev_side.get(level_key) or [[None, None]])[0]
            new_top  = (new_side.get(level_key) or [[None, None]])[0]
            if prev_top[0] is None and new_top[0] is None:
                continue
            if prev_top[0] is None or new_top[0] is None:
                return False
            if abs(float(new_top[0]) - float(prev_top[0])) >= MIN_CHANGE_TICK:
                return False
            if abs(float(new_top[1] or 0) - float(prev_top[1] or 0)) > 0.5:  # size changed by >0.5 share
                return False
    return True


def _prune_old_files(now: float) -> None:
    """Delete snapshot files older than RETENTION_DAYS. Throttled to once/hour."""
    global _last_prune_ts
    if now - _last_prune_ts < 3600:
        return
    _last_prune_ts = now
    cutoff = now - RETENTION_DAYS * 86400
    try:
        for fp in SNAPSHOT_DIR.glob("*.jsonl"):
            try:
                if fp.stat().st_mtime < cutoff:
                    fp.unlink(missing_ok=True)
            except Exception:
                continue
    except Exception:
        pass


def record_snapshot(
    *,
    window_ts: int,
    yes_token: Optional[str],
    no_token:  Optional[str],
    yes_book:  Any = None,              # polymarket_ws.L2Book or None
    no_book:   Any = None,
    yes_bid:   Optional[float] = None,  # REST fallback values
    yes_ask:   Optional[float] = None,
    no_bid:    Optional[float] = None,
    no_ask:    Optional[float] = None,
) -> bool:
    """
    Append one CLOB snapshot to data/clob_snapshots/{window_ts}.jsonl.

    Prefers full L2 books (yes_book/no_book). Falls back to top-of-book REST
    values (yes_bid/yes_ask/no_bid/no_ask) when the WS book is unavailable.

    Returns True if a row was written, False otherwise (silent). Never raises.
    """
    if window_ts is None or window_ts <= 0:
        return False
    try:
        now = time.time()
        _prune_old_files(now)

        # Read TOP_N_LEVELS at call time so monkeypatch-driven tests see the override.
        _n = TOP_N_LEVELS
        yes_side = _l2book_to_dict(yes_book, n=_n) or _top_of_book_dict(yes_bid, yes_ask)
        no_side  = _l2book_to_dict(no_book,  n=_n) or _top_of_book_dict(no_bid,  no_ask)
        if yes_side is None and no_side is None:
            return False

        snap = {
            "ts_ms":     int(now * 1000),
            "window_ts": int(window_ts),
            "yes_token": str(yes_token) if yes_token else None,
            "no_token":  str(no_token)  if no_token  else None,
            "yes":       yes_side,
            "no":        no_side,
        }

        if _should_skip(int(window_ts), snap):
            return False

        _ensure_dir()
        fp = SNAPSHOT_DIR / f"{int(window_ts)}.jsonl"
        with fp.open("a") as f:
            f.write(json.dumps(snap, separators=(",", ":")) + "\n")

        _last_snapshot[str(window_ts)] = snap
        return True
    except Exception:
        return False


def iter_snapshots(window_ts: int):
    """
    Yield recorded snapshots for a given window in chronological order.
    Used by the backtest engine to replay real CLOB state.
    """
    fp = SNAPSHOT_DIR / f"{int(window_ts)}.jsonl"
    if not fp.exists():
        return
    try:
        with fp.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except Exception:
        return


def list_recorded_windows(limit: int = 500) -> list[int]:
    """Return recorded window_ts values (newest first)."""
    try:
        files = sorted(SNAPSHOT_DIR.glob("*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        out: list[int] = []
        for fp in files:
            try:
                out.append(int(fp.stem))
            except Exception:
                continue
        return out
    except Exception:
        return []
