"""
Tier 3 #12: Economic-calendar entry block.

Pauses NEW entries in the ±N-minute window around scheduled high-impact macro
events (FOMC, CPI, NFP, PPI, Fed speakers). Institutional practice: MM spreads
widen and realized vol spikes around prints; running an entry rule through that
is lighting money on fire. Monster-conviction trades (>0.90) bypass the block
since those setups are rare and typically robust to event noise.

Why hard-coded schedule?
- FOMC / CPI / NFP / PPI dates are known months in advance.
- Tradingeconomics-free and FRED don't have a reliable public futures-aware API
  at 15m cadence; a baked-in schedule is more reliable than a fragile HTTP dep.
- The list is editable; monthly maintenance is trivial relative to the risk.

Events stored in eastern time as (year, month, day, hour, minute, label).
All blocking is computed in UTC at call time.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# ── Default block radius (minutes) ─────────────────────────────────────────
DEFAULT_PRE_BLOCK_MIN  = 30
DEFAULT_POST_BLOCK_MIN = 30

# ── High-impact events (America/New_York local time) ───────────────────────
# Format: (YYYY, M, D, HH, MM, label)
# Keep this list lean — only events that historically move BTC > 0.3% on print.
_SCHEDULED_ET: list[tuple[int, int, int, int, int, str]] = [
    # 2026 FOMC meetings (standard 2:00 PM ET)
    (2026, 1, 28, 14, 0, "FOMC"),
    (2026, 3, 18, 14, 0, "FOMC"),
    (2026, 4, 29, 14, 0, "FOMC"),
    (2026, 6, 10, 14, 0, "FOMC"),
    (2026, 7, 29, 14, 0, "FOMC"),
    (2026, 9, 16, 14, 0, "FOMC"),
    (2026, 10, 28, 14, 0, "FOMC"),
    (2026, 12, 9, 14, 0, "FOMC"),

    # 2026 CPI releases (~2nd Wed 8:30 AM ET, representative schedule)
    (2026, 1, 14, 8, 30, "CPI"),
    (2026, 2, 11, 8, 30, "CPI"),
    (2026, 3, 11, 8, 30, "CPI"),
    (2026, 4, 14, 8, 30, "CPI"),
    (2026, 5, 12, 8, 30, "CPI"),
    (2026, 6, 10, 8, 30, "CPI"),
    (2026, 7, 15, 8, 30, "CPI"),
    (2026, 8, 12, 8, 30, "CPI"),
    (2026, 9, 10, 8, 30, "CPI"),
    (2026, 10, 14, 8, 30, "CPI"),
    (2026, 11, 12, 8, 30, "CPI"),
    (2026, 12, 10, 8, 30, "CPI"),

    # 2026 NFP (~1st Fri 8:30 AM ET)
    (2026, 1, 2, 8, 30, "NFP"),
    (2026, 2, 6, 8, 30, "NFP"),
    (2026, 3, 6, 8, 30, "NFP"),
    (2026, 4, 3, 8, 30, "NFP"),
    (2026, 5, 1, 8, 30, "NFP"),
    (2026, 6, 5, 8, 30, "NFP"),
    (2026, 7, 3, 8, 30, "NFP"),
    (2026, 8, 7, 8, 30, "NFP"),
    (2026, 9, 4, 8, 30, "NFP"),
    (2026, 10, 2, 8, 30, "NFP"),
    (2026, 11, 6, 8, 30, "NFP"),
    (2026, 12, 4, 8, 30, "NFP"),
]


def _et_to_utc(year: int, month: int, day: int, hour: int, minute: int) -> _dt.datetime:
    """Convert (year, month, day, hour, minute) in America/New_York to UTC."""
    try:
        from zoneinfo import ZoneInfo
        tz_et = ZoneInfo("America/New_York")
        local = _dt.datetime(year, month, day, hour, minute, tzinfo=tz_et)
        return local.astimezone(_dt.timezone.utc)
    except Exception:
        # Fallback: treat ET as UTC-5 (EST) — crude, but avoids import error.
        return _dt.datetime(year, month, day, hour + 5, minute, tzinfo=_dt.timezone.utc)


def active_event(
    now: Optional[_dt.datetime] = None,
    pre_min: int = DEFAULT_PRE_BLOCK_MIN,
    post_min: int = DEFAULT_POST_BLOCK_MIN,
) -> Optional[dict]:
    """
    Return {'label', 'event_ts', 'minutes_until', 'minutes_since'} if the
    current time falls inside any event's blocking window. Otherwise None.
    """
    t = now or _dt.datetime.now(_dt.timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)

    for (y, m, d, hh, mm, label) in _SCHEDULED_ET:
        event_utc = _et_to_utc(y, m, d, hh, mm)
        delta_sec = (event_utc - t).total_seconds()
        if -post_min * 60 <= delta_sec <= pre_min * 60:
            return {
                "label": label,
                "event_ts": int(event_utc.timestamp()),
                "event_utc_iso": event_utc.isoformat(),
                "minutes_until": round(delta_sec / 60.0, 2),
                "minutes_since": round(-delta_sec / 60.0, 2),
            }
    return None


def is_blocked(
    now: Optional[_dt.datetime] = None,
    pre_min: int = DEFAULT_PRE_BLOCK_MIN,
    post_min: int = DEFAULT_POST_BLOCK_MIN,
    *,
    monster_conviction: float = 0.0,
    monster_bypass_threshold: float = 0.90,
) -> tuple[bool, Optional[dict]]:
    """
    Returns (blocked, event_dict). Monster-conviction trades above the bypass
    threshold slip through the block since those are rare & high-quality setups.
    """
    ev = active_event(now=now, pre_min=pre_min, post_min=post_min)
    if ev is None:
        return False, None
    if monster_conviction >= float(monster_bypass_threshold):
        return False, ev   # event still recorded for observability, but blocking bypassed
    return True, ev


def load_custom_schedule(path: Optional[str] = None) -> int:
    """
    Optional: merge an additional user-provided schedule from a simple CSV
    (one event per line: YYYY,M,D,HH,MM,LABEL). Returns count added.
    Silent no-op if file is missing.
    """
    if not path:
        path = os.environ.get("CALENDAR_EXTRA_PATH")
    if not path or not os.path.isfile(path):
        return 0
    added = 0
    try:
        with open(path, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                try:
                    tup = (int(parts[0]), int(parts[1]), int(parts[2]),
                           int(parts[3]), int(parts[4]), str(parts[5]))
                    _SCHEDULED_ET.append(tup)
                    added += 1
                except Exception:
                    continue
    except Exception:
        pass
    return added
