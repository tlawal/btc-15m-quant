"""
Phase 2 P1.4: Deribit implied-vol feed.

Pulls BTC options summary from Deribit every N seconds and tracks the ATM IV
for the nearest expiry. Exposes a function that converts (spot, strike,
minutes_remaining) → implied_P(touch) = N(d1) using the Black–Scholes normal.

This gives the Bayesian blend a *theoretically-grounded* market-vol prior in
addition to the existing Polymarket-price prior. Deribit prices the BTC
vol surface out to 0DTE; on any 15m binary we now have a proper σ-based
forecast to triangulate against the Polymarket CLOB mid.

Graceful degradation:
- No web3 / no httpx on machine? Class still imports; `get_iv()` just returns
  0.0 and the caller falls back to the current 2-input blend.
- Deribit unreachable? Retries with expo backoff; last-good IV stays cached.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import math
import time
from typing import Optional

log = logging.getLogger(__name__)


DERIBIT_BOOK_SUMMARY_URL = (
    "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    "?currency=BTC&kind=option"
)

_POLL_INTERVAL_SEC = 60   # Deribit book-summary rate-limits aren't an issue at 60s
_SECONDS_PER_YEAR  = 365.0 * 24 * 3600.0


class DeribitIVFeed:
    """
    Polls Deribit nearest-expiry ATM straddle IV in the background.

    Usage:
        feed = DeribitIVFeed()
        await feed.start()
        iv = feed.get_atm_iv()            # annualised volatility
        p  = feed.implied_p_up(spot=76000, strike=76500, min_rem=5.0)
    """

    def __init__(self, poll_interval: int = _POLL_INTERVAL_SEC):
        self._poll_interval = int(poll_interval)
        self._running = False
        self._task: Optional[asyncio.Task] = None

        self.atm_iv: float = 0.0           # annualised (e.g. 0.55 = 55%)
        self.last_update_ts: float = 0.0
        self.expiry_ms: int = 0
        self.reconnects: int = 0

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def _run_forever(self):
        backoff = 2.0
        while self._running:
            try:
                await self._poll_once()
                backoff = 2.0
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.reconnects += 1
                log.warning("DeribitIVFeed error: %s (retry in %.0fs)", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2.0)

    async def _poll_once(self):
        """Fetch Deribit book summary and extract nearest-expiry ATM IV."""
        try:
            import httpx  # lazy import
        except ImportError:
            log.debug("httpx not installed — DeribitIVFeed disabled")
            await asyncio.sleep(self._poll_interval)
            return

        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(DERIBIT_BOOK_SUMMARY_URL)
            r.raise_for_status()
            data = r.json()

        result = data.get("result") or []
        if not result:
            return

        # Instrument names are like "BTC-26APR26-76000-C". We want the nearest
        # non-expired expiry and the ATM strike closest to spot.
        now_ms = int(time.time() * 1000)
        spot = _extract_underlying_price(result)

        best: tuple[int, float, float] = (10**18, 0.0, 0.0)  # (expiry_ms, iv, |K-spot|)
        best_expiry: int = 0
        best_iv: float = 0.0

        # Two-pass: (1) find nearest future expiry, (2) find strike closest to spot.
        expiries: dict[int, list[dict]] = {}
        for row in result:
            inst = row.get("instrument_name") or ""
            parts = inst.split("-")
            if len(parts) < 4:
                continue
            try:
                _expiry_tag = parts[1]
                _strike     = float(parts[2])
                _iv         = float(row.get("mark_iv") or 0.0) / 100.0  # % → decimal
            except Exception:
                continue
            if _iv <= 0:
                continue
            expiry_ms = _parse_deribit_expiry(_expiry_tag)
            if expiry_ms is None or expiry_ms <= now_ms:
                continue
            expiries.setdefault(expiry_ms, []).append({"strike": _strike, "iv": _iv})

        if not expiries:
            return

        # Nearest expiry
        soonest = min(expiries.keys())
        rows = expiries[soonest]
        # ATM by closest strike
        if spot and spot > 0:
            atm_row = min(rows, key=lambda r: abs(r["strike"] - spot))
        else:
            # Fallback: median strike
            rows_sorted = sorted(rows, key=lambda r: r["strike"])
            atm_row = rows_sorted[len(rows_sorted) // 2]

        self.atm_iv = float(atm_row["iv"])
        self.expiry_ms = int(soonest)
        self.last_update_ts = time.time()

    def get_atm_iv(self) -> float:
        """Return the last-known annualised ATM IV (0.0 if never populated)."""
        return float(self.atm_iv or 0.0)

    def is_fresh(self, max_age_sec: float = 300.0) -> bool:
        return (
            self.atm_iv > 0
            and (time.time() - self.last_update_ts) <= max_age_sec
        )

    def implied_p_up(
        self,
        spot: float,
        strike: float,
        min_rem: float,
        *,
        iv_override: Optional[float] = None,
    ) -> Optional[float]:
        """
        Compute implied P(spot > strike at expiry) = N(d1) under GBM.

        Args:
            spot:   current BTC price.
            strike: binary strike.
            min_rem: minutes to expiry.
            iv_override: use this σ instead of the cached ATM IV (for tests).

        Returns None when inputs are invalid or no IV is available.
        """
        iv = float(iv_override if iv_override is not None else self.atm_iv or 0.0)
        if iv <= 0 or spot <= 0 or strike <= 0 or min_rem is None:
            return None
        T = max(1e-6, float(min_rem) * 60.0 / _SECONDS_PER_YEAR)
        try:
            # Risk-neutral drift ~0 at 15m horizon; assume r=q=0.
            denom = iv * math.sqrt(T)
            if denom <= 0:
                return None
            d1 = (math.log(spot / strike) + 0.5 * iv * iv * T) / denom
            return _phi(d1)
        except (ValueError, ZeroDivisionError):
            return None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _phi(x: float) -> float:
    """Standard-normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _extract_underlying_price(summary_rows: list[dict]) -> Optional[float]:
    """Deribit book summary carries underlying_price on each option row."""
    for row in summary_rows:
        up = row.get("underlying_price")
        if up and up > 0:
            return float(up)
    return None


_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_deribit_expiry(tag: str) -> Optional[int]:
    """
    Parse a Deribit expiry tag like '26APR26' → unix_ms at 08:00 UTC.
    Returns None on failure.
    """
    if not tag or len(tag) < 5:
        return None
    # Strip leading zeros & pull parts
    try:
        # tag can be "6APR26" or "26APR26"
        day_len = 1 if not tag[1].isdigit() else 2
        day   = int(tag[:day_len])
        mon   = _MONTHS.get(tag[day_len:day_len + 3].upper())
        year2 = int(tag[day_len + 3:day_len + 5])
        if mon is None:
            return None
        year = 2000 + year2
        import datetime as _dt
        d = _dt.datetime(year, mon, day, 8, 0, 0, tzinfo=_dt.timezone.utc)
        return int(d.timestamp() * 1000)
    except Exception:
        return None
