"""
Data feeds:
  - Binance REST: klines (5m/1m), 15m candle open (for strike), aggTrades (real CVD)
  - Coinbase REST: 15m candle open (strike fallback #2 — FIX #2)
  - Hyperliquid REST: deep 20-level L2 order book (OFI, VPIN, microprice)
  - WebSocket: Binance aggTrades stream for live CVD delta
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
import aiohttp
from config import Config

log = logging.getLogger(__name__)

BINANCE_REST    = "https://api.binance.com"
BINANCE_ALT     = "https://api1.binance.com"   # fallback mirror
BINANCE_ALT2    = "https://api2.binance.com"   # fallback mirror 2
COINBASE_REST   = "https://api.exchange.coinbase.com"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Candle:
    ts_ms:  int
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


@dataclass
class DeepBook:
    bid_depth20:    float = 0.0
    ask_depth20:    float = 0.0
    deep_imbalance: float = 0.5   # cumBid / (cumBid + cumAsk)
    vpin_proxy:     float = 0.0   # |deepImbalance - 0.5| * 2
    deep_ofi:       float = 0.0   # depth delta vs previous snapshot
    best_bid_px:    Optional[float] = None
    best_ask_px:    Optional[float] = None
    best_bid_sz:    Optional[float] = None
    best_ask_sz:    Optional[float] = None
    mid:            Optional[float] = None
    microprice:     Optional[float] = None
    tob_crossed:    bool = False
    is_stale:       bool = False


@dataclass
class CVDResult:
    cvd_delta:  float   # buy_vol - sell_vol for this 60s window
    buy_vol:    float
    sell_vol:   float


# ── Main DataFeeds class ──────────────────────────────────────────────────────

class DataFeeds:
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        timeout = aiohttp.ClientTimeout(total=8)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": "btc-15m-quant/1.0"},
        )

    async def close(self):
        if self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            raise RuntimeError("DataFeeds.start() not called")
        return self._session

    # ── Klines ────────────────────────────────────────────────────────────────

    async def get_klines(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        url = f"{BINANCE_REST}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        alt = f"{BINANCE_ALT}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        raw = await self._fetch_json_with_fallback(url, alt)
        if not raw:
            return []
        return [
            Candle(
                ts_ms=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in raw
        ]

    # ── Strike price (FIX #2) ─────────────────────────────────────────────────
    # Priority: Binance 15m open → Coinbase 15m open → Binance mid → None
    # Live EMA / current price is NEVER used as strike fallback.

    async def get_binance_15m_open(self, window_start_ms: int) -> Optional[float]:
        """Priority 1: Binance 15m kline open at window start."""
        url = (
            f"{BINANCE_REST}/api/v3/klines"
            f"?symbol=BTCUSDT&interval=15m"
            f"&startTime={window_start_ms}&limit=1"
        )
        alt = url.replace(BINANCE_REST, BINANCE_ALT)
        try:
            raw = await self._fetch_json_with_fallback(url, alt)
            if raw and len(raw) > 0:
                open_price = float(raw[0][1])
                if open_price > 0:
                    return open_price
        except Exception as e:
            log.warning(f"Binance 15m open error: {e}")
        return None

    async def get_coinbase_15m_open(self, window_start_iso: str, window_end_iso: str) -> Optional[float]:
        """Priority 2: Coinbase Exchange 15m candle open (FIX #2 new fallback)."""
        url = (
            f"{COINBASE_REST}/products/BTC-USD/candles"
            f"?granularity=900&start={window_start_iso}&end={window_end_iso}"
        )
        try:
            async with self.session.get(url) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                # Coinbase format: [[ts, low, high, open, close, volume], ...]
                if data and len(data) > 0:
                    # Coinbase returns newest first. Window might have multiple if overlaps.
                    # We want the one that MATCHES window_start exactly.
                    ws_ts = int(datetime.fromisoformat(window_start_iso.replace("Z", "+00:00")).timestamp())
                    for row in data:
                        if int(row[0]) == ws_ts:
                            open_price = float(row[3])
                            if open_price > 0:
                                return open_price
        except Exception as e:
            log.warning(f"Coinbase 15m open error: {e}")
        return None

    # ── Coinbase Klines ───────────────────────────────────────────────────────

    async def get_coinbase_klines(self, interval: str, limit: int) -> list[Candle]:
        """
        Fetch klines from Coinbase.
        granularity: 60 (1m), 300 (5m).
        Returns list of Candle, newest first (Coinbase default).
        """
        gran = 300 if interval == "5m" else 60
        url = f"{COINBASE_REST}/products/BTC-USD/candles?granularity={gran}"
        try:
            async with self.session.get(url) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                # [[ts, low, high, open, close, volume], ...]
                candles = [
                    Candle(
                        ts_ms=int(row[0]) * 1000,
                        open=float(row[3]),
                        high=float(row[2]),
                        low=float(row[1]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
                    for row in data[:limit]
                ]
                candles.reverse()  # Coinbase returns newest-first; we need oldest-first
                return candles
        except Exception as e:
            log.warning(f"Coinbase klines error: {e}")
            return []

    # ── Coinbase CVD ──────────────────────────────────────────────────────────

    async def get_coinbase_cvd(self, start_ms: int, end_ms: int) -> CVDResult:
        """
        CVD from Coinbase public /trades.
        Fetches latest page and filters by window.
        """
        url = f"{COINBASE_REST}/products/BTC-USD/trades"
        try:
            async with self.session.get(url) as r:
                if r.status != 200:
                    return CVDResult(0.0, 0.0, 0.0)
                trades = await r.json()
            
            # Coinbase format: [{"time": "...", "trade_id": ..., "price": "...", "size": "...", "side": "buy"|"sell"}, ...]
            buy_vol = sell_vol = 0.0
            found = 0
            for t in trades:
                ts = int(datetime.fromisoformat(t["time"].replace("Z", "+00:00")).timestamp() * 1000)
                if start_ms <= ts <= end_ms:
                    qty = float(t["size"])
                    if t["side"] == "buy": # buyer aggressor
                        buy_vol += qty
                    else:
                        sell_vol += qty
                    found += 1
            
            cvd = buy_vol - sell_vol
            if found > 0:
                log.debug(f"get_coinbase_cvd: {found} trades in window, delta={cvd:.4f}")
            return CVDResult(cvd, buy_vol, sell_vol)
        except Exception as e:
            log.warning(f"Coinbase CVD error: {e}")
            return CVDResult(0.0, 0.0, 0.0)

    # ── Real CVD via aggTrades (FIX #4) ───────────────────────────────────────

    async def get_real_cvd(self, start_ms: int, end_ms: int) -> CVDResult:
        """
        Binance aggTrades-based CVD.
        Returns (cvd_delta, buy_vol, sell_vol).
        """
        # Add 1s buffer for Binance clock skew
        end_ms -= 1000
        url = f"{BINANCE_REST}/api/v3/aggTrades?symbol=BTCUSDT&startTime={start_ms}&endTime={end_ms}&limit=1000"
        try:
            raw = await self._fetch_json_with_fallback(url, BINANCE_ALT)
            if not raw:
                return CVDResult(0.0, 0.0, 0.0)

            buy_vol = 0.0
            sell_vol = 0.0
            for t in raw:
                qty = float(t['q'])
                if t['m']:  # m=True means buyer was maker -> sell aggressor
                    sell_vol += qty
                else:
                    buy_vol += qty
            return CVDResult(buy_vol - sell_vol, buy_vol, sell_vol)
        except Exception as e:
            log.warning(f"get_real_cvd error: {e}")
            return CVDResult(0.0, 0.0, 0.0)

    async def get_cvd_with_cb_fallback(self) -> tuple[CVDResult, CVDResult, CVDResult]:
        """
        Phase 4: Multi-source CVD fetch with fallback.
        Returns (best_cvd, bin_cvd, cb_cvd).
        """
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - 60_000
        
        bin_task = asyncio.create_task(self.get_real_cvd(start_ms, now_ms))
        cb_task  = asyncio.create_task(self.get_coinbase_cvd(start_ms, now_ms))
        
        bin_cvd, cb_cvd = await asyncio.gather(bin_task, cb_task)
        
        bin_vol = bin_cvd.buy_vol + bin_cvd.sell_vol
        cb_vol  = cb_cvd.buy_vol + cb_cvd.sell_vol
        
        # Select source with more volume
        best_cvd = bin_cvd if bin_vol >= cb_vol else cb_cvd
        return best_cvd, bin_cvd, cb_cvd

    # ── Binance L2 order book ─────────────────────────────────────────────────
    # Replacing Hyperliquid with Binance as the primary source for microstructure.

    async def get_binance_book(
        self,
        symbol: str = "BTCUSDT",
        prev_bid_depth20: Optional[float] = None,
        prev_ask_depth20: Optional[float] = None,
    ) -> DeepBook:
        """
        Fetch 20-level L2 depth from Binance.
        schema: {"lastUpdateId":..., "bids": [["px","qty"],...], "asks": [...]}
        """
        url = f"{BINANCE_REST}/api/v3/depth?symbol={symbol}&limit=20"
        alt = f"{BINANCE_ALT}/api/v3/depth?symbol={symbol}&limit=20"
        try:
            data = await self._fetch_json_with_fallback(url, alt)
            if not data or "bids" not in data:
                return DeepBook(is_stale=True)
        except Exception as e:
            log.warning(f"Binance book error: {e}")
            return DeepBook(is_stale=True)

        # Binance returns lists of [price, qty]
        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])

        def parse_levels(raw):
            out = []
            for lvl in raw:
                try:
                    out.append((float(lvl[0]), float(lvl[1])))
                except (IndexError, ValueError):
                    pass
            return out

        bids = parse_levels(bids_raw) # already desc
        asks = parse_levels(asks_raw) # already asc

        bid20 = sum(sz for _, sz in bids)
        ask20 = sum(sz for _, sz in asks)
        total = bid20 + ask20
        deep_imbalance = bid20 / total if total > 0 else 0.5
        vpin_proxy = abs(deep_imbalance - 0.5) * 2.0

        # OFI: depth delta vs previous snapshot
        deep_ofi = 0.0
        if prev_bid_depth20 is not None and prev_ask_depth20 is not None:
            deep_ofi = (bid20 - prev_bid_depth20) - (ask20 - prev_ask_depth20)

        best_bid_px = bids[0][0] if bids else None
        best_ask_px = asks[0][0] if asks else None
        best_bid_sz = bids[0][1] if bids else None
        best_ask_sz = asks[0][1] if asks else None

        mid = None
        if best_bid_px and best_ask_px:
            mid = (best_bid_px + best_ask_px) / 2.0

        microprice = None
        if best_bid_px and best_ask_px and best_bid_sz and best_ask_sz:
            denom = best_bid_sz + best_ask_sz
            if denom > 0:
                microprice = (best_ask_px * best_bid_sz + best_bid_px * best_ask_sz) / denom

        tob_crossed = bool(best_bid_px and best_ask_px and best_bid_px >= best_ask_px)

        return DeepBook(
            bid_depth20    = bid20,
            ask_depth20    = ask20,
            deep_imbalance = deep_imbalance,
            vpin_proxy     = vpin_proxy,
            deep_ofi       = deep_ofi,
            best_bid_px    = best_bid_px,
            best_ask_px    = best_ask_px,
            best_bid_sz    = best_bid_sz,
            best_ask_sz    = best_ask_sz,
            mid            = mid,
            microprice     = microprice,
            tob_crossed    = tob_crossed,
            is_stale       = False,
        )

    # ── Simple price fallback ─────────────────────────────────────────────────

    async def get_btc_price(self) -> Optional[float]:
        """Priority 3 fallback: Binance simple ticker price."""
        url = f"{BINANCE_REST}/api/v3/ticker/price?symbol=BTCUSDT"
        alt = f"{BINANCE_ALT}/api/v3/ticker/price?symbol=BTCUSDT"
        try:
            raw = await self._fetch_json_with_fallback(url, alt)
            if raw and "price" in raw:
                return float(raw["price"])
        except Exception as e:
            log.warning(f"Simple BTC price error: {e}")
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _fetch_json_with_fallback(self, url: str, alt: str = None):
        try:
            async with self.session.get(url) as r:
                if r.status == 200:
                    return await r.json()
        except Exception as e:
            log.debug(f"Primary fetch failed ({url}): {e}")
        if alt:
            try:
                async with self.session.get(alt) as r:
                    if r.status == 200:
                        return await r.json()
            except Exception as e:
                log.warning(f"Fallback fetch failed ({alt}): {e}")
        return None
