"""
Data feeds:
  - Binance REST: klines (5m/1m), 15m candle open (for strike), aggTrades (real CVD)
  - Coinbase REST: 15m candle open (strike fallback #2 — FIX #2)
  - Binance REST: deep L2 order book (OFI, VPIN, microprice)
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
BINANCE_FAPI    = "https://fapi.binance.com"
BYBIT_REST      = "https://api.bybit.com"       # secondary source (US-friendly)
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
    bids:           list = field(default_factory=list)  # list[(price, size)] best→worse
    asks:           list = field(default_factory=list)  # list[(price, size)] best→worse
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
    tob_imbalance:  float = 0.5   # best_bid_sz / (best_bid_sz + best_ask_sz)
    is_stale:       bool = False


@dataclass
class CVDResult:
    cvd_delta:  float   # buy_vol - sell_vol for this 60s window
    buy_vol:    float
    sell_vol:   float


# ── Real-time CVD WebSocket (Bybit public trades) ─────────────────────────────

class BinanceCVDWebsocket:
    """
    Streams live Binance BTCUSDT aggTrade feed and computes true taker-side CVD.
    Uses wss://stream.binance.com:9443/ws/btcusdt@aggTrade — no subscribe handshake,
    highest liquidity source. Falls back to Bybit on connection failure.
    Interface is identical — no changes needed elsewhere.
    """

    def __init__(self):
        self.cvd = 0.0
        self.buy_vol = 0.0
        self.sell_vol = 0.0
        self.last_price = None
        self.running = False
        self._task = None
        self.cvd_history = []  # rolling buffer for slope calculation
        self.cvd_timestamps = []  # timestamps for linear regression

    async def start(self):
        """Start the WebSocket in the background with auto-reconnect."""
        self._task = asyncio.create_task(self._run_forever())

    async def _run_forever(self):
        """Reconnect loop — tries Binance first, falls back to Bybit on repeated failure."""
        import websockets
        import json as _json
        self.running = True

        BINANCE_URI = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
        BYBIT_URI   = "wss://stream.bybit.com/v5/public/linear"
        _binance_failures = 0

        while self.running:
            use_bybit = _binance_failures >= 3
            try:
                if not use_bybit:
                    async with websockets.connect(BINANCE_URI, ping_interval=20, ping_timeout=10) as ws:
                        log.info("Binance aggTrade CVD WebSocket connected")
                        _binance_failures = 0
                        while self.running:
                            msg = await ws.recv()
                            data = _json.loads(msg)
                            # aggTrade schema: e, T, p, q, m (isBuyerMaker)
                            if data.get("e") == "aggTrade":
                                trade_ts = int(data["T"])
                                if getattr(self, 'window_start_ms', 0) > 0 and trade_ts < self.window_start_ms:
                                    continue
                                qty = float(data["q"])
                                is_buyer_maker = data["m"]  # True = sell taker, False = buy taker
                                if not is_buyer_maker:
                                    self.cvd += qty
                                    self.buy_vol += qty
                                else:
                                    self.cvd -= qty
                                    self.sell_vol += qty
                                self.last_price = float(data["p"])
                                self.cvd_history.append(self.cvd)
                                self.cvd_timestamps.append(trade_ts)
                                if len(self.cvd_history) > 30:
                                    self.cvd_history.pop(0)
                                    self.cvd_timestamps.pop(0)
                else:
                    async with websockets.connect(BYBIT_URI, ping_interval=20, ping_timeout=10) as ws:
                        log.info("Bybit CVD WebSocket connected (Binance fallback)")
                        await ws.send(_json.dumps({"op": "subscribe", "args": ["publicTrade.BTCUSDT"]}))
                        while self.running:
                            msg = await ws.recv()
                            data = _json.loads(msg)
                            if data.get("topic") == "publicTrade.BTCUSDT" and "data" in data:
                                for trade in data["data"]:
                                    trade_ts = int(trade["T"])
                                    if getattr(self, 'window_start_ms', 0) > 0 and trade_ts < self.window_start_ms:
                                        continue
                                    qty = float(trade["v"])
                                    side = trade["S"]  # "Buy" or "Sell"
                                    if side == "Buy":
                                        self.cvd += qty
                                        self.buy_vol += qty
                                    else:
                                        self.cvd -= qty
                                        self.sell_vol += qty
                                    self.last_price = float(trade["p"])
                                    self.cvd_history.append(self.cvd)
                                    self.cvd_timestamps.append(trade_ts)
                                    if len(self.cvd_history) > 30:
                                        self.cvd_history.pop(0)
                                        self.cvd_timestamps.pop(0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not use_bybit:
                    _binance_failures += 1
                    log.warning(f"Binance CVD WebSocket error (attempt {_binance_failures}/3): {e} — reconnecting in 3s")
                    await asyncio.sleep(3)
                else:
                    log.warning(f"Bybit CVD WebSocket error: {e} — reconnecting in 5s")
                    _binance_failures = 0  # retry Binance next time
                    await asyncio.sleep(5)

    def get_cvd(self) -> float:
        """Return current cumulative volume delta."""
        return self.cvd

    def get_cvd_slope(self) -> float:
        """Return linear regression slope of CVD over rolling window.
        Units: CVD change per second (velocity)."""
        n = len(self.cvd_history)
        if n < 5:
            return 0.0
        # Use timestamps for proper time-weighted regression
        ts = self.cvd_timestamps[-n:]
        vals = self.cvd_history[-n:]
        t0 = ts[0]
        xs = [(t - t0) / 1000.0 for t in ts]  # seconds since start
        mean_x = sum(xs) / n
        mean_y = sum(vals) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, vals))
        den = sum((x - mean_x) ** 2 for x in xs)
        if den < 1e-9:
            return 0.0
        return num / den  # CVD per second

    def get_volumes(self) -> tuple:
        """Return (cvd, buy_vol, sell_vol)."""
        return self.cvd, self.buy_vol, self.sell_vol

    def reset(self, window_start_ms: int = 0):
        """Reset CVD and volumes (call on each new 15m window)."""
        self.cvd = 0.0
        self.buy_vol = 0.0
        self.sell_vol = 0.0
        self.cvd_history.clear()
        self.cvd_timestamps.clear()
        self.window_start_ms = window_start_ms

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()


# ── Main DataFeeds class ──────────────────────────────────────────────────────

class DataFeeds:
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self.kline_ws = BinanceKlineWebsocket()
        self.funding_ws = BinanceFundingWebsocket()
        self.oracle = ChainlinkOraclePolygon()
        self.last_kline_fetch_ts: int = 0  # unix ts of last successful kline fetch

    async def start(self):
        timeout = aiohttp.ClientTimeout(total=8)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": "btc-15m-quant/1.0"},
        )
        await self.funding_ws.start()
        await self.oracle.start()

    async def close(self):
        if self._session:
            await self._session.close()
        self.funding_ws.stop()
        self.oracle.stop()

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            raise RuntimeError("DataFeeds.start() not called")
        return self._session

    # ── Liquidation Cascade ────────────────────────────────────────────────────

    async def get_liquidation_cascade(self, symbol: str = "BTCUSDT", lookback_s: int = 60) -> float:
        """Fetch recent forced liquidations from Binance Futures.
        Returns net USD liquidation value: positive = long liquidations (bearish),
        negative = short liquidations (bullish).
        Endpoint: GET https://fapi.binance.com/fapi/v1/forceOrders
        """
        try:
            cutoff_ms = int((time.time() - lookback_s) * 1000)
            url = f"https://fapi.binance.com/fapi/v1/forceOrders?symbol={symbol}&limit=50"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                if r.status != 200:
                    return 0.0
                orders = await r.json()
            net = 0.0
            for o in orders:
                ts = int(o.get("time", 0))
                if ts < cutoff_ms:
                    continue
                qty = float(o.get("origQty", 0))
                price = float(o.get("price", 0))
                usd = qty * price
                side = o.get("side", "")
                # SELL = long liquidated (bearish), BUY = short liquidated (bullish)
                net += usd if side == "SELL" else -usd
            return net
        except Exception as e:
            log.debug(f"liquidation_cascade: {e}")
            return 0.0

    async def get_binance_perp_basis_pct(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """Return perp basis as a percentage proxy: (mark - index) / index.

        Uses Binance USDT-margined futures premiumIndex endpoint.
        """
        url = f"{BINANCE_FAPI}/fapi/v1/premiumIndex?symbol={symbol}"
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                if r.status != 200:
                    return None
                data = await r.json()
            if not isinstance(data, dict):
                return None
            mark = float(data.get("markPrice") or 0.0)
            index = float(data.get("indexPrice") or 0.0)
            if index <= 0 or mark <= 0:
                return None
            return (mark - index) / index
        except Exception as e:
            log.debug(f"perp_basis_pct: {e}")
            return None

    # ── Klines ────────────────────────────────────────────────────────────────

    def is_kline_stale(self, threshold_sec: int = 30) -> bool:
        """Return True if last successful kline fetch was more than threshold_sec ago."""
        if self.last_kline_fetch_ts == 0:
            return False  # never fetched yet — don't block on first cycle
        import time as _time
        return (_time.time() - self.last_kline_fetch_ts) > threshold_sec

    async def get_klines(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        """Fetch klines: Priority 1 = Binance REST (works from NL Railway) → Fallbacks."""
        import time as _time
        # Priority 1: Binance REST
        url = f"{BINANCE_REST}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        alt = f"{BINANCE_ALT}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        raw = await self._fetch_json_with_fallback(url, alt)
        if raw:
            self.last_kline_fetch_ts = int(_time.time())
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

        # Priority 2: Bybit REST (US-friendly)
        bybit_candles = await self._get_bybit_klines(symbol, interval, limit)
        if bybit_candles:
            self.last_kline_fetch_ts = int(_time.time())
            return bybit_candles

        # Priority 3: Coinbase
        cb_interval = interval  # "5m" or "1m"
        cb_candles = await self.get_coinbase_klines(cb_interval, limit)
        if cb_candles:
            self.last_kline_fetch_ts = int(_time.time())
            return cb_candles

        log.warning(f"All kline sources failed for {symbol} {interval}")
        return []

    async def _get_bybit_klines(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        """Fetch klines from Bybit v5 public API."""
        # Bybit interval format: "5" for 5m, "1" for 1m, "15" for 15m
        bybit_interval = interval.replace("m", "").replace("h", "0")
        bybit_symbol = symbol  # BTCUSDT works on Bybit too
        url = (
            f"{BYBIT_REST}/v5/market/kline"
            f"?category=linear&symbol={bybit_symbol}&interval={bybit_interval}&limit={limit}"
        )
        try:
            async with self.session.get(url) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                result = data.get("result", {}).get("list", [])
                if not result:
                    return []
                # Bybit format: [startTime, open, high, low, close, volume, turnover]
                # Bybit returns newest first, so reverse
                candles = [
                    Candle(
                        ts_ms=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
                    for row in reversed(result)
                ]
                return candles
        except Exception as e:
            log.debug(f"Bybit klines error: {e}")
            return []

    # ── Strike price (FIX #2) ─────────────────────────────────────────────────
    # Priority: Binance 15m open → Coinbase 15m open → Binance mid → None
    # Live EMA / current price is NEVER used as strike fallback.

    async def get_binance_15m_open(self, window_start_ms: int) -> Optional[float]:
        """Priority 1: Binance 15m kline open at window start."""
        # Binance can occasionally return an empty set when queried with an exact boundary.
        # Query a small lookback window and select the exact candle by openTime.
        lookback_ms = 15 * 60 * 1000
        start_ms = max(0, window_start_ms - lookback_ms)
        url = (
            f"{BINANCE_REST}/api/v3/klines"
            f"?symbol=BTCUSDT&interval=15m"
            f"&startTime={start_ms}&limit=3"
        )
        alt = url.replace(BINANCE_REST, BINANCE_ALT)
        try:
            raw = await self._fetch_json_with_fallback(url, alt)
            if raw and len(raw) > 0:
                chosen = None
                for row in raw:
                    try:
                        if int(row[0]) == int(window_start_ms):
                            chosen = row
                            break
                    except Exception:
                        continue

                # Fallback: if we didn't get the exact openTime, take the latest returned candle.
                if chosen is None:
                    chosen = raw[-1]
                    try:
                        log.debug(
                            "Binance 15m open: exact candle not found (wanted=%s got=%s..%s)",
                            window_start_ms,
                            raw[0][0],
                            raw[-1][0],
                        )
                    except Exception:
                        pass

                open_price = float(chosen[1])
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
                if data and len(data) > 0:
                    # Try exact match first
                    ws_ts = int(datetime.fromisoformat(window_start_iso.replace("Z", "+00:00")).timestamp())
                    for row in data:
                        if int(row[0]) == ws_ts:
                            open_price = float(row[3])
                            if open_price > 0:
                                return open_price
                    # Fallback: use the latest candle's open if no exact match
                    open_price = float(data[0][3])  # Coinbase returns newest first
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
            # IMPORTANT: Coinbase "side" = maker's side. Taker (aggressor) is the opposite.
            #   side="sell" → maker was seller → taker BOUGHT (buy aggressor)
            #   side="buy"  → maker was buyer → taker SOLD (sell aggressor)
            buy_vol = sell_vol = 0.0
            found = 0
            for t in trades:
                ts = int(datetime.fromisoformat(t["time"].replace("Z", "+00:00")).timestamp() * 1000)
                if start_ms <= ts <= end_ms:
                    qty = float(t["size"])
                    if t["side"] == "sell":  # maker was seller → taker bought
                        buy_vol += qty
                    else:                    # maker was buyer → taker sold
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
        alt = url.replace(BINANCE_REST, BINANCE_ALT)
        try:
            raw = await self._fetch_json_with_fallback(url, alt)
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

    async def get_cvd_with_cb_fallback(self, start_ms: int, end_ms: int) -> tuple[CVDResult, CVDResult, CVDResult]:
        """
        Phase 4: Multi-source CVD fetch with fallback.
        Returns (best_cvd, bin_cvd, cb_cvd).
        """
        bin_task = asyncio.create_task(self.get_real_cvd(start_ms, end_ms))
        cb_task  = asyncio.create_task(self.get_coinbase_cvd(start_ms, end_ms))
        
        bin_cvd, cb_cvd = await asyncio.gather(bin_task, cb_task)
        
        bin_vol = bin_cvd.buy_vol + bin_cvd.sell_vol
        cb_vol  = cb_cvd.buy_vol + cb_cvd.sell_vol
        
        # Select source with more volume
        best_cvd = bin_cvd if bin_vol >= cb_vol else cb_cvd
        return best_cvd, bin_cvd, cb_cvd

    # ── Binance L2 order book ─────────────────────────────────────────────────
    # Binance L2 order book — primary source for microstructure.

    async def get_binance_book(
        self,
        symbol: str = "BTCUSDT",
        prev_bid_depth20: Optional[float] = None,
        prev_ask_depth20: Optional[float] = None,
        limit: int = 20,
    ) -> DeepBook:
        """
        Fetch N-level L2 depth from Binance.
        schema: {"lastUpdateId":..., "bids": [["px","qty"],...], "asks": [...]}
        """
        _limit = 100 if int(limit) >= 100 else (20 if int(limit) >= 20 else 5)
        url = f"{BINANCE_REST}/api/v3/depth?symbol={symbol}&limit={_limit}"
        alt = f"{BINANCE_ALT}/api/v3/depth?symbol={symbol}&limit={_limit}"
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

        # TOB imbalance: level-1 size ratio (separate from 20-level deep_imbalance)
        tob_imbalance = 0.5
        if best_bid_sz and best_ask_sz:
            tob_total = best_bid_sz + best_ask_sz
            if tob_total > 0:
                tob_imbalance = best_bid_sz / tob_total

        return DeepBook(
            bid_depth20    = bid20,
            ask_depth20    = ask20,
            bids           = bids,
            asks           = asks,
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
            tob_imbalance  = tob_imbalance,
            is_stale       = False,
        )

    # ── Dedicated ATR from Binance 5m klines ──────────────────────────────────

    async def calculate_atr_binance(self, periods: int = 14) -> float:
        """
        Compute ATR from Binance 5m klines directly.
        Returns a safe fallback (150.0) if insufficient data.
        """
        try:
            klines = await self.get_klines("BTCUSDT", "5m", periods + 1)
            if not klines or len(klines) < periods + 1:
                return 150.0  # safe fallback

            highs  = [c.high for c in klines]
            lows   = [c.low for c in klines]
            closes = [c.close for c in klines]

            tr = [
                max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1])
                )
                for i in range(1, len(klines))
            ]
            atr = sum(tr[-periods:]) / periods
            return atr
        except Exception as e:
            log.warning(f"ATR calculation failed: {e}")
            return 150.0  # safe fallback

    # ── Dedicated MACD from Binance 5m klines ─────────────────────────────────

    async def calculate_macd_histogram(self) -> float:
        """
        Compute MACD histogram from Binance 5m klines directly.
        Returns 0.0 if insufficient data.
        """
        try:
            klines = await self.get_klines("BTCUSDT", "5m", 50)
            if not klines or len(klines) < 35:
                return 0.0

            closes = [c.close for c in klines]

            def ema(data, period):
                alpha = 2.0 / (period + 1)
                val = data[0]
                for px in data[1:]:
                    val = alpha * px + (1 - alpha) * val
                return val

            # Compute running MACD line for signal EMA
            macd_values = []
            for i in range(26, len(closes)):
                e12 = ema(closes[i - 12 + 1 : i + 1], 12)
                e26 = ema(closes[i - 26 + 1 : i + 1], 26)
                macd_values.append(e12 - e26)

            if len(macd_values) < 9:
                return 0.0

            signal_line = ema(macd_values[-9:], 9)
            histogram = macd_values[-1] - signal_line
            return histogram
        except Exception as e:
            log.warning(f"MACD calculation failed: {e}")
            return 0.0

    # ── Simple price fallback ─────────────────────────────────────────────────

    async def get_btc_price(self) -> Optional[float]:
        """Simple spot price fallback (Binance → Coinbase)."""
        url = f"{BINANCE_REST}/api/v3/ticker/price?symbol=BTCUSDT"
        alt = f"{BINANCE_ALT}/api/v3/ticker/price?symbol=BTCUSDT"
        try:
            raw = await self._fetch_json_with_fallback(url, alt)
            if raw and "price" in raw:
                return float(raw["price"])
        except Exception as e:
            log.warning(f"Simple BTC price error: {e}")
        # Coinbase public ticker (often works when Binance is geo-blocked)
        try:
            cb_url = f"{COINBASE_REST}/products/BTC-USD/ticker"
            async with self.session.get(cb_url) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                px = float(data.get("price") or 0.0)
                return px if px > 0 else None
        except Exception as e:
            log.warning(f"Coinbase spot price error: {e}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _fetch_json_with_fallback(self, url: str, alt: str = None):
        try:
            async with self.session.get(url) as r:
                if r.status == 200:
                    return await r.json()
                try:
                    body = await r.text()
                    log.debug(f"Primary fetch non-200 ({r.status}) {url}: {body[:200]}")
                except Exception:
                    log.debug(f"Primary fetch non-200 ({r.status}) {url}")
        except Exception as e:
            log.debug(f"Primary fetch failed ({url}): {e}")
        if alt:
            try:
                async with self.session.get(alt) as r:
                    if r.status == 200:
                        return await r.json()
                    try:
                        body = await r.text()
                        log.debug(f"Fallback fetch non-200 ({r.status}) {alt}: {body[:200]}")
                    except Exception:
                        log.debug(f"Fallback fetch non-200 ({r.status}) {alt}")
            except Exception as e:
                log.warning(f"Fallback fetch failed ({alt}): {e}")
        return None
import asyncio
import json
import logging
import time

log = logging.getLogger("oracles")

class BinanceKlineWebsocket:
    """Streams live 1m and 5m klines from Binance."""
    def __init__(self):
        self.klines_1m = []
        self.klines_5m = []
        self.running = False
        self._task = None

    async def start(self):
        self._task = asyncio.create_task(self._run_forever())

    async def _run_forever(self):
        import websockets
        self.running = True
        uri = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m/btcusdt@kline_5m"
        while self.running:
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                    log.info("Binance Kline WS connected")
                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if "k" in data:
                            k = data["k"]
                            candle = {
                                "ts_ms": int(k["t"]),
                                "open": float(k["o"]),
                                "high": float(k["h"]),
                                "low": float(k["l"]),
                                "close": float(k["c"]),
                                "volume": float(k["v"])
                            }
                            # Update exactly the candle that matches the open time
                            interval = k["i"]
                            target_list = self.klines_1m if interval == "1m" else self.klines_5m
                            
                            # Maintain up to 50 candles
                            if not target_list or target_list[-1]["ts_ms"] < candle["ts_ms"]:
                                target_list.append(candle)
                                if len(target_list) > 50:
                                    target_list.pop(0)
                            elif target_list[-1]["ts_ms"] == candle["ts_ms"]:
                                target_list[-1] = candle
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"Binance Kline WS error: {e} - reconnecting in 5s")
                await asyncio.sleep(5)

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()

class BinanceFundingWebsocket:
    """Streams live perp funding rate from Binance."""
    def __init__(self):
        self.funding_rate = 0.0
        self.running = False
        self._task = None

    async def start(self):
        self._task = asyncio.create_task(self._run_forever())

    async def _run_forever(self):
        import websockets
        self.running = True
        uri = "wss://fstream.binance.com/ws/btcusdt@markPrice"
        while self.running:
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                    log.info("Binance Funding WS connected")
                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if "r" in data:
                            self.funding_rate = float(data["r"])
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"Binance Funding WS error: {e}")
                await asyncio.sleep(5)

    def get_rate(self) -> float:
        return self.funding_rate

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()

class ChainlinkOraclePolygon:
    """Tracks Chainlink BTC/USD oracle on Polygon for Binance lag detection."""
    def __init__(self):
        self.oracle_px = 0.0
        self.last_update = 0
        self.running = False
        self._task = None

    async def start(self):
        self._task = asyncio.create_task(self._run_forever())

    async def _run_forever(self):
        from web3 import AsyncWeb3
        from config import Config
        self.running = True
        
        # BTC/USD on Polygon
        contract_addr = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
        # minimal ABI for latestRoundData
        abi = [{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"}]
        
        while self.running:
            try:
                rpc_url = Config.POLYGON_RPC_URL or "https://polygon-rpc.com"
                w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
                contract = w3.eth.contract(address=contract_addr, abi=abi)
                log.info(f"Chainlink Oracle tracker started ({rpc_url})")
                
                while self.running:
                    try:
                        log.debug("Calling Chainlink latestRoundData...")
                        data = await asyncio.wait_for(
                            contract.functions.latestRoundData().call(),
                            timeout=5.0
                        )
                        log.debug("Chainlink latestRoundData returned")
                        # answer is data[1], updatedAt is data[3]
                        self.oracle_px = float(data[1]) / 1e8
                        self.last_update = int(data[3])
                    except Exception as e:
                        log.debug(f"Chainlink fetch error: {e}")
                    # Poll efficiently since oracle only updates every ~27s or 0.5% deviation
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"Chainlink Oracle setup error: {e}")
                await asyncio.sleep(10)

    def get_price(self) -> float:
        return self.oracle_px

    def get_last_update(self) -> int:
        return self.last_update

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
