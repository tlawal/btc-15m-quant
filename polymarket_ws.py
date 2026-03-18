import asyncio
try:
    import orjson as _json
except ImportError:
    import json as _json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import websockets


log = logging.getLogger(__name__)


WS_MARKET_ENDPOINT = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class L2Book:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_ts_ms: int = 0
    tick_size: Optional[float] = None

    def best_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None

    def mid(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2

    def total_bid_size(self) -> float:
        return float(sum(self.bids.values()))

    def total_ask_size(self) -> float:
        return float(sum(self.asks.values()))


class PolymarketMarketWSClient:
    def __init__(self, *, ping_interval: int = 20, ping_timeout: int = 10):
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

        self._books: dict[str, L2Book] = {}
        self._subscribed_assets: set[str] = set()
        self._desired_assets: set[str] = set()

        self.connected = False
        self.last_msg_ts = 0.0
        self.reconnects = 0
        self.resyncs = 0

        self._lock = asyncio.Lock()

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
        await self._close_ws()

    async def set_active_assets(self, asset_ids: list[str]):
        asset_set = {str(a) for a in asset_ids if a}
        async with self._lock:
            self._desired_assets = asset_set
        await self._maybe_update_subscription()

    def get_book(self, asset_id: str) -> Optional[L2Book]:
        return self._books.get(asset_id)

    async def _close_ws(self):
        ws = self._ws
        self._ws = None
        self.connected = False
        if ws:
            try:
                await ws.close()
            except Exception:
                pass

    async def _run_forever(self):
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(
                    WS_MARKET_ENDPOINT,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    max_queue=2048,
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    self.reconnects += 1
                    backoff = 1.0

                    await self._send_initial_subscription()

                    while self._running:
                        msg = await ws.recv()
                        self.last_msg_ts = time.time()
                        await self._handle_message(msg)
                        await self._maybe_update_subscription()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    log.warning(f"Polymarket market WS error: {e} (reconnecting in {backoff:.1f}s)")
                    await self._close_ws()
                    await asyncio.sleep(backoff)
                    backoff = min(15.0, backoff * 1.7)

    async def _send_initial_subscription(self):
        async with self._lock:
            assets = list(self._desired_assets)
        if not assets:
            return
        await self._send_json({"assets_ids": assets, "type": "market", "custom_feature_enabled": True})
        async with self._lock:
            self._subscribed_assets = set(assets)

    async def _maybe_update_subscription(self):
        ws = self._ws
        if not ws or not self.connected:
            return

        async with self._lock:
            desired = set(self._desired_assets)
            subscribed = set(self._subscribed_assets)

        to_sub = sorted(desired - subscribed)
        to_unsub = sorted(subscribed - desired)

        if to_sub:
            await self._send_json({"assets_ids": to_sub, "operation": "subscribe", "custom_feature_enabled": True})
            async with self._lock:
                self._subscribed_assets.update(to_sub)

        if to_unsub:
            await self._send_json({"assets_ids": to_unsub, "operation": "unsubscribe"})
            async with self._lock:
                for a in to_unsub:
                    self._subscribed_assets.discard(a)

    async def _send_json(self, payload: dict):
        ws = self._ws
        if not ws:
            return
        try:
            _d = _json.dumps(payload)
            await ws.send(_d.decode() if isinstance(_d, bytes) else _d)
        except Exception as e:
            log.debug(f"Polymarket WS send failed: {e}")

    async def _handle_message(self, msg: str):
        try:
            data = _json.loads(msg)
        except Exception:
            return

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    await self._handle_event_dict(item)
            return

        if isinstance(data, dict):
            await self._handle_event_dict(data)
            return

    async def _handle_event_dict(self, data: dict):
        if not data:
            return

        event_type = data.get("event_type") or data.get("type")
        if not event_type:
            return

        if event_type == "book":
            await self._handle_book_snapshot(data)
            return

        if event_type == "price_change":
            await self._handle_price_change(data)
            return

        if event_type == "tick_size_change":
            asset_id = str(data.get("asset_id") or "")
            if not asset_id:
                return
            tick = data.get("new_tick_size")
            try:
                tick_f = float(tick)
            except Exception:
                tick_f = None
            book = self._books.setdefault(asset_id, L2Book())
            book.tick_size = tick_f
            book.last_ts_ms = _safe_ts_ms(data.get("timestamp"))
            return

    async def _handle_book_snapshot(self, data: dict):
        asset_id = str(data.get("asset_id") or "")
        if not asset_id:
            return

        bids = {}
        for lvl in data.get("bids", []) or []:
            try:
                if isinstance(lvl, dict):
                    p = float(lvl.get("price"))
                    s = float(lvl.get("size"))
                else:
                    p = float(lvl[0])
                    s = float(lvl[1])
            except Exception:
                continue
            if p > 0 and s > 0:
                bids[p] = s

        asks = {}
        for lvl in data.get("asks", []) or []:
            try:
                if isinstance(lvl, dict):
                    p = float(lvl.get("price"))
                    s = float(lvl.get("size"))
                else:
                    p = float(lvl[0])
                    s = float(lvl[1])
            except Exception:
                continue
            if p > 0 and s > 0:
                asks[p] = s

        book = self._books.setdefault(asset_id, L2Book())
        book.bids = bids
        book.asks = asks
        book.last_ts_ms = _safe_ts_ms(data.get("timestamp"))

    async def _handle_price_change(self, data: dict):
        ts_ms = _safe_ts_ms(data.get("timestamp"))
        for ch in data.get("price_changes", []) or []:
            asset_id = str(ch.get("asset_id") or "")
            if not asset_id:
                continue
            side = (ch.get("side") or "").upper()
            try:
                price = float(ch.get("price"))
                size = float(ch.get("size"))
            except Exception:
                continue

            book = self._books.setdefault(asset_id, L2Book())
            if side == "BUY":
                _apply_level(book.bids, price, size)
            elif side == "SELL":
                _apply_level(book.asks, price, size)
            book.last_ts_ms = ts_ms


def _apply_level(levels: dict[float, float], price: float, size: float):
    if price <= 0:
        return
    if size <= 0:
        levels.pop(price, None)
    else:
        levels[price] = size


def _safe_ts_ms(x) -> int:
    try:
        v = int(float(x))
        return v
    except Exception:
        return int(time.time() * 1000)
