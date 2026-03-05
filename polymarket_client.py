"""
Polymarket CLOB client wrapper around py-clob-client.

Handles:
  - Market discovery (slug → condition_id → token IDs)
  - Order book fetch (YES + NO mid/ask/bid)
  - Balance check
  - Position query
  - Limit buy/sell and market IOC orders
  - ensure_approvals on startup
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional
import aiohttp

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds, AssetType, BalanceAllowanceParams,
    OpenOrderParams, OrderArgs, OrderType, MarketOrderArgs,
)

POLYMARKET_HOST = "https://clob.polymarket.com"

from config import Config

log = logging.getLogger(__name__)

GAMMA_API   = "https://gamma-api.polymarket.com"
BTC_PREFIX  = "btc-updown-15m-"


@dataclass
class MarketInfo:
    condition_id:     str
    slug:             str
    yes_token_id:     str
    no_token_id:      str
    expiry_ts:        int
    start_ts:         int
    accepting_orders: bool
    question:         Optional[str] = None
    url:              Optional[str] = None


@dataclass
class OrderBook:
    yes_mid:        Optional[float] = None
    yes_ask:        Optional[float] = None
    yes_bid:        Optional[float] = None
    no_mid:         Optional[float] = None
    no_ask:         Optional[float] = None
    no_bid:         Optional[float] = None
    yes_tick:       float = 0.01
    no_tick:        float = 0.01
    total_bid_size: float = 0.0
    total_ask_size: float = 0.0


class PolymarketClient:
    def __init__(self):
        # Fix base64 padding if needed
        secret = Config.POLYMARKET_API_SECRET or ""
        if secret and len(secret) % 4 != 0:
            secret += "=" * (4 - (len(secret) % 4))

        self.client = ClobClient(
            host     = POLYMARKET_HOST,
            key      = Config.POLYMARKET_PRIVATE_KEY,
            chain_id = Config.CHAIN_ID,
            creds    = ApiCreds(
                api_key        = Config.POLYMARKET_API_KEY,
                api_secret     = secret,
                api_passphrase = Config.POLYMARKET_API_PASSPHRASE,
            ),
        )
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=8),
            headers={"User-Agent": "btc-15m-quant/1.0"},
        )

    async def close(self):
        if self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            raise RuntimeError("PolymarketClient.start() not called")
        return self._session

    async def ensure_approvals(self):
        """Run once on startup to approve USDC collateral on Polygon."""
        try:
            loop = asyncio.get_event_loop()
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            await loop.run_in_executor(
                None, lambda: self.client.update_balance_allowance(params)
            )
            log.info("Polymarket approvals OK")
        except Exception as e:
            log.error(f"ensure_approvals failed: {e}")

    # ── Market discovery ──────────────────────────────────────────────────────

    async def search_market_by_slug(self, slug: str) -> Optional[MarketInfo]:
        """Fetch event by slug via /events/slug/{slug} and parse the nested market."""
        url = f"{GAMMA_API}/events/slug/{slug}"
        try:
            async with self.session.get(url) as r:
                if r.status != 200:
                    log.debug(f"search_market_by_slug({slug}): HTTP {r.status}")
                    return None
                event = await r.json()
                for m in event.get("markets", []):
                    info = _parse_market_info(m)
                    if info:
                        return info
        except Exception as e:
            log.warning(f"search_market_by_slug({slug}): {e}")
        return None

    async def get_market_by_condition(self, condition_id: str) -> Optional[MarketInfo]:
        """Look up a market by condition_id via the /markets endpoint."""
        url = f"{GAMMA_API}/markets/{condition_id}"
        try:
            async with self.session.get(url) as r:
                if r.status != 200:
                    return None
                m = await r.json()
                return _parse_market_info(m)
        except Exception as e:
            log.warning(f"get_market_by_condition({condition_id}): {e}")
        return None

    async def list_ending_soon(self) -> list[MarketInfo]:
        url = f"{GAMMA_API}/markets?active=true&limit=50&order=endDate&ascending=true"
        try:
            async with self.session.get(url) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                markets = data if isinstance(data, list) else data.get("markets", [])
                return [info for m in markets if (info := _parse_market_info(m))]
        except Exception as e:
            log.warning(f"list_ending_soon: {e}")
            return []

    async def discover_market(
        self,
        window_start: int,
        cached_slug: Optional[str] = None,
        cached_expiry: Optional[int] = None,
        cached_condition_id: Optional[str] = None,
    ) -> Optional[MarketInfo]:
        """Multi-pass market discovery with caching."""
        window_end = window_start + Config.WINDOW_SEC

        # Cache hit
        if cached_condition_id and cached_expiry:
            if abs(cached_expiry - window_end) <= 180:
                m = await self.get_market_by_condition(cached_condition_id)
                if m:
                    return m

        # Try exact slug
        slug = f"{BTC_PREFIX}{window_start}"
        m = await self.search_market_by_slug(slug)
        if m and m.accepting_orders:
            return m

        # Try next-window slug
        next_slug = f"{BTC_PREFIX}{window_start + Config.WINDOW_SEC}"
        m = await self.search_market_by_slug(next_slug)
        if m and m.accepting_orders:
            return m

        # Fallback: ending_soon scan
        candidates = await self.list_ending_soon()
        for m in candidates:
            if m.slug.startswith(BTC_PREFIX) and m.accepting_orders:
                if abs(m.expiry_ts - window_end) <= 3600:
                    return m

        return None

    # ── Order book ────────────────────────────────────────────────────────────

    async def get_order_books(
        self, yes_token_id: str, no_token_id: str
    ) -> OrderBook:
        yes_ob, no_ob = await asyncio.gather(
            self._get_single_ob(yes_token_id),
            self._get_single_ob(no_token_id),
            return_exceptions=True,
        )
        ob = OrderBook()
        if isinstance(yes_ob, dict):
            bids = yes_ob.get("bids", [])
            asks = yes_ob.get("asks", [])
            if bids:
                ob.yes_bid = _best_price(bids, best="high")
            if asks:
                ob.yes_ask = _best_price(asks, best="low")
            if ob.yes_bid and ob.yes_ask:
                ob.yes_mid = (ob.yes_bid + ob.yes_ask) / 2
            ob.total_bid_size = sum(float(l.get("size", 0)) for l in bids)
            ob.total_ask_size = sum(float(l.get("size", 0)) for l in asks)
        if isinstance(no_ob, dict):
            bids = no_ob.get("bids", [])
            asks = no_ob.get("asks", [])
            if bids:
                ob.no_bid = _best_price(bids, best="high")
            if asks:
                ob.no_ask = _best_price(asks, best="low")
            if ob.no_bid and ob.no_ask:
                ob.no_mid = (ob.no_bid + ob.no_ask) / 2
        return ob

    async def _get_single_ob(self, token_id: str):
        try:
            url = f"{POLYMARKET_HOST}/book?token_id={token_id}"
            async with self.session.get(url) as r:
                data = await r.json()
                log.debug(f"Polymarket OB {token_id}: {len(data.get('bids', []))} bids, {len(data.get('asks', []))} asks")
                return data
        except Exception as e:
            log.debug(f"order book {token_id}: {e}")
            return None

    # ── Balance ───────────────────────────────────────────────────────────────

    async def get_balance(self) -> Optional[float]:
        """Fetch USDC collateral balance via get_balance_allowance."""
        try:
            loop = asyncio.get_event_loop()
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            result = await loop.run_in_executor(
                None, lambda: self.client.get_balance_allowance(params)
            )
            # result is a dict like {"balance": "123456789", ...}
            # Polygon USDC always uses 6 decimals — raw value is always in micro-USDC
            if isinstance(result, dict):
                raw_bal = float(result.get("balance", 0))
                bal = raw_bal / 1_000_000.0  # Always 6 decimals on Polygon
                log.debug(f"get_balance: raw={raw_bal} parsed=${bal:.6f}")
                return bal
            raw_bal = float(result) if result is not None else None
            if raw_bal is None:
                return None
            bal = raw_bal / 1_000_000.0  # Always 6 decimals on Polygon
            log.debug(f"get_balance: raw={raw_bal} parsed=${bal:.6f}")
            return bal
        except Exception as e:
            log.warning(f"get_balance: {e}")
            return None

    # ── Positions ─────────────────────────────────────────────────────────────

    async def get_positions(self) -> list[dict]:
        """Fetch positions via CLOB REST API (no SDK method in 0.34.x)."""
        try:
            url = f"{POLYMARKET_HOST}/positions"
            async with self.session.get(url) as r:
                if r.status != 200:
                    return []
                return await r.json()
        except Exception as e:
            log.warning(f"get_positions: {e}")
            return []

    # ── Open orders / cancel ──────────────────────────────────────────────────

    async def get_open_orders(self, condition_id: str) -> list[str]:
        try:
            loop = asyncio.get_event_loop()
            params = OpenOrderParams(market=condition_id)
            orders = await loop.run_in_executor(
                None, lambda: self.client.get_orders(params)
            )
            return [o["id"] for o in (orders or [])]
        except Exception as e:
            log.warning(f"get_open_orders: {e}")
            return []

    async def cancel_order(self, order_id: str):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.client.cancel(order_id))
        except Exception as e:
            log.warning(f"cancel_order {order_id}: {e}")

    async def get_order_status(self, order_id: str) -> Optional[dict]:
        """Check if an order has been filled, partially filled, or is still open."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.client.get_order(order_id)
            )
            if isinstance(result, dict):
                return {
                    "status": result.get("status", "UNKNOWN"),
                    "size_matched": float(result.get("size_matched", 0) or 0),
                    "original_size": float(result.get("original_size", 0) or result.get("size", 0) or 0),
                    "price": float(result.get("price", 0) or 0),
                }
            return None
        except Exception as e:
            log.debug(f"get_order_status {order_id}: {e}")
            return None

    @staticmethod
    def smart_entry_price(bid: Optional[float], ask: Optional[float], tick: float = 0.01) -> Optional[float]:
        """
        Compute a smart limit price: bid + 1 tick (improves over crossing spread to ask).
        Falls back to ask if no bid available.
        """
        if bid is not None and ask is not None:
            smart_px = round(bid + tick, 4)
            # Don't exceed the ask — that would be worse than just using the ask
            return min(smart_px, ask)
        return ask

    # ── Order execution ───────────────────────────────────────────────────────

    async def limit_buy(
        self, token_id: str, price: float, size: float, order_type: str = "GTC"
    ) -> Optional[str]:
        """Place a limit buy order. Returns order_id or None."""
        try:
            args = OrderArgs(
                token_id = token_id,
                price    = round(price, 4),
                size     = round(size, 2),
                side     = "BUY",
            )
            ot = OrderType.GTC if order_type == "GTC" else OrderType.FOK
            loop = asyncio.get_event_loop()
            signed = await loop.run_in_executor(
                None, lambda: self.client.create_order(args)
            )
            resp = await loop.run_in_executor(
                None, lambda: self.client.post_order(signed, ot)
            )
            return resp.get("orderID") or resp.get("id")
        except Exception as e:
            log.error(f"limit_buy failed: {e}")
            return None

    async def market_buy(
        self, token_id: str, amount_usd: float
    ) -> Optional[str]:
        """Market IOC buy for `amount_usd` dollars of a token."""
        try:
            args = MarketOrderArgs(
                token_id = token_id,
                amount   = amount_usd,
                side     = "BUY",
            )
            loop = asyncio.get_event_loop()
            signed = await loop.run_in_executor(
                None, lambda: self.client.create_market_order(args)
            )
            resp = await loop.run_in_executor(
                None, lambda: self.client.post_order(signed, OrderType.FOK)
            )
            return resp.get("orderID") or resp.get("id")
        except Exception as e:
            log.error(f"market_buy failed: {e}")
            return None

    async def limit_sell(
        self, token_id: str, price: float, size: float, order_type: str = "GTC"
    ) -> Optional[str]:
        try:
            args = OrderArgs(
                token_id = token_id,
                price    = round(price, 4),
                size     = round(size, 2),
                side     = "SELL",
            )
            ot = OrderType.IOC if order_type == "IOC" else OrderType.GTC
            loop = asyncio.get_event_loop()
            signed = await loop.run_in_executor(
                None, lambda: self.client.create_order(args)
            )
            resp = await loop.run_in_executor(
                None, lambda: self.client.post_order(signed, ot)
            )
            return resp.get("orderID") or resp.get("id")
        except Exception as e:
            log.error(f"limit_sell failed: {e}")
            return None

    async def market_sell(
        self, token_id: str, size: float
    ) -> Optional[str]:
        try:
            args = MarketOrderArgs(
                token_id = token_id,
                amount   = size,
                side     = "SELL",
            )
            loop = asyncio.get_event_loop()
            signed = await loop.run_in_executor(
                None, lambda: self.client.create_market_order(args)
            )
            resp = await loop.run_in_executor(
                None, lambda: self.client.post_order(signed, OrderType.FOK)
            )
            return resp.get("orderID") or resp.get("id")
        except Exception as e:
            log.error(f"market_sell failed: {e}")
            return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_market_info(m: dict) -> Optional[MarketInfo]:
    """Parse a Gamma API market dict into MarketInfo.

    Handles two formats:
      - clobTokenIds + outcomes as JSON strings (modern Gamma API)
      - tokens array with outcome/token_id dicts (legacy)
    """
    try:
        slug         = m.get("slug", "")
        condition_id = m.get("conditionId") or m.get("condition_id", "")
        if not condition_id:
            return None

        yes_id = None
        no_id  = None

        # Modern format: clobTokenIds and outcomes are JSON-encoded strings
        raw_token_ids = m.get("clobTokenIds", "")
        raw_outcomes  = m.get("outcomes", "")
        if isinstance(raw_token_ids, str) and raw_token_ids:
            try:
                token_ids = json.loads(raw_token_ids)
                outcomes  = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
                for i, outcome in enumerate(outcomes or []):
                    ol = outcome.lower()
                    tid = token_ids[i] if i < len(token_ids) else None
                    if not tid:
                        continue
                    if "up" in ol or "yes" in ol:
                        yes_id = tid
                    elif "down" in ol or "no" in ol:
                        no_id = tid
            except (json.JSONDecodeError, IndexError):
                pass

        # Legacy fallback: tokens array
        if not yes_id or not no_id:
            for t in m.get("tokens", []):
                outcome = (t.get("outcome") or "").lower()
                tid     = t.get("token_id") or t.get("tokenId")
                if not tid:
                    continue
                if "yes" in outcome or "up" in outcome:
                    yes_id = tid
                elif "no" in outcome or "down" in outcome:
                    no_id  = tid

        if not yes_id or not no_id:
            return None

        from datetime import datetime, timezone
        end_date = m.get("endDate") or m.get("endDateIso")
        if end_date:
            dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            expiry_ts = int(dt.timestamp())
        else:
            return None

        accepting = bool(m.get("acceptingOrders") or m.get("active"))
        question  = m.get("question", "")

        return MarketInfo(
            condition_id     = condition_id,
            slug             = slug,
            yes_token_id     = yes_id,
            no_token_id      = no_id,
            expiry_ts        = expiry_ts,
            start_ts         = expiry_ts - Config.WINDOW_SEC,
            accepting_orders = accepting,
            question         = question,
            url              = f"https://polymarket.com/event/{slug}",
        )
    except Exception as e:
        log.debug(f"parse_market_info error: {e}")
        return None


def _best_price(levels: list, best: str = "low") -> Optional[float]:
    prices = []
    for lvl in levels:
        try:
            prices.append(float(lvl.get("price") or lvl.get("p") or 0))
        except (TypeError, ValueError):
            pass
    valid = [p for p in prices if p > 0]
    if not valid:
        return None
    return min(valid) if best == "low" else max(valid)
