"""
Polymarket CLOB client wrapper around py-clob-client.

Handles:
  - Market discovery (slug → condition_id → token IDs)
  - Order book fetch (YES + NO mid/ask/bid)
  - Balance check
  - Position query
  - Limit buy/sell and market IOC orders
  - ensure_approvals on startup
  - Mid-window stop-loss handling (Option A):
    - Enforce min-hold after entry
    - Add persistence (consecutive checks or seconds) for STOP_LOSS_15PCT
    - Enforce post-stop cooldown to prevent immediate re-entry/exits for same market/side
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import aiohttp
import httpx
from eth_account import Account
from web3 import AsyncWeb3

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds, AssetType, BalanceAllowanceParams,
    OpenOrderParams, OrderArgs, OrderType, MarketOrderArgs,
)
from py_clob_client.exceptions import PolyApiException

POLYMARKET_HOST = "https://clob.polymarket.com"

from config import Config

log = logging.getLogger(__name__)

GAMMA_API   = "https://gamma-api.polymarket.com"
BTC_PREFIX  = "btc-updown-15m-"

# CTF Exchange on Polygon (v2) for redemption
CTF_EXCHANGE = "0x4bFbB701cd4a0bba82C318CcEd1b4Ebc115A36de"
CTF_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"}
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]


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
    fetch_ms:       int   = 0

    source:         str   = "unknown"  # "ws" | "rest" | "unknown"
    yes_age_ms:     Optional[int] = None
    no_age_ms:      Optional[int] = None
    ws_connected:   Optional[bool] = None
    ws_last_msg_age_ms: Optional[int] = None
    ws_fallback_to_rest: int = 0


class PolymarketClient:
    def __init__(self):
        self._warned_missing_creds = False

        self.client = ClobClient(
            host=Config.POLYMARKET_HOST,
            key=Config.POLYMARKET_PRIVATE_KEY,
            chain_id=Config.CHAIN_ID,
            funder=Config.FUNDER_ADDRESS or None,
        )

        try:
            clob_usdc_addr = self._get_clob_usdc_address()
            log.info(
                "polymarket_client: POLY_USDC_ADDR=%s clob_client_addr=%s",
                (Config.POLYGON_USDC_ADDRESS or "").lower(),
                (clob_usdc_addr or "unknown"),
            )
        except Exception:
            pass

        # Derive fresh API creds each startup (library-intended flow)
        try:
            creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(creds)
            log.info("Polymarket CLOB credentials derived successfully")
            self.can_trade = True
        except Exception as e:
            log.error(f"Polymarket CLOB credential derivation failed: {e}")
            self.can_trade = False

        self._session: Optional[aiohttp.ClientSession] = None
        self._redeem_cooldown_sec = 60 * 30
        self._last_redeem_attempt_ts_by_condition: dict[str, float] = {}

        self._ws_market = None
        self.ws_fallback_to_rest = 0
        self._last_book_source_log_ts = 0.0

    def _warn_no_creds_once(self, caller: str):
        if self.can_trade or self._warned_missing_creds:
            return
        self._warned_missing_creds = True
        log.warning(
            "%s: Polymarket trading creds missing; skipping trading endpoint calls.",
            caller,
        )

    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=8),
            headers={"User-Agent": "btc-15m-quant/1.0"},
        )

        try:
            from polymarket_ws import PolymarketMarketWSClient

            self._ws_market = PolymarketMarketWSClient()
            await self._ws_market.start()
        except Exception as e:
            self._ws_market = None
            log.warning(f"Polymarket market WS disabled: {e}")

    async def close(self):
        if self._ws_market:
            try:
                await self._ws_market.stop()
            except Exception:
                pass
        if self._session:
            await self._session.close()

    async def set_active_market_assets(self, yes_token_id: Optional[str], no_token_id: Optional[str]):
        if not self._ws_market:
            return
        try:
            assets = [a for a in [yes_token_id, no_token_id] if a]
            await self._ws_market.set_active_assets(assets)
        except Exception:
            return

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            raise RuntimeError("PolymarketClient.start() not called")
        return self._session

    async def ensure_approvals(self):
        """Run once on startup to approve USDC collateral on Polygon."""
        if not self.can_trade:
            self._warn_no_creds_once("ensure_approvals")
            return
        try:
            loop = asyncio.get_event_loop()
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.client.update_balance_allowance(params)
                ),
                timeout=5.0
            )
            log.info("Polymarket approvals OK")
        except asyncio.TimeoutError:
            log.warning("ensure_approvals timed out (5s), API might be hanging.")
        except Exception as e:
            log.error(f"ensure_approvals failed: {e}")

    def _get_clob_usdc_address(self) -> Optional[str]:
        """Best-effort extraction of the USDC collateral token address used by the SDK."""
        for name in (
            "usdc_address",
            "USDC_ADDRESS",
            "collateral_address",
            "COLLATERAL_ADDRESS",
            "collateralToken",
            "collateral_token",
        ):
            v = getattr(self.client, name, None)
            if isinstance(v, str) and v.startswith("0x") and len(v) >= 10:
                return v.lower()
        return None

    async def _ensure_onchain_allowance(self, min_required_usd: float = 1.0) -> bool:
        """Ensure the wallet has sufficient collateral allowance on-chain.

        Returns True if allowance appears sufficient, else False.
        """
        if not self.can_trade:
            self._warn_no_creds_once("_ensure_onchain_allowance")
            return False

        clob_usdc_addr = self._get_clob_usdc_address()
        cfg_usdc_addr = (Config.POLYGON_USDC_ADDRESS or "").lower()
        if clob_usdc_addr and cfg_usdc_addr and clob_usdc_addr != cfg_usdc_addr:
            log.error(
                "USDC address mismatch: config=%s clob_client=%s",
                cfg_usdc_addr,
                clob_usdc_addr,
            )
            return False

        def _parse_allowance(result: dict) -> Optional[float]:
            if not isinstance(result, dict):
                return None
            allowance_raw = result.get("allowance")
            if allowance_raw is None:
                allowances_dict = result.get("allowances")
                if isinstance(allowances_dict, dict) and allowances_dict:
                    allowance_raw = next(iter(allowances_dict.values()), None)
            if allowance_raw is None:
                allowance_raw = result.get("approved") or result.get("spend")
            return self._parse_usdc(allowance_raw)

        async def _fetch_allowance() -> Optional[float]:
            try:
                loop = asyncio.get_event_loop()
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self.client.get_balance_allowance(params)),
                    timeout=5.0,
                )
                return _parse_allowance(result)
            except asyncio.TimeoutError:
                log.error("Allowance check timed out (5s)")
                return None
            except Exception as e:
                log.error("Allowance check failed: %s", e)
                return None

        allowance = await _fetch_allowance()
        if allowance is None:
            return False
        if allowance >= float(min_required_usd or 0.0):
            return True

        # Try to set allowance/approvals synchronously and re-check.
        try:
            loop = asyncio.get_event_loop()
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.client.update_balance_allowance(params)),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            log.error("Allowance update timed out (10s)")
            return False
        except Exception as e:
            log.error("Allowance update failed: %s", e)
            return False

        allowance2 = await _fetch_allowance()
        if allowance2 is None:
            return False
        if allowance2 >= float(min_required_usd or 0.0):
            log.info("Allowance updated: %.4f -> %.4f", allowance, allowance2)
            return True

        log.error("Allowance still insufficient after update: %.4f < %.4f", allowance2, float(min_required_usd or 0.0))
        return False

    async def _ensure_conditional_allowance(self, token_id: str, min_required_shares: float = 1.0) -> bool:
        """Ensure the wallet has sufficient conditional-token allowance on-chain for SELLs.

        Polymarket CLOB differentiates collateral (USDC) approvals from conditional-token approvals.
        For SELL orders, the SDK may require conditional-token allowance/approval even when
        collateral allowance is fine.

        Returns True if allowance appears sufficient, else False.
        """
        if not self.can_trade:
            self._warn_no_creds_once("_ensure_conditional_allowance")
            return False
        if not token_id:
            log.error("_ensure_conditional_allowance: missing token_id")
            return False

        # Conditional-token approvals are not naturally expressed in "shares". Some SDK/API
        # responses return `approved: true/false` instead of a numeric allowance.
        # We treat `approved=True` as sufficient and avoid comparing allowance to share count.
        min_required = 1.0

        def _parse_allowance(result: dict) -> Optional[float]:
            if not isinstance(result, dict):
                return None
            approved = result.get("approved")
            if isinstance(approved, bool):
                return 1.0 if approved else 0.0
            allowance_raw = result.get("allowance")
            if allowance_raw is None:
                allowances_dict = result.get("allowances")
                if isinstance(allowances_dict, dict) and allowances_dict:
                    allowance_raw = next(iter(allowances_dict.values()), None)
            if allowance_raw is None:
                allowance_raw = result.get("spend")
            try:
                return float(allowance_raw)
            except (TypeError, ValueError):
                return None

        async def _fetch_allowance() -> Optional[float]:
            try:
                loop = asyncio.get_event_loop()
                params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=str(token_id))
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self.client.get_balance_allowance(params)),
                    timeout=5.0,
                )
                return _parse_allowance(result)
            except asyncio.TimeoutError:
                log.error("Conditional allowance check timed out (5s)")
                return None
            except Exception as e:
                log.error("Conditional allowance check failed: %s", e)
                return None

        allowance = await _fetch_allowance()
        if allowance is None:
            return False
        if allowance >= float(min_required):
            return True

        try:
            loop = asyncio.get_event_loop()
            params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=str(token_id))
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.client.update_balance_allowance(params)),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            log.error("Conditional allowance update timed out (10s)")
            return False
        except Exception as e:
            log.error("Conditional allowance update failed: %s", e)
            return False

        allowance2 = await _fetch_allowance()
        if allowance2 is None:
            return False
        if allowance2 >= float(min_required):
            log.info(
                "Conditional allowance updated for token_id=%s: %.4f -> %.4f",
                str(token_id),
                allowance,
                allowance2,
            )
            return True

        log.error(
            "Conditional allowance still insufficient after update for token_id=%s: %.4f < %.4f",
            str(token_id),
            allowance2,
            float(min_required),
        )
        return False

    def _get_auth_headers_best_effort(self) -> dict:
        """Best-effort extraction of auth headers for direct REST calls.

        py-clob-client exposes slightly different helpers across versions.
        We probe a few common method/property names so the bot keeps working
        across library upgrades.
        """
        try:
            for name in ("get_auth_headers", "create_auth_headers", "auth_headers"):
                attr = getattr(self.client, name, None)
                if callable(attr):
                    h = attr()
                    return h if isinstance(h, dict) else {}
                if isinstance(attr, dict):
                    return attr
        except Exception:
            return {}
        return {}

    @staticmethod
    def _parse_usdc(raw, decimals: int = 6) -> Optional[float]:
        """Parse raw USDC value from CLOB/on-chain.

        py-clob-client's get_balance_allowance returns the raw balanceOf
        value which uses 6 decimals (micro-USDC).  Values ≥ 1_000_000 are
        almost certainly raw on-chain units; values < 100 are likely
        already in USDC.  We use a conservative threshold to decide.
        """
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return None
        if x < 0:
            return None
        # Threshold: if >= 1e5, treat as raw micro-USDC (i.e. ≥ $0.10 in raw units).
        # This safely handles balances from $0.10 up to billions.
        if x >= 1e5:
            return x / (10 ** decimals)
        # Small values are ambiguous but more often already USDC.
        return x

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

        # Cache hit — verify it still matches the current window AND is accepting orders
        if cached_condition_id and cached_expiry:
            if abs(cached_expiry - window_end) <= 60:
                m = await self.get_market_by_condition(cached_condition_id)
                if m and m.accepting_orders:
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

        # Fallback: ending_soon scan — tight tolerance to avoid matching wrong contract
        candidates = await self.list_ending_soon()
        for m in candidates:
            if m.slug.startswith(BTC_PREFIX) and m.accepting_orders:
                if abs(m.expiry_ts - window_end) <= 120:
                    return m

        return None

    # ── Order book ────────────────────────────────────────────────────────────

    async def get_order_books(
        self, yes_token_id: str, no_token_id: str
    ) -> OrderBook:
        now_ms = int(time.time() * 1000)
        ob = OrderBook(fetch_ms=now_ms)

        ob.ws_fallback_to_rest = int(self.ws_fallback_to_rest)
        if self._ws_market:
            try:
                ob.ws_connected = bool(getattr(self._ws_market, "connected", False))
                last_msg_ts = float(getattr(self._ws_market, "last_msg_ts", 0.0) or 0.0)
                ob.ws_last_msg_age_ms = int((time.time() - last_msg_ts) * 1000) if last_msg_ts > 0 else None
            except Exception:
                ob.ws_connected = None
                ob.ws_last_msg_age_ms = None

        # Prefer WS local books when fresh; otherwise fall back to REST.
        ws_ok = False
        if self._ws_market:
            try:
                yes_book = self._ws_market.get_book(str(yes_token_id))
                no_book = self._ws_market.get_book(str(no_token_id))
                if yes_book and no_book:
                    yes_age = now_ms - int(yes_book.last_ts_ms or 0)
                    no_age = now_ms - int(no_book.last_ts_ms or 0)
                    ob.yes_age_ms = int(yes_age)
                    ob.no_age_ms = int(no_age)
                    if yes_age <= 2500 and no_age <= 2500:
                        ob.yes_bid = yes_book.best_bid()
                        ob.yes_ask = yes_book.best_ask()
                        ob.yes_mid = yes_book.mid()
                        ob.no_bid = no_book.best_bid()
                        ob.no_ask = no_book.best_ask()
                        ob.no_mid = no_book.mid()
                        ob.total_bid_size = yes_book.total_bid_size()
                        ob.total_ask_size = yes_book.total_ask_size()
                        if yes_book.tick_size:
                            ob.yes_tick = yes_book.tick_size
                        if no_book.tick_size:
                            ob.no_tick = no_book.tick_size
                        ws_ok = True
            except Exception:
                ws_ok = False

        if ws_ok:
            ob.source = "ws"
            now_s = time.time()
            if now_s - self._last_book_source_log_ts >= 30:
                log.info(
                    "pm_ob_source=ws yes_age_ms=%s no_age_ms=%s ws_connected=%s ws_last_msg_age_ms=%s ws_fallback_to_rest=%s",
                    ob.yes_age_ms,
                    ob.no_age_ms,
                    ob.ws_connected,
                    ob.ws_last_msg_age_ms,
                    self.ws_fallback_to_rest,
                )
                self._last_book_source_log_ts = now_s
            return ob

        self.ws_fallback_to_rest += 1
        ob.ws_fallback_to_rest = int(self.ws_fallback_to_rest)
        ob.source = "rest"
        now_s = time.time()
        if now_s - self._last_book_source_log_ts >= 15:
            log.warning(
                "pm_ob_source=rest yes_age_ms=%s no_age_ms=%s ws_connected=%s ws_last_msg_age_ms=%s ws_fallback_to_rest=%s",
                ob.yes_age_ms,
                ob.no_age_ms,
                ob.ws_connected,
                ob.ws_last_msg_age_ms,
                self.ws_fallback_to_rest,
            )
            self._last_book_source_log_ts = now_s

        yes_ob, no_ob = await asyncio.gather(
            self._get_single_ob(yes_token_id),
            self._get_single_ob(no_token_id),
            return_exceptions=True,
        )
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
        if not self.can_trade:
            self._warn_no_creds_once("get_balance")
            return None
        try:
            loop = asyncio.get_event_loop()
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.client.get_balance_allowance(params)
                ),
                timeout=5.0
            )
            # Typical result is a dict like {"balance": "123456789", "allowance": "..."}
            # Polygon USDC uses 6 decimals (micro-USDC). We defensively probe common keys.
            if isinstance(result, dict):
                raw = (
                    result.get("balance")
                    or result.get("available")
                    or result.get("availableBalance")
                    or result.get("collateral")
                    or 0
                )
                bal = self._parse_usdc(raw)
                log.debug("get_balance: result=%s raw=%s parsed_usdc=%s", result, raw, bal)
                return bal or 0.0

            if result is None:
                return None
            bal = self._parse_usdc(result)
            log.debug("get_balance: raw=%s parsed_usdc=%s", result, bal)
            return bal
        except asyncio.TimeoutError:
            log.warning("get_balance timed out (5s)")
            return None
        except Exception as e:
            log.warning(f"get_balance: {e}")
            return None

    async def get_margin(self) -> dict:
        """Return collateral info as {balance_usdc, allowance_usdc, available_usdc} when possible."""
        if not self.can_trade:
            self._warn_no_creds_once("get_margin")
            return {"balance_usdc": None, "allowance_usdc": None, "available_usdc": None}
        try:
            loop = asyncio.get_event_loop()
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.client.get_balance_allowance(params)
                ),
                timeout=5.0
            )
            if not isinstance(result, dict):
                bal = await self.get_balance()
                return {"balance_usdc": bal, "allowance_usdc": None, "available_usdc": bal}

            bal_raw = result.get("balance")
            # 'allowances' can be a dict of {spender: amount} — extract first non-zero value
            allowance_raw = result.get("allowance")
            if allowance_raw is None:
                allowances_dict = result.get("allowances")
                if isinstance(allowances_dict, dict) and allowances_dict:
                    # Take the first allowance value
                    allowance_raw = next(iter(allowances_dict.values()), None)
            if allowance_raw is None:
                allowance_raw = result.get("approved") or result.get("spend")
            available_raw = result.get("available") or result.get("availableBalance")

            bal = self._parse_usdc(bal_raw)
            allowance = self._parse_usdc(allowance_raw)
            available = self._parse_usdc(available_raw)

            log.info(
                "get_margin: raw_balance=%s parsed_balance=%s parsed_available=%s",
                bal_raw, bal, available,
            )
            if available is None:
                available = bal
            return {
                "balance_usdc": bal,
                "allowance_usdc": allowance,
                "available_usdc": available,
            }
        except asyncio.TimeoutError:
            log.warning("get_margin timed out (5s)")
            return {"balance_usdc": None, "allowance_usdc": None, "available_usdc": None}
        except Exception as e:
            log.warning(f"get_margin: {e}")
            return {"balance_usdc": None, "allowance_usdc": None, "available_usdc": None}

    async def get_wallet_usdc_balance(self) -> Optional[float]:
        """Fetch ERC20 USDC balance for the trading wallet directly from Polygon RPC.

        Checks both USDC.e (bridged, used by Polymarket) and native USDC,
        returning the sum of both.
        """
        if not Config.POLYGON_RPC_URL:
            return None
        pk = Config.POLYMARKET_PRIVATE_KEY
        if not pk:
            return None
        try:
            if not pk.startswith("0x"):
                pk = "0x" + pk
            acct = Account.from_key(pk)
            wallet = acct.address

            async def _check_token(token_addr: str) -> float:
                owner = wallet.lower().replace("0x", "")
                data = "0x70a08231" + owner.rjust(64, "0")
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "eth_call",
                    "params": [{"to": token_addr.lower(), "data": data}, "latest"],
                }
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
                    async with s.post(Config.POLYGON_RPC_URL, json=payload) as r:
                        if r.status != 200:
                            return 0.0
                        resp = await r.json()
                raw_hex = (resp or {}).get("result", "0x0")
                if not isinstance(raw_hex, str) or not raw_hex.startswith("0x"):
                    return 0.0
                return float(int(raw_hex, 16)) / 1_000_000.0

            # Check both USDC.e (bridged, Polymarket default) and native USDC
            usdc_e = await _check_token(Config.POLYGON_USDC_ADDRESS)  # USDC.e
            usdc_native = await _check_token(Config.POLYGON_USDC_NATIVE) if hasattr(Config, 'POLYGON_USDC_NATIVE') else 0.0
            total = usdc_e + usdc_native
            if total > 0:
                log.info("wallet_usdc: USDC.e=$%.4f native=$%.4f total=$%.4f", usdc_e, usdc_native, total)
            return total if total > 0 else None
        except Exception as e:
            log.debug("get_wallet_usdc_balance: %s", e)
            return None

    # ── Positions ─────────────────────────────────────────────────────────────

    async def get_positions(self) -> list[dict]:
        """Fetch positions via CLOB REST API (no SDK method in 0.34.x)."""
        if not self.can_trade:
            return []
        try:
            url = f"{POLYMARKET_HOST}/positions"
            headers = self._get_auth_headers_best_effort()
            async with self.session.get(url, headers=headers or None) as r:
                if r.status != 200:
                    if r.status in (401, 403):
                        log.warning("get_positions: unauthorized (%s) — missing/invalid auth headers", r.status)
                    return []
                return await r.json()
        except Exception as e:
            log.warning(f"get_positions: {e}")
            return []

    async def get_market_trades(self, token_id: str, since_ts: int = 0) -> dict:
        """Fetch recent trades on a specific token from the CLOB.
        Returns {yes_volume, no_volume, net_flow, whale_flow, whale_yes, whale_no}
        where whale_flow > 0 means large-fill YES-heavy (top-10% fills = 'whale' threshold).
        """
        result = {
            "yes_volume": 0.0, "no_volume": 0.0, "net_flow": 0.0,
            "whale_flow": 0.0, "whale_yes": 0.0, "whale_no": 0.0,
        }
        try:
            url = f"{POLYMARKET_HOST}/trades?asset_id={token_id}&limit=200"
            async with self.session.get(url) as r:
                if r.status != 200:
                    return result
                trades = await r.json()

            # Collect all fills in window
            fills = []
            for t in trades:
                ts = int(t.get("timestamp", 0))
                if ts < since_ts:
                    continue
                size = float(t.get("size", 0))
                side = t.get("side", "").upper()
                price = float(t.get("price", 0))
                fills.append({"size": size, "side": side, "usd": size * price})
                if side == "BUY":
                    result["yes_volume"] += size
                elif side == "SELL":
                    result["no_volume"] += size

            result["net_flow"] = result["yes_volume"] - result["no_volume"]

            # Whale detection: fills > $50 USD notional
            for f in fills:
                if f["usd"] >= 50.0:
                    if f["side"] == "BUY":
                        result["whale_yes"] += f["usd"]
                    elif f["side"] == "SELL":
                        result["whale_no"] += f["usd"]
            result["whale_flow"] = result["whale_yes"] - result["whale_no"]

        except Exception as e:
            log.debug(f"get_market_trades: {e}")
        return result

    async def get_trade_history(self, limit: int = 50):
        try:
            from eth_account import Account
            account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
            wallet = account.address.lower()
            url = f"https://data-api.polymarket.com/positions?user={wallet}&limit={limit}"
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
            
            trades = []
            for p in data:
                if float(p.get("size", 0)) == 0: continue
                trades.append({
                    "slug": p.get("marketSlug") or p.get("slug", "?"),
                    "outcome": p.get("outcome", "?"),
                    "size": float(p.get("size", 0)),
                    "entry_price": float(p.get("avgPrice", 0)),
                    "exit_price": float(p.get("currentValue", 0)) / float(p.get("size", 1)),
                    "realized_pnl": float(p.get("realizedPnl", 0)),
                    "percent_pnl": float(p.get("percentRealizedPnl", 0)),
                    "timestamp": p.get("endDate", "?")
                })
            return trades[:20]  # last 20 only
        except Exception as e:
            log.warning(f"Trade history fetch failed: {e}")
            return []

    async def redeem_winning_positions(self) -> float:
        """Detects resolved winners via Data API (redeemable=true) and redeems via CTF contract."""
        if not self.can_trade or not Config.POLYGON_RPC_URL:
            return 0.0

        # Use Data API (this is the only source that shows resolved winners)
        try:
            from eth_account import Account
            account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
            wallet = account.address.lower()
            url = f"https://data-api.polymarket.com/positions?user={wallet}&redeemable=true"
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                positions = r.json()
        except Exception as e:
            log.error(f"Data API redeem check failed: {e}")
            return 0.0

        if not positions:
            return 0.0

        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(Config.POLYGON_RPC_URL))
        account = w3.eth.account.from_key(Config.POLYMARKET_PRIVATE_KEY)
        
        # We must use the base Gnosis CTF directly (4 arguments) to bypass proxy bugs
        gnosis_ctf_address = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
        gnosis_abi = [{
            "inputs": [
                {"internalType": "contract IERC20", "name": "collateralToken", "type": "address"},
                {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
                {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
                {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"}
            ],
            "name": "redeemPositions",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        }]
        
        ctf = w3.eth.contract(address=w3.to_checksum_address(gnosis_ctf_address), abi=gnosis_abi)
        usdc_e = w3.to_checksum_address(Config.POLYGON_USDC_ADDRESS)
        parent_col = b'\x00' * 32

        import time

        total_redeemed = 0.0
        seen_conditions: set[str] = set()
        min_redeem_usd = float(getattr(Config, "MIN_REDEEM_USD", 0.05) or 0.05)

        for p in positions:
            condition_id = p.get("conditionId")
            if not condition_id or condition_id in seen_conditions:
                continue
            seen_conditions.add(condition_id)

            # Data API uses:
            # - size: shares
            # - currentValue: USDC value (typically what you can redeem when resolved)
            redeem_usd = float(p.get("currentValue", 0.0) or 0.0)
            if redeem_usd < min_redeem_usd:
                continue

            now = time.time()
            last_ts = float(self._last_redeem_attempt_ts_by_condition.get(condition_id, 0.0) or 0.0)
            if (now - last_ts) < self._redeem_cooldown_sec:
                continue
            self._last_redeem_attempt_ts_by_condition[condition_id] = now

            log.info(
                f"Attempting redemption of ${redeem_usd:.2f} for {p.get('marketSlug','?')} (condition {condition_id[:10]}...)"
            )
            try:
                tx = await ctf.functions.redeemPositions(
                    usdc_e,
                    parent_col,
                    w3.to_bytes(hexstr=condition_id),
                    [1, 2]  # YES/NO binary
                ).build_transaction({
                    'from': account.address,
                    'nonce': await w3.eth.get_transaction_count(account.address),
                    'gas': 200000,
                    'gasPrice': await w3.eth.gas_price
                })
                signed = account.sign_transaction(tx)
                tx_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)
                receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                
                if receipt.status == 1:
                    log.info(f"✅ CLAIMED ${redeem_usd:.2f} SUCCESSFULLY!")
                    total_redeemed += redeem_usd
                    
                    try:
                        from state import StateManager
                        import time
                        sm = StateManager()
                        entry_price = float(p.get("avgPrice", p.get("averagePrice", 0.5)))
                        size_shares = float(p.get("size", 0.0) or 0.0)
                        pnl = redeem_usd - (size_shares * entry_price)
                        await sm.record_closed_trade(
                            ts=int(time.time()),
                            market_slug=p.get("marketSlug", "unknown"),
                            size=size_shares,
                            entry_price=entry_price,
                            exit_price=1.0,
                            pnl_usd=pnl,
                            outcome_win=1
                        )
                    except Exception as e:
                        log.error(f"Failed to record closed trade to DB: {e}")
                else:
                    log.error(f"Redemption failed (status {receipt.status})")
            except Exception as e:
                log.error(f"Redemption error: {e}")

        return total_redeemed

    async def log_unclaimed_positions(self) -> float:
        """Dashboard visibility: shows REAL unclaimed winnings from Data API (resolved CTF tokens)."""
        try:
            # Derive wallet address exactly like your other methods
            from eth_account import Account
            account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
            wallet_address = account.address.lower()

            url = f"https://data-api.polymarket.com/positions?user={wallet_address}&redeemable=true"
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                positions = r.json()
            
            if not positions:
                log.info("Unclaimed positions: $0.00 (none found)")
                return 0.0

            total = 0.0
            details = []
            for p in positions:
                amount = float(p.get("currentValue", p.get("size", 0)))
                total += amount
                slug = p.get("marketSlug", p.get("slug", "?"))
                details.append(f"{slug} (${amount:.2f})")
            
            log.info(f"Unclaimed positions: ${total:.2f} → {' | '.join(details)}")
            return total
        except Exception as e:
            log.warning(f"Unclaimed check failed: {e}")
            return 0.0

    async def get_open_positions(self) -> list[dict]:
        """Fetch all currently open (unresolved, non-redeemable) positions via the Data API."""
        try:
            from eth_account import Account
            account = Account.from_key(Config.POLYMARKET_PRIVATE_KEY)
            wallet_address = account.address.lower()

            url = f"https://data-api.polymarket.com/positions?user={wallet_address}"
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
            
            open_pos = []
            for p in data:
                # Active open positions are generally not redeemable yet and have size > 0
                if p.get("redeemable") is True: 
                    continue
                size = float(p.get("size", 0))
                if size <= 0.01:
                    continue
                    
                open_pos.append({
                    "conditionId": p.get("conditionId"),
                    "marketSlug": p.get("marketSlug", p.get("slug", "?")),
                    "size": size,
                    "currentValue": float(p.get("currentValue", 0)),
                    "entryValue": float(p.get("initialValue", 0)),
                    "asset": p.get("asset"),
                    "outcome": p.get("outcome", "unknown")
                })
            return open_pos
        except Exception as e:
            log.error(f"Failed to fetch open positions from Data API: {e}")
            return []
    async def monitor_and_exit_open_positions(self, open_positions: list[dict], sig, btc_price: float, atr14: float, state=None) -> list[dict]:
        """DEPRECATED: All exit logic consolidated into exit_policy.evaluate_exit().

        Conditions migrated:
        - SCORE_REVERSAL → ABS_SCORE_REVERSAL in exit_policy L6
        - STOP_LOSS_15PCT → FORCED_DRAWDOWN (now 15%) in exit_policy L5
        - STRIKE_DISTANCE_EXCEEDED → exit_policy L1
        - MICROSTRUCTURE_FLIP → MICRO_REVERSAL in exit_policy L6

        This method is kept for backward compatibility but returns empty list.
        """
        log.debug("monitor_and_exit_open_positions: DEPRECATED — all exits via exit_policy.evaluate_exit()")
        return []
        # Original implementation below (unreachable):
        exits = []
        if not self.can_trade or not open_positions:
            return exits

        strike = sig.strike_price
        if not strike or not atr14:
            return exits

        for pos in open_positions:
            market_slug = pos.get("marketSlug", "unknown")
            side = pos.get("outcome", "").upper() # YES or NO
            size = pos.get("size", 0.0)
            entry_val = pos.get("entryValue", 0.0)
            curr_val = pos.get("currentValue", 0.0)
            token_id = pos.get("asset")

            key = f"{market_slug}:{side}"

            posterior = None
            try:
                if side == "YES":
                    posterior = float(getattr(sig, "posterior_final_up", None))
                elif side == "NO":
                    posterior = float(getattr(sig, "posterior_final_down", None))
            except Exception:
                posterior = None

            if size <= 0 or entry_val <= 0:
                continue

            unrealized_pct = (curr_val / entry_val - 1)
            current_distance = abs(btc_price - strike)

            trigger_reason = None

            # ── Option A: stop-loss min-hold + persistence + cooldown ─────────
            now_ts = int(time.time())
            min_hold = int(getattr(Config, "MIN_HOLD_SECONDS", 30) or 30)
            stop_checks_req = int(getattr(Config, "STOP_LOSS_CONSEC_CHECKS", 3) or 3)
            stop_persist_req = int(getattr(Config, "STOP_LOSS_PERSIST_SECONDS", 12) or 12)
            cooldown_sec = int(getattr(Config, "STOP_LOSS_COOLDOWN_SECONDS", 120) or 120)

            # Entry timestamp is taken from engine-held_position if available.
            entry_ts = None
            if state is not None:
                hp = getattr(state, "held_position", None)
                if hp and getattr(hp, "side", None) == side and str(getattr(hp, "condition_id", "")) == str(pos.get("conditionId")):
                    entry_ts = getattr(hp, "placed_at_ts", None)
            hold_age_s = (now_ts - int(entry_ts)) if entry_ts else None

            if state is not None:
                cd_until = int(getattr(state, "stop_loss_cooldown_until_ts", {}).get(key, 0) or 0)
                if cd_until and now_ts < cd_until:
                    continue

            protective_loss = float(getattr(Config, "MID_WINDOW_PROTECTIVE_LOSS_PCT", 0.05) or 0.05)
            post_ceil = float(getattr(Config, "MID_WINDOW_POSTERIOR_CEIL", 0.55) or 0.55)
            posterior_decayed = (posterior is not None and posterior <= post_ceil)
            protective_ok = (unrealized_pct <= -protective_loss and posterior_decayed)

            # 1. SignedScore reversal — only exit on strong reversal AND losing position
            # Score oscillates rapidly (±3–7 within seconds); don't panic-sell winners.
            if side == "YES" and sig.signed_score < -5 and protective_ok:
                trigger_reason = "SCORE_REVERSAL"
            elif side == "NO" and sig.signed_score > 5 and protective_ok:
                trigger_reason = "SCORE_REVERSAL"

            # 2. Unrealized loss > 15%
            elif unrealized_pct < -0.15:
                # Respect min-hold window if known.
                if hold_age_s is not None and hold_age_s < min_hold:
                    if state is not None:
                        log.info(
                            "STOP_LOSS_BLOCKED_MIN_HOLD key=%s age_s=%s min_hold=%s unreal=%.2f%%",
                            key,
                            hold_age_s,
                            min_hold,
                            unrealized_pct * 100.0,
                        )
                    continue

                if state is None:
                    trigger_reason = "STOP_LOSS_15PCT"
                else:
                    breach_count = int(getattr(state, "stop_loss_breach_count", {}).get(key, 0) or 0)
                    breach_start = int(getattr(state, "stop_loss_breach_start_ts", {}).get(key, 0) or 0)

                    breach_count += 1
                    if breach_start <= 0:
                        breach_start = now_ts
                    persist_s = now_ts - breach_start

                    state.stop_loss_breach_count[key] = breach_count
                    state.stop_loss_breach_start_ts[key] = breach_start

                    if breach_count >= stop_checks_req or persist_s >= stop_persist_req:
                        trigger_reason = "STOP_LOSS_15PCT_PERSIST"
                    else:
                        log.info(
                            "STOP_LOSS_PERSIST_WAIT key=%s checks=%s/%s persist_s=%s/%s unreal=%.2f%%",
                            key,
                            breach_count,
                            stop_checks_req,
                            persist_s,
                            stop_persist_req,
                            unrealized_pct * 100.0,
                        )
                        continue

            # Reset stop-loss persistence counters when breach condition clears.
            elif state is not None:
                if key in state.stop_loss_breach_count or key in state.stop_loss_breach_start_ts:
                    state.stop_loss_breach_count.pop(key, None)
                    state.stop_loss_breach_start_ts.pop(key, None)

            # 3. Distance from strike > 0.6 * ATR — only if LOSING
            # If BTC moved far from strike in our favor, that's a winning position.
            elif current_distance > 0.6 * atr14 and protective_ok:
                trigger_reason = "STRIKE_DISTANCE_EXCEEDED"

            # 4. CVD or OFI flip against position
            # (Sensitive: CVD/OFI flip while losing)
            elif protective_ok:
                if side == "YES" and (sig.cvd < -0.5 or sig.deep_ofi < -5):
                    trigger_reason = "MICROSTRUCTURE_FLIP"
                elif side == "NO" and (sig.cvd > 0.5 or sig.deep_ofi > 5):
                    trigger_reason = "MICROSTRUCTURE_FLIP"

            if trigger_reason:
                log.warning(f"MID-WINDOW EXIT TRIGGERED for {market_slug}: {trigger_reason} (Unrealized: {unrealized_pct*100:.2f}%)")
                if not token_id:
                    log.error(f"Cannot exit {market_slug}: missing token_id / asset")
                    continue

                api_est_px = (curr_val / size) if size else 0.0
                vpin = float(getattr(sig, "vpin_proxy", 0.0) or 0.0)
                deep_imb = float(getattr(sig, "deep_imbalance", 0.0) or 0.0)
                obi = float(getattr(sig, "obi", 0.0) or 0.0)
                toxic = (vpin >= getattr(Config, "VPIN_BLOCK_THRESHOLD", 0.95)) or (abs(deep_imb) >= 0.80) or (abs(obi) >= 0.30)

                order_id = None
                if toxic and api_est_px > 0:
                    limit_px = max(0.01, min(0.99, round(api_est_px - 0.01, 2)))
                    order_id = await self.limit_sell(token_id, limit_px, size, order_type="FOK")
                    if order_id:
                        log.info(
                            "MID-WINDOW EXIT liquidity-aware: toxic_flow vpin=%.3f obi=%.3f imb=%.3f -> FOK limit_sell @ %.2f",
                            vpin,
                            obi,
                            deep_imb,
                            limit_px,
                        )

                if not order_id:
                    order_id = await self.market_sell(token_id, size)
                if order_id:
                    # Fetch actual fill price from order status instead of using stale currentValue
                    fill_px = api_est_px  # fallback to API estimate
                    await asyncio.sleep(1.0)  # let fill propagate
                    fill_status = await self.get_order_status(order_id)
                    if fill_status and fill_status.get("price", 0) > 0:
                        fill_px = fill_status["price"]
                        log.info(f"MID-WINDOW EXIT fill price: {fill_px:.4f} (API estimate was {curr_val/size:.4f})")

                    if api_est_px and fill_px:
                        slippage_against = (fill_px - api_est_px) / api_est_px
                        if slippage_against < -0.02:
                            log.warning(
                                "MID-WINDOW EXIT HIGH SLIPPAGE: %.2f%% (fill=%.4f vs est=%.4f)",
                                slippage_against * 100.0,
                                fill_px,
                                api_est_px,
                            )

                    entry_px_per_share = entry_val / size
                    actual_pnl_usd = (fill_px - entry_px_per_share) * size
                    exits.append({
                        "market_slug": market_slug,
                        "reason": trigger_reason,
                        "order_id": order_id,
                        "size": size,
                        "entry_price": entry_px_per_share,
                        "exit_price": fill_px,
                        "pnl_usd": actual_pnl_usd,
                        "condition_id": pos.get("conditionId")
                    })

                    # On stop-loss exits, start cooldown and reset persistence counters.
                    if state is not None and trigger_reason.startswith("STOP_LOSS"):
                        state.stop_loss_cooldown_until_ts[key] = now_ts + cooldown_sec
                        state.stop_loss_breach_count.pop(key, None)
                        state.stop_loss_breach_start_ts.pop(key, None)
        
        return exits

    @staticmethod
    def summarize_exposure(positions: list[dict], condition_id: Optional[str] = None) -> dict:
        """Return lightweight exposure metrics from the positions payload."""
        exposure_usd = 0.0
        open_positions = 0
        for p in positions or []:
            try:
                cid = p.get("conditionId") or p.get("condition_id")
                if condition_id and cid and cid != condition_id:
                    continue
                bal = float(p.get("balance") or 0)
                if bal <= 0:
                    continue
                # CLOB positions commonly represent share count; notional is share * avg_price.
                avg_px = float(p.get("averagePrice") or p.get("avgPrice") or p.get("avg_price") or 0)
                if avg_px > 0:
                    exposure_usd += bal * avg_px
                open_positions += 1
            except Exception:
                continue
        return {"open_positions": open_positions, "exposure_usd": exposure_usd}

    # ── Open orders / cancel ──────────────────────────────────────────────────

    async def get_open_orders(self, condition_id: str) -> list[str]:
        if not self.can_trade:
            self._warn_no_creds_once("get_open_orders")
            return []
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
        if not self.can_trade:
            self._warn_no_creds_once("cancel_order")
            return
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.client.cancel(order_id))
        except Exception as e:
            log.warning(f"cancel_order {order_id}: {e}")

    async def replace_order(self, old_order_id: str, token_id: str, new_price: float, size: float, side: str = "BUY", order_type: str = "GTC") -> Optional[str]:
        """Cancel an existing order and place a new one at the updated price.
        Returns the new order_id, or None if the replacement failed."""
        if not self.can_trade:
            self._warn_no_creds_once("replace_order")
            return None
        try:
            await self.cancel_order(old_order_id)
            # Small delay to let the cancel propagate
            await asyncio.sleep(0.5)
            if side == "BUY":
                return await self.limit_buy(token_id, new_price, size, order_type=order_type)
            else:
                return await self.market_sell(token_id, new_price, size)
        except Exception as e:
            log.warning(f"replace_order failed: {e}")
            return None

    async def get_order_status(self, order_id: str) -> Optional[dict]:
        """Check if an order has been filled, partially filled, or is still open."""
        if not self.can_trade:
            self._warn_no_creds_once("get_order_status")
            return None
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
    def smart_entry_price(bid: Optional[float], ask: Optional[float], tick: float = 0.01, aggressive: bool = False) -> Optional[float]:
        """
        aggressive=True  (FOK): use ask to guarantee immediate fill.
        aggressive=False (GTC): bid+tick for passive queue entry.
        Falls back to ask if no bid available.
        """
        if aggressive:
            # Cap at 0.98 — buying at 0.99+ has <1% edge, negative EV after fees
            return round(min(ask, 0.98), 2) if ask is not None else None
        if bid is not None and ask is not None:
            smart_px = round(bid + tick, 2)
            return round(min(smart_px, ask, 0.98), 2)
        return round(min(ask, 0.98), 2) if ask is not None else None

    # ── Order execution ───────────────────────────────────────────────────────

    async def limit_buy(
        self, token_id: str, price: float, size: float, order_type: str = "GTC", nonce: int = 0
    ) -> Optional[str]:
        """Place a limit buy order. Returns order_id or None."""
        if not self.can_trade:
            self._warn_no_creds_once("limit_buy")
            return None
        ok = await self._ensure_onchain_allowance(min_required_usd=Config.MIN_TRADE_USD)
        if not ok:
            log.error("Order blocked: allowance check failed for configured USDC address.")
            raise PolyApiException(error_msg="allowance_missing")
        try:
            # CLOB requires: maker_amount max 2 decimals, taker_amount max 4 decimals
            # For BUY: maker_amount = price * size (USDC cost), taker_amount = size (shares)
            # With 2-decimal prices, integer sizes guarantee clean maker_amounts
            clean_price = round(price, 2)
            clean_size = int(size)  # floor to integer — guarantees price*size has ≤2 decimals
            if clean_size < 1:
                log.warning(f"limit_buy: size {size} floors to 0 shares at price {clean_price}")
                return None
            log.info(f"limit_buy: price={clean_price} size={clean_size} notional=${clean_price * clean_size:.2f} type={order_type}")
            args = OrderArgs(
                token_id = token_id,
                price    = clean_price,
                size     = float(clean_size),
                side     = "BUY",
                nonce    = nonce,
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
        self, token_id: str, amount_usd: float, nonce: int = 0
    ) -> Optional[str]:
        """Market IOC buy for `amount_usd` dollars of a token."""
        if not self.can_trade:
            self._warn_no_creds_once("market_buy")
            return None
        ok = await self._ensure_onchain_allowance(min_required_usd=Config.MIN_TRADE_USD)
        if not ok:
            log.error("Order blocked: allowance check failed for configured USDC address.")
            raise PolyApiException(error_msg="allowance_missing")
        try:
            args = MarketOrderArgs(
                token_id = token_id,
                amount   = round(amount_usd, 2),
                side     = "BUY",
                nonce    = nonce,
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

    async def market_sell(
        self, token_id: str, size: float, nonce: int = 0
    ) -> Optional[str]:
        """Market IOC sell for `amount_shares` of a token."""
        if not self.can_trade:
            self._warn_no_creds_once("market_sell")
            return None
        ok = await self._ensure_conditional_allowance(token_id=token_id, min_required_shares=float(size or 0.0))
        if not ok:
            log.error("Order blocked: conditional token allowance check failed for token_id=%s", str(token_id))
            raise PolyApiException(error_msg="conditional_allowance_missing")
        try:
            args = MarketOrderArgs(
                token_id = token_id,
                amount   = size,
                side     = "SELL",
                nonce    = nonce,
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

    async def limit_sell(
        self, token_id: str, price: float, size: float, order_type: str = "GTC", nonce: int = 0
    ) -> Optional[str]:
        if not self.can_trade:
            self._warn_no_creds_once("limit_sell")
            return None
        ok = await self._ensure_conditional_allowance(token_id=token_id, min_required_shares=float(size or 0.0))
        if not ok:
            log.error("Order blocked: conditional token allowance check failed for token_id=%s", str(token_id))
            raise PolyApiException(error_msg="conditional_allowance_missing")
        try:
            clean_price = round(price, 2)
            clean_size = int(size) if size > 1 else round(size, 2)
            if clean_size < 1 and int(size) == 0:
                # Selling fractional position — keep as-is, CLOB may accept for sells
                clean_size = round(size, 2)
            log.info(f"limit_sell: price={clean_price} size={clean_size} type={order_type}")
            args = OrderArgs(
                token_id = token_id,
                price    = clean_price,
                size     = float(clean_size),
                side     = "SELL",
                nonce    = nonce,
            )
            ot = OrderType.FOK if order_type in ("IOC", "FOK") else OrderType.GTC
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
