"""
SQLite async state persistence via SQLAlchemy + aiosqlite.

All state lives in a single 'kv' table (key TEXT, value JSON TEXT).
On startup, load() returns the full state dict.
save() updates individual keys atomically.
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

from config import Config

log = logging.getLogger(__name__)


# ── State schema as a dataclass ───────────────────────────────────────────────

@dataclass
class HeldPosition:
    side: Optional[str]             = None   # "YES" | "NO" | None
    token_id: Optional[str]         = None
    entry_price: Optional[float]    = None
    avg_entry_price: Optional[float] = None
    size: Optional[float]           = None   # shares
    size_usd: Optional[float]       = None
    entry_signed_score: Optional[float] = None
    condition_id: Optional[str]     = None
    # Phase 3: order fill tracking
    order_id: Optional[str]         = None
    is_pending: bool                = False  # True until fill confirmed
    placed_at_ts: Optional[int]     = None   # when order was placed


@dataclass
class BeliefVolSample:
    delta_x: float
    ts: int   # unix timestamp


@dataclass
class TradeRecord:
    ts: int
    side: str
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    outcome: Optional[str]   # "WIN" | "LOSS" | "OPEN"
    score: float
    window: int


@dataclass
class EngineState:
    # ── Window tracking ───────────────────────────────────────────────────────
    last_window_start_sec: Optional[int]  = None
    last_market_slug: Optional[str]       = None
    last_condition_id: Optional[str]      = None
    last_market_expiry: Optional[int]     = None
    trades_this_window: int               = 0

    # ── Strike locking (FIX #2) ───────────────────────────────────────────────
    locked_strike_price: Optional[float]  = None
    strike_source: Optional[str]          = None
    strike_window_start: Optional[int]    = None

    # ── Loss / halt tracking ──────────────────────────────────────────────────
    loss_streak: int                      = 0
    trading_halted: bool                  = False

    # ── Position tracking ─────────────────────────────────────────────────────
    held_position: HeldPosition           = field(default_factory=HeldPosition)
    trade_history: List[TradeRecord]      = field(default_factory=list)

    # ── Microstructure memory ─────────────────────────────────────────────────
    prev_bid_depth20: Optional[float]     = None
    prev_ask_depth20: Optional[float]     = None
    prev_deep_imbalance: Optional[float]  = None
    last_vpin_proxy: Optional[float]      = None
    prev_bid_px: Optional[float]          = None
    prev_ask_px: Optional[float]          = None
    prev_bid_sz: Optional[float]          = None
    prev_ask_sz: Optional[float]          = None
    last_hl_fetch_ts: int                 = 0
    prev_hl_mid: Optional[float]          = None
    prev_ofi_recent: Optional[float]      = None
    ofi_15m_by_window: Dict[str, float]   = field(default_factory=dict)
    zero_ofi_cycles: int                  = 0

    # ── Micro score memory (stale micro fallback) ──────────────────────────────
    last_cvd_score: float                 = 0.0
    last_ofi_score: float                 = 0.0
    last_imbalance_score: float           = 0.0
    last_flow_accel_score: float          = 0.0

    # ── CVD accumulator (reset per window) ────────────────────────────────────
    cvd: float                            = 0.0
    prev_cvd_slope: float                 = 0.0

    # ── Phase 2: Accumulated OFI + cross-exchange ─────────────────────────────
    accumulated_ofi: float                = 0.0   # sum of OFI deltas within window
    cross_cvd_agree: bool                 = True   # Binance & Coinbase CVD same direction
    prev_cvd_total_vol: float             = 0.0   # previous cycle's total CVD volume

    # ── Price / indicator memory ──────────────────────────────────────────────
    prev_price: Optional[float]           = None
    prev_price2: Optional[float]          = None
    prev_price3: Optional[float]          = None
    prev_volume_1m: Optional[float]       = None
    prev_obv: Optional[float]             = None
    prev_obv2: Optional[float]            = None
    prev_obv3: Optional[float]            = None
    prev_obv4: Optional[float]            = None
    prev_mfi: Optional[float]             = None

    # ── Polymarket price memory ────────────────────────────────────────────────
    last_pm_px_yes: Optional[float]       = None
    last_pm_px_no: Optional[float]        = None
    last_sizing: float                    = 0.0

    # ── Delta tracking ────────────────────────────────────────────────────────
    prev_cycle_score: Optional[float]     = None
    prev_cycle_price: Optional[float]     = None

    # ── Belief volatility ─────────────────────────────────────────────────────
    prev_x: Optional[float]               = None
    belief_vol_samples: List[BeliefVolSample] = field(default_factory=list)


# ── State manager ─────────────────────────────────────────────────────────────

class StateManager:
    def __init__(self, db_url: str = None):
        url = db_url or Config.DATABASE_URL
        self.engine = create_async_engine(url, echo=False)
        self._session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self._state: Optional[EngineState] = None

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.execute(text(
                "CREATE TABLE IF NOT EXISTS kv "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            ))
        log.info("State DB initialised")

    async def load(self) -> EngineState:
        await self.init_db()
        async with self._session_factory() as session:
            result = await session.execute(
                text("SELECT key, value FROM kv")
            )
            rows = {row.key: json.loads(row.value) for row in result}

        state = EngineState()
        if not rows:
            log.info("Fresh state — no persisted data")
            self._state = state
            return state

        # Deserialise each field
        for k, v in rows.items():
            if k == "held_position" and isinstance(v, dict):
                state.held_position = HeldPosition(**{
                    kk: vv for kk, vv in v.items() if kk in HeldPosition.__dataclass_fields__
                })
            elif k == "trade_history" and isinstance(v, list):
                state.trade_history = [
                    TradeRecord(**{kk: vv for kk, vv in t.items() if kk in TradeRecord.__dataclass_fields__})
                    for t in v
                ]
            elif k == "belief_vol_samples" and isinstance(v, list):
                state.belief_vol_samples = [
                    BeliefVolSample(**s) for s in v
                ]
            elif hasattr(state, k):
                setattr(state, k, v)

        log.info(
            f"State loaded: streak={state.loss_streak} "
            f"halted={state.trading_halted} "
            f"position={state.held_position.side or 'FLAT'}"
        )
        self._state = state
        return state

    async def save(self, state: EngineState):
        """Persist full state to SQLite atomically."""
        self._state = state
        data = {
            "last_window_start_sec":  state.last_window_start_sec,
            "last_market_slug":       state.last_market_slug,
            "last_condition_id":      state.last_condition_id,
            "last_market_expiry":     state.last_market_expiry,
            "trades_this_window":     state.trades_this_window,
            "locked_strike_price":    state.locked_strike_price,
            "strike_source":          state.strike_source,
            "strike_window_start":    state.strike_window_start,
            "loss_streak":            state.loss_streak,
            "trading_halted":         state.trading_halted,
            "held_position":          asdict(state.held_position),
            "trade_history":          [asdict(t) for t in state.trade_history[-100:]],
            "belief_vol_samples":     [asdict(s) for s in state.belief_vol_samples],
            "prev_bid_depth20":       state.prev_bid_depth20,
            "prev_ask_depth20":       state.prev_ask_depth20,
            "prev_deep_imbalance":    state.prev_deep_imbalance,
            "last_vpin_proxy":        state.last_vpin_proxy,
            "prev_bid_px":            state.prev_bid_px,
            "prev_ask_px":            state.prev_ask_px,
            "prev_bid_sz":            state.prev_bid_sz,
            "prev_ask_sz":            state.prev_ask_sz,
            "last_hl_fetch_ts":       state.last_hl_fetch_ts,
            "prev_hl_mid":            state.prev_hl_mid,
            "prev_ofi_recent":        state.prev_ofi_recent,
            "ofi_15m_by_window":      state.ofi_15m_by_window,
            "zero_ofi_cycles":        state.zero_ofi_cycles,
            "last_cvd_score":         state.last_cvd_score,
            "last_ofi_score":         state.last_ofi_score,
            "last_imbalance_score":   state.last_imbalance_score,
            "last_flow_accel_score":  state.last_flow_accel_score,
            "cvd":                    state.cvd,
            "prev_cvd_slope":         state.prev_cvd_slope,
            "accumulated_ofi":        state.accumulated_ofi,
            "cross_cvd_agree":        state.cross_cvd_agree,
            "prev_cvd_total_vol":     state.prev_cvd_total_vol,
            "prev_price":             state.prev_price,
            "prev_price2":            state.prev_price2,
            "prev_price3":            state.prev_price3,
            "prev_volume_1m":         state.prev_volume_1m,
            "prev_obv":               state.prev_obv,
            "prev_obv2":              state.prev_obv2,
            "prev_obv3":              state.prev_obv3,
            "prev_obv4":              state.prev_obv4,
            "prev_mfi":               state.prev_mfi,
            "last_pm_px_yes":         state.last_pm_px_yes,
            "last_pm_px_no":          state.last_pm_px_no,
            "last_sizing":            state.last_sizing,
            "prev_cycle_score":       state.prev_cycle_score,
            "prev_cycle_price":       state.prev_cycle_price,
            "prev_x":                 state.prev_x,
        }
        async with self._session_factory() as session:
            async with session.begin():
                for key, value in data.items():
                    await session.execute(
                        text("INSERT OR REPLACE INTO kv (key, value) VALUES (:k, :v)"),
                        {"k": key, "v": json.dumps(value)}
                    )

    def get_cached(self) -> Optional[EngineState]:
        return self._state
