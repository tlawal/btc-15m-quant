"""
SQLite async state persistence via SQLAlchemy + aiosqlite.

All state lives in a single 'kv' table (key TEXT, value JSON TEXT).
On startup, load() returns the full state dict.
save() updates individual keys atomically.
"""

import json
import os
import logging
import asyncio
from dataclasses import dataclass, field, is_dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from alembic.config import Config as AlembicConfig
from alembic import command

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
    intended_entry_price: Optional[float] = None
    intended_exit_price: Optional[float]  = None
    # Phase 3: order fill tracking
    order_id: Optional[str]         = None
    is_pending: bool                = False  # True until fill confirmed
    placed_at_ts: Optional[int]     = None   # when order was placed
    exit_reason: Optional[str]      = None   # for tracking why an exit was placed


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
    slippage: Optional[float] = None
    size: Optional[float] = None


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
    open_positions_api: List[Dict[str, Any]] = field(default_factory=list)

    # ── Performance tracking ──────────────────────────────────────────────────
    total_trades: int                     = 0
    total_wins: int                       = 0
    total_losses: int                     = 0
    total_pnl_usd: float                  = 0.0
    unclaimed_usdc: float                 = 0.0
    session_start_balance: Optional[float] = None
    performance_metrics: dict             = field(default_factory=dict)
    last_tuned_time: Optional[float]      = None

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
    prev_cycle_cvd: float                 = 0.0
    latencies: dict[str, float]           = field(default_factory=dict)  # Phase 4: Name -> ms

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
    last_posterior_up: Optional[float]    = None
    last_posterior_down: Optional[float]  = None

    # ── Delta tracking ────────────────────────────────────────────────────────
    prev_cycle_score: Optional[float]     = None
    prev_cycle_price: Optional[float]     = None
    prev_x: Optional[float]               = None  # Bayesian z-score memory

    # ── Belief volatility ─────────────────────────────────────────────────────
    belief_vol_samples: List[BeliefVolSample] = field(default_factory=list)

    # ── Nightly AI review tracking ─────────────────────────────────────────────
    last_review_date: Optional[str]       = None

    # ── Tier 1: Cross-window memory ───────────────────────────────────────────
    window_outcomes: List[str]            = field(default_factory=list)  # ["UP","DOWN",...]
    last_funding_rate: Optional[float]    = None  # previous cycle's funding rate


# ── State manager ─────────────────────────────────────────────────────────────

class StateManager:
    def __init__(self, db_url: str = None):
        url = db_url or Config.DATABASE_URL
        
        # Robust path check for SQLite
        if url.startswith("sqlite+aiosqlite:///"):
            db_path = url.replace("sqlite+aiosqlite:///", "")
            db_dir = os.path.dirname(db_path)
            
            # If path is absolute (starts with /), check directory access
            if db_path.startswith("/") and db_dir:
                if not os.path.exists(db_dir):
                    try:
                        os.makedirs(db_dir, exist_ok=True)
                    except Exception as e:
                        log.error(f"Cannot create DB directory {db_dir}: {e}")
                        
                if not os.access(db_dir if os.path.exists(db_dir) else ".", os.W_OK):
                    fallback_url = "sqlite+aiosqlite:///./state.db"
                    log.warning(f"🚨 PERMISSION DENIED: Directory {db_dir} is not writable. Falling back to {fallback_url}")
                    url = fallback_url
                    Config.DATABASE_URL = url  # Sync globally
        
        self.engine = create_async_engine(url, echo=False)
        self._session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self._state: Optional[EngineState] = None

    async def init_db(self):
        # Run Alembic migrations synchronously BEFORE acquiring any async connections 
        # to prevent SQLite database locked errors.
        try:
            _HERE = os.path.dirname(os.path.abspath(__file__))
            alembic_ini_path = os.path.join(_HERE, "alembic.ini")
            alembic_cfg = AlembicConfig(alembic_ini_path)
            alembic_cfg.set_main_option("script_location", os.path.join(_HERE, "alembic"))
            
            # Sync Alembic with our engine's actual URL (handles fallback automatically)
            alembic_cfg.set_main_option("sqlalchemy.url", str(self.engine.url).replace("sqlite+aiosqlite:///", "sqlite:///"))
            
            # Alembic's command.upgrade is synchronous; use run_in_executor for safety inside async
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, command.upgrade, alembic_cfg, "head")
            log.info(f"Alembic migrations applied successfully to {self.engine.url}")
        except Exception as e:
            log.error(f"Failed to run Alembic migrations on {self.engine.url}: {e}")
            # If migrations fail on a writable fallback, we have a bigger problem
            if "state.db" in str(self.engine.url) and "sqlite" in str(self.engine.url):
                log.error("CRITICAL: Alembic failed on local fallback state.db")
            
        if "sqlite" in str(self.engine.url):
            async with self.engine.begin() as conn:
                await conn.execute(text("PRAGMA journal_mode=WAL;"))
                await conn.execute(text("PRAGMA synchronous=NORMAL;"))
        
        log.info("State DB initialised with Alembic")

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
            elif k == "latencies" and isinstance(v, dict):
                state.latencies = v
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
            "total_trades":           state.total_trades,
            "total_wins":             state.total_wins,
            "total_losses":           state.total_losses,
            "total_pnl_usd":          state.total_pnl_usd,
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
            "prev_x":                 state.prev_x,
            "latencies":              state.latencies,
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
            "last_posterior_up":      state.last_posterior_up,
            "last_posterior_down":    state.last_posterior_down,
            "prev_cycle_score":       state.prev_cycle_score,
            "prev_cycle_price":       state.prev_cycle_price,
            "performance_metrics":    state.performance_metrics,
            "last_review_date":       state.last_review_date,
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

    async def record_closed_trade(self, ts: int, market_slug: str, size: float, entry_price: float, exit_price: float, pnl_usd: float, outcome_win: int, regime: str = None, features: dict = None, kelly_fraction: float = None):
        async with self.engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO closed_trades (timestamp, market_slug, size, entry_price, exit_price, pnl_usd, outcome_win, regime, features, kelly_fraction) "
                "VALUES (:ts, :slug, :sz, :ep, :xp, :pnl, :win, :regime, :feats, :kelly)"
            ), {
                "ts": ts, "slug": market_slug, "sz": size,
                "ep": entry_price, "xp": exit_price, "pnl": pnl_usd, "win": outcome_win,
                "regime": regime, "feats": json.dumps(features) if features else None, "kelly": kelly_fraction
            })

    async def calculate_performance_metrics(self) -> dict:
        async with self._session_factory() as session:
            result = await session.execute(text("SELECT pnl_usd, outcome_win, regime FROM closed_trades"))
            rows = result.fetchall()
            
        total_trades = len(rows)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "avg_pnl_per_trade": 0.0,
                "regimes": {}
            }
            
        wins = sum(1 for r in rows if r.outcome_win == 1)
        win_rate = round(wins / total_trades, 2)
        
        gross_profit = sum(r.pnl_usd for r in rows if r.pnl_usd > 0)
        gross_loss = abs(sum(r.pnl_usd for r in rows if r.pnl_usd < 0))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0
        
        pnl_list = [r.pnl_usd for r in rows]
        avg_pnl = sum(pnl_list) / total_trades
        
        # Per-regime stats
        regimes = {}
        for r in rows:
            reg = r.regime or "unknown"
            if reg not in regimes:
                regimes[reg] = {"trades": 0, "wins": 0, "pnl": 0.0}
            regimes[reg]["trades"] += 1
            if r.outcome_win == 1:
                regimes[reg]["wins"] += 1
            regimes[reg]["pnl"] += r.pnl_usd

        for reg in regimes:
            regimes[reg]["win_rate"] = round(regimes[reg]["wins"] / regimes[reg]["trades"], 2)
            regimes[reg]["avg_pnl"] = round(regimes[reg]["pnl"] / regimes[reg]["trades"], 2)

        import statistics
        try:
            stdev = statistics.stdev(pnl_list)
            sharpe = round((avg_pnl / stdev) * (len(pnl_list) ** 0.5), 2) if stdev > 0 else 0.0
        except statistics.StatisticsError:
            sharpe = 0.0
            
        metrics = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "avg_pnl_per_trade": round(avg_pnl, 2),
            "regimes": regimes
        }
        
        if self._state:
            self._state.performance_metrics = metrics
            await self.save(self._state)
            
        return metrics
