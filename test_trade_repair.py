import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import text

import dashboard
from config import Config
from state import EngineState, StateManager, TradeRecord
from trade_utils import compute_auto_settle_outcome


def _run(coro):
    return asyncio.run(coro)


async def _prime_db(sm: StateManager):
    async with sm.engine.begin() as conn:
        await conn.execute(text("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)"))
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS closed_trades ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "timestamp INTEGER, market_slug VARCHAR, size FLOAT, entry_price FLOAT, exit_price FLOAT, "
                "pnl_usd FLOAT, outcome_win INTEGER, slippage FLOAT, exit_reason VARCHAR, "
                "regime VARCHAR, features VARCHAR, kelly_fraction FLOAT)"
            )
        )


def _repair_inputs():
    return {
        "buy": {
            "timestamp": 1775675353,
            "size": 8.274315,
            "usdcSize": 6.159999,
            "price": 0.7299999822240235,
            "asset": "yes-token",
            "side": "BUY",
            "slug": "btc-updown-15m-1775674800",
            "type": "TRADE",
        },
        "sells": [
            {
                "timestamp": 1775675403,
                "size": 8.27,
                "usdcSize": 6.78009,
                "price": 0.83,
                "asset": "yes-token",
                "side": "SELL",
                "slug": "btc-updown-15m-1775674800",
                "type": "TRADE",
            }
        ],
        "event": {"markets": [{"outcomes": '["Up","Down"]', "outcomePrices": '["1","0"]'}]},
        "winning_outcome": "Up",
        "buy_size": 8.274315,
        "buy_usdc": 6.159999,
        "buy_effective_price": 6.159999 / 8.274315,
        "sell_size": 8.27,
        "sell_usdc": 6.78009,
        "sell_effective_price": 6.78009 / 8.27,
        "sell_ts": 1775675403,
    }


class DummyEngine:
    def __init__(self, state_mgr, state):
        self.state_mgr = state_mgr
        self.state = state

    def rebuild_state_totals_from_trade_history(self):
        summary = dashboard._summarize_trade_history(self.state.trade_history)
        self.state.total_trades = summary["total_trades"]
        self.state.total_wins = summary["total_wins"]
        self.state.total_losses = summary["total_losses"]
        self.state.total_pnl_usd = summary["total_pnl_usd"]
        self.state.loss_streak = summary["loss_streak"]

    def _closed_trade_row_from_record(self, tr, market_slug, timestamp=None):
        return {
            "timestamp": int(timestamp or 1775675403),
            "market_slug": market_slug,
            "size": tr.size,
            "entry_price": tr.entry_price,
            "exit_price": tr.exit_price,
            "pnl_usd": (tr.pnl or 0.0) * (tr.entry_price or 0.0) * (tr.size or 0.0),
            "outcome_win": 1 if tr.outcome == "WIN" else 0,
            "slippage": tr.slippage,
            "exit_reason": tr.exit_reason,
            "regime": "normal",
            "features": None,
            "kelly_fraction": None,
        }


def test_compute_auto_settle_outcome_marks_profitable_loss_settlement_as_win():
    result = compute_auto_settle_outcome(
        entry_price=0.77,
        entry_size=8.438355,
        partial_exits=[{"size": 8.27, "price": 0.83}],
        position_won=False,
    )
    assert result["outcome"] == "WIN"
    assert result["exit_reason"] == "AUTO_SETTLE_WIN"
    assert result["blended_pnl"] > 0


def test_resync_closed_trades_updates_existing_row_without_duplicate(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/repair.db"
    sm = StateManager(db_url=db_url)
    _run(sm.init_db())
    _run(_prime_db(sm))
    _run(
        sm.record_closed_trade(
            ts=1775675403,
            market_slug="btc-updown-15m-1775674800",
            size=8.438355,
            entry_price=0.77,
            exit_price=0.8134,
            pnl_usd=-0.91,
            outcome_win=0,
            exit_reason="AUTO_SETTLE_LOSS",
        )
    )
    result = _run(
        sm.resync_closed_trades_from_canonical(
            [
                {
                    "timestamp": 1775675403,
                    "market_slug": "btc-updown-15m-1775674800",
                    "size": 8.274315,
                    "entry_price": 0.7444723823059674,
                    "exit_price": 0.8203632595621286,
                    "pnl_usd": 0.624406,
                    "outcome_win": 1,
                    "exit_reason": "AUTO_SETTLE_WIN",
                }
            ],
            market_slug="btc-updown-15m-1775674800",
            delete_extra=True,
        )
    )
    rows = _run(sm.fetch_closed_trades(market_slug="btc-updown-15m-1775674800"))
    assert result["inserted"] == 0
    assert len(rows) == 1
    assert rows[0]["outcome_win"] == 1
    assert rows[0]["exit_reason"] == "AUTO_SETTLE_WIN"
    assert rows[0]["pnl_usd"] > 0
    _run(sm.engine.dispose())


def test_repair_trade_dry_run_returns_repaired_snapshot(monkeypatch):
    state = EngineState()
    state.trade_history = [
        TradeRecord(
            ts=1775675350,
            side="YES",
            entry_price=0.77,
            exit_price=0.8134,
            pnl=0.056416278340456705,
            outcome="LOSS",
            score=0.0,
            window=1775674800,
            size=8.438355,
            exit_reason="AUTO_SETTLE_LOSS",
            partial_exits=[{"size": 8.27, "price": 0.83, "reason": "TP_FULL", "ts": 1775675701}],
        )
    ]
    engine = DummyEngine(
        state_mgr=SimpleNamespace(fetch_closed_trades=lambda market_slug=None: asyncio.sleep(0, result=[])),
        state=state,
    )
    monkeypatch.setattr(dashboard, "engine", engine, raising=False)
    monkeypatch.setattr(Config, "DASHBOARD_ADMIN_TOKEN", "secret")
    monkeypatch.setattr(Config, "POLYMARKET_PRIVATE_KEY", "0x" + "1" * 64)

    async def _fake_fetch(wallet, slug):
        assert slug == "btc-updown-15m-1775674800"
        return _repair_inputs()

    monkeypatch.setattr(dashboard, "_fetch_trade_repair_inputs", _fake_fetch)

    client = TestClient(dashboard.app)
    resp = client.post(
        "/api/repair-trade",
        json={"market_slug": "btc-updown-15m-1775674800", "dry_run": True},
        headers={"x-admin-token": "secret"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["dry_run"] is True
    assert data["trade_before"]["outcome"] == "LOSS"
    assert data["trade_after"]["outcome"] == "WIN"
    assert data["trade_after"]["exit_reason"] == "AUTO_SETTLE_WIN"
    assert state.trade_history[0].outcome == "LOSS"


def test_repair_trade_endpoint_repairs_current_trade_and_dedupes_rows(tmp_path, monkeypatch):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/repair_live.db"
    sm = StateManager(db_url=db_url)
    _run(sm.init_db())
    _run(_prime_db(sm))

    state = EngineState()
    state.trade_history = [
        TradeRecord(
            ts=1775675350,
            side="YES",
            entry_price=0.77,
            exit_price=0.8134,
            pnl=0.056416278340456705,
            outcome="LOSS",
            score=0.0,
            window=1775674800,
            size=8.438355,
            exit_reason="AUTO_SETTLE_LOSS",
            partial_exits=[{"size": 8.27, "price": 0.83, "reason": "TP_FULL", "ts": 1775675701}],
        )
    ]
    sm._state = state
    _run(sm.save(state))
    _run(
        sm.record_closed_trade(
            ts=1775675403,
            market_slug="btc-updown-15m-1775674800",
            size=8.438355,
            entry_price=0.77,
            exit_price=0.8134,
            pnl_usd=-0.91,
            outcome_win=0,
            exit_reason="AUTO_SETTLE_LOSS",
        )
    )
    _run(
        sm.record_closed_trade(
            ts=1775675401,
            market_slug="btc-updown-15m-1775674800",
            size=8.27,
            entry_price=0.77,
            exit_price=0.83,
            pnl_usd=0.4962,
            outcome_win=1,
            exit_reason="TP_FULL",
        )
    )

    engine = DummyEngine(sm, state)
    monkeypatch.setattr(dashboard, "engine", engine, raising=False)
    monkeypatch.setattr(Config, "DASHBOARD_ADMIN_TOKEN", "secret")
    monkeypatch.setattr(Config, "POLYMARKET_PRIVATE_KEY", "0x" + "1" * 64)

    async def _fake_fetch(wallet, slug):
        assert slug == "btc-updown-15m-1775674800"
        return _repair_inputs()

    monkeypatch.setattr(dashboard, "_fetch_trade_repair_inputs", _fake_fetch)

    client = TestClient(dashboard.app)
    resp = client.post(
        "/api/repair-trade",
        json={"market_slug": "btc-updown-15m-1775674800", "dry_run": False},
        headers={"x-admin-token": "secret"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["trade_after"]["outcome"] == "WIN"
    assert data["duplicates_removed"] == 1

    rows = _run(sm.fetch_closed_trades(market_slug="btc-updown-15m-1775674800"))
    assert len(rows) == 1
    assert rows[0]["outcome_win"] == 1
    assert rows[0]["pnl_usd"] > 0

    repaired_trade = state.trade_history[0]
    assert repaired_trade.outcome == "WIN"
    assert repaired_trade.exit_reason == "AUTO_SETTLE_WIN"
    assert round(repaired_trade.size, 6) == round(8.274315, 6)
    assert state.total_wins == 1
    assert state.total_losses == 0
    _run(sm.engine.dispose())
