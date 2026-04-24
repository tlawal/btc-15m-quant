"""
Backtest Engine — unified replay harness for the dashboard Backtest page.

Design notes
------------
- Three fill models: `perfect` (theoretical max; fills at logged mid), `synthetic`
  (the legacy `0.50 + edge * 0.5` approximation retained for back-compat), and
  `clob_replay` (walks a recorded CLOB snapshot). Snapshot-based replay only
  works when `data/clob_snapshots/` has been populated — falls back to
  `synthetic` + a warning if absent.
- Source of truth for replayable windows is `logs/features.jsonl` (per-cycle
  feature rows) joined to `logs/outcomes.jsonl` (one row per resolved window)
  on `window_start`.
- No import of `main.py` / live network clients — pure replay, safe to run
  inside the dashboard process without touching trading state.
- Metrics: Sharpe, Sortino, Calmar, max drawdown, hit rate, expectancy, PF,
  avg win / loss, decomposed PnL (signal / prior / execution / residual).
- Strategy flags (`tier1`, `payoff`, `calib`, `free`) are applied as simple
  gates on top of the replayed decisions. `tier1` = BTC-confirmation gate on
  REVERSE_CONVERGENCE / TIME_DECAY exits, approximated in replay by refusing
  to exit a position that is still ITM on the underlying.

This is a first-pass engine intended to make the dashboard Backtest page
functional end-to-end. The CLOB-replay path and richer exit simulation are
layered on later.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(HERE, "logs")
FEATURES_PATH = os.path.join(LOGS_DIR, "features.jsonl")
OUTCOMES_PATH = os.path.join(LOGS_DIR, "outcomes.jsonl")
SNAPSHOTS_DIR = os.path.join(HERE, "data", "clob_snapshots")
BACKTEST_DIR = os.path.join(HERE, "data", "backtests")
os.makedirs(BACKTEST_DIR, exist_ok=True)


# ───────────────────────── dataclasses ─────────────────────────

@dataclass
class BacktestParams:
    fill_model: str = "synthetic"      # perfect | synthetic | clob_replay
    source: str = "trade_features"     # trade_features | clob_snapshots | logs
    start_ts: Optional[int] = None     # unix seconds
    end_ts: Optional[int] = None
    score_offset: float = 0.0
    edge_offset: float = 0.0
    flags: dict = field(default_factory=dict)  # tier1, payoff, calib, free
    notional_usd: float = 10.0         # USD per trade
    min_edge: float = 0.005            # below this edge, skip
    min_score: float = 2.5             # abs-score gate
    min_rem_max: float = 7.5           # only enter when min_rem <= this
    min_rem_min: float = 0.5           # don't enter in the final 30s
    fee_rate: float = 0.02             # Polymarket taker fee on price (2%)
    slippage_bps: float = 50           # 0.5% assumed slippage per side


@dataclass
class BacktestTrade:
    window_ts: int
    side: str                 # YES | NO
    entry_px: float
    exit_px: float
    size: float               # shares (notional_usd / entry_px)
    pnl_pct: float
    pnl_usd: float
    slippage_pct: float
    exit_reason: str
    outcome: str              # WIN | LOSS
    theoretical_px: Optional[float] = None
    # PnL decomposition (all in % of notional)
    pnl_signal: float = 0.0
    pnl_prior: float = 0.0
    pnl_execution: float = 0.0
    pnl_residual: float = 0.0
    min_rem_at_entry: float = 0.0
    posterior: float = 0.0
    edge: float = 0.0


# ───────────────────────── data loaders ─────────────────────────

def _iter_jsonl(path: str) -> Iterable[dict]:
    if not os.path.isfile(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_outcomes() -> dict:
    """Return {window_start: btc_close_at_settlement}."""
    out: dict = {}
    for rec in _iter_jsonl(OUTCOMES_PATH):
        ws = rec.get("window_start")
        bc = rec.get("btc_close")
        if ws is None or bc is None:
            continue
        out[int(ws)] = float(bc)
    return out


def _pick_entry_row(rows: list[dict]) -> Optional[dict]:
    """
    From all feature rows within a single window, pick one representative
    'entry candidate' row: the earliest row where |signed_score| >= 2.0 AND
    min_rem is between (0.5, 7.5). Falls back to the last row if none qualifies.
    """
    if not rows:
        return None
    rows_sorted = sorted(rows, key=lambda r: float(r.get("ts", 0)))
    for r in rows_sorted:
        mr = float(r.get("min_rem", 99))
        ss = abs(float(r.get("signed_score", 0)))
        if 0.5 <= mr <= 7.5 and ss >= 2.0:
            return r
    return rows_sorted[-1]


# ───────────────────────── replay math ─────────────────────────

def _post_up(row: dict, apply_calib: bool) -> float:
    """Return the posterior (optionally passed through bucketed calibration)."""
    raw = float(row.get("posterior_final_up", row.get("posterior_fair_up", 0.5)))
    if not apply_calib:
        return raw
    try:
        from calibration import calibrate
        return calibrate(raw, min_rem=float(row.get("min_rem", 7.0)))
    except Exception:
        return raw


def _decide_side(row: dict, apply_calib: bool = False) -> Optional[str]:
    """Return 'YES' | 'NO' | None based on signed_score + posterior."""
    ss = float(row.get("signed_score", 0))
    post_up = _post_up(row, apply_calib)
    if ss > 0 and post_up > 0.52:
        return "YES"
    if ss < 0 and post_up < 0.48:
        return "NO"
    return None


def _row_edge(row: dict, side: str, apply_calib: bool = False) -> float:
    """Edge = model_prob - market_prob."""
    post_up = _post_up(row, apply_calib)
    yes_mid = float(row.get("yes_mid", 0.5))
    if side == "YES":
        return post_up - yes_mid
    return (1.0 - post_up) - (1.0 - yes_mid)


def _entry_price(row: dict, side: str, fill_model: str, slippage_bps: float) -> float:
    yes_mid = float(row.get("yes_mid", 0.5))
    px = yes_mid if side == "YES" else float(row.get("no_mid", 1.0 - yes_mid))
    if fill_model == "perfect":
        return px
    if fill_model == "synthetic":
        # legacy backtest_historical.py formula, preserved for continuity
        edge = _row_edge(row, side)
        return max(0.02, min(0.98, 0.50 + edge * 0.5))
    # clob_replay or any other → mid + slippage
    return max(0.02, min(0.98, px * (1 + slippage_bps / 10_000.0)))


def _exit_price(won: bool, fill_model: str, slippage_bps: float) -> float:
    """YES wins → $1.00; YES loses → $0.00 (symmetric for NO)."""
    if fill_model == "perfect":
        return 1.0 if won else 0.0
    haircut = slippage_bps / 10_000.0
    if won:
        return max(0.0, 1.0 - haircut)
    return haircut  # loser gets a tiny haircut added


def _gate_passes(row: dict, params: BacktestParams, side: str, edge: float) -> Optional[str]:
    """Return None if passes, else a string reason for skip."""
    abs_score = abs(float(row.get("signed_score", 0)))
    min_rem = float(row.get("min_rem", 99))
    if min_rem > params.min_rem_max:
        return "EARLY_WINDOW"
    if min_rem < params.min_rem_min:
        return "FINAL_30S"
    if abs_score < params.min_score + params.score_offset:
        return "SCORE_GATE"
    if edge < params.min_edge + params.edge_offset:
        return "EDGE_GATE"

    # Payoff-geometry gate (Tier 2 #6): margin_σ = dist / (ATR × √(min_rem/15))
    if params.flags.get("payoff"):
        atr = float(row.get("atr14", 150.0)) or 150.0
        btc_px = float(row.get("btc_price", 0.0)) or 0.0
        strike = float(row.get("strike", btc_px)) or btc_px
        dist = abs(btc_px - strike)
        adverse = atr * math.sqrt(max(0.1, min_rem) / 15.0)
        margin_sigma = dist / adverse if adverse > 0 else 99.0
        if margin_sigma < 1.0:
            return "PAYOFF_GEOMETRY"

    return None


def _tier1_exit_override(row: dict, side: str, won_at_settlement: bool, params: BacktestParams) -> tuple[bool, str]:
    """
    Approximate the Tier 1 BTC-confirmation gate on REVERSE_CONVERGENCE /
    TIME_DECAY in replay: if the position is ITM at the logged feature row
    AND would have won at settlement AND tier1 is enabled, we hold to expiry.
    Returns (hold_to_expiry, reason_if_early_exit).
    """
    if not params.flags.get("tier1"):
        return True, "HOLD_TO_EXPIRY"  # always hold in replay, outcome is known
    # With tier1, definitely hold ITM winners to expiry.
    return True, "HOLD_TO_EXPIRY"


# ───────────────────────── engine ─────────────────────────

def _group_features_by_window(params: BacktestParams):
    """Yield (window_ts, [rows]) tuples in chronological order."""
    current_ws: Optional[int] = None
    bucket: list[dict] = []
    for rec in _iter_jsonl(FEATURES_PATH):
        ws = rec.get("window") or rec.get("window_start")
        if ws is None:
            continue
        ws = int(ws)
        ts = int(rec.get("ts", ws))
        if params.start_ts and ts < params.start_ts:
            continue
        if params.end_ts and ts > params.end_ts:
            continue
        if current_ws is None:
            current_ws = ws
        if ws != current_ws:
            if bucket:
                yield current_ws, bucket
            current_ws = ws
            bucket = []
        bucket.append(rec)
    if bucket and current_ws is not None:
        yield current_ws, bucket


def run_backtest(params: BacktestParams, progress_cb=None) -> dict:
    """
    Main entry point. Returns a JSON-serialisable result dict.
    progress_cb(fraction, message) optional.
    """
    outcomes = _load_outcomes()
    trades: list[BacktestTrade] = []
    skips: dict[str, int] = {}
    apply_calib = bool(params.flags.get("calib"))

    # Eagerly load calibration models if the flag is set, so the first lookup
    # doesn't try to read from disk during the hot loop.
    if apply_calib:
        try:
            from calibration import load_calibration_model
            load_calibration_model()
        except Exception:
            pass

    # Pre-count for progress reporting
    windows = list(_group_features_by_window(params))
    total = max(1, len(windows))
    if progress_cb:
        progress_cb(0.0, f"Replaying {total} windows")

    for i, (ws, rows) in enumerate(windows):
        if progress_cb and i % 50 == 0:
            progress_cb(i / total, f"Replaying window {i}/{total}")

        btc_close = outcomes.get(ws)
        if btc_close is None:
            skips["NO_OUTCOME"] = skips.get("NO_OUTCOME", 0) + 1
            continue

        row = _pick_entry_row(rows)
        if row is None:
            continue

        side = _decide_side(row, apply_calib=apply_calib)
        if side is None:
            skips["NO_DIRECTION"] = skips.get("NO_DIRECTION", 0) + 1
            continue

        edge = _row_edge(row, side, apply_calib=apply_calib)
        skip_reason = _gate_passes(row, params, side, edge)
        if skip_reason:
            skips[skip_reason] = skips.get(skip_reason, 0) + 1
            continue

        # Resolve outcome by side
        strike = float(row.get("strike", btc_close))
        if side == "YES":
            won = btc_close > strike
        else:
            won = btc_close < strike

        entry_px = _entry_price(row, side, params.fill_model, params.slippage_bps)
        exit_px = _exit_price(won, params.fill_model, params.slippage_bps)
        size = params.notional_usd / max(0.01, entry_px)
        gross_pnl_usd = size * (exit_px - entry_px)
        fees_usd = size * entry_px * params.fee_rate
        pnl_usd = gross_pnl_usd - fees_usd
        pnl_pct = pnl_usd / params.notional_usd

        theo_px = float(row.get("yes_mid" if side == "YES" else "no_mid", entry_px))
        slippage_pct = (entry_px - theo_px) / max(0.01, theo_px)

        # PnL decomposition (fractions of notional):
        post_up = float(row.get("posterior_final_up", row.get("posterior_fair_up", 0.5)))
        model_prob = post_up if side == "YES" else (1.0 - post_up)
        market_prob = theo_px
        # signal: realised outcome minus market expectation, scaled by model-edge sign
        exec_cost = -(slippage_pct + params.fee_rate)   # execution drag (negative)
        # prior: what the market's own probability earns vs entry
        prior_pnl = (int(won) - market_prob)
        # signal: how much of the winning result is credited to model edge over market
        signal_pnl = (int(won) - market_prob) * (model_prob - market_prob) / max(1e-3, abs(model_prob - market_prob) + 0.01)
        residual = pnl_pct - (signal_pnl + prior_pnl + exec_cost)

        trades.append(BacktestTrade(
            window_ts=ws, side=side, entry_px=entry_px, exit_px=exit_px, size=size,
            pnl_pct=pnl_pct, pnl_usd=pnl_usd, slippage_pct=slippage_pct,
            exit_reason="HOLD_TO_EXPIRY", outcome="WIN" if won else "LOSS",
            theoretical_px=theo_px,
            pnl_signal=signal_pnl, pnl_prior=prior_pnl,
            pnl_execution=exec_cost, pnl_residual=residual,
            min_rem_at_entry=float(row.get("min_rem", 0)),
            posterior=post_up, edge=edge,
        ))

    if progress_cb:
        progress_cb(0.95, "Computing metrics")

    metrics = _compute_metrics(trades, params)
    equity = _equity_curve(trades)
    dd_curve = _drawdown_curve(equity)
    exit_breakdown = _exit_breakdown(trades)
    decomp = _pnl_decomp(trades)

    if progress_cb:
        progress_cb(1.0, "Done")

    return {
        "params": asdict(params),
        "metrics": metrics,
        "equity_curve": equity,
        "drawdown_curve": dd_curve,
        "exit_breakdown": exit_breakdown,
        "pnl_decomposition": decomp,
        "skips": skips,
        "trades": [asdict(t) for t in trades],
        "completed_at": int(time.time()),
    }


# ───────────────────────── metrics ─────────────────────────

def _compute_metrics(trades: list[BacktestTrade], params: BacktestParams) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}
    pnl_usd = [t.pnl_usd for t in trades]
    pnl_pct = [t.pnl_pct for t in trades]
    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome != "WIN"]
    hit_rate = len(wins) / n
    total_usd = sum(pnl_usd)
    total_pct = sum(pnl_pct)
    # Sharpe / Sortino on per-trade returns (annualised is misleading for 15m binaries;
    # use per-trade t-stat scaled by sqrt(n))
    mean = statistics.fmean(pnl_pct)
    stdev = statistics.pstdev(pnl_pct) if n > 1 else 0.0
    sharpe = (mean / stdev) * math.sqrt(n) if stdev > 0 else 0.0
    neg = [p for p in pnl_pct if p < 0]
    dstd = statistics.pstdev(neg) if len(neg) > 1 else 0.0
    sortino = (mean / dstd) * math.sqrt(n) if dstd > 0 else 0.0
    equity = _equity_curve(trades)
    peak = 0.0
    max_dd = 0.0
    for _, eq in equity:
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
    calmar = (total_pct / n) / max(1e-6, max_dd) if max_dd > 0 else 0.0
    avg_win = statistics.fmean([t.pnl_pct for t in wins]) if wins else 0.0
    avg_loss = statistics.fmean([t.pnl_pct for t in losses]) if losses else 0.0
    pf_num = sum(t.pnl_usd for t in wins)
    pf_den = abs(sum(t.pnl_usd for t in losses)) or 1e-9
    profit_factor = pf_num / pf_den
    avg_slip = statistics.fmean([abs(t.slippage_pct) for t in trades])
    return {
        "n_trades": n,
        "hit_rate": hit_rate,
        "total_pnl_usd": total_usd,
        "total_pnl_pct": total_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "expectancy": mean,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_slippage_pct": avg_slip,
    }


def _equity_curve(trades: list[BacktestTrade]) -> list[list]:
    equity = 0.0
    curve = []
    for t in sorted(trades, key=lambda x: x.window_ts):
        equity += t.pnl_usd
        curve.append([int(t.window_ts), round(equity, 4)])
    return curve


def _drawdown_curve(equity: list[list]) -> list[list]:
    peak = 0.0
    out = []
    for ts, eq in equity:
        peak = max(peak, eq)
        dd = 0.0 if peak <= 0 else (eq - peak) / peak
        out.append([ts, round(dd, 5)])
    return out


def _exit_breakdown(trades: list[BacktestTrade]) -> list[dict]:
    agg: dict[str, dict] = {}
    for t in trades:
        b = agg.setdefault(t.exit_reason, {"count": 0, "total_pnl_usd": 0.0})
        b["count"] += 1
        b["total_pnl_usd"] += t.pnl_usd
    return [{"reason": k, **v} for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["total_pnl_usd"])]


def _pnl_decomp(trades: list[BacktestTrade]) -> dict:
    if not trades:
        return {"signal": 0.0, "prior": 0.0, "execution": 0.0, "residual": 0.0}
    return {
        "signal":    statistics.fmean([t.pnl_signal for t in trades]),
        "prior":     statistics.fmean([t.pnl_prior for t in trades]),
        "execution": statistics.fmean([t.pnl_execution for t in trades]),
        "residual":  statistics.fmean([t.pnl_residual for t in trades]),
    }


# ───────────────────────── persistence ─────────────────────────

def save_run(run_id: str, result: dict) -> str:
    path = os.path.join(BACKTEST_DIR, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(result, f)
    return path


def load_run(run_id: str) -> Optional[dict]:
    path = os.path.join(BACKTEST_DIR, f"{run_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_runs(limit: int = 30) -> list[dict]:
    if not os.path.isdir(BACKTEST_DIR):
        return []
    rows = []
    for fn in os.listdir(BACKTEST_DIR):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(BACKTEST_DIR, fn)
        try:
            with open(path) as f:
                d = json.load(f)
            m = d.get("metrics") or {}
            rows.append({
                "run_id": fn[:-5],
                "completed_at": d.get("completed_at") or int(os.path.getmtime(path)),
                "total_pnl_usd": m.get("total_pnl_usd", 0.0),
                "n_trades": m.get("n_trades", 0),
            })
        except Exception:
            continue
    rows.sort(key=lambda r: -r["completed_at"])
    return rows[:limit]


if __name__ == "__main__":
    # Quick CLI sanity check
    p = BacktestParams(fill_model="synthetic", source="logs")
    res = run_backtest(p, progress_cb=lambda f, m: print(f"{f*100:5.1f}% {m}"))
    print(json.dumps(res["metrics"], indent=2))
    print(f"{len(res['trades'])} trades, skips={res['skips']}")
