"""
Phase 4: Strategy Optimizer

- Signal decay detection (per-feature rolling accuracy)
- Adaptive threshold engine (auto-tune gate thresholds from recent performance)
- Sharpe-optimized Kelly calibration (use live Sharpe to scale Kelly fraction)
"""

import json
import math
import logging
import time
from collections import defaultdict
from config import Config

log = logging.getLogger(__name__)


class StrategyOptimizer:
    def __init__(self, state):
        self.state = state
        self.score_offset = 0.0
        self.edge_offset = 0.0

        # Per-signal accuracy tracking (signal_name -> list of (prediction_correct, ts))
        self._signal_accuracy: dict[str, list[tuple[bool, int]]] = defaultdict(list)
        # Disabled signals (decayed below threshold)
        self._disabled_signals: set[str] = set()
        # Sharpe-calibrated Kelly multiplier
        self._kelly_multiplier = 1.0
        # Rolling PnL for Sharpe calculation
        self._pnl_history: list[float] = []
        # Last optimization timestamp
        self._last_optimize_ts = 0

    def detect_decay(self, window_size: int = 10):
        """
        Detect signal decay by looking at the last N trades.
        If win rate is below 40%, increase selectivity.
        Also runs periodic optimization every 30 minutes.
        """
        now = int(time.time())
        if now - self._last_optimize_ts > 1800:  # every 30 min
            self._run_periodic_optimization()
            self._last_optimize_ts = now

        closed_trades = [t for t in self.state.trade_history if t.outcome in ("WIN", "LOSS")]
        if len(closed_trades) < window_size:
            return self.score_offset, self.edge_offset

        recent = closed_trades[-window_size:]
        wins = sum(1 for t in recent if t.outcome == "WIN")
        win_rate = wins / window_size

        if win_rate < 0.40:
            log.warning(f"Signal decay detected: win_rate={win_rate*100:.1f}% over last {window_size} trades.")
            self.score_offset = min(self.score_offset + 1.0, 3.0)   # cap at +3
            self.edge_offset = min(self.edge_offset + 0.010, 0.03)  # cap at +3%
        elif win_rate > 0.60:
            self.score_offset = max(0.0, self.score_offset - 0.5)
            self.edge_offset = max(0.0, self.edge_offset - 0.005)

        return self.score_offset, self.edge_offset

    # ── #22: Per-Signal Decay Detection ──────────────────────────────────────

    def record_signal_outcome(self, features: dict, outcome_win: bool, ts: int = None):
        """Record whether each signal's direction agreed with the outcome."""
        ts = ts or int(time.time())
        directional_signals = [
            "ema_score", "vwap_score", "rsi_score", "macd_score", "stoch_score",
            "mfi_score", "obv_score", "cvd_score", "ofi_score", "imbalance_score",
            "flow_accel_score", "tob_score", "cvd_velocity_score", "pm_flow_score",
            "liq_vacuum_score", "bb_position_score", "oracle_lag_score",
            "funding_rate_score", "spread_pressure_score", "accum_ofi_score",
            "cross_exch_score", "mtf_momentum_score", "adx_stoch_boost",
            "misprice_score",
        ]
        for sig_name in directional_signals:
            val = features.get(sig_name, 0.0)
            if val == 0.0:
                continue  # signal was neutral, skip
            # Signal predicted UP if positive, DOWN if negative
            # outcome_win True means the model-chosen direction was correct
            predicted_correct = (val > 0 and outcome_win) or (val < 0 and not outcome_win)
            self._signal_accuracy[sig_name].append((predicted_correct, ts))

    def get_signal_accuracies(self, lookback_sec: int = 7 * 86400) -> dict[str, dict]:
        """Return per-signal accuracy over the lookback window."""
        cutoff = int(time.time()) - lookback_sec
        result = {}
        for sig_name, history in self._signal_accuracy.items():
            recent = [(correct, ts) for correct, ts in history if ts >= cutoff]
            if len(recent) < 5:
                continue
            correct_count = sum(1 for c, _ in recent if c)
            accuracy = correct_count / len(recent)
            result[sig_name] = {
                "accuracy": round(accuracy, 3),
                "n_samples": len(recent),
                "decayed": accuracy < 0.45,
            }
        return result

    def get_disabled_signals(self) -> set[str]:
        """Return signals that have decayed below threshold."""
        return self._disabled_signals

    # ── #23: Adaptive Threshold Engine ───────────────────────────────────────

    def _run_periodic_optimization(self):
        """Auto-tune parameters based on recent performance."""
        # Per-signal decay check
        accuracies = self.get_signal_accuracies()
        newly_disabled = set()
        newly_enabled = set()

        for sig_name, stats in accuracies.items():
            if stats["decayed"] and stats["n_samples"] >= 10:
                newly_disabled.add(sig_name)
                if sig_name not in self._disabled_signals:
                    log.warning(f"SIGNAL_DECAY: {sig_name} accuracy={stats['accuracy']:.1%} "
                                f"(n={stats['n_samples']}) — disabling")
            elif not stats["decayed"] and sig_name in self._disabled_signals:
                newly_enabled.add(sig_name)
                log.info(f"SIGNAL_RECOVERY: {sig_name} accuracy={stats['accuracy']:.1%} — re-enabling")

        self._disabled_signals = (self._disabled_signals | newly_disabled) - newly_enabled

        # Sharpe-optimized Kelly recalibration
        self._recalibrate_kelly()

    # ── #24: Sharpe-Optimized Kelly Calibration ──────────────────────────────

    def record_trade_pnl(self, pnl_pct: float):
        """Record a trade's PnL percentage for Sharpe calculation."""
        self._pnl_history.append(pnl_pct)
        if len(self._pnl_history) > 200:
            self._pnl_history = self._pnl_history[-200:]

    def _recalibrate_kelly(self):
        """Adjust Kelly fraction based on realized Sharpe ratio."""
        if len(self._pnl_history) < 10:
            self._kelly_multiplier = 1.0
            return

        pnls = self._pnl_history[-50:]  # last 50 trades
        mean_pnl = sum(pnls) / len(pnls)
        var = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(var) if var > 0 else 1e-6

        # Annualized Sharpe (assuming ~96 trades/day for 15min windows)
        sharpe = (mean_pnl / std) * math.sqrt(96) if std > 0 else 0.0

        # Kelly multiplier: scale down if Sharpe is poor, scale up if excellent
        if sharpe < 0:
            self._kelly_multiplier = 0.25  # quarter size on negative Sharpe
        elif sharpe < 0.5:
            self._kelly_multiplier = 0.50
        elif sharpe < 1.0:
            self._kelly_multiplier = 0.75
        elif sharpe < 2.0:
            self._kelly_multiplier = 1.0   # normal quarter-Kelly
        else:
            self._kelly_multiplier = 1.25  # slightly larger on strong Sharpe

        log.info(f"Kelly recalibrated: sharpe={sharpe:.2f} multiplier={self._kelly_multiplier:.2f}")

    def get_kelly_multiplier(self) -> float:
        """Return the Sharpe-calibrated Kelly multiplier."""
        return self._kelly_multiplier

    def get_adjusted_thresholds(self, base_score: float, base_edge: float):
        """Return thresholds adjusted by decay detection."""
        return base_score + self.score_offset, base_edge + self.edge_offset
