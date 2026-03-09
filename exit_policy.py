"""
Exit evaluation logic and policies.
Extracted from logic.py during Phase 3 refactor.
"""

import logging
from typing import Optional
from config import Config

log = logging.getLogger(__name__)

def evaluate_exit(
    *,
    held_side:        str,
    entry_price:      float,
    current_price:    Optional[float],
    minutes_remaining: float,
    signed_score:     float,
    entry_score:      float,
    distance:         Optional[float],
    cvd_delta:        float = 0.0,
    posterior:        Optional[float] = None,
    prev_posterior:   Optional[float] = None,
    entry_posterior:  Optional[float] = None,
    hold_seconds:     float = 999.0,
) -> Optional[str]:
    """
    Returns exit reason string or None.
    Handles momentum reversal, probability decay, and time-decay exits.
    """
    if current_price is None or entry_price <= 0:
        return None

    unrealized_pct = (current_price - entry_price) / entry_price

    # Use a safe entry_posterior fallback: if not recorded, treat as 0.5 (neutral).
    # This prevents the trailing guard from being silently skipped on reloaded positions.
    _entry_post = entry_posterior if entry_posterior is not None else 0.5

    # ── HARD CIRCUIT BREAKERS (bypass ALL posterior gating) ───────────────────
    # These fire unconditionally regardless of what the Bayesian model says.
    # Lesson: the trailing posterior guard caused a -65% loss by holding while
    # the model lagged BTC's real price action. Hard stops must always take priority.

    # 0a. Absolute max loss per trade — unconditional at -25%.
    # Binary positions CAN recover from -15% but rarely from -25%+. Cut ruthlessly.
    if unrealized_pct < -Config.HARD_STOP_PCT:
        log.warning(
            f"HARD_STOP: unrealized={unrealized_pct*100:.1f}% breached -{Config.HARD_STOP_PCT*100:.0f}% "
            f"— exiting unconditionally (posterior={posterior:.3f if posterior is not None else 'N/A'} ignored)"
        )
        return "HARD_STOP"

    # 0b. Late-window hard stop: near expiry, any loss beyond small threshold.
    # At <5 min remaining, the binary is repricing fast — the model is stale.
    # Tighter threshold for late losers: exit at -10% if < 5 min remain.
    if minutes_remaining < Config.FORCED_LATE_EXIT_MIN_REM and unrealized_pct < -Config.FORCED_LATE_LOSS_PCT:
        return "FORCED_LATE_EXIT"

    # 1. Forced drawdown — posterior-gated up to -20%, unconditional beyond.
    # The posterior gate prevents cutting on normal BTC noise when model still believes.
    # But if posterior has also dropped, it's a genuine adverse move — cut it.
    # NOTE: HARD_STOP_PCT (-25%) above already catches anything truly catastrophic.
    if unrealized_pct < -Config.MAX_DRAWDOWN_PCT:
        if unrealized_pct < -0.20:
            return "FORCED_DRAWDOWN"   # unconditional beyond 20% (belt+suspenders)
        # Gate with posterior: only hold if model is still convinced (< 5pp drop from entry)
        if posterior is None or posterior <= _entry_post - 0.05:
            return "FORCED_DRAWDOWN"
        log.info(
            f"DRAWDOWN_HELD: unrealized={unrealized_pct*100:.1f}% but posterior={posterior:.3f}"
            f" still near entry={_entry_post:.3f} — holding"
        )

    # ── TRAILING POSTERIOR GUARD (suppresses soft exits 2-8 only) ─────────────
    # Holds the position while the Bayesian model still believes in the trade.
    # Only runs after hard breakers above — it CANNOT override HARD_STOP or FORCED_LATE_EXIT.
    # Tolerance scales with profitability — hold winners tighter, give modest losers room.
    # When losing >10%, tighten tolerance: the model is increasingly suspect vs market price.
    if posterior is not None:
        if unrealized_pct > 0.05:
            _tol = 0.02   # winning >5%: tight hold
        elif unrealized_pct > 0:
            _tol = 0.03   # small win
        elif unrealized_pct > -0.10:
            _tol = 0.05   # modest loss: give room for oscillation
        else:
            _tol = 0.03   # losing >10%: tighten — model is lagging reality
        if posterior > _entry_post - _tol:
            return None   # model still believes — hold for soft exit conditions

    # 2. Forced distance exit (near expiry, out-of-range, losing)
    if (minutes_remaining < Config.FORCED_DISTANCE_EXIT_MIN_REM
            and distance is not None
            and abs(distance) < Config.FORCED_DISTANCE_MAX
            and unrealized_pct < 0):
        return "FORCED_DISTANCE"

    # 3. Forced profit lock (near expiry, strong profit)
    if (minutes_remaining < Config.FORCED_PROFIT_LOCK_MIN_REM
            and unrealized_pct > Config.FORCED_PROFIT_PCT):
        return "FORCED_PROFIT_LOCK"

    # 4. Take profit
    if current_price >= Config.TAKE_PROFIT_PRICE:
        return "TAKE_PROFIT"

    # Minimum hold gate: conditions 5-8 require at least 60s in position.
    # Avoids whipsaw exits on EMA lag and first-candle noise immediately post-fill.
    if hold_seconds < 60.0:
        return None

    # 5. Alpha decay (score reversed significantly vs entry)
    score_delta = signed_score - entry_score
    if held_side == "YES" and score_delta < -Config.STOP_LOSS_DELTA:
        return "ALPHA_DECAY"
    if held_side == "NO" and score_delta > Config.STOP_LOSS_DELTA:
        return "ALPHA_DECAY"

    # 6. Momentum reversal — CVD flipped against position
    if held_side == "YES" and cvd_delta < -0.5 and unrealized_pct < 0 and minutes_remaining < 8.0:
        return "MOMENTUM_REVERSAL"
    if held_side == "NO" and cvd_delta > 0.5 and unrealized_pct < 0 and minutes_remaining < 8.0:
        return "MOMENTUM_REVERSAL"

    # 7. Probability decay — posterior declining while losing AND cvd reverses
    if posterior is not None and prev_posterior is not None:
        post_decline = prev_posterior - posterior
        cvd_reversal = (held_side == "YES" and cvd_delta < -0.5) or (held_side == "NO" and cvd_delta > 0.5)
        if post_decline > 0.08 and cvd_reversal:
            return "PROBABILITY_DECAY"

    # 8. Time-decay exit — only exit LOSING positions very near expiry.
    # Tightened to <2min: at 2-3min remaining a losing binary can still recover;
    # exiting at 0.78 forfeits the chance to settle at $1.00 if the posterior is high.
    if minutes_remaining < 2.0 and unrealized_pct < -0.02:
        # Final posterior check: if model is still >60% confident, hold to settlement
        if posterior is not None and posterior > 0.60:
            log.info(
                f"TIME_DECAY_HELD: {minutes_remaining:.1f}min rem, "
                f"unrealized={unrealized_pct*100:.1f}%, posterior={posterior:.3f} — holding for settlement"
            )
            return None
        return "TIME_DECAY"

    return None
