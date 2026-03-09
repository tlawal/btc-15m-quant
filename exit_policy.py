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
    hold_seconds:     float = 999.0,
) -> Optional[str]:
    """
    Returns exit reason string or None.
    Handles momentum reversal, probability decay, and time-decay exits.
    """
    if current_price is None or entry_price <= 0:
        return None

    unrealized_pct = (current_price - entry_price) / entry_price

    # 1. Forced drawdown
    if unrealized_pct < -Config.MAX_DRAWDOWN_PCT:
        return "FORCED_DRAWDOWN"

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

    # 4. Forced late exit (5 min rem, losing)
    if (minutes_remaining < Config.FORCED_LATE_EXIT_MIN_REM
            and unrealized_pct < -Config.FORCED_LATE_LOSS_PCT):
        return "FORCED_LATE_EXIT"

    # 5. Take profit
    if current_price >= Config.TAKE_PROFIT_PRICE:
        return "TAKE_PROFIT"

    # 6. Alpha decay (score reversed significantly vs entry)
    # Require minimum 30s hold to avoid firing on EMA lag immediately post-fill
    if hold_seconds >= 30.0:
        score_delta = signed_score - entry_score
        if held_side == "YES" and score_delta < -Config.STOP_LOSS_DELTA:
            return "ALPHA_DECAY"
        if held_side == "NO" and score_delta > Config.STOP_LOSS_DELTA:
            return "ALPHA_DECAY"

    # 7. Momentum reversal — CVD flipped against position
    if held_side == "YES" and cvd_delta < -0.5 and unrealized_pct < 0 and minutes_remaining < 8.0:
        return "MOMENTUM_REVERSAL"
    if held_side == "NO" and cvd_delta > 0.5 and unrealized_pct < 0 and minutes_remaining < 8.0:
        return "MOMENTUM_REVERSAL"

    # 8. Probability decay — posterior declining while losing
    if posterior is not None and prev_posterior is not None:
        post_decline = prev_posterior - posterior
        if post_decline > 0.08 and unrealized_pct < -0.03:
            return "PROBABILITY_DECAY"

    # 9. Time-decay exit
    if minutes_remaining < 2.0 and abs(unrealized_pct) < 0.05:
        return "TIME_DECAY"

    return None
