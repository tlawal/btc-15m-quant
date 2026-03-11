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
    cvd_velocity:     float = 0.0,
    deep_ofi:         float = 0.0,
    obi:              float = 0.0,
    atr14:            Optional[float] = None,
    vpin:             float = 0.0,
    posterior:        Optional[float] = None,
    prev_posterior:   Optional[float] = None,
    entry_posterior:  Optional[float] = None,
    peak_posterior:   Optional[float] = None,
    book_flip_count:  int = 0,
    hold_seconds:     float = 999.0,
    entry_min_rem:    Optional[float] = None,
    yes_mid:          float = 0.5,
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
    _peak_post  = peak_posterior if peak_posterior is not None else (_entry_post if _entry_post is not None else 0.5)

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
    # At <1 min remaining, the binary is repricing fast — the model is stale.
    # Tighter threshold for late losers: exit at -10% if < 1 min remain.
    # Skip for late-entered positions to allow holding to expiry.
    # Hold if posterior >0.70 (model still believes).
    if minutes_remaining < Config.FORCED_LATE_EXIT_MIN_REM and unrealized_pct < -Config.FORCED_LATE_LOSS_PCT:
        if entry_min_rem is not None and entry_min_rem < 5:
            log.info(f"FORCED_LATE_EXIT_SKIPPED: late entry (entry_min_rem={entry_min_rem:.1f}) — holding to expiry")
            pass  # skip forced exit for late entries
        elif posterior is not None and posterior > 0.70:
            log.info(f"FORCED_LATE_EXIT_HELD: posterior={posterior:.3f} > 0.70 — holding to expiry")
            pass  # hold if model confident
        else:
            return "FORCED_LATE_EXIT"

    # 0c. Outside preferred hours: aggressive profit-taking at 5%
    if not Config.is_preferred_trading_time() and unrealized_pct >= 0.05:
        return "TAKE_SMALL_PROFIT_OUTSIDE"

    # 0e. Adverse microstructure reversal (Hawkes OFI for adverse selection)
    # Force exit if OFI indicates strong adverse flow despite model confidence
    if minutes_remaining <= 1.0 and posterior is not None and posterior > 0.7 and unrealized_pct < 0:
        ofi_threshold = Config.EXIT_DEEP_OFI_REV_THRESH
        if (held_side == "YES" and deep_ofi < -ofi_threshold) or (held_side == "NO" and deep_ofi > ofi_threshold):
            log.warning(
                f"FORCED_ADVERSE_OFI: deep_ofi={deep_ofi:.1f} threshold={ofi_threshold} "
                f"(unrealized={unrealized_pct*100:.1f}%) — adverse selection via OFI reversal"
            )
            return "FORCED_ADVERSE_OFI"

    # 0f. Distance-based forced exit near expiry (replaces posterior-based Forced Late Exit)
    # If price distance from strike is >5 under 1 minute, hold for potential recovery.
    # Otherwise, force exit losing positions to avoid catastrophic losses.
    if minutes_remaining <= 1.0 and distance is not None and unrealized_pct < -Config.FORCED_LATE_LOSS_PCT:
        if abs(distance) <= 5.0:
            log.warning(
                f"FORCED_DISTANCE_LATE: distance={abs(distance):.1f} <= 5, unrealized={unrealized_pct*100:.1f}% — forcing exit near expiry"
            )
            return "FORCED_DISTANCE_LATE"
        else:
            log.info(
                f"DISTANCE_HELD: distance={abs(distance):.1f} > 5, unrealized={unrealized_pct*100:.1f}% — holding for expiry"
            )

    # 1. Forced drawdown — posterior-gated up to -20%, unconditional beyond.
    # The posterior gate prevents cutting on normal BTC noise when model still believes.
    # But if posterior has also dropped, it's a genuine adverse move — cut it.
    # NOTE: HARD_STOP_PCT (-25%) above already catches anything truly catastrophic.
    if unrealized_pct < -Config.MAX_DRAWDOWN_PCT:
        # 0.95+ conviction exemption: if model is near-certain and market is far from strike near expiry,
        # ignore order book mid-price dips (which can trigger false drawdown at low liquidity).
        if posterior is not None and posterior > 0.95 and minutes_remaining < 2.0 and distance is not None and abs(distance) > 50.0:
            log.info(
                f"DRAWDOWN_EXEMPT: posterior={posterior:.3f} > 0.95, distance={abs(distance):.1f} > 50, rem={minutes_remaining:.1f} "
                f"— ignoring drawdown exit near expiry"
            )
            pass
        else:
            if unrealized_pct < -0.20:
                return "FORCED_DRAWDOWN"   # unconditional beyond 20% (belt+suspenders)
            # Gate with posterior: only hold if model is still convinced (< 5pp drop from entry)
            if posterior is None or posterior <= _entry_post - 0.05:
                return "FORCED_DRAWDOWN"
            log.info(
                f"DRAWDOWN_HELD: unrealized={unrealized_pct*100:.1f}% but posterior={posterior:.3f}"
                f" still near entry={_entry_post:.3f} — holding"
            )

    # 1b. Explicit adverse selection / book flip.
    # If the book flips meaningfully against our side for multiple cycles, exit.
    if book_flip_count >= Config.BOOK_FLIP_CONFIRM_CYCLES and abs(obi) >= Config.BOOK_FLIP_IMB_THRESH:
        if held_side == "YES" and obi < 0:
            return "BOOK_FLIP"
        if held_side == "NO" and obi > 0:
            return "BOOK_FLIP"

    # ── TRAILING POSTERIOR GUARD (suppresses soft exits 2-8 only) ─────────────
    # Holds the position while the Bayesian model still believes in the trade.
    # Only runs after hard breakers above — it CANNOT override HARD_STOP or FORCED_LATE_EXIT.
    # Tolerance scales with profitability — hold winners tighter, give modest losers room.
    # When losing >10%, tighten tolerance: the model is increasingly suspect vs market price.
    # High confidence override: if posterior > 0.95, hold unless it drops significantly (>10pp).
    if posterior is not None:
        if posterior > 0.95:
             # Near-certain: very loose tolerance (0.10) to avoid being shaken out by noise.
             _tol = 0.10
        elif unrealized_pct > 0.05:
            _tol = 0.02   # winning >5%: tight hold
        elif unrealized_pct > 0:
            _tol = 0.03   # small win
        elif unrealized_pct > -0.10:
            _tol = 0.03   # modest loss: give room for oscillation (tightened from 0.05)
        else:
            _tol = 0.02   # losing >10%: tighten — model is lagging reality (tightened from 0.03)

        if posterior > _entry_post - _tol:
            return None   # model still believes — hold for soft exit conditions

    # 1c. Volatility-adjusted trailing (ATR-aware) using peak posterior.
    # Only arm once in profit and after minimal holding time to avoid whipsaw.
    if (
        posterior is not None
        and hold_seconds >= Config.TRAIL_MIN_HOLD_SEC
        and unrealized_pct >= Config.TRAIL_ARM_MIN_PROFIT_PCT
        and _peak_post is not None
    ):
        eff_atr = atr14 if (atr14 is not None and atr14 > 0) else Config.TRAIL_ATR_REF
        atr_scale = max(0.0, min(2.0, eff_atr / max(Config.TRAIL_ATR_REF, 1e-6)))
        allow_drop = Config.TRAIL_BASE_POST_DROP + Config.TRAIL_ATR_SCALE * (atr_scale - 1.0)
        allow_drop = max(Config.TRAIL_MIN_POST_DROP, min(Config.TRAIL_MAX_POST_DROP, allow_drop))
        if posterior <= _peak_post - allow_drop:
            return "TRAIL_POSTERIOR"

    # 2. Forced profit lock (near expiry, strong profit)
    if (minutes_remaining < Config.FORCED_PROFIT_LOCK_MIN_REM
            and unrealized_pct > Config.FORCED_PROFIT_PCT):
        return "FORCED_PROFIT_LOCK"

    # 4. Take profit
    if current_price >= Config.TAKE_PROFIT_PRICE:
        return "TAKE_PROFIT"

    # 4a. Take profit at 10% from entry
    if unrealized_pct >= 0.10:
        log.info(f"TAKE_PROFIT_10PCT: unrealized={unrealized_pct*100:.1f}% >= 10% from entry")
        return "TAKE_PROFIT_10PCT"

    # 4b. Take small profit: lock in 3% gain quickly
    if unrealized_pct > 0.03:
        log.info(f"TAKE_SMALL_PROFIT: unrealized={unrealized_pct*100:.1f}% > 3%")
        return "TAKE_SMALL_PROFIT"

    # 4b. Dynamic profit-taking based on signal strength, time, and microstructure
    # Posterior-gated (trailing guard above suppresses if model still believes)
    abs_score = abs(signed_score)
    entry_abs_score = abs(entry_score)
    profit_threshold = None

    # Strong signals: 25% in early window
    if (abs_score >= Config.TAKE_PROFIT_STRONG_SCORE and
        posterior is not None and posterior >= Config.TAKE_PROFIT_STRONG_POSTERIOR and
        minutes_remaining <= Config.TAKE_PROFIT_STRONG_MAX_MIN):
        profit_threshold = Config.TAKE_PROFIT_STRONG_PCT

    # Moderate signals: 10% in mid-window
    elif (abs_score >= Config.TAKE_PROFIT_MODERATE_SCORE and
          posterior is not None and posterior >= Config.TAKE_PROFIT_MODERATE_POSTERIOR and
          minutes_remaining > Config.TAKE_PROFIT_STRONG_MAX_MIN and
          minutes_remaining <= Config.TAKE_PROFIT_MODERATE_MAX_MIN):
        profit_threshold = Config.TAKE_PROFIT_MODERATE_PCT

    # Weak signals: 5% in late window or toxic microstructure
    elif (abs_score >= Config.TAKE_PROFIT_WEAK_SCORE and
          posterior is not None and posterior >= Config.TAKE_PROFIT_WEAK_POSTERIOR and
          (minutes_remaining > Config.TAKE_PROFIT_MODERATE_MAX_MIN or
           (vpin >= Config.VPIN_TOXIC_THRESHOLD and abs(deep_ofi) >= Config.EXIT_DEEP_OFI_REV_THRESH))):
        profit_threshold = Config.TAKE_PROFIT_WEAK_PCT

    if profit_threshold is not None:
        # Microstructure scaling: reduce threshold in toxic flow
        if vpin >= Config.VPIN_TOXIC_THRESHOLD:
            profit_threshold *= Config.TAKE_PROFIT_TOXIC_MULTIPLIER

        # Time scaling: tighten near expiry
        if minutes_remaining < Config.FORCED_PROFIT_LOCK_MIN_REM:
            profit_threshold *= Config.TAKE_PROFIT_LATE_MULTIPLIER

        if unrealized_pct > profit_threshold:
            log.info(
                f"TAKE_PROFIT_DYNAMIC: unrealized={unrealized_pct*100:.1f}% > {profit_threshold*100:.1f}% "
                f"(score={abs_score:.1f}, posterior={posterior:.3f if posterior else 'N/A'}, "
                f"min_rem={minutes_remaining:.1f}, vpin={vpin:.3f})"
            )
            return "TAKE_PROFIT_DYNAMIC"

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

    # 6b. Microstructure confirmation exit: reverse CVD velocity and deep OFI.
    # Trigger only when position is not clearly winning — this is primarily adverse-selection defense.
    # Consolidated with adverse OFI: includes near-expiry losing positions with high posterior.
    microstructure_trigger = False
    if unrealized_pct < 0.01 and minutes_remaining < 10.0:
        microstructure_trigger = True
    elif minutes_remaining <= 1.0 and posterior is not None and posterior > 0.7 and unrealized_pct < 0:
        microstructure_trigger = True

    if microstructure_trigger:
        ofi_rev = abs(deep_ofi) > 0 and ((held_side == "YES" and deep_ofi < -Config.EXIT_DEEP_OFI_REV_THRESH) or (held_side == "NO" and deep_ofi > Config.EXIT_DEEP_OFI_REV_THRESH))
        cvd_vel_rev = abs(cvd_velocity) > 0 and ((held_side == "YES" and cvd_velocity < -Config.EXIT_CVD_VEL_REV_THRESH) or (held_side == "NO" and cvd_velocity > Config.EXIT_CVD_VEL_REV_THRESH))
        if ofi_rev or cvd_vel_rev:  # Consolidated: trigger on either OFI or CVD reversal
            return "MICRO_REVERSAL"

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
