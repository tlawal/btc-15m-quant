"""
Exit evaluation logic and policies.
Extracted from logic.py during Phase 3 refactor.

Production-grade exit architecture with 6 layered mechanisms:
  1. Tiered take-profits (percentage-based, with late-entry override)
  2. Volatility-adapted stop-loss (ATR-normalized drawdown)
  3. Spread-aware exiting (Avellaneda-Stoikov maker-preference)
  4. Probability-convergence exits (bid >= posterior => take the money)
  5. Structural model reversal (posterior collapse from entry)
  6. Exponential time-decay (sensitivity multiplier in final 2 min)
"""

import math
import logging
from typing import Optional

from config import Config

log = logging.getLogger(__name__)


# ── Exit result helper ───────────────────────────────────────────────────────

def _exit(reason: str, *, partial_pct: float = 1.0, use_maker: bool = False) -> dict:
    """Build a structured exit result.

    Args:
        reason:      Human-readable exit tag (e.g. "HARD_STOP", "TP1").
        partial_pct: Fraction of position to sell (1.0 = full, 0.333 = one-third).
        use_maker:   If True, caller should place maker limit instead of crossing spread.
    """
    return {"reason": reason, "partial_pct": partial_pct, "use_maker": use_maker}


def _check_spread_aware(
    bid_price: Optional[float],
    ask_price: Optional[float],
    minutes_remaining: float,
) -> bool:
    """Determine whether the exit should use a maker limit (True) or cross the spread.

    Avellaneda-Stoikov principle: if spread is wide and there is time, prefer
    capturing spread via a maker order rather than paying it via a taker order.
    """
    if bid_price is None or ask_price is None or bid_price <= 0:
        return False
    spread_pct = (ask_price - bid_price) / bid_price
    seconds_remaining = minutes_remaining * 60.0
    if spread_pct > Config.SPREAD_AGGRESSIVE_THRESH and seconds_remaining > Config.SPREAD_EXPIRY_OVERRIDE_SEC:
        return True
    return False


def _time_decay_multiplier(minutes_remaining: float) -> float:
    """Exponential sensitivity multiplier for the final TIME_DECAY_WINDOW_MIN minutes.

    Returns 1.0 outside the window.  Inside, returns an increasing multiplier
    that makes adverse OFI / distance thresholds tighter (easier to trigger exits).

    At exactly 0 minutes remaining the multiplier equals TIME_DECAY_EXP_BASE.
    """
    if minutes_remaining >= Config.TIME_DECAY_WINDOW_MIN:
        return 1.0
    # fraction: 0.0 at window edge → 1.0 at expiry
    frac = (Config.TIME_DECAY_WINDOW_MIN - minutes_remaining) / max(Config.TIME_DECAY_WINDOW_MIN, 1e-9)
    frac = max(0.0, min(1.0, frac))
    return Config.TIME_DECAY_EXP_BASE ** frac


# ── Main evaluator ────────────────────────────────────────────────────────────

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
    # ── New params for production exits ──
    bid_price:        Optional[float] = None,
    ask_price:        Optional[float] = None,
    tp1_hit:          bool = False,
    tp2_hit:          bool = False,
    tp3_hit:          bool = False,
) -> Optional[dict]:
    """Evaluate whether to exit the current position.

    Returns a dict ``{"reason": str, "partial_pct": float, "use_maker": bool}``
    or ``None`` if the position should be held.
    """
    if current_price is None or entry_price <= 0:
        return None

    unrealized_pct = (current_price - entry_price) / entry_price

    # Safe fallbacks for posteriors
    _entry_post = entry_posterior if entry_posterior is not None else 0.5
    _peak_post  = peak_posterior if peak_posterior is not None else (_entry_post if _entry_post is not None else 0.5)

    # Pre-compute effective ATR ratio for volatility-adapted stops
    eff_atr = atr14 if (atr14 is not None and atr14 > 0) else Config.VOL_STOP_ATR_BASELINE
    atr_ratio = eff_atr / max(Config.VOL_STOP_ATR_BASELINE, 1e-6)

    # Pre-compute spread-awareness flag for non-emergency exits
    use_maker = _check_spread_aware(bid_price, ask_price, minutes_remaining)

    # Pre-compute time-decay multiplier for microstructure exits
    td_mult = _time_decay_multiplier(minutes_remaining)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1: HARD CIRCUIT BREAKERS — bypass ALL posterior gating
    # These fire unconditionally regardless of what the Bayesian model says.
    # ══════════════════════════════════════════════════════════════════════════

    # 1a. VOLATILITY-ADAPTED HARD STOP (Req #2)
    # Instead of a static -25%, scale the base drawdown by BTC ATR.
    # If ATR is 1.5x baseline, the allowable drawdown widens proportionally.
    # Example: base -15%, ATR ratio 1.5 → allowable = -22.5%.
    vol_stop_pct = min(
        Config.VOL_STOP_BASE_PCT * max(1.0, atr_ratio),
        Config.VOL_STOP_MAX_PCT,
    )
    if unrealized_pct < -vol_stop_pct:
        log.warning(
            "VOL_HARD_STOP: unrealized=%.1f%% breached -%.1f%% "
            "(atr=%.1f, ratio=%.2f, base=%.0f%%, widened=%.1f%%) — exiting unconditionally",
            unrealized_pct * 100, vol_stop_pct * 100,
            eff_atr, atr_ratio,
            Config.VOL_STOP_BASE_PCT * 100, vol_stop_pct * 100,
        )
        return _exit("VOL_HARD_STOP")  # emergency — ignore spread

    # 1b. Late-window hard stop: near expiry, any loss beyond small threshold.
    # Skip for late-entered positions to allow holding to expiry.
    if minutes_remaining < Config.FORCED_LATE_EXIT_MIN_REM and unrealized_pct < -Config.FORCED_LATE_LOSS_PCT:
        if entry_min_rem is not None and entry_min_rem < 5:
            log.info(
                "FORCED_LATE_EXIT_SKIPPED: late entry (entry_min_rem=%.1f) — holding to expiry",
                entry_min_rem,
            )
        else:
            return _exit("FORCED_LATE_EXIT")  # emergency — ignore spread

    # 1c. Outside preferred hours: aggressive profit-taking
    if not Config.is_preferred_trading_time() and unrealized_pct >= Config.OUTSIDE_HOURS_TAKE_PROFIT_PCT:
        return _exit("TAKE_SMALL_PROFIT_OUTSIDE", use_maker=use_maker)

    # 1d. Adverse microstructure reversal (Hawkes OFI)
    # Apply time-decay multiplier: lower threshold near expiry.
    if minutes_remaining <= 1.0 and unrealized_pct < 0:
        ofi_threshold = Config.EXIT_DEEP_OFI_REV_THRESH / td_mult  # tighter near expiry
        if (held_side == "YES" and deep_ofi < -ofi_threshold) or (held_side == "NO" and deep_ofi > ofi_threshold):
            log.warning(
                "FORCED_ADVERSE_OFI: deep_ofi=%.1f threshold=%.3f td_mult=%.2f "
                "(unrealized=%.1f%%) — adverse selection via OFI reversal",
                deep_ofi, ofi_threshold, td_mult, unrealized_pct * 100,
            )
            return _exit("FORCED_ADVERSE_OFI")  # emergency — ignore spread

    # 1e. Distance-based forced exit near expiry
    if minutes_remaining <= 1.0 and distance is not None and unrealized_pct < -Config.FORCED_LATE_LOSS_PCT:
        if abs(distance) <= 5.0:
            log.warning(
                "FORCED_DISTANCE_LATE: distance=%.1f <= 5, unrealized=%.1f%% — forcing exit near expiry",
                abs(distance), unrealized_pct * 100,
            )
            return _exit("FORCED_DISTANCE_LATE")
        else:
            log.info(
                "DISTANCE_HELD: distance=%.1f > 5, unrealized=%.1f%% — holding for expiry",
                abs(distance), unrealized_pct * 100,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 2: TIERED TAKE-PROFITS (Req #1)
    # ══════════════════════════════════════════════════════════════════════════

    # Late-entry override: entry >= $0.95 → single tight TP at +2%
    if entry_price >= Config.TP_LATE_ENTRY_THRESH:
        tp_target = Config.TP_LATE_ENTRY_PCT
        if unrealized_pct >= tp_target:
            log.info(
                "TP_LATE_ENTRY: entry=%.3f >= %.2f, unrealized=%.1f%% >= %.1f%% — exiting",
                entry_price, Config.TP_LATE_ENTRY_THRESH,
                unrealized_pct * 100, tp_target * 100,
            )
            return _exit("TP_LATE_ENTRY", use_maker=use_maker)
    else:
        # Normal tiered take-profits
        if Config.TP_PARTIAL_ENABLED:
            # Partial exits: 1/3 at each tier
            if not tp3_hit and unrealized_pct >= Config.TP3_PCT:
                log.info("TP3: unrealized=%.1f%% >= %.1f%% — selling remaining", unrealized_pct * 100, Config.TP3_PCT * 100)
                return _exit("TP3", partial_pct=1.0, use_maker=use_maker)
            if not tp2_hit and unrealized_pct >= Config.TP2_PCT:
                log.info("TP2: unrealized=%.1f%% >= %.1f%% — selling 1/3", unrealized_pct * 100, Config.TP2_PCT * 100)
                return _exit("TP2", partial_pct=0.333, use_maker=use_maker)
            if not tp1_hit and unrealized_pct >= Config.TP1_PCT:
                log.info("TP1: unrealized=%.1f%% >= %.1f%% — selling 1/3", unrealized_pct * 100, Config.TP1_PCT * 100)
                return _exit("TP1", partial_pct=0.333, use_maker=use_maker)
        else:
            # Full exit at TP1 threshold (fallback when partial disabled)
            if unrealized_pct >= Config.TP1_PCT:
                log.info(
                    "TP_FULL: unrealized=%.1f%% >= %.1f%% — full exit (partial disabled)",
                    unrealized_pct * 100, Config.TP1_PCT * 100,
                )
                return _exit("TP_FULL", use_maker=use_maker)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 3: PROBABILITY-CONVERGENCE EXIT (Req #4)
    # Market bid >= model posterior => market is paying our expected value early.
    # ══════════════════════════════════════════════════════════════════════════

    if Config.PROB_CONVERGENCE_ENABLED and posterior is not None and bid_price is not None:
        if bid_price >= posterior and unrealized_pct > 0:
            log.info(
                "PROB_CONVERGENCE: bid=%.3f >= posterior=%.3f, unrealized=%.1f%% — taking early EV",
                bid_price, posterior, unrealized_pct * 100,
            )
            return _exit("PROB_CONVERGENCE", use_maker=use_maker)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 4: STRUCTURAL MODEL REVERSAL (Req #5)
    # Posterior collapsed from entry — more reliable than price-based stops
    # on wide-spread 15m binaries.
    # ══════════════════════════════════════════════════════════════════════════

    if posterior is not None and _entry_post is not None:
        posterior_drop = _entry_post - posterior
        if posterior_drop >= Config.MODEL_REVERSAL_DROP_PCT:
            log.warning(
                "MODEL_REVERSAL: posterior=%.3f dropped %.1fpp from entry=%.3f (threshold=%.0fpp) — exiting",
                posterior, posterior_drop * 100, _entry_post, Config.MODEL_REVERSAL_DROP_PCT * 100,
            )
            return _exit("MODEL_REVERSAL", use_maker=use_maker)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 5: EXISTING POSTERIOR-GATED EXITS (with ATR-adapted drawdown)
    # ══════════════════════════════════════════════════════════════════════════

    # 5a. Forced drawdown — volatility-adapted, posterior-gated
    # Use ATR-scaled drawdown instead of the old static MAX_DRAWDOWN_PCT.
    adapted_drawdown = min(
        Config.MAX_DRAWDOWN_PCT * max(1.0, atr_ratio),
        Config.VOL_STOP_MAX_PCT,
    )
    if unrealized_pct < -adapted_drawdown:
        # 0.95+ conviction exemption near expiry with large distance
        if (posterior is not None and posterior > 0.95 and minutes_remaining < 2.0
                and distance is not None and abs(distance) > 50.0):
            log.info(
                "DRAWDOWN_EXEMPT: posterior=%.3f > 0.95, distance=%.1f > 50, rem=%.1f "
                "— ignoring drawdown exit near expiry",
                posterior, abs(distance), minutes_remaining,
            )
        else:
            if unrealized_pct < -0.20:
                return _exit("FORCED_DRAWDOWN", use_maker=use_maker)
            # Gate with posterior: only hold if model still convinced
            if posterior is None or posterior <= _entry_post - 0.05:
                return _exit("FORCED_DRAWDOWN", use_maker=use_maker)
            log.info(
                "DRAWDOWN_HELD: unrealized=%.1f%% but posterior=%.3f"
                " still near entry=%.3f — holding",
                unrealized_pct * 100, posterior, _entry_post,
            )

    # 5b. Explicit adverse selection / book flip.
    _book_flip_cycles = getattr(Config, "BOOK_FLIP_CONFIRM_CYCLES", None)
    _book_flip_imb = getattr(Config, "BOOK_FLIP_IMB_THRESH", None)
    if _book_flip_cycles is not None and _book_flip_imb is not None:
        if book_flip_count >= _book_flip_cycles and abs(obi) >= _book_flip_imb:
            if held_side == "YES" and obi < 0:
                return _exit("BOOK_FLIP", use_maker=use_maker)
            if held_side == "NO" and obi > 0:
                return _exit("BOOK_FLIP", use_maker=use_maker)

    # ── TRAILING POSTERIOR GUARD (suppresses soft exits below) ────────────────
    if posterior is not None and unrealized_pct >= 0:
        if posterior > 0.95:
            _tol = 0.10
        elif unrealized_pct > 0.05:
            _tol = 0.02
        elif unrealized_pct > 0:
            _tol = 0.03
        else:
            _tol = 0.05

        if posterior > _entry_post - _tol:
            return None   # model still believes — hold

    # 5c. Volatility-adjusted trailing (ATR-aware) using peak posterior.
    if (
        posterior is not None
        and hold_seconds >= Config.TRAIL_MIN_HOLD_SEC
        and unrealized_pct >= Config.TRAIL_ARM_MIN_PROFIT_PCT
        and _peak_post is not None
    ):
        trail_atr_scale = max(0.0, min(2.0, eff_atr / max(Config.TRAIL_ATR_REF, 1e-6)))
        allow_drop = Config.TRAIL_BASE_POST_DROP + Config.TRAIL_ATR_SCALE * (trail_atr_scale - 1.0)
        allow_drop = max(Config.TRAIL_MIN_POST_DROP, min(Config.TRAIL_MAX_POST_DROP, allow_drop))
        if posterior <= _peak_post - allow_drop:
            return _exit("TRAIL_POSTERIOR", use_maker=use_maker)

    # 5d. Forced profit lock (near expiry, strong profit)
    if minutes_remaining <= Config.FORCED_PROFIT_LOCK_MIN_REM:
        if unrealized_pct > Config.FORCED_PROFIT_PCT:
            return _exit("FORCED_PROFIT_LOCK", use_maker=use_maker)

    # 5e. Absolute price take-profit
    if current_price >= Config.TAKE_PROFIT_PRICE:
        return _exit("TAKE_PROFIT", use_maker=use_maker)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 6: TIME-DECAY-ENHANCED MICROSTRUCTURE EXITS (Req #6)
    # In the final 2 minutes, exponential multiplier makes adverse OFI/distance
    # thresholds tighter — less adverse flow triggers protective exits.
    # ══════════════════════════════════════════════════════════════════════════

    # Dynamic profit-taking based on signal strength, time, and microstructure
    abs_score = abs(signed_score)
    profit_threshold = None

    # Strong signals
    if (abs_score >= Config.TAKE_PROFIT_STRONG_SCORE and
        posterior is not None and posterior >= Config.TAKE_PROFIT_STRONG_POSTERIOR and
        minutes_remaining <= Config.TAKE_PROFIT_STRONG_MAX_MIN):
        profit_threshold = Config.TAKE_PROFIT_STRONG_PCT

    # Moderate signals
    elif (abs_score >= Config.TAKE_PROFIT_MODERATE_SCORE and
          posterior is not None and posterior >= Config.TAKE_PROFIT_MODERATE_POSTERIOR and
          minutes_remaining > Config.TAKE_PROFIT_STRONG_MAX_MIN and
          minutes_remaining <= Config.TAKE_PROFIT_MODERATE_MAX_MIN):
        profit_threshold = Config.TAKE_PROFIT_MODERATE_PCT

    # Weak signals
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
                "TAKE_PROFIT_DYNAMIC: unrealized=%.1f%% > %.1f%% "
                "(score=%.1f, posterior=%s, min_rem=%.1f, vpin=%.3f)",
                unrealized_pct * 100, profit_threshold * 100,
                abs_score,
                f"{posterior:.3f}" if posterior else "N/A",
                minutes_remaining, vpin,
            )
            return _exit("TAKE_PROFIT_DYNAMIC", use_maker=use_maker)

    # Minimum hold gate: conditions below require at least 60s in position.
    if hold_seconds < 60.0:
        return None

    # 6a. Alpha decay (score reversed significantly vs entry)
    score_delta = signed_score - entry_score
    if held_side == "YES" and score_delta < -Config.STOP_LOSS_DELTA:
        return _exit("ALPHA_DECAY", use_maker=use_maker)
    if held_side == "NO" and score_delta > Config.STOP_LOSS_DELTA:
        return _exit("ALPHA_DECAY", use_maker=use_maker)

    # 6b. Momentum reversal — CVD flipped against position
    if held_side == "YES" and cvd_delta < -0.5 and unrealized_pct < 0 and minutes_remaining < 8.0:
        return _exit("MOMENTUM_REVERSAL", use_maker=use_maker)
    if held_side == "NO" and cvd_delta > 0.5 and unrealized_pct < 0 and minutes_remaining < 8.0:
        return _exit("MOMENTUM_REVERSAL", use_maker=use_maker)

    # 6c. Microstructure confirmation exit with time-decay-scaled thresholds (Req #6)
    microstructure_trigger = False
    if unrealized_pct < 0.01 and minutes_remaining < 10.0:
        microstructure_trigger = True
    elif minutes_remaining <= 1.0 and posterior is not None and posterior > 0.7 and unrealized_pct < 0:
        microstructure_trigger = True

    if microstructure_trigger:
        # Apply time-decay multiplier: divide thresholds by td_mult to make them tighter
        _ofi_thresh = Config.EXIT_DEEP_OFI_REV_THRESH / td_mult
        _cvd_thresh = Config.EXIT_CVD_VEL_REV_THRESH / td_mult

        ofi_rev = abs(deep_ofi) > 0 and (
            (held_side == "YES" and deep_ofi < -_ofi_thresh) or
            (held_side == "NO" and deep_ofi > _ofi_thresh)
        )
        cvd_vel_rev = abs(cvd_velocity) > 0 and (
            (held_side == "YES" and cvd_velocity < -_cvd_thresh) or
            (held_side == "NO" and cvd_velocity > _cvd_thresh)
        )
        if ofi_rev or cvd_vel_rev:
            return _exit("MICRO_REVERSAL", use_maker=use_maker)

    # 6d. Probability decay — posterior declining while losing AND cvd reverses
    if posterior is not None and prev_posterior is not None:
        post_decline = prev_posterior - posterior
        cvd_reversal = (held_side == "YES" and cvd_delta < -0.5) or (held_side == "NO" and cvd_delta > 0.5)
        if post_decline > 0.08 and cvd_reversal:
            return _exit("PROBABILITY_DECAY", use_maker=use_maker)

    # 6e. Time-decay exit — only exit LOSING positions very near expiry.
    if minutes_remaining < 2.0 and unrealized_pct < -0.02:
        # Final posterior check: if model is still >60% confident, hold
        if posterior is not None and posterior > 0.60:
            log.info(
                "TIME_DECAY_HELD: %.1fmin rem, unrealized=%.1f%%, posterior=%.3f — holding for settlement",
                minutes_remaining, unrealized_pct * 100, posterior,
            )
            return None
        return _exit("TIME_DECAY", use_maker=use_maker)

    return None
