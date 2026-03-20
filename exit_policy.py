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
import time
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
    entry_edge:       Optional[float] = None,
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
    # ── MAE/MFE tracking ──
    mae_pct:          float = 0.0,
    mfe_pct:          float = 0.0,
    deep_drawdown_ts: Optional[int] = None,
    # ── Fix #7: opposing side prices for reverse convergence ──
    no_bid:           Optional[float] = None,
    yes_bid:          Optional[float] = None,
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
    # LAYER 0: ABSOLUTE HARD STOP — unconditional circuit breaker, no gates
    # At -25% the position is unrecoverable on a 15m binary; cut always.
    # Exception: for late-window entries (< 3 min remaining at entry) within the first 60s of holding,
    # binary price swings of ±30% are CLOB noise, not outcome-probability change.
    # Kelly (1956) + Thorp (2006): at < 3 min, variance is maximum — don't cut noise as signal.
    # ══════════════════════════════════════════════════════════════════════════
    if unrealized_pct < -Config.HARD_STOP_PCT:
        _suppress_hard_stop = (
            entry_min_rem is not None
            and entry_min_rem < 3.0
            and hold_seconds < 60
            and posterior >= 0.65
        )
        if _suppress_hard_stop:
            log.info(
                "HARD_STOP_LATE_SUPPRESSED: unrealized=%.1f%% but entry_min_rem=%.1f<3.0 hold=%ds<60s posterior=%.3f≥0.65 — noise, not signal",
                unrealized_pct * 100, entry_min_rem, hold_seconds, posterior,
            )
        else:
            log.warning(
                "HARD_STOP: unrealized=%.1f%% breached -%.1f%% — unconditional exit",
                unrealized_pct * 100, Config.HARD_STOP_PCT * 100,
            )
            return _exit("HARD_STOP")

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
    _min_hold = getattr(Config, "MIN_HOLD_BEFORE_DRAWDOWN_SEC", 60)
    if unrealized_pct < -vol_stop_pct:
        if hold_seconds < _min_hold:
            log.info(
                "VOL_HARD_STOP_SUPPRESSED: unrealized=%.1f%% breached -%.1f%% but hold_seconds=%.0f < %ds — suppressing (transient impact)",
                unrealized_pct * 100, vol_stop_pct * 100, hold_seconds, _min_hold,
            )
        else:
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

    # 1d. Strike distance exceeded — migrated from monitor_and_exit_open_positions
    # When BTC has moved far from strike and position is losing, cut losses.
    # ATR-adaptive multiplier: tighter in high-vol regimes (fast moves), looser in low-vol.
    # Ref: Avellaneda & Stoikov (2008) — optimal threshold adapts to current σ, not historical σ.
    # Fix #1: STRIKE_DISTANCE uses a shorter 20s grace (not the general 60s _min_hold).
    # Forensic audit: 60s allowed 17 consecutive suppressions while BTC moved against the trade.
    _strike_grace_sec = 20
    if (
        distance is not None and atr14 is not None and atr14 > 0
        and unrealized_pct < -0.05
    ):
        if atr14 > Config.ATR_HIGH_THRESHOLD:
            _strike_mult = 0.40   # high-vol: BTC moves fast — cut sooner
        elif atr14 < Config.ATR_LOW_THRESHOLD:
            _strike_mult = 0.80   # low-vol: give more room, small moves are noise
        else:
            _strike_mult = 0.60   # normal: baseline
        if abs(distance) > _strike_mult * atr14:
            if hold_seconds < _strike_grace_sec:
                log.info(
                    "STRIKE_DISTANCE_SUPPRESSED: distance=%.1f > %.2f*ATR=%.1f unrealized=%.1f%% "
                    "but hold_seconds=%.0f < %ds — suppressing (transient impact)",
                    abs(distance), _strike_mult, _strike_mult * atr14, unrealized_pct * 100, hold_seconds, _strike_grace_sec,
                )
            else:
                log.warning(
                    "STRIKE_DISTANCE_EXCEEDED: distance=%.1f > %.2f*ATR=%.1f, unrealized=%.1f%% — exiting",
                    abs(distance), _strike_mult, _strike_mult * atr14, unrealized_pct * 100,
                )
                return _exit("STRIKE_DISTANCE_EXCEEDED")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1.5: MAE-CONDITIONED EXITS
    # After deep drawdown, exit aggressively on recovery instead of waiting
    # for full TP. Deep-V recoveries are rare on 15-min binaries.
    # ══════════════════════════════════════════════════════════════════════════
    if mae_pct >= Config.MAE_RECOVERY_EXIT_THRESHOLD:
        # MAE_RECOVERY_EXIT: deep drawdown recovered near entry → exit 100%
        if unrealized_pct >= -Config.MAE_RECOVERY_NEAR_ENTRY_PCT:
            log.warning(
                "MAE_RECOVERY_EXIT: mae=%.1f%% recovered to unrealized=%.1f%% (within %.1f%% of entry) — full exit",
                mae_pct * 100, unrealized_pct * 100, Config.MAE_RECOVERY_NEAR_ENTRY_PCT * 100,
            )
            return _exit("MAE_RECOVERY_EXIT")

        # MAE_LATE_RECOVERY: deep drawdown + time elapsed → accept partial recovery
        if deep_drawdown_ts is not None:
            _time_since_deep = time.time() - deep_drawdown_ts
            if _time_since_deep >= Config.MAE_RECOVERY_TIME_LATE_SEC and unrealized_pct >= -0.10:
                log.warning(
                    "MAE_LATE_RECOVERY: mae=%.1f%% + %.0fs since deep drawdown, unrealized=%.1f%% — accepting partial recovery",
                    mae_pct * 100, _time_since_deep, unrealized_pct * 100,
                )
                return _exit("MAE_LATE_RECOVERY")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1.8: PRICE-BASED TRAILING STOP
    # Locks in profits if price drops significantly from the highest recorded peak.
    # ══════════════════════════════════════════════════════════════════════════
    _trail_active = getattr(Config, "TRAIL_PRICE_ACTIVATION_PCT", 0.05)
    _trail_dist   = getattr(Config, "TRAIL_PRICE_DISTANCE_PCT", 0.10)
    
    if mfe_pct >= _trail_active:
        highest_price = entry_price * (1.0 + mfe_pct)
        trail_trigger_price = highest_price * (1.0 - _trail_dist)
        if current_price <= trail_trigger_price:
            log.info(
                "TRAIL_PRICE_STOP: mfe=%.1f%%, price=%.3f fell %.1f%% from peak=%.3f — protecting profit",
                mfe_pct * 100, current_price, _trail_dist * 100, highest_price
            )
            return _exit("TRAIL_PRICE_STOP", use_maker=use_maker)

    # ── Post-TP1 remainder protection: tighter trailing once TP1 has filled ──
    # After scaling out, protect the runner more aggressively to avoid giving
    # back a winner into HARD_STOP.
    _tp1_trail_active = float(getattr(Config, "TP1_TRAIL_PRICE_ACTIVATION_PCT", 0.02) or 0.02)
    _tp1_trail_dist   = float(getattr(Config, "TP1_TRAIL_PRICE_DISTANCE_PCT", 0.06) or 0.06)
    if tp1_hit and mfe_pct >= _tp1_trail_active:
        highest_price = entry_price * (1.0 + mfe_pct)
        trail_trigger_price = highest_price * (1.0 - _tp1_trail_dist)
        if current_price <= trail_trigger_price:
            log.info(
                "TP1_TRAIL_PRICE_STOP: tp1_hit=1 mfe=%.1f%%, price=%.3f fell %.1f%% from peak=%.3f — protecting remainder",
                mfe_pct * 100, current_price, _tp1_trail_dist * 100, highest_price
            )
            return _exit("TP1_TRAIL_PRICE_STOP", use_maker=use_maker)

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
        tp1_post_ceil = float(getattr(Config, "TP1_POSTERIOR_CEIL", 0.93) or 0.93)
        tp1_allowed = (posterior is None) or (posterior < tp1_post_ceil)
        if Config.TP_PARTIAL_ENABLED:
            # Partial exits: 1/3 at each tier
            if not tp3_hit and unrealized_pct >= Config.TP3_PCT:
                log.info("TP3: unrealized=%.1f%% >= %.1f%% — selling remaining", unrealized_pct * 100, Config.TP3_PCT * 100)
                return _exit("TP3", partial_pct=1.0, use_maker=use_maker)
            if not tp2_hit and unrealized_pct >= Config.TP2_PCT:
                log.info("TP2: unrealized=%.1f%% >= %.1f%% — selling 1/3", unrealized_pct * 100, Config.TP2_PCT * 100)
                return _exit("TP2", partial_pct=0.333, use_maker=use_maker)
            if not tp1_hit and unrealized_pct >= Config.TP1_PCT:
                if tp1_allowed:
                    # Adaptive TP1 sizing (Option B): increase TP1 sell fraction when
                    # proximity/time are risky, or signal quality degraded vs entry.
                    _tp1_frac_base = float(getattr(Config, "TP1_PARTIAL_BASE", 0.333) or 0.333)
                    _tp1_frac_mid  = float(getattr(Config, "TP1_PARTIAL_MID", 0.666) or 0.666)
                    _tp1_frac_max  = float(getattr(Config, "TP1_PARTIAL_MAX", 1.0) or 1.0)

                    _close_dist = float(getattr(Config, "TP1_CLOSE_DIST", 80.0) or 80.0)
                    _late_min_rem = float(getattr(Config, "TP1_LATE_MIN_REM", 8.0) or 8.0)
                    _edge_drop_thresh = float(getattr(Config, "TP1_EDGE_DROP_THRESH", 0.005) or 0.005)
                    _post_drop_thresh = float(getattr(Config, "TP1_POST_DROP_THRESH", 0.05) or 0.05)

                    close_to_strike = (distance is not None) and (abs(distance) <= _close_dist)
                    late_in_window = minutes_remaining <= _late_min_rem

                    entry_post = _entry_post if _entry_post is not None else 0.5
                    post_drop = (entry_post - float(posterior)) if posterior is not None else 0.0
                    edge_drop = 0.0
                    if entry_edge is not None and getattr(Config, "TP1_USE_EDGE_DEGRADATION", True):
                        # entry_edge is expected edge at entry; current edge is approximated by (posterior - current_price)
                        # for the held side.
                        curr_edge = None
                        if posterior is not None and current_price is not None:
                            curr_edge = float(posterior) - float(current_price)
                        if curr_edge is not None:
                            edge_drop = float(entry_edge) - float(curr_edge)

                    degrade = (post_drop >= _post_drop_thresh) or (edge_drop >= _edge_drop_thresh)

                    # MAE override stays strongest: after deep drawdown, take full profit.
                    if mae_pct >= Config.MAE_RECOVERY_EXIT_THRESHOLD:
                        _tp1_pct = 1.0
                        log.info("TP1_MAE_OVERRIDE: mae=%.1f%% — upgrading TP1 to full exit", mae_pct * 100)
                    else:
                        if close_to_strike and late_in_window:
                            _tp1_pct = _tp1_frac_max
                        elif degrade or close_to_strike or late_in_window:
                            _tp1_pct = _tp1_frac_mid
                        else:
                            _tp1_pct = _tp1_frac_base
                        _tp1_pct = max(0.0, min(1.0, float(_tp1_pct)))

                        log.info(
                            "TP1_ADAPTIVE: unrealized=%.1f%%>=%.1f%% sell=%.0f%% close=%s late=%s post_drop=%.1fpp edge_drop=%.1fpp",
                            unrealized_pct * 100,
                            Config.TP1_PCT * 100,
                            _tp1_pct * 100,
                            str(bool(close_to_strike)),
                            str(bool(late_in_window)),
                            post_drop * 100,
                            edge_drop * 100,
                        )

                    return _exit("TP1", partial_pct=_tp1_pct, use_maker=use_maker)
                log.info(
                    "TP1_SKIPPED_HIGH_CONVICTION: posterior=%.3f >= %.3f — holding despite unrealized=%.1f%%",
                    float(posterior),
                    tp1_post_ceil,
                    unrealized_pct * 100,
                )
        else:
            # Full exit at TP1 threshold (fallback when partial disabled)
            if not tp3_hit and unrealized_pct >= Config.TP3_PCT:
                log.info(
                    "TP3: unrealized=%.1f%% >= %.1f%% — full exit (partial disabled)",
                    unrealized_pct * 100,
                    Config.TP3_PCT * 100,
                )
                return _exit("TP3", partial_pct=1.0, use_maker=use_maker)
            if not tp2_hit and unrealized_pct >= Config.TP2_PCT:
                log.info(
                    "TP2: unrealized=%.1f%% >= %.1f%% — selling 50%% (partial even when partial disabled)",
                    unrealized_pct * 100,
                    Config.TP2_PCT * 100,
                )
                return _exit("TP2", partial_pct=0.5, use_maker=use_maker)
            if unrealized_pct >= Config.TP1_PCT:
                if tp1_allowed:
                    log.info(
                        "TP_FULL: unrealized=%.1f%% >= %.1f%% — full exit (partial disabled)",
                        unrealized_pct * 100, Config.TP1_PCT * 100,
                    )
                    return _exit("TP_FULL", use_maker=use_maker)
                log.info(
                    "TP_FULL_SKIPPED_HIGH_CONVICTION: posterior=%.3f >= %.3f — holding despite unrealized=%.1f%%",
                    float(posterior),
                    tp1_post_ceil,
                    unrealized_pct * 100,
                )

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

    # 3b. Near-certain profit lock — bid ≥ $0.98 means market is virtually decided;
    # lock in the win rather than risk a last-minute reversal for $0.01-0.02 upside.
    if bid_price is not None and bid_price >= 0.98 and unrealized_pct > 0:
        log.info(
            "NEAR_CERTAIN_LOCK: bid=%.3f >= $0.98, unrealized=%.1f%% — locking profit",
            bid_price, unrealized_pct * 100,
        )
        return _exit("NEAR_CERTAIN_LOCK", use_maker=use_maker)

    # 3c. Fix #7: Reverse convergence — opposing side near certain means our side is losing.
    # Exit YES when NO_bid >= 0.85 (market says DOWN is ~85% likely) and vice versa.
    _rev_conv_threshold = 0.85
    if held_side == "YES" and no_bid is not None and no_bid >= _rev_conv_threshold:
        log.warning(
            "REVERSE_CONVERGENCE: NO bid=%.3f >= %.2f while holding YES, unrealized=%.1f%% — exiting",
            no_bid, _rev_conv_threshold, unrealized_pct * 100,
        )
        return _exit("REVERSE_CONVERGENCE", use_maker=False)
    if held_side == "NO" and yes_bid is not None and yes_bid >= _rev_conv_threshold:
        log.warning(
            "REVERSE_CONVERGENCE: YES bid=%.3f >= %.2f while holding NO, unrealized=%.1f%% — exiting",
            yes_bid, _rev_conv_threshold, unrealized_pct * 100,
        )
        return _exit("REVERSE_CONVERGENCE", use_maker=False)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 4: STRUCTURAL MODEL REVERSAL (Req #5)
    # Posterior collapsed from entry — more reliable than price-based stops
    # on wide-spread 15m binaries.
    # ══════════════════════════════════════════════════════════════════════════

    if posterior is not None and _entry_post is not None:
        posterior_drop = _entry_post - posterior
        # Time-dependent threshold: early drops are informative, late drops are noisier
        minutes_elapsed = 15.0 - minutes_remaining
        if minutes_elapsed < 5.0:
            _reversal_threshold = 0.10   # tighter early — posterior updates carry more signal
        elif minutes_elapsed < 10.0:
            _reversal_threshold = Config.MODEL_REVERSAL_DROP_PCT  # 0.15 mid-window
        else:
            _reversal_threshold = 0.20   # looser late — spread noise dominates near expiry
        if posterior_drop >= _reversal_threshold:
            log.warning(
                "MODEL_REVERSAL: posterior=%.3f dropped %.1fpp from entry=%.3f "
                "(threshold=%.0fpp, %.1fmin elapsed) — exiting",
                posterior, posterior_drop * 100, _entry_post, _reversal_threshold * 100, minutes_elapsed,
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
    # MAE tightening: if position has already shown vulnerability (MAE >= 10%),
    # reduce drawdown tolerance by 40% to exit faster on second adverse move.
    # Only apply after grace period — instant post-fill drawdowns are price discovery,
    # not a "second adverse move" (Tetlock 2004: prediction markets overshoot then revert).
    _grace_sec = _min_hold  # unified: same as VOL_HARD_STOP suppression window
    _in_grace = hold_seconds < _grace_sec
    mae_tightened = False
    if mae_pct >= Config.MAE_TIGHTEN_THRESHOLD and not _in_grace:
        adapted_drawdown *= 0.60
        mae_tightened = True
    log.debug(
        "DRAWDOWN_THRESHOLD: adapted=%.1f%% (base=%.1f%% atr_ratio=%.2f mae_tight=%s) "
        "unrealized=%.1f%% mae=%.1f%% hold=%.0fs grace=%s",
        adapted_drawdown * 100, Config.MAX_DRAWDOWN_PCT * 100, atr_ratio,
        mae_tightened, unrealized_pct * 100, mae_pct * 100, hold_seconds, _in_grace,
    )
    if unrealized_pct < -adapted_drawdown:
        # Grace period: within first N seconds, only fire if loss > HARD_STOP
        # or model conviction dropped significantly (>10pp from entry).
        # Hard stops (Layer 0/1) still fire normally — this only gates Layer 5.
        if _in_grace and unrealized_pct > -Config.HARD_STOP_PCT:
            if posterior is not None and posterior > _entry_post - 0.10:
                log.info(
                    "DRAWDOWN_GRACE: unrealized=%.1f%% but hold=%.0fs < %.0fs grace "
                    "and posterior=%.3f near entry=%.3f — holding",
                    unrealized_pct * 100, hold_seconds, _grace_sec,
                    posterior, _entry_post,
                )
            else:
                return _exit("FORCED_DRAWDOWN", use_maker=use_maker)
        # 0.95+ conviction exemption near expiry with large distance
        elif (posterior is not None and posterior > 0.95 and minutes_remaining < 2.0
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


    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 6: TIME-DECAY-ENHANCED MICROSTRUCTURE EXITS (Req #6)
    # In the final 2 minutes, exponential multiplier makes adverse OFI/distance
    # thresholds tighter — less adverse flow triggers protective exits.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Micro-exit early grace period ────────────────────────────────────────
    # Suppress ALL Layer-6 microstructure / alpha-decay / momentum exits for
    # the first MICRO_EXIT_GRACE_SEC seconds (default 180s = 3 minutes).
    # Rationale: sub-3-minute microstructure on a 15m binary prediction market
    # is dominated by bid-ask noise and order-book transient imbalances, not
    # genuine directional information. Cutting positions this early converts
    # potential expiry wins into realized losses (confirmed by morning trade
    # audit: all 7 early-exited trades lost; the 2 held-to-expiry trades won).
    # Hard stops (Layers 0–1) and TP exits (Layer 2) are NOT affected.
    _micro_grace = getattr(Config, "MICRO_EXIT_GRACE_SEC", 180)
    if hold_seconds < _micro_grace:
        log.debug(
            "MICRO_EXIT_GRACE: hold_seconds=%.0fs < %.0fs — suppressing Layer-6 micro exits",
            hold_seconds, _micro_grace,
        )
        return None

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

    # 6a2. Absolute score reversal (migrated from monitor_and_exit_open_positions)
    # Fires on absolute score opposing the held side, not just delta from entry.
    if held_side == "YES" and signed_score < -5.0 and unrealized_pct < 0:
        return _exit("ABS_SCORE_REVERSAL", use_maker=use_maker)
    if held_side == "NO" and signed_score > 5.0 and unrealized_pct < 0:
        return _exit("ABS_SCORE_REVERSAL", use_maker=use_maker)

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
        # Phase 8: Reduce sensitivity during early trade phase (Fix premature exits)
        mid_window_factor = 1.0
        if minutes_remaining > 5.0:
            mid_window_factor = getattr(Config, "MICRO_EXIT_MID_WINDOW_SENSITIVITY", 2.0)
            
        _ofi_thresh = (Config.EXIT_DEEP_OFI_REV_THRESH * mid_window_factor) / td_mult
        _cvd_thresh = (Config.EXIT_CVD_VEL_REV_THRESH * mid_window_factor) / td_mult

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
