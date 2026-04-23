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
    # ── Audit Apr 1: BTC cross-validation & MM bait detection ──
    btc_price:        Optional[float] = None,
    strike_price:     Optional[float] = None,
    held_direction:   Optional[str]   = None,  # "UP" or "DOWN"
    entry_spread_pct: Optional[float] = None,  # spread % at entry time
    entry_bid_px:     Optional[float] = None,  # bid price at entry time
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

    # R15: Universal endgame grace cap. At < 3 min remaining, grace/hold periods
    # must scale with available time — a 45s grace at 1.67 min remaining means
    # the stop physically can't fire until 45% of remaining time has elapsed.
    _endgame_grace_cap = max(3, int(minutes_remaining * 5)) if minutes_remaining < 3.0 else 999

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 0: ABSOLUTE HARD STOP — unconditional circuit breaker, no gates
    # At -25% the position is unrecoverable on a 15m binary; cut always.
    # Exception: for late-window entries (< 3 min remaining at entry) within the first 60s of holding,
    # binary price swings of ±30% are CLOB noise, not outcome-probability change.
    # Kelly (1956) + Thorp (2006): at < 3 min, variance is maximum — don't cut noise as signal.
    # ══════════════════════════════════════════════════════════════════════════
    if unrealized_pct < -Config.HARD_STOP_PCT:
        # Late-entry suppression (original): for entries with < 3 min remaining,
        # binary price swings of ±30% are CLOB noise in the first 60s.
        _suppress_late = (
            entry_min_rem is not None
            and entry_min_rem < 3.0
            and hold_seconds < 60
            and posterior >= 0.65
            and unrealized_pct > -0.35  # R12: absolute cap — no suppression past -35%
        )
        if _suppress_late:
            log.info(
                "HARD_STOP_LATE_SUPPRESSED: unrealized=%.1f%% but entry_min_rem=%.1f<3.0 hold=%ds<60s posterior=%.3f≥0.65 — noise, not signal",
                unrealized_pct * 100, entry_min_rem, hold_seconds, posterior,
            )
        else:
            # Posterior-gated grace period (Audit 3 P3):
            # When model still supports position, CLOB drop may be microstructure noise.
            # posterior >= HIGH → full grace; LOW-HIGH → reduced grace; < LOW → immediate.
            _grace_high = float(getattr(Config, "HARD_STOP_GRACE_POST_HIGH", 0.70))
            _grace_low = float(getattr(Config, "HARD_STOP_GRACE_POST_LOW", 0.50))
            _grace_full = int(getattr(Config, "HARD_STOP_GRACE_SEC", 45))
            _grace_reduced = int(getattr(Config, "HARD_STOP_GRACE_REDUCED_SEC", 25))

            if posterior >= _grace_high:
                _grace_sec = min(_grace_full, _endgame_grace_cap)    # R15
            elif posterior >= _grace_low:
                _grace_sec = min(_grace_reduced, _endgame_grace_cap)  # R15
            else:
                _grace_sec = 0  # model says we're wrong — fire immediately

            if _grace_sec > 0 and hold_seconds < _grace_sec:
                log.info(
                    "HARD_STOP_GRACE: unrealized=%.1f%% breached -%.1f%% but posterior=%.3f "
                    "→ grace=%ds, hold=%ds < grace — suppressing",
                    unrealized_pct * 100, Config.HARD_STOP_PCT * 100,
                    posterior, _grace_sec, hold_seconds,
                )
            else:
                log.warning(
                    "HARD_STOP: unrealized=%.1f%% breached -%.1f%% — %s (posterior=%.3f, hold=%ds)",
                    unrealized_pct * 100, Config.HARD_STOP_PCT * 100,
                    "grace expired" if _grace_sec > 0 else "no grace (posterior too low)",
                    posterior, hold_seconds,
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
    _base_hold = getattr(Config, "MIN_HOLD_BEFORE_DRAWDOWN_SEC", 30)
    _min_hold = min(_base_hold, max(5, minutes_remaining * 15), _endgame_grace_cap)  # R15: endgame cap overrides
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
    # Fix 2 (Forensic Apr 15): Monster entries get extended grace — high-conviction
    # signals should not be cut by distance alone in the first 45s (CLOB whipsaw window).
    _is_monster_entry = (
        (entry_score is not None and abs(entry_score) >= 8.0)
        or (entry_posterior is not None and entry_posterior >= 0.90)
    )
    if _is_monster_entry:
        _strike_grace_sec = int(getattr(Config, "MONSTER_STRIKE_GRACE_SEC", 45))
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
                # Fix 1 (Forensic Apr 15): BTC cross-validation gate.
                # When BTC confirms the held direction, distance is a confirmation signal,
                # not a loss signal — the CLOB price drop is microstructure noise.
                # Same pattern as FORCED_DRAWDOWN BTC confirmation (line ~740-754).
                if (btc_price is not None and strike_price is not None
                        and held_direction is not None and posterior is not None
                        and posterior > 0.80):
                    _btc_confirms = (
                        (held_direction == "DOWN" and btc_price < strike_price)
                        or (held_direction == "UP" and btc_price > strike_price)
                    )
                    if _btc_confirms:
                        log.info(
                            "STRIKE_DISTANCE_BTC_CONFIRMED: distance=%.1f > %.2f*ATR=%.1f unrealized=%.1f%% "
                            "but BTC=$%.2f %s strike=$%.2f posterior=%.3f — CLOB noise, holding",
                            abs(distance), _strike_mult, _strike_mult * atr14, unrealized_pct * 100,
                            btc_price, "<" if held_direction == "DOWN" else ">",
                            strike_price, posterior,
                        )
                        # Skip exit — BTC confirms direction, CLOB drawdown is noise
                    else:
                        log.warning(
                            "STRIKE_DISTANCE_EXCEEDED: distance=%.1f > %.2f*ATR=%.1f, unrealized=%.1f%% "
                            "BTC does NOT confirm — exiting",
                            abs(distance), _strike_mult, _strike_mult * atr14, unrealized_pct * 100,
                        )
                        return _exit("STRIKE_DISTANCE_EXCEEDED")
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
    # LAYER 1.75: MID_WINDOW_PROFIT_LOCK (R6)
    # Big peak + time still left + still meaningfully profitable → cash out rather
    # than wait for strike-flip roulette. This is the user's exact ask after Trade 1:
    # "it was at 0.98 with 6+ min remaining, why didn't it take it?"
    # Forced taker (R7): profit-protection exits MUST cross the spread to avoid
    # maker-failure → DUST_WRITEOFF chains when books vaporize.
    # ══════════════════════════════════════════════════════════════════════════
    _mw_lock_mfe      = float(getattr(Config, "MID_WINDOW_LOCK_MFE_PCT",     0.10) or 0.10)
    _mw_lock_max_rem  = float(getattr(Config, "MID_WINDOW_LOCK_MAX_REM_MIN", 8.0)  or 8.0)
    _mw_lock_min_pnl  = float(getattr(Config, "MID_WINDOW_LOCK_MIN_PNL_PCT", 0.06) or 0.06)
    if (
        mfe_pct >= _mw_lock_mfe
        and minutes_remaining <= _mw_lock_max_rem
        and unrealized_pct >= _mw_lock_min_pnl
    ):
        log.info(
            "MID_WINDOW_PROFIT_LOCK: mfe=%.1f%% min_rem=%.1f unrealized=%.1f%% — locking mid-window profit",
            mfe_pct * 100, minutes_remaining, unrealized_pct * 100,
        )
        return _exit("MID_WINDOW_PROFIT_LOCK", use_maker=False)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1.78: PROFIT_GIVEBACK_50PCT (R5)
    # Independent of TRAIL_PRICE_DISTANCE: if we gave back ≥50% of a meaningful
    # peak, exit. Catches the slow-drift failure mode that TRAIL_PRICE_STOP misses
    # on thin books (and would have fired on Trade 1 before the crash leg).
    # ══════════════════════════════════════════════════════════════════════════
    _giveback_min_mfe  = float(getattr(Config, "PROFIT_GIVEBACK_MIN_MFE_PCT", 0.08) or 0.08)
    _giveback_frac     = float(getattr(Config, "PROFIT_GIVEBACK_FRAC", 0.5) or 0.5)
    if mfe_pct >= _giveback_min_mfe and unrealized_pct < (mfe_pct * _giveback_frac):
        log.info(
            "PROFIT_GIVEBACK_50PCT: mfe=%.1f%% unrealized=%.1f%% (<%.0f%% of peak) — locking remaining profit",
            mfe_pct * 100, unrealized_pct * 100, _giveback_frac * 100,
        )
        return _exit("PROFIT_GIVEBACK_50PCT", use_maker=False)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1.8: PRICE-BASED TRAILING STOP
    # Locks in profits if price drops significantly from the highest recorded peak.
    # R7: forced taker (use_maker=False) — maker on a collapsing book → DUST.
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
            return _exit("TRAIL_PRICE_STOP", use_maker=False)

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
            return _exit("TP1_TRAIL_PRICE_STOP", use_maker=False)

    # ── Earlier profit protection on runner: posterior collapse after TP1 ──
    # If we've already taken TP1 and then the model conviction collapses, exit the remainder
    # rather than waiting for full reversal or expiry.
    _runner_post_drop = float(getattr(Config, "RUNNER_POSTERIOR_DROP_PCT", 0.10) or 0.10)
    _runner_min_profit = float(getattr(Config, "RUNNER_MIN_PROFIT_PCT", 0.03) or 0.03)
    if tp1_hit and posterior is not None and _entry_post is not None:
        if unrealized_pct >= _runner_min_profit and (_entry_post - float(posterior)) >= _runner_post_drop:
            log.warning(
                "RUNNER_POSTERIOR_COLLAPSE: tp1_hit=1 unrealized=%.1f%% entry_post=%.3f posterior=%.3f drop=%.1fpp>=%.1fpp — exiting remainder",
                unrealized_pct * 100,
                float(_entry_post),
                float(posterior),
                (float(_entry_post) - float(posterior)) * 100,
                _runner_post_drop * 100,
            )
            return _exit("RUNNER_POSTERIOR_COLLAPSE", use_maker=False)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 2: TIERED TAKE-PROFITS (Req #1)
    # ══════════════════════════════════════════════════════════════════════════

    # Late-entry override: entry in the 0.88–0.95+ "expensive" band gets a single
    # price-scaled TP. This closes the old dead zone where entries at 0.90–0.94
    # had no reachable profit target (TP1 at +5% required 0.945–0.987, which crosses
    # strike-flip territory). Target scales from +4% at 0.88 down to +2% at 0.95.
    if entry_price >= Config.TP_LATE_ENTRY_THRESH:
        _late_floor = float(getattr(Config, "TP_LATE_ENTRY_PCT", 0.02) or 0.02)
        _late_ceil  = float(getattr(Config, "TP_LATE_ENTRY_PCT_MAX", 0.04) or 0.04)
        _band_lo    = float(getattr(Config, "TP_LATE_ENTRY_THRESH", 0.88) or 0.88)
        _band_hi    = float(getattr(Config, "TP_LATE_ENTRY_BAND_TOP", 0.95) or 0.95)
        if _band_hi <= _band_lo:
            tp_target = _late_floor
        else:
            # Linear interp: entry=_band_lo → _late_ceil, entry=_band_hi+ → _late_floor.
            frac = max(0.0, min(1.0, (entry_price - _band_lo) / (_band_hi - _band_lo)))
            tp_target = _late_ceil + frac * (_late_floor - _late_ceil)
        if unrealized_pct >= tp_target:
            log.info(
                "TP_LATE_ENTRY: entry=%.3f >= %.2f, unrealized=%.1f%% >= %.1f%% — exiting",
                entry_price, _band_lo,
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
            # TP-state hardening: if TP1 already hit, do NOT require additional gates; allow TP2/TP3
            # to fire normally when thresholds are reached.
            # (This also makes behavior resilient to partial-fill timing/order failures.)
        else:
            # Full exit at TP1 threshold (fallback when partial disabled).
            # R2/R3: When partial exits are disabled, there is no "runner" to protect,
            # so we MUST exit fully at every TP tier and ignore TP1_POSTERIOR_CEIL
            # (the ceiling only makes sense when partials have already banked profit).
            if not tp3_hit and unrealized_pct >= Config.TP3_PCT:
                log.info(
                    "TP3: unrealized=%.1f%% >= %.1f%% — full exit (partial disabled)",
                    unrealized_pct * 100,
                    Config.TP3_PCT * 100,
                )
                return _exit("TP3", partial_pct=1.0, use_maker=use_maker)
            if not tp2_hit and unrealized_pct >= Config.TP2_PCT:
                log.info(
                    "TP2: unrealized=%.1f%% >= %.1f%% — full exit (partial disabled)",
                    unrealized_pct * 100,
                    Config.TP2_PCT * 100,
                )
                return _exit("TP2", partial_pct=1.0, use_maker=use_maker)
            if unrealized_pct >= Config.TP1_PCT:
                log.info(
                    "TP_FULL: unrealized=%.1f%% >= %.1f%% — full exit (partial disabled, posterior ceiling bypassed)",
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

    # 3b. Near-certain profit lock — bid ≥ $0.98 means market is virtually decided;
    # lock in the win rather than risk a last-minute reversal for $0.01-0.02 upside.
    if bid_price is not None and bid_price >= 0.98 and unrealized_pct > 0:
        log.info(
            "NEAR_CERTAIN_LOCK: bid=%.3f >= $0.98, unrealized=%.1f%% — locking profit",
            bid_price, unrealized_pct * 100,
        )
        return _exit("NEAR_CERTAIN_LOCK", use_maker=use_maker)

    # 3c. Fix #7: Reverse convergence — opposing side near certain means our side is losing.
    # Exit YES when NO_bid >= threshold (market says DOWN is near-certain) and vice versa.
    #
    # Guards added after audit of 2026-03-23 10:27 AM trade:
    #   - 30s minimum hold: prevents 1-second panic exits on post-entry order book noise
    #   - Sustained convergence: opposing bid must exceed threshold for ≥3 consecutive cycles
    #     to filter transient whipsaws near expiry (De Long et al., 1990)
    #   - Time-adaptive threshold: require higher certainty when >1.5 min remains (0.93 vs 0.85)
    #     because BTC can cross strike multiple times in final 2-3 minutes of a 15m binary
    _REV_CONV_MIN_HOLD_SEC = 30
    _REV_CONV_BASE_THRESHOLD = 0.85
    _REV_CONV_EARLY_THRESHOLD = 0.93  # when >1.5 min remains, require near-certainty
    _REV_CONV_EARLY_MIN_REM = 1.5
    _REV_CONV_SUSTAINED_CYCLES = 3    # must persist for 3 cycles (~15s at 5s cadence)

    _rev_conv_threshold = (
        _REV_CONV_EARLY_THRESHOLD if minutes_remaining > _REV_CONV_EARLY_MIN_REM
        else _REV_CONV_BASE_THRESHOLD
    )
    _rev_conv_hold_ok = hold_seconds >= _REV_CONV_MIN_HOLD_SEC

    _opp_bid = no_bid if held_side == "YES" else yes_bid
    _opp_label = "NO" if held_side == "YES" else "YES"

    # Forensic 2026-04-17 (btc-updown-15m-1776432600): REVERSE_CONVERGENCE fired on a
    # YES position at 2s past expiry while BTC was +$296 above strike. YES then settled
    # at $1.00. The rule had no BTC-confirmation gate and no proximity-to-expiry
    # suppression. Two gates added below, mirroring STRIKE_DISTANCE_EXCEEDED (~line 284)
    # and the SUB_3_MIN_HOLD pattern in Layer 5 (~line 702).
    _rev_conv_btc_confirms = False
    _rev_conv_distance_atr = 0.0
    if (btc_price is not None and strike_price is not None
            and held_direction is not None and eff_atr > 0):
        _rev_conv_btc_confirms = (
            (held_direction == "UP" and btc_price > strike_price)
            or (held_direction == "DOWN" and btc_price < strike_price)
        )
        _rev_conv_distance_atr = abs(btc_price - strike_price) / eff_atr

    if _opp_bid is not None and _opp_bid >= _rev_conv_threshold and _rev_conv_hold_ok and unrealized_pct < -0.02:
        # Gate 1: BTC-confirmation. If BTC is clearly ITM (≥ 0.5×ATR in our favor),
        # the opposing-bid spike is MM inventory / retail panic, not information arrival.
        if _rev_conv_btc_confirms and _rev_conv_distance_atr >= 0.50:
            log.info(
                "REVERSE_CONVERGENCE_BTC_CONFIRMED: %s bid=%.3f >= %.2f but BTC=$%.2f %s strike=$%.2f "
                "(distance=%.2f×ATR, %.1fmin rem) — CLOB noise on ITM position, holding to settlement",
                _opp_label, _opp_bid, _rev_conv_threshold,
                btc_price, ">" if held_direction == "UP" else "<",
                strike_price, _rev_conv_distance_atr, minutes_remaining,
            )
            evaluate_exit._rev_conv_count = 0  # type: ignore[attr-defined]
        # Gate 2: final-30s suppression on ITM. Inside 30s, Chainlink is about to print;
        # any CLOB swing is untimely and settlement is the dominant source of truth.
        elif minutes_remaining * 60.0 <= 30.0 and _rev_conv_btc_confirms:
            log.info(
                "REVERSE_CONVERGENCE_FINAL_30S: %s bid=%.3f >= %.2f but %.0fs to expiry and BTC=$%.2f "
                "confirms %s — holding for oracle settlement",
                _opp_label, _opp_bid, _rev_conv_threshold,
                minutes_remaining * 60.0, btc_price, held_direction,
            )
            evaluate_exit._rev_conv_count = 0  # type: ignore[attr-defined]
        else:
            # Track sustained convergence via counter stored on function (cheap stateless approach)
            _prev = getattr(evaluate_exit, "_rev_conv_count", 0)
            evaluate_exit._rev_conv_count = _prev + 1  # type: ignore[attr-defined]
            if evaluate_exit._rev_conv_count >= _REV_CONV_SUSTAINED_CYCLES:
                log.warning(
                    "REVERSE_CONVERGENCE: %s bid=%.3f >= %.2f (sustained %d cycles) while holding %s, "
                    "unrealized=%.1f%%, hold=%.0fs — exiting",
                    _opp_label, _opp_bid, _rev_conv_threshold,
                    evaluate_exit._rev_conv_count, held_side,
                    unrealized_pct * 100, hold_seconds,
                )
                evaluate_exit._rev_conv_count = 0  # type: ignore[attr-defined]
                return _exit("REVERSE_CONVERGENCE", use_maker=False)
            else:
                log.info(
                    "REVERSE_CONVERGENCE: %s bid=%.3f >= %.2f but only %d/%d sustained cycles — holding",
                    _opp_label, _opp_bid, _rev_conv_threshold,
                    evaluate_exit._rev_conv_count, _REV_CONV_SUSTAINED_CYCLES,
                )
    else:
        # Reset sustained counter when condition not met
        evaluate_exit._rev_conv_count = 0  # type: ignore[attr-defined]
        if _opp_bid is not None and _opp_bid >= _REV_CONV_BASE_THRESHOLD and not _rev_conv_hold_ok:
            log.info(
                "REVERSE_CONVERGENCE: %s bid=%.3f >= %.2f SUPPRESSED — hold=%.0fs < %ds min hold",
                _opp_label, _opp_bid, _rev_conv_threshold, hold_seconds, _REV_CONV_MIN_HOLD_SEC,
            )

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
        # Audit Apr 3: Suppress MODEL_REVERSAL for monster entries near expiry.
        # At < 5 min remaining with a monster signal, posterior noise from CLOB spread
        # widening dominates — hold to settlement instead of panic-selling.
        _monster_suppress_min_rem = float(getattr(Config, "MODEL_REVERSAL_MONSTER_SUPPRESS_MIN_REM", 5.0))
        _is_monster_entry = (
            (entry_score is not None and abs(entry_score) >= 8.0)
            or (_entry_post is not None and _entry_post >= 0.90)
        )
        if posterior_drop >= _reversal_threshold:
            if _is_monster_entry and minutes_remaining < _monster_suppress_min_rem:
                log.info(
                    "MODEL_REVERSAL_SUPPRESSED_MONSTER: drop=%.1fpp (threshold=%.0fpp) but monster entry "
                    "(score=%.1f, entry_post=%.3f) with %.1f min rem — holding to expiry",
                    posterior_drop * 100, _reversal_threshold * 100,
                    entry_score or 0, _entry_post or 0, minutes_remaining,
                )
            else:
                log.warning(
                    "MODEL_REVERSAL: posterior=%.3f dropped %.1fpp from entry=%.3f "
                    "(threshold=%.0fpp, %.1fmin elapsed) — exiting",
                    posterior, posterior_drop * 100, _entry_post, _reversal_threshold * 100, minutes_elapsed,
                )
                return _exit("MODEL_REVERSAL", use_maker=use_maker)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 5: EXISTING POSTERIOR-GATED EXITS (with ATR-adapted drawdown)
    # ══════════════════════════════════════════════════════════════════════════

    # ── 5.pre: Market-Maker Bait detection (Audit Apr 1) ─────────────────────
    # If bid/ask spread has widened >15% from entry, classify as adversarial
    # market-making and suppress ALL Layer 5 exits for 60s.
    # Biais, Hillion & Spatt (1995): CLOB spread widening near events is
    # market-maker inventory management, not information arrival.
    _mm_bait_suppressed = False
    if hold_seconds < 60:
        if (entry_spread_pct is not None and entry_spread_pct > 0
                and bid_price is not None and ask_price is not None
                and bid_price > 0):
            _curr_spread_pct = (ask_price - bid_price) / bid_price
            _spread_widen_ratio = _curr_spread_pct / entry_spread_pct if entry_spread_pct > 0.001 else 1.0
            if _spread_widen_ratio > 1.15:
                _mm_bait_suppressed = True
                log.info(
                    "MM_BAIT_DETECTED: spread widened %.0f%% (entry=%.1f%% now=%.1f%%) "
                    "hold=%.0fs < 60s — suppressing Layer 5 exits",
                    (_spread_widen_ratio - 1) * 100, entry_spread_pct * 100,
                    _curr_spread_pct * 100, hold_seconds,
                )
        
        # Bid-pull detection: if the bid has completely collapsed (>20% drop from entry bid)
        if not _mm_bait_suppressed and entry_bid_px is not None and bid_price is not None:
            if bid_price < entry_bid_px * 0.80:
                _mm_bait_suppressed = True
                log.info(
                    "MM_BAIT_DETECTED: bid pulled >20%% (entry=%.3f now=%.3f) "
                    "hold=%.0fs < 60s — suppressing Layer 5 exits",
                    entry_bid_px, bid_price, hold_seconds,
                )

    # ── 5.pre2: Sub-3-minute BTC confirmation hold (Audit Apr 2) ─────────────
    # If < 3 mins remain, posterior is high (> 0.85), and BTC confirms direction,
    # suppress ALL Layer 5 exits (including FORCED_DRAWDOWN, TRAIL_POSTERIOR, BOOK_FLIP).
    _sub3_suppressed = False
    if minutes_remaining < 3.0 and posterior is not None and posterior > 0.85:
        if (btc_price is not None and strike_price is not None and held_direction is not None):
            _btc_confirms_sub3 = (
                (held_direction == "DOWN" and btc_price < strike_price)
                or (held_direction == "UP" and btc_price > strike_price)
            )
            if _btc_confirms_sub3:
                _sub3_suppressed = True
                log.info(
                    "SUB_3_MIN_HOLD: rem=%.1fm < 3.0, post=%.3f > 0.85, BTC confirms — "
                    "suppressing Layer 5 exits",
                    minutes_remaining, posterior
                )

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
        "unrealized=%.1f%% mae=%.1f%% hold=%.0fs grace=%s mm_bait=%s",
        adapted_drawdown * 100, Config.MAX_DRAWDOWN_PCT * 100, atr_ratio,
        mae_tightened, unrealized_pct * 100, mae_pct * 100, hold_seconds, _in_grace,
        _mm_bait_suppressed,
    )
    if not _sub3_suppressed and unrealized_pct < -adapted_drawdown:
        # ── MM Bait suppression: skip all Layer 5 exits if spread widened adversarially ──
        if _mm_bait_suppressed:
            log.info(
                "DRAWDOWN_MM_BAIT_HOLD: unrealized=%.1f%% but spread widened — holding",
                unrealized_pct * 100,
            )
        # Grace period: within first N seconds, only fire if loss > HARD_STOP
        # or model conviction dropped significantly (>10pp from entry).
        # Hard stops (Layer 0/1) still fire normally — this only gates Layer 5.
        elif _in_grace and unrealized_pct > -Config.HARD_STOP_PCT:
            if posterior is not None and posterior > _entry_post - 0.10:
                log.info(
                    "DRAWDOWN_GRACE: unrealized=%.1f%% but hold=%.0fs < %.0fs grace "
                    "and posterior=%.3f near entry=%.3f — holding",
                    unrealized_pct * 100, hold_seconds, _grace_sec,
                    posterior, _entry_post,
                )
            else:
                return _exit("FORCED_DRAWDOWN", use_maker=use_maker)
        # 0.95+ conviction exemption near expiry — hold to settlement when model
        # is near-certain.  Audit: FORCED_DRAWDOWN panic-sold winning positions
        # at deep losses that settled at $1.00.  (Trade #1 Mar 30, Trade Apr 1 4:10)
        elif (posterior is not None and posterior > 0.95 and minutes_remaining < 5.0):
            log.info(
                "DRAWDOWN_EXEMPT: posterior=%.3f > 0.95, rem=%.1f < 5.0 "
                "— suppressing drawdown near expiry (hold to settlement)",
                posterior, minutes_remaining,
            )
        # ── BTC cross-validation (Audit Apr 1 Fix #3) ────────────────────────
        # If the underlying BTC price still confirms our position direction,
        # the CLOB drawdown is microstructure noise — suppress exit.
        # De Long et al. (1990): noise trader deviations revert near terminal time.
        elif (btc_price is not None and strike_price is not None
                and held_direction is not None and posterior is not None
                and posterior > 0.80):
            _btc_confirms = (
                (held_direction == "DOWN" and btc_price < strike_price)
                or (held_direction == "UP" and btc_price > strike_price)
            )
            if _btc_confirms:
                log.info(
                    "DRAWDOWN_BTC_CONFIRMED: unrealized=%.1f%% but BTC=$%.2f %s strike=$%.2f "
                    "posterior=%.3f — CLOB noise, holding",
                    unrealized_pct * 100, btc_price,
                    "<" if held_direction == "DOWN" else ">",
                    strike_price, posterior,
                )
            else:
                # BTC does NOT confirm — drawdown may be real
                return _exit("FORCED_DRAWDOWN", use_maker=use_maker)
        else:
            # Gate the -20% hard override on posterior (Audit Apr 1 Fix #2):
            # If model is 85%+ confident, a -20% paper loss is CLOB noise.
            # De Long et al. (1990): noise-driven deviations are value-destroying
            # to trade on when fundamentals are clear.
            if unrealized_pct < -0.20:
                if posterior is None or posterior < 0.85:
                    return _exit("FORCED_DRAWDOWN", use_maker=use_maker)
                else:
                    log.info(
                        "DRAWDOWN_POSTERIOR_HOLD: unrealized=%.1f%% < -20%% but "
                        "posterior=%.3f >= 0.85 — trusting model over CLOB",
                        unrealized_pct * 100, posterior,
                    )
            # Gate with posterior: only hold if model still convinced
            elif posterior is None or posterior <= _entry_post - 0.05:
                return _exit("FORCED_DRAWDOWN", use_maker=use_maker)
            else:
                log.info(
                    "DRAWDOWN_HELD: unrealized=%.1f%% but posterior=%.3f"
                    " still near entry=%.3f — holding",
                    unrealized_pct * 100, posterior, _entry_post,
                )

    # 5b. Explicit adverse selection / book flip.
    _book_flip_cycles = getattr(Config, "BOOK_FLIP_CONFIRM_CYCLES", None)
    _book_flip_imb = getattr(Config, "BOOK_FLIP_IMB_THRESH", None)
    if _book_flip_cycles is not None and _book_flip_imb is not None:
        if not _sub3_suppressed and book_flip_count >= _book_flip_cycles and abs(obi) >= _book_flip_imb:
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
        if not _sub3_suppressed and posterior <= _peak_post - allow_drop:
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
        # BTC-confirmation gate (Tier 1 #2): if the underlying is ITM on the held
        # side, NEVER fire TIME_DECAY — Chainlink settlement is the ground truth
        # and short-dated CLOB marks do not reflect oracle resolution.
        if (btc_price is not None and strike_price is not None
                and held_direction is not None):
            _td_btc_confirms = (
                (held_direction == "UP" and btc_price > strike_price)
                or (held_direction == "DOWN" and btc_price < strike_price)
            )
            # In the final 30s, suppress unconditionally when BTC confirms —
            # CLOB noise is maximal, decisions are untimely, oracle is imminent.
            if _td_btc_confirms and minutes_remaining * 60.0 <= 30.0:
                log.info(
                    "TIME_DECAY_FINAL_30S_ITM: %.1fs rem, btc=%.2f %s strike=%.2f — holding for oracle settlement",
                    minutes_remaining * 60.0, btc_price,
                    ">" if held_direction == "UP" else "<",
                    strike_price,
                )
                return None
            # With >30s remaining but BTC still ITM, require confirming ATR margin.
            if _td_btc_confirms and eff_atr > 0:
                _td_dist_atr = abs(btc_price - strike_price) / eff_atr
                if _td_dist_atr >= 0.50:
                    log.info(
                        "TIME_DECAY_BTC_CONFIRMED: %.1fmin rem, btc=%.2f %s strike=%.2f, dist=%.2f×ATR — holding to settlement",
                        minutes_remaining, btc_price,
                        ">" if held_direction == "UP" else "<",
                        strike_price, _td_dist_atr,
                    )
                    return None
        # Final posterior check: if model is still >60% confident, hold
        if posterior is not None and posterior > 0.60:
            log.info(
                "TIME_DECAY_HELD: %.1fmin rem, unrealized=%.1f%%, posterior=%.3f — holding for settlement",
                minutes_remaining, unrealized_pct * 100, posterior,
            )
            return None
        return _exit("TIME_DECAY", use_maker=use_maker)

    return None
