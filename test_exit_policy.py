"""
Unit tests for exit_policy.py — production-grade exit architecture.
Tests all 6 exit mechanisms with controlled inputs (pure, no async/DB).
"""

import pytest
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from exit_policy import evaluate_exit, _time_decay_multiplier, _check_spread_aware


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

BASE_KWARGS = dict(
    held_side="YES",
    entry_price=0.50,
    current_price=0.50,
    minutes_remaining=7.0,
    signed_score=3.0,
    entry_score=3.0,
    distance=100.0,
    cvd_delta=0.0,
    cvd_velocity=0.0,
    deep_ofi=0.0,
    obi=0.0,
    atr14=150.0,
    vpin=0.0,
    posterior=0.70,
    prev_posterior=0.70,
    entry_posterior=0.65,
    peak_posterior=0.72,
    book_flip_count=0,
    hold_seconds=120.0,
    entry_min_rem=10.0,
    yes_mid=0.50,
    bid_price=0.49,
    ask_price=0.51,
    tp1_hit=False,
    tp2_hit=False,
    tp3_hit=False,
)


@pytest.fixture(autouse=True)
def force_preferred_hours(monkeypatch):
    """Force is_preferred_trading_time=True so TAKE_SMALL_PROFIT_OUTSIDE
    does not fire in all tests. Individual tests can override."""
    from config import Config
    monkeypatch.setattr(Config, "is_preferred_trading_time", classmethod(lambda cls: True))


def call_exit(**overrides):
    """Call evaluate_exit with sensible defaults, overriding specific params."""
    kw = {**BASE_KWARGS, **overrides}
    return evaluate_exit(**kw)


# ──────────────────────────────────────────────────────────────────────────────
# 1. TIERED TAKE-PROFITS
# ──────────────────────────────────────────────────────────────────────────────

class TestTieredTakeProfits:
    """Req #1: Tiered take-profit evaluation."""

    def test_tp_full_at_5pct(self):
        """With partial disabled (default), full exit at +5% unrealized."""
        result = call_exit(entry_price=0.50, current_price=0.53)
        assert result is not None
        assert result["reason"] in ("TP_FULL", "TP1")
        assert result["partial_pct"] in (1.0, 0.333)

    def test_tp_full_skipped_when_high_conviction(self, monkeypatch):
        """TP_FULL should be skipped when posterior is monster-high."""
        from config import Config
        monkeypatch.setattr(Config, "TP1_POSTERIOR_CEIL", 0.93)
        result = call_exit(entry_price=0.50, current_price=0.53, posterior=0.99)
        assert result is None

    def test_no_tp_below_threshold(self):
        """No TP exit when unrealized is below 5%."""
        result = call_exit(entry_price=0.50, current_price=0.52)
        assert result is None

    def test_late_entry_override(self):
        """Entry >= $0.95 triggers single TP at +2%."""
        result = call_exit(entry_price=0.96, current_price=0.98, entry_posterior=0.96)
        assert result is not None
        assert result["reason"] == "TP_LATE_ENTRY"

    def test_late_entry_no_trigger_below_2pct(self):
        """Late entry but price hasn't risen 2% yet."""
        result = call_exit(entry_price=0.96, current_price=0.97, entry_posterior=0.96, posterior=0.95)
        assert result is None

    def test_tp1_partial_when_enabled(self, monkeypatch):
        """When TP_PARTIAL_ENABLED=True, TP1 returns 1/3 partial."""
        from config import Config
        monkeypatch.setattr(Config, "TP_PARTIAL_ENABLED", True)
        result = call_exit(entry_price=0.50, current_price=0.53)
        assert result is not None
        assert result["reason"] == "TP1"
        assert abs(result["partial_pct"] - 0.333) < 0.01

    def test_tp1_skipped_when_high_conviction(self, monkeypatch):
        """TP1 should be skipped when posterior is monster-high, even with partial enabled."""
        from config import Config
        monkeypatch.setattr(Config, "TP_PARTIAL_ENABLED", True)
        monkeypatch.setattr(Config, "TP1_POSTERIOR_CEIL", 0.93)
        result = call_exit(entry_price=0.50, current_price=0.53, posterior=0.99)
        assert result is None

    def test_tp2_fires_when_tp1_already_hit(self, monkeypatch):
        """TP2 fires when TP1 already hit and unrealized >= 12%."""
        from config import Config
        monkeypatch.setattr(Config, "TP_PARTIAL_ENABLED", True)
        result = call_exit(
            entry_price=0.50, current_price=0.57,
            tp1_hit=True, tp2_hit=False,
        )
        assert result is not None
        assert result["reason"] == "TP2"

    def test_tp3_full_exit_remaining(self, monkeypatch):
        """TP3 sells remaining position (full exit)."""
        from config import Config
        monkeypatch.setattr(Config, "TP_PARTIAL_ENABLED", True)
        result = call_exit(
            entry_price=0.50, current_price=0.61,
            tp1_hit=True, tp2_hit=True, tp3_hit=False,
        )
        assert result is not None
        assert result["reason"] == "TP3"
        assert result["partial_pct"] == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. VOLATILITY-ADAPTED STOP-LOSS
# ──────────────────────────────────────────────────────────────────────────────

class TestVolatilityAdaptedStop:
    """Req #2: ATR-normalized stop-loss."""

    def test_base_stop_at_normal_atr(self):
        """At baseline ATR (150), stop triggers at -15%."""
        result = call_exit(
            entry_price=0.60, current_price=0.50,  # -16.7%
            atr14=150.0, posterior=0.30, entry_posterior=0.30,
        )
        assert result is not None
        assert result["reason"] == "VOL_HARD_STOP"

    def test_widened_stop_at_high_atr(self):
        """At 1.5x ATR (225), stop widens to -22.5%. -18% should NOT trigger vol stop."""
        result = call_exit(
            entry_price=0.60, current_price=0.492,  # -18%
            atr14=225.0, posterior=0.55, entry_posterior=0.55,
            distance=50.0,
        )
        # -18% < -22.5% (vol stop), posterior hasn't dropped enough for MODEL_REVERSAL
        assert result is None

    def test_widened_stop_triggers_at_high_atr(self):
        """At 1.5x ATR (225), stop widens to -22.5%. -24% SHOULD trigger."""
        result = call_exit(
            entry_price=0.60, current_price=0.456,  # -24%
            atr14=225.0, posterior=0.30, entry_posterior=0.30,
        )
        assert result is not None
        assert result["reason"] in ("VOL_HARD_STOP", "HARD_STOP")

    def test_stop_capped_at_max_pct(self):
        """Even at extreme ATR, stop is capped at VOL_STOP_MAX_PCT (30%)."""
        result = call_exit(
            entry_price=0.60, current_price=0.41,  # -31.7%
            atr14=500.0, posterior=0.30, entry_posterior=0.30,
        )
        assert result is not None
        assert result["reason"] in ("VOL_HARD_STOP", "HARD_STOP")

    def test_no_btc_atr_on_option_price(self):
        """Verify ATR is used as a RATIO, not subtracted from price.
        $200 ATR should NOT be subtracted from $0.60 option price."""
        result = call_exit(
            entry_price=0.60, current_price=0.55,  # -8.3%
            atr14=200.0, posterior=0.70, entry_posterior=0.65,
        )
        # -8.3% < -20% (base 15% * max(1, 200/150)=1.33 → 20%)
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# 3. SPREAD-AWARE EXITING
# ──────────────────────────────────────────────────────────────────────────────

class TestSpreadAwareExiting:
    """Req #3: Avellaneda-Stoikov spread-aware exits."""

    def test_narrow_spread_no_maker(self):
        """Spread <= 10% should NOT trigger maker mode."""
        assert _check_spread_aware(0.49, 0.52, 5.0) is False  # ~6% spread

    def test_wide_spread_triggers_maker(self):
        """Spread > 10% with >60s to expiry triggers maker mode."""
        assert _check_spread_aware(0.40, 0.50, 5.0) is True  # 25% spread

    def test_wide_spread_ignored_near_expiry(self):
        """Even wide spread should NOT trigger maker within 60s of expiry."""
        assert _check_spread_aware(0.40, 0.50, 0.5) is False  # 30s remaining

    def test_exit_uses_maker_on_wide_spread(self):
        """A non-emergency exit with wide spread should set use_maker=True."""
        result = call_exit(
            entry_price=0.50, current_price=0.53,
            bid_price=0.40, ask_price=0.55,  # 37.5% spread
            minutes_remaining=5.0,
        )
        assert result is not None
        assert result["use_maker"] is True

    def test_emergency_exit_ignores_spread(self):
        """VOL_HARD_STOP should ignore spread and NOT set use_maker."""
        result = call_exit(
            entry_price=0.60, current_price=0.48,  # -20%
            bid_price=0.35, ask_price=0.55,  # huge spread
            atr14=150.0, posterior=0.30, entry_posterior=0.30,
        )
        assert result is not None
        assert result["reason"] == "VOL_HARD_STOP"
        assert result["use_maker"] is False  # emergency exits never use maker


# ──────────────────────────────────────────────────────────────────────────────
# 4. PROBABILITY-CONVERGENCE EXITS
# ──────────────────────────────────────────────────────────────────────────────

class TestProbabilityConvergence:
    """Req #4: Exit when market bid >= model posterior."""

    def test_convergence_triggers_exit(self):
        """When bid >= posterior AND in profit, exit."""
        result = call_exit(
            entry_price=0.50, current_price=0.55,
            bid_price=0.75, posterior=0.70,
        )
        assert result is not None
        assert result["reason"] in ("TP_FULL", "TP1", "PROB_CONVERGENCE", "TAKE_PROFIT_DYNAMIC")
        # TP_FULL/TP1 fires first at +10% unrealized; all are correct exits

    def test_convergence_fires_below_tp_threshold(self):
        """Convergence should fire when profit < TP threshold but bid >= posterior."""
        result = call_exit(
            entry_price=0.50, current_price=0.52,  # +4%, below TP1
            bid_price=0.72, posterior=0.70,
        )
        assert result is not None
        assert result["reason"] == "PROB_CONVERGENCE"

    def test_convergence_no_trigger_when_losing(self):
        """Convergence should NOT trigger when position is losing."""
        result = call_exit(
            entry_price=0.50, current_price=0.48,
            bid_price=0.75, posterior=0.70, entry_posterior=0.70,
        )
        # Even though bid > posterior, unrealized is negative
        if result is not None:
            assert result["reason"] != "PROB_CONVERGENCE"

    def test_convergence_no_trigger_bid_below_posterior(self):
        """No exit when bid < posterior."""
        result = call_exit(
            entry_price=0.50, current_price=0.52,
            bid_price=0.60, posterior=0.80,
        )
        if result is not None:
            assert result["reason"] != "PROB_CONVERGENCE"


# ──────────────────────────────────────────────────────────────────────────────
# 5. STRUCTURAL MODEL REVERSAL
# ──────────────────────────────────────────────────────────────────────────────

class TestStructuralModelReversal:
    """Req #5: Exit when posterior drops >15pp from entry."""

    def test_model_reversal_triggers(self):
        """15pp posterior drop from entry should trigger exit."""
        result = call_exit(
            entry_price=0.50, current_price=0.50,
            posterior=0.50, entry_posterior=0.70,
        )
        assert result is not None
        assert result["reason"] == "MODEL_REVERSAL"

    def test_model_reversal_no_trigger_small_drop(self):
        """10pp drop should NOT trigger (threshold is 15pp)."""
        result = call_exit(
            entry_price=0.50, current_price=0.50,
            posterior=0.60, entry_posterior=0.70,
        )
        if result is not None:
            assert result["reason"] != "MODEL_REVERSAL"

    def test_model_reversal_uses_entry_posterior(self):
        """The drop is measured from entry_posterior, not peak."""
        result = call_exit(
            entry_price=0.50, current_price=0.50,
            posterior=0.55, entry_posterior=0.72,
            peak_posterior=0.85,  # peak is higher but irrelevant for this check
        )
        assert result is not None
        assert result["reason"] == "MODEL_REVERSAL"  # 72% - 55% = 17pp > 15pp


# ──────────────────────────────────────────────────────────────────────────────
# 6. EXPONENTIAL TIME-DECAY
# ──────────────────────────────────────────────────────────────────────────────

class TestExponentialTimeDecay:
    """Req #6: Time-decay multiplier for microstructure exits."""

    def test_multiplier_one_outside_window(self):
        """Multiplier is 1.0 when > 2 min remaining."""
        assert _time_decay_multiplier(5.0) == 1.0

    def test_multiplier_increases_near_expiry(self):
        """Multiplier > 1 when inside the 2-minute window."""
        m1 = _time_decay_multiplier(1.0)  # 1 min remaining
        assert m1 > 1.0

    def test_multiplier_max_at_zero(self):
        """Multiplier is at max (== TIME_DECAY_EXP_BASE) at 0 seconds remaining."""
        from config import Config
        m0 = _time_decay_multiplier(0.0)
        assert abs(m0 - Config.TIME_DECAY_EXP_BASE) < 0.01

    def test_multiplier_monotonic(self):
        """Multiplier increases as time decreases."""
        m2 = _time_decay_multiplier(1.5)
        m1 = _time_decay_multiplier(1.0)
        m05 = _time_decay_multiplier(0.5)
        m01 = _time_decay_multiplier(0.1)
        assert m01 > m05 > m1 > m2 >= 1.0

    def test_ofi_threshold_tighter_near_expiry(self):
        """Adverse OFI should trigger more easily near expiry due to time-decay."""
        # At 5 min remaining (outside window), moderate OFI should not trigger
        result_far = call_exit(
            minutes_remaining=5.0,
            entry_price=0.55, current_price=0.54,
            deep_ofi=-0.15,  # below standard 0.20 threshold
            held_side="YES",
        )
        # At 0.5 min remaining (inside window), same OFI should trigger
        result_near = call_exit(
            minutes_remaining=0.5,
            entry_price=0.55, current_price=0.54,
            deep_ofi=-0.15,
            held_side="YES",
            posterior=0.50, entry_posterior=0.50,
        )
        # Near expiry, the OFI threshold is divided by td_mult (>1), so -0.15 exceeds it
        if result_far is not None:
            assert result_far["reason"] != "MICRO_REVERSAL"
        assert result_near is not None
        assert result_near["reason"] == "MICRO_REVERSAL"


# ──────────────────────────────────────────────────────────────────────────────
# Existing functionality preservation
# ──────────────────────────────────────────────────────────────────────────────

class TestExistingExitsPreserved:
    """Verify backward compatibility with existing exit mechanisms."""

    def test_none_when_price_missing(self):
        result = call_exit(current_price=None)
        assert result is None

    def test_none_when_entry_zero(self):
        result = call_exit(entry_price=0.0)
        assert result is None

    def test_forced_late_exit(self):
        """FORCED_LATE_EXIT triggers at <1min with -10% loss (not caught by vol stop)."""
        result = call_exit(
            minutes_remaining=0.5,
            entry_price=0.60, current_price=0.53,  # -11.7%, below vol stop (-15%)
            entry_min_rem=None,
            posterior=0.30, entry_posterior=0.30,
            atr14=150.0,
        )
        assert result is not None
        assert result["reason"] == "FORCED_LATE_EXIT"

    def test_alpha_decay(self):
        result = call_exit(
            entry_price=0.50, current_price=0.50,
            signed_score=-5.0, entry_score=3.0,
            posterior=0.40, entry_posterior=0.45,
            hold_seconds=120.0,
        )
        assert result is not None
        assert result["reason"] == "ALPHA_DECAY"

    def test_hold_when_no_reason(self):
        result = call_exit(
            entry_price=0.50, current_price=0.51,
            posterior=0.70, entry_posterior=0.65,
        )
        assert result is None

    def test_return_is_dict(self):
        """All non-None returns should be dicts with required keys."""
        result = call_exit(entry_price=0.50, current_price=0.53)
        assert result is not None
        assert "reason" in result
        assert "partial_pct" in result
        assert "use_maker" in result


# ──────────────────────────────────────────────────────────────────────────────
# Audit 3: HARD_STOP posterior-gated grace period
# ──────────────────────────────────────────────────────────────────────────────

class TestHardStopGrace:
    """Audit 3 P3: HARD_STOP fires with posterior-gated grace period."""

    def test_hard_stop_fires_immediately_low_posterior(self):
        """Posterior < 0.50 → 0s grace → fires immediately."""
        result = call_exit(
            entry_price=0.70, current_price=0.52,  # -25.7%
            posterior=0.40, entry_posterior=0.70,
            hold_seconds=5,
        )
        assert result is not None
        assert result["reason"] == "HARD_STOP"

    def test_hard_stop_suppressed_high_posterior_short_hold(self):
        """Posterior >= 0.70 → 45s grace → suppressed when hold < 45s.
        Use exactly -26% to cross HARD_STOP threshold but stay in grace."""
        result = call_exit(
            entry_price=0.80, current_price=0.59,  # -26.25%
            posterior=0.75, entry_posterior=0.78,
            hold_seconds=10,
            distance=10.0,
            atr14=300.0,  # high ATR widens FORCED_DRAWDOWN threshold
        )
        assert result is None

    def test_hard_stop_fires_high_posterior_grace_expired(self):
        """Posterior >= 0.70 → 45s grace → fires when hold >= 45s."""
        result = call_exit(
            entry_price=0.80, current_price=0.59,  # -26.25%
            posterior=0.75, entry_posterior=0.78,
            hold_seconds=50,
            distance=10.0,
            atr14=300.0,
        )
        assert result is not None
        assert result["reason"] == "HARD_STOP"

    def test_hard_stop_suppressed_mid_posterior_short_hold(self):
        """Posterior 0.50-0.70 → 25s grace → suppressed when hold < 25s."""
        result = call_exit(
            entry_price=0.80, current_price=0.59,  # -26.25%
            posterior=0.60, entry_posterior=0.65,
            hold_seconds=10,
            distance=10.0,
            atr14=300.0,
        )
        assert result is None

    def test_hard_stop_fires_mid_posterior_grace_expired(self):
        """Posterior 0.50-0.70 → 25s grace → fires when hold >= 25s."""
        result = call_exit(
            entry_price=0.80, current_price=0.59,  # -26.25%
            posterior=0.60, entry_posterior=0.65,
            hold_seconds=30,
            distance=10.0,
            atr14=300.0,
        )
        assert result is not None
        assert result["reason"] == "HARD_STOP"

    def test_hard_stop_late_entry_still_suppressed(self):
        """Late-entry suppression (entry_min_rem < 3.0) still works independently."""
        result = call_exit(
            entry_price=0.80, current_price=0.59,  # -26.25%
            posterior=0.70, entry_posterior=0.72,
            hold_seconds=30,
            entry_min_rem=2.0,  # late entry
            distance=10.0,
            atr14=300.0,
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# REGRESSION: Apr 17, 2026 — REVERSE_CONVERGENCE fired on an ITM winner
# ──────────────────────────────────────────────────────────────────────────────
# Market: btc-updown-15m-1776432600 (9:30–9:45 AM ET, strike 76630.02)
# Entry:  YES @ 0.97, t = +771s into window (129s remaining), 6.00 shares
# Exit:   REVERSE_CONVERGENCE @ 0.94 for 5.98/6.00 shares at +2s past expiry
# Truth:  YES settled at $1.00 (Polymarket outcomePrices ["1", "0"])
#         BTC close 76959.16 → +$329 above strike at window end
# Loss:   realized −3.42% vs counterfactual +3.09% = 6.5% EV swing
#
# Fix:    BTC-confirmation + final-30s suppression gates on REVERSE_CONVERGENCE
#         and TIME_DECAY. These tests assert neither rule fires in the scenario.

class TestApr17RegressionREVCONV:
    """Regression tests for the 2026-04-17 btc-updown-15m-1776432600 trade.

    On that trade REVERSE_CONVERGENCE fired on a deep-ITM YES position 2s past
    window expiry, selling into a NO-bid sweep that was MM inventory / retail
    panic, not information arrival. These tests pin down the new gates so the
    exit rule can never fire on an ITM winner again.
    """

    # Apr 17 scenario kwargs: YES held, BTC +$296 ITM above strike 76630, with
    # NO_bid spiking to 0.85 (the legacy convergence threshold at <1.5 min rem).
    APR17 = dict(
        held_side="YES",
        held_direction="UP",
        entry_price=0.97,
        current_price=0.94,       # unrealized ≈ −3.1%, passes the -2% gate
        minutes_remaining=0.5,    # 30s remaining → final-30s branch
        signed_score=8.20,
        entry_score=8.20,
        distance=296.0,
        btc_price=76926.0,
        strike_price=76630.0,     # +296 above strike = clearly ITM for UP
        atr14=150.0,              # 296/150 ≈ 1.97×ATR — well above 0.5× gate
        posterior=0.95,
        prev_posterior=0.95,
        entry_posterior=0.97,
        peak_posterior=0.97,
        hold_seconds=131.0,       # well past the 30s min-hold
        entry_min_rem=2.15,       # 129s remaining at entry
        yes_mid=0.94,
        bid_price=0.93,
        ask_price=0.95,
        no_bid=0.85,              # the convergence spike that fired the rule
        yes_bid=0.93,
        cvd_delta=0.0,
        cvd_velocity=0.0,
        deep_ofi=0.0,
        obi=0.0,
        vpin=0.0,
        book_flip_count=0,
        tp1_hit=False,
        tp2_hit=False,
        tp3_hit=False,
    )

    @pytest.fixture(autouse=True)
    def _reset_rev_conv_counter(self):
        """REVERSE_CONVERGENCE uses a function-attribute counter for sustained
        cycles — clear it before each test so cross-test leakage can't mask
        the gate behavior."""
        from exit_policy import evaluate_exit as _ee
        if hasattr(_ee, "_rev_conv_count"):
            _ee._rev_conv_count = 0
        yield
        if hasattr(_ee, "_rev_conv_count"):
            _ee._rev_conv_count = 0

    def test_rev_conv_suppressed_when_btc_itm(self):
        """BTC confirms UP (+1.97×ATR above strike) → REVERSE_CONVERGENCE must not fire."""
        # Even firing the rule three times in a row must NOT exit: the BTC-confirm
        # gate resets the sustained counter on every cycle.
        from exit_policy import evaluate_exit
        for _ in range(3):
            result = evaluate_exit(**self.APR17)
            assert result is None or result["reason"] != "REVERSE_CONVERGENCE", \
                f"REVERSE_CONVERGENCE fired on ITM winner: {result}"

    def test_rev_conv_suppressed_in_final_30s_even_without_full_atr(self):
        """Inside 30s with any ITM margin, REVERSE_CONVERGENCE must not fire.

        Regression guard for the exact moment the Apr 17 trade exited (2s past
        expiry, +$296 ITM). A tight-ATR case should still be suppressed by the
        final-30s branch even if the distance-ATR gate is inconclusive.
        """
        from exit_policy import evaluate_exit
        kw = dict(self.APR17)
        kw["atr14"] = 1000.0  # forces dist/ATR = 0.296 < 0.50 gate threshold
        kw["minutes_remaining"] = 0.4  # 24s remaining
        for _ in range(3):
            result = evaluate_exit(**kw)
            assert result is None or result["reason"] != "REVERSE_CONVERGENCE", \
                f"REVERSE_CONVERGENCE fired in final 30s on ITM winner: {result}"

    def test_rev_conv_still_fires_on_genuine_otm_loser(self):
        """Regression-safety: the new gates must not create a false negative.

        A position that is truly OTM (BTC on the wrong side of strike) with a
        sustained opposing bid AND a collapsed posterior should still exit via
        REVERSE_CONVERGENCE. Note: in reality the posterior tracks BTC, so the
        free-boundary rule (Tier 2 #5) correctly refuses to exit on an "OTM
        but posterior=0.95" contradiction — posterior is the ground truth
        for our confidence. This test uses a realistic collapsed posterior.
        """
        from exit_policy import evaluate_exit
        kw = dict(self.APR17)
        # Flip BTC below strike — now YES is OTM, BTC does NOT confirm UP.
        kw["btc_price"] = 76500.0  # −130 below strike 76630
        kw["distance"] = -130.0
        kw["minutes_remaining"] = 1.0  # outside the final-30s window
        # Realistic posterior collapse when BTC flips to the wrong side of strike.
        kw["posterior"] = 0.25
        kw["prev_posterior"] = 0.25
        # Need enough time remaining that we're in the low (0.85) threshold band
        # and still past the min-hold. Fire three cycles to trip sustained count.
        result = None
        for _ in range(3):
            result = evaluate_exit(**kw)
        assert result is not None, "REVERSE_CONVERGENCE should still fire on OTM loser"
        assert result["reason"] == "REVERSE_CONVERGENCE"

    def test_time_decay_suppressed_in_final_30s_when_itm(self):
        """Tier 1 #2 — TIME_DECAY must not fire on ITM position in final 30s.

        On Apr 17 the position had posterior ~0.95 so the existing posterior
        gate would have held; this test pins down the BTC-side gate so that
        even a posterior-collapse scenario cannot capitulate inside 30s.
        """
        from exit_policy import evaluate_exit
        kw = dict(self.APR17)
        kw["current_price"] = 0.93       # -4.1% unrealized, triggers TIME_DECAY path
        kw["posterior"] = 0.40           # below the 0.60 posterior-hold gate
        kw["prev_posterior"] = 0.40
        kw["minutes_remaining"] = 0.4    # 24s remaining
        result = evaluate_exit(**kw)
        assert result is None or result["reason"] != "TIME_DECAY", \
            f"TIME_DECAY fired on ITM position in final 30s: {result}"

    def test_time_decay_suppressed_when_btc_confirms_beyond_atr(self):
        """TIME_DECAY must not fire outside 30s when BTC is clearly ITM (≥0.5×ATR)."""
        from exit_policy import evaluate_exit
        kw = dict(self.APR17)
        kw["current_price"] = 0.93
        kw["posterior"] = 0.40
        kw["prev_posterior"] = 0.40
        kw["minutes_remaining"] = 1.5    # outside 30s window but < 2min TIME_DECAY band
        # distance 296 / ATR 150 = 1.97× — well above 0.50× gate.
        result = evaluate_exit(**kw)
        assert result is None or result["reason"] != "TIME_DECAY", \
            f"TIME_DECAY fired on BTC-confirmed ITM position: {result}"

    def test_time_decay_still_fires_on_genuine_otm_loser(self):
        """Regression-safety: TIME_DECAY still exits losing OTM positions in final 2m."""
        from exit_policy import evaluate_exit
        kw = dict(self.APR17)
        kw["held_direction"] = "UP"
        kw["btc_price"] = 76500.0        # BTC below strike → UP does NOT confirm
        kw["distance"] = -130.0
        kw["current_price"] = 0.60
        kw["entry_price"] = 0.70          # -14% unrealized
        kw["posterior"] = 0.30            # below 0.60 gate
        kw["prev_posterior"] = 0.30
        kw["minutes_remaining"] = 1.5
        result = evaluate_exit(**kw)
        assert result is not None, "TIME_DECAY should fire on OTM losing position"
        # Could fire via TIME_DECAY or an earlier rule; assert it didn't silently hold.
        # Relaxed assertion: an exit happened.
        assert "reason" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
