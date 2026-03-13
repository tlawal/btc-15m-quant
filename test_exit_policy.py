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
        assert result["reason"] == "TP_FULL"
        assert result["partial_pct"] == 1.0  # full exit

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
        assert result["reason"] == "VOL_HARD_STOP"

    def test_stop_capped_at_max_pct(self):
        """Even at extreme ATR, stop is capped at VOL_STOP_MAX_PCT (30%)."""
        result = call_exit(
            entry_price=0.60, current_price=0.41,  # -31.7%
            atr14=500.0, posterior=0.30, entry_posterior=0.30,
        )
        assert result is not None
        assert result["reason"] == "VOL_HARD_STOP"

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
        assert result["reason"] in ("TP_FULL", "PROB_CONVERGENCE")
        # TP_FULL fires first at +10% unrealized; both are correct exits

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
            assert result_far["reason"] != "FORCED_ADVERSE_OFI"
        assert result_near is not None
        assert result_near["reason"] == "FORCED_ADVERSE_OFI"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
