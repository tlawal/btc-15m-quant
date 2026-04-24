"""
Tier 3 #12: Tests for the economic-calendar block.
"""

import datetime as dt
import pytest

import calendar_block as cb


def _t(year, month, day, hour, minute):
    """Build a UTC datetime."""
    return dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)


class TestActiveEvent:
    def test_fomc_day_2pm_et_blocked(self):
        """Mar 18, 2026 18:00 UTC = 14:00 ET (FOMC decision). Block active."""
        # 2:00 PM ET = 18:00 UTC (ET = UTC-4 during DST)
        ev = cb.active_event(now=_t(2026, 3, 18, 18, 0))
        assert ev is not None
        assert ev["label"] == "FOMC"

    def test_fomc_30min_before_blocked(self):
        """30 min before FOMC is still inside the pre-window."""
        ev = cb.active_event(now=_t(2026, 3, 18, 17, 30))
        assert ev is not None

    def test_fomc_31min_before_allowed(self):
        """31 min before FOMC is outside the default 30-min pre-window."""
        ev = cb.active_event(now=_t(2026, 3, 18, 17, 29))
        assert ev is None

    def test_fomc_30min_after_blocked(self):
        """Inside the post-window."""
        ev = cb.active_event(now=_t(2026, 3, 18, 18, 30))
        assert ev is not None

    def test_random_day_not_blocked(self):
        """Mid-month random date with no event scheduled."""
        ev = cb.active_event(now=_t(2026, 3, 19, 14, 0))
        assert ev is None

    def test_unknown_year_not_blocked(self):
        ev = cb.active_event(now=_t(2030, 6, 15, 14, 0))
        assert ev is None


class TestIsBlocked:
    def test_blocked_returns_true_with_event(self):
        blocked, ev = cb.is_blocked(now=_t(2026, 3, 18, 18, 0))
        assert blocked is True
        assert ev is not None

    def test_monster_bypass(self):
        """Monster-conviction trades (≥0.90) slip through the block."""
        blocked, ev = cb.is_blocked(
            now=_t(2026, 3, 18, 18, 0),
            monster_conviction=0.95,
        )
        assert blocked is False   # bypassed
        assert ev is not None      # still reported for observability

    def test_sub_monster_still_blocked(self):
        blocked, ev = cb.is_blocked(
            now=_t(2026, 3, 18, 18, 0),
            monster_conviction=0.85,   # below default bypass threshold 0.90
        )
        assert blocked is True

    def test_no_event_not_blocked_regardless_of_conviction(self):
        blocked, ev = cb.is_blocked(
            now=_t(2026, 3, 19, 14, 0),
            monster_conviction=0.0,
        )
        assert blocked is False
        assert ev is None


class TestWindowTuning:
    def test_shrunk_pre_window_allows_earlier_entry(self):
        """With pre_min=5, 20 min before event is no longer blocked."""
        ev = cb.active_event(now=_t(2026, 3, 18, 17, 40), pre_min=5)
        assert ev is None

    def test_widened_post_window_blocks_longer(self):
        """With post_min=120, we're still blocked 1h after event."""
        ev = cb.active_event(now=_t(2026, 3, 18, 19, 0), post_min=120)
        assert ev is not None


class TestMinutesUntilSince:
    def test_minutes_until_positive_before_event(self):
        ev = cb.active_event(now=_t(2026, 3, 18, 17, 50))
        assert ev is not None
        assert ev["minutes_until"] > 0
        assert ev["minutes_until"] <= 30

    def test_minutes_since_positive_after_event(self):
        ev = cb.active_event(now=_t(2026, 3, 18, 18, 20))
        assert ev is not None
        assert ev["minutes_since"] > 0
        assert ev["minutes_until"] < 0
