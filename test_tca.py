"""
Tier 3 #14: Tests for the TCA logger.
"""

import json
import time
from pathlib import Path

import pytest

import tca


@pytest.fixture(autouse=True)
def isolated_log(monkeypatch, tmp_path):
    p = tmp_path / "tca.jsonl"
    monkeypatch.setattr(tca, "TCA_LOG_PATH", p)
    tca._last_prune_ts = 0.0
    yield
    # Cleanup handled by tmp_path


class TestLogFill:
    def test_buy_worse_than_intended(self):
        """Buy at intended=0.50 realized=0.52 → +4% slippage (paid more = bad)."""
        ok = tca.log_fill(
            token_id="0xTOKEN", order_type="BUY",
            intended_px=0.50, realized_px=0.52, size=10,
            strategy="FOK", tag="entry",
        )
        assert ok is True
        row = next(tca.iter_records())
        assert abs(row["slippage_pct"] - 0.04) < 1e-6
        assert abs(row["slippage_bps"] - 400.0) < 1e-4
        assert row["notional_usd"] == 0.52 * 10

    def test_sell_worse_than_intended(self):
        """Sell at intended=0.94 realized=0.92 → +2.13% slippage (got less = bad)."""
        ok = tca.log_fill(
            token_id="0xTOKEN", order_type="SELL",
            intended_px=0.94, realized_px=0.92, size=6,
        )
        assert ok is True
        row = next(tca.iter_records())
        # (0.92 - 0.94) / 0.94 = -0.0213; negate for SELL → +0.0213
        assert row["slippage_pct"] > 0
        assert abs(row["slippage_pct"] - 0.02127659) < 1e-6

    def test_sell_better_than_intended(self):
        """Sell at intended=0.94 realized=0.95 → negative (improvement)."""
        ok = tca.log_fill(
            token_id="0xTOKEN", order_type="SELL",
            intended_px=0.94, realized_px=0.95, size=6,
        )
        assert ok is True
        row = next(tca.iter_records())
        assert row["slippage_pct"] < 0

    def test_invalid_inputs_silent_skip(self):
        assert tca.log_fill(
            token_id="0xTOKEN", order_type="BUY",
            intended_px=None, realized_px=0.50, size=10,
        ) is False
        assert tca.log_fill(
            token_id="0xTOKEN", order_type="BUY",
            intended_px=0.50, realized_px=None, size=10,
        ) is False
        assert tca.log_fill(
            token_id="0xTOKEN", order_type="BUY",
            intended_px=0.0, realized_px=0.50, size=10,
        ) is False
        assert tca.log_fill(
            token_id="0xTOKEN", order_type="BUY",
            intended_px=0.50, realized_px=0.50, size=0,
        ) is False


class TestSummary:
    def test_empty_summary(self):
        s = tca.summary()
        assert s["total_records"] == 0
        assert s["buy"]["n"] == 0
        assert s["sell"]["n"] == 0

    def test_summary_stats(self):
        # 5 BUYs at different slippage
        for intended, realized in [(0.50, 0.50), (0.50, 0.51), (0.50, 0.52),
                                    (0.50, 0.53), (0.50, 0.54)]:
            tca.log_fill(
                token_id="t", order_type="BUY",
                intended_px=intended, realized_px=realized, size=10,
                strategy="FOK",
            )
        s = tca.summary(last_n=100)
        assert s["buy"]["n"] == 5
        assert s["sell"]["n"] == 0
        # Median should be the middle record (0.52 → 400bps)
        assert abs(s["buy"]["median_bps"] - 400.0) < 1e-4
        # p95 should be near the max (0.54 → 800bps)
        assert s["buy"]["p95_bps"] >= 600.0
        # Strategy breakdown
        assert "FOK" in s["by_strategy"]
        assert s["by_strategy"]["FOK"]["n"] == 5

    def test_summary_last_n_cap(self):
        for i in range(10):
            tca.log_fill(
                token_id="t", order_type="BUY",
                intended_px=0.50, realized_px=0.50 + 0.001 * i, size=1,
            )
        s = tca.summary(last_n=3)
        assert s["buy"]["n"] == 3   # only last 3
