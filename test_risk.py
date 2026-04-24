"""
Tier 4 #18 + #19: Tests for risk.py (Sharpe-decay monitor + VaR/ES).
"""

import json
from pathlib import Path

import pytest

import risk
import trade_attribution as ta


@pytest.fixture(autouse=True)
def isolated_attr(monkeypatch, tmp_path):
    p = tmp_path / "attr.jsonl"
    monkeypatch.setattr(ta, "ATTRIBUTION_LOG_PATH", p)
    yield p


def _write_pnls(path: Path, pnls: list[float]):
    lines = [json.dumps({"pnl_usd": p}) for p in pnls]
    path.write_text("\n".join(lines) + "\n")


class TestSharpeDecay:
    def test_collecting_under_10(self, isolated_attr):
        _write_pnls(isolated_attr, [1.0, 2.0])
        s = risk.sharpe_decay()
        assert s["status"] == "collecting"

    def test_no_decay_when_recent_matches_baseline(self, isolated_attr):
        pnls = [0.1, -0.05, 0.12, -0.04] * 100   # 400 trades
        _write_pnls(isolated_attr, pnls)
        s = risk.sharpe_decay(recent_window=100, baseline_window=400)
        assert s["status"] == "ok"
        # recent and baseline share the same distribution → near-zero decay.
        assert abs(s["decay_pct"]) < 0.1
        assert s["halt"] is False

    def test_detects_regime_break(self, isolated_attr):
        """Baseline wins, recent losses → high decay, halt=True."""
        # 300 good trades followed by 100 bad trades.
        pnls = [0.20] * 300 + [-0.15] * 100
        _write_pnls(isolated_attr, pnls)
        s = risk.sharpe_decay(recent_window=100, baseline_window=400)
        assert s["status"] == "ok"
        assert s["recent_sharpe"] <= 0   # recent is all losses
        assert s["baseline_sharpe"] > 0  # baseline still positive overall
        assert s["decay_pct"] > 0.30
        assert s["halt"] is True

    def test_negative_baseline_does_not_halt(self, isolated_attr):
        """You can't decay from a losing baseline — that's just a bad strategy."""
        pnls = [-0.10] * 400
        _write_pnls(isolated_attr, pnls)
        s = risk.sharpe_decay(recent_window=100, baseline_window=400)
        assert s["status"] == "ok"
        assert s["halt"] is False    # guarded


class TestVaR:
    def test_collecting_under_10(self, isolated_attr):
        _write_pnls(isolated_attr, [1.0, 2.0])
        s = risk.var_es()
        assert s["status"] == "collecting"

    def test_no_losses(self, isolated_attr):
        _write_pnls(isolated_attr, [1.0] * 20)
        s = risk.var_es()
        assert s["status"] == "no_losses"
        assert s["var"]["0.95"] == 0.0

    def test_var_p95_and_p99_ordering(self, isolated_attr):
        # Mixed wins/losses with a couple of tail-losses.
        pnls = [1.0] * 80 + [-0.5] * 15 + [-2.0, -3.0, -4.0, -5.0, -8.0]
        _write_pnls(isolated_attr, pnls)
        s = risk.var_es(alphas=(0.95, 0.99))
        assert s["status"] == "ok"
        # VaR(99) >= VaR(95) always
        assert s["var"]["0.99"] >= s["var"]["0.95"]
        # ES >= VaR at same alpha
        assert s["es"]["0.95"] >= s["var"]["0.95"]
        assert s["es"]["0.99"] >= s["var"]["0.99"]

    def test_evt_fits_with_enough_data(self, isolated_attr):
        import random
        random.seed(0)
        # 200 trades: 70% small wins, 30% losses with heavy tail.
        pnls = [0.1 if random.random() < 0.7 else -abs(random.gauss(1.0, 2.0))
                for _ in range(200)]
        _write_pnls(isolated_attr, pnls)
        s = risk.var_es()
        assert s["status"] == "ok"
        assert s["evt"] is not None
        assert "xi" in s["evt"] and "beta" in s["evt"]
        assert s["evt"]["beta"] > 0     # scale must be positive

    def test_evt_none_on_small_sample(self, isolated_attr):
        pnls = [1.0] * 10 + [-0.5] * 5
        _write_pnls(isolated_attr, pnls)
        s = risk.var_es()
        # Small sample: VaR still computed, EVT skipped
        assert s["status"] == "ok"
        assert s["evt"] is None
