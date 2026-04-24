"""
Unit tests for the Tier 2 #6 payoff-geometry entry gate.

Replicates the gate math inline — mirrors the exact formula used in
main.py:_handle_entry at the PAYOFF_GEOMETRY_BLOCK branch. The gate rejects
entries where the distance-to-strike margin (in units of remaining BTC vol)
is below PAYOFF_GEOMETRY_MIN_SIGMA.
"""

import math
from config import Config


def payoff_gate_blocks(
    *,
    distance_usd: float,
    atr: float,
    min_rem: float,
    monster_conviction: float = 0.0,
    min_sigma: float = None,
) -> bool:
    """Return True iff the payoff-geometry gate BLOCKS this entry."""
    if monster_conviction >= 0.85:
        return False
    sigma_req = min_sigma if min_sigma is not None else float(Config.PAYOFF_GEOMETRY_MIN_SIGMA)
    adverse = atr * math.sqrt(max(0.1, min_rem) / 15.0)
    if adverse <= 0:
        return False
    margin_sigma = abs(distance_usd) / adverse
    return margin_sigma < sigma_req


class TestPayoffGeometry:

    def test_apr17_passes(self):
        """Apr 17 winner: dist=$298 ITM, ATR=150, 2.15 min → ~5.2σ margin → pass."""
        # adverse = 150 * sqrt(2.15/15) = 57 → margin = 298/57 = 5.23σ
        assert not payoff_gate_blocks(distance_usd=298.0, atr=150.0, min_rem=2.15)

    def test_at_strike_blocks(self):
        """Zero margin always blocks (except monster)."""
        assert payoff_gate_blocks(distance_usd=0.0, atr=150.0, min_rem=7.0)

    def test_near_strike_blocks(self):
        """Position within 0.5σ of strike should be blocked."""
        # adverse = 150 * sqrt(5/15) = 86.6, margin = 30/86.6 = 0.35σ
        assert payoff_gate_blocks(distance_usd=30.0, atr=150.0, min_rem=5.0)

    def test_monster_override(self):
        """Monster conviction ≥ 0.85 bypasses the gate."""
        # Would normally block (margin 0.35σ)
        assert payoff_gate_blocks(distance_usd=30.0, atr=150.0, min_rem=5.0)
        # Monster bypasses it
        assert not payoff_gate_blocks(
            distance_usd=30.0, atr=150.0, min_rem=5.0, monster_conviction=0.90
        )

    def test_sigma_threshold_tunable(self):
        """Raising min_sigma tightens the gate."""
        # margin = 100/86.6 = 1.15σ — passes at 1.0σ, blocks at 2.0σ
        assert not payoff_gate_blocks(distance_usd=100.0, atr=150.0, min_rem=5.0, min_sigma=1.0)
        assert payoff_gate_blocks(distance_usd=100.0, atr=150.0, min_rem=5.0, min_sigma=2.0)

    def test_low_vol_regime_lenient(self):
        """Low ATR shrinks adverse → modest distance still passes."""
        # ATR=50, min_rem=3, dist=50 → adverse=22.4 → margin=2.23σ → pass
        assert not payoff_gate_blocks(distance_usd=50.0, atr=50.0, min_rem=3.0)

    def test_high_vol_regime_strict(self):
        """High ATR expands adverse → modest distance fails."""
        # ATR=400, min_rem=6, dist=100 → adverse=253 → margin=0.40σ → block
        assert payoff_gate_blocks(distance_usd=100.0, atr=400.0, min_rem=6.0)
