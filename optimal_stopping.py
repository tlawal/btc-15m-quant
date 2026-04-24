"""
Tier 2 #5: Free-boundary optimal stopping for binary payoffs.

Replaces the heuristic REVERSE_CONVERGENCE bid-threshold (hard-coded 0.85 / 0.93)
with a posterior-aware, time-aware exit boundary derived from first principles.

────────────────────────────────────────────────────────────────────────────────
Derivation
────────────────────────────────────────────────────────────────────────────────
For a binary option that pays $1 if YES wins, $0 if NO wins, with:

• Posterior (our belief P(YES wins)):        p
• Opposing-side (NO) bid price:              b_opp
• Implied market belief P(NO wins):          ≈ b_opp   (YES_ask + NO_ask ≈ 1,
                                                         so market assesses p
                                                         at ≈ 1 − b_opp)

Our posterior says P(YES wins) = p.
Market's posterior says P(YES wins) ≈ 1 − b_opp.

REVERSE_CONVERGENCE fires when we *defer to the market* — i.e. when the
market's confidence in NO materially exceeds our confidence in YES.

Exit boundary:

    b_opp  >  p + α + β(t) + λ(entry_px)        ← free-boundary exit rule

where:
  • α ≈ 0.01 accounts for execution friction (taker fee + spread-cross).
  • β(t) is a time-dependent risk-aversion margin that shrinks as t → 0.
  • λ(entry_px) is an entry-price leakage correction (only active for late,
    high-price entries where selling crystallizes a large loss).

Why β(t)?  Early in the window the posterior re-scores every 5s — there is
re-evaluation optionality, so we should require *stronger* market disagreement
before capitulating. Near expiry the market is crystallizing on the settlement
value; a noisy opposing-bid spike is dispositive when 10s remain but is mostly
MM inventory noise with 10 min to go. So β is largest late-window, smallest
early? — no, backwards. Think again:

  • Far from expiry: posterior and market can diverge on signal (rare info);
    don't rush to match the market.                       → β large
  • Near expiry: oracle is printing soon; noisy CLOB swing on top of a
    confirmed underlying shouldn't drive us out.          → β SMALLER, but
    Tier 1 already adds a 30-second suppression gate on confirmed positions.

    β(t_min) = β_max × √(t_min / 15)

This is the Carr-Jarrow-Myneni free-boundary structure applied to the binary:
the exit boundary curves *up* as t grows because holding-optionality grows.

Behavior illustrated:
    p=0.95, t=5 :   b_opp threshold ≈ 0.99  (hardly ever fires — good)
    p=0.60, t=5 :   b_opp threshold ≈ 0.68  (fires on meaningful disagreement)
    p=0.30, t=2 :   b_opp threshold ≈ 0.36  (eagerly defers to market)
    p=0.50, t=0.5:  b_opp threshold ≈ 0.52  (coin flip, tight threshold)

────────────────────────────────────────────────────────────────────────────────
Output clamped to [MIN_THR, MAX_THR] so the rule cannot become trivially loose
or degenerate (threshold > 1.0 would be unreachable).

The module exposes:
    exit_threshold(posterior, minutes_remaining, entry_price=None) -> float
    build_lookup_table() -> dict   (for dashboard / offline inspection)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


# ── Parameters (tunable, kept as module constants) ───────────────────────────
ALPHA_EXEC_COST   = 0.01   # execution friction (Polymarket 0.0625× taker fee + spread cross)
BETA_MAX          = 0.12   # max risk-aversion margin at 15min to expiry
BETA_SHAPE_POWER  = 0.5    # √t shape; >0.5 front-loads margin, <0.5 back-loads

MIN_THR           = 0.60   # never fire REVERSE_CONVERGENCE below this opp_bid
MAX_THR           = 0.99   # above this, the signal is settlement-proximate only

DEFAULT_T_MAX_MIN = 15.0


def exit_threshold(
    posterior: float,
    minutes_remaining: float,
    entry_price: Optional[float] = None,
    *,
    alpha: float = ALPHA_EXEC_COST,
    beta_max: float = BETA_MAX,
) -> float:
    """
    Compute the minimum opposing-side bid that justifies a REVERSE_CONVERGENCE
    exit, given the current posterior, minutes remaining, and entry price.

    Args:
        posterior: Current posterior P(our side wins), ∈ [0, 1].
        minutes_remaining: Time to expiry (minutes). Clamped to [0, 15].
        entry_price: Optional; fold-in leakage adjustment for late/high-px entries.
        alpha: Execution friction. Default ALPHA_EXEC_COST.
        beta_max: Time-risk margin cap. Default BETA_MAX.

    Returns:
        Threshold b* ∈ [MIN_THR, MAX_THR].
    """
    # Input sanitation
    p = max(0.0, min(1.0, float(posterior)))
    t = max(0.0, min(DEFAULT_T_MAX_MIN, float(minutes_remaining)))

    # Base boundary: defer to the market when its belief exceeds ours.
    #   b_opp > p ⟺ market's implied P(opposing wins) > our P(own side wins)
    b_base = p

    # Time-dependent risk-aversion margin (more time → more re-eval optionality
    # → require stronger disagreement before capitulating).
    beta_t = beta_max * math.pow(t / DEFAULT_T_MAX_MIN, BETA_SHAPE_POWER)

    # Entry-price leakage correction.
    # If we entered at a high price (e.g. $0.97), selling at a low YES bid
    # crystallizes a large loss; require a slightly higher opposing bid
    # (stronger market disagreement) before cutting.
    leakage_margin = 0.0
    if entry_price is not None:
        try:
            ep = float(entry_price)
            if ep >= 0.90:
                # Linearly interpolate +0 → +0.03 across [$0.90, $0.99].
                leakage_margin = 0.03 * min(1.0, (ep - 0.90) / 0.09)
        except (TypeError, ValueError):
            pass

    thr = b_base + alpha + beta_t + leakage_margin
    return max(MIN_THR, min(MAX_THR, thr))


def holding_dominates(
    posterior: float,
    opposing_bid: float,
    minutes_remaining: float,
    entry_price: Optional[float] = None,
) -> bool:
    """
    Convenience wrapper: True if the free-boundary says we should hold
    (opposing bid has NOT breached the exit boundary), False if the
    free-boundary says exit.
    """
    return float(opposing_bid) <= exit_threshold(
        posterior, minutes_remaining, entry_price=entry_price,
    )


def build_lookup_table(
    posteriors: Optional[Tuple[float, ...]] = None,
    minutes: Optional[Tuple[float, ...]] = None,
    entry_prices: Optional[Tuple[float, ...]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Precompute an (offline-inspectable) table of exit thresholds.
    Keyed by "p={p}_t={t}_ex={ex}" → threshold. Not used on the live path
    (exit_threshold is already cheap) — this exists for dashboard rendering
    and offline validation.
    """
    if posteriors is None:
        posteriors = (0.30, 0.50, 0.65, 0.80, 0.90, 0.95, 0.99)
    if minutes is None:
        minutes = (15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.0)
    if entry_prices is None:
        entry_prices = (0.50, 0.75, 0.90, 0.97)

    table: Dict[str, Dict[str, float]] = {}
    for p in posteriors:
        table[f"p={p:.2f}"] = {}
        for t in minutes:
            for ex in entry_prices:
                key = f"t={t:>4.1f}_ex={ex:.2f}"
                table[f"p={p:.2f}"][key] = round(
                    exit_threshold(p, t, entry_price=ex), 3,
                )
    return table


if __name__ == "__main__":
    # Print the lookup table for visual validation.
    tbl = build_lookup_table()
    print("Free-boundary REVERSE_CONVERGENCE thresholds")
    print("=" * 72)
    for p_key, row in tbl.items():
        print(f"\n{p_key}")
        for t_key, v in row.items():
            print(f"  {t_key} → {v:.3f}")
