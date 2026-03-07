"""
Kelly criterion and position sizing math.
Extracted from logic.py during Phase 3 refactor.
"""

import logging
from typing import Optional
from config import Config

log = logging.getLogger(__name__)

def compute_position_size(
    *,
    posterior:      float,
    entry_price:    float,
    balance:        float,
    loss_streak:    int,
    monster_signal: bool = False,
) -> Optional[float]:
    """
    Returns position size in USD, or None if below minimum.
    Uses quarter-Kelly with tiered risk_pct cap.
    """
    # === MONSTER SIZING OVERRIDE ===
    is_monster = monster_signal and posterior >= 0.90

    if is_monster and balance >= 6.00:
        log.info(f"MONSTER_FORCE_MIN: forced $6.00 on 90%+ conviction (balance=${balance:.2f})")
        return 6.00

    risk_pct = Config.get_risk_pct(balance)
    max_loss_usd = balance * risk_pct

    # Streak de-risk: halve after 2 consecutive losses
    if Config.STREAK_HALVE and loss_streak >= 2:
        max_loss_usd = max(max_loss_usd * 0.5, Config.MIN_TRADE_USD)

    # Kelly fraction: f* = (p*b - q) / b  where b = (1-price)/price
    b = (1.0 - entry_price) / entry_price if entry_price > 0 else 1.0
    q = 1.0 - posterior
    full_kelly = max(0.0, (posterior * b - q) / b)

    # True quarter-Kelly
    quarter_kelly = full_kelly * 0.25

    # Position = fraction of bankroll to bet, capped at risk budget
    position_usd = balance * quarter_kelly
    position_usd = min(position_usd, max_loss_usd)        # Never exceed risk budget
    position_usd = min(position_usd, Config.MAX_TRADE_USD) # Absolute per-trade cap

    log.debug(f"sizing info: posterior={posterior:.4f} px={entry_price:.4f} bal={balance:.2f} riskPct={risk_pct:.4f} streak={loss_streak} -> {position_usd:.2f} usd")
    if position_usd < Config.MIN_TRADE_USD:
        return None
    if balance < Config.MIN_TRADE_USD:
        return None
    return round(position_usd, 2)
