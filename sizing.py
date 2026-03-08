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
    win_rate:       Optional[float] = None,
    profit_factor:  Optional[float] = None,
    kelly_multiplier: float = 1.0,
    book: Optional[object] = None,
) -> Optional[float]:
    """
    Returns position size in USD, or None if below minimum.
    Uses quarter-Kelly with tiered risk_pct cap.
    """
    # === MONSTER SIZING OVERRIDE ===
    is_monster = monster_signal and posterior >= 0.90

    if is_monster:
        # Scale monster floor: min($6, 90% of balance) — works at any balance level
        monster_floor = min(6.00, round(balance * 0.90, 2))
        if balance >= Config.MIN_TRADE_USD and monster_floor >= Config.MIN_TRADE_USD:
            log.info(f"MONSTER_FORCE_MIN: forced ${monster_floor:.2f} on 90%+ conviction (balance=${balance:.2f})")
            return monster_floor

    risk_pct = Config.get_risk_pct(balance)
    max_loss_usd = balance * risk_pct

    # Recalibrate Kelly using live performance if available
    # Modified Kelly: f* = p - (1-p)/b
    # If we have live profit_factor, use it as 'b'
    b = profit_factor if profit_factor and profit_factor > 0 else (1.0 - entry_price) / entry_price if entry_price > 0 else 1.0
    p = win_rate if win_rate and win_rate > 0 else posterior
    
    q = 1.0 - p
    full_kelly = max(0.0, (p * b - q) / b) if b > 0 else 0.0

    # True quarter-Kelly, scaled by Sharpe-calibrated multiplier
    quarter_kelly = full_kelly * 0.25 * kelly_multiplier

    # Position = fraction of bankroll to bet, capped at risk budget
    position_usd = balance * quarter_kelly
    
    # Phase 6: Institutional execution buffer (haircut)
    position_usd *= (1.0 - Config.SLIPPAGE_BUFFER_PCT)

    # Phase 6: Depth-aware limit (max 50% of top-of-book depth)
    if book and hasattr(book, 'bid_sz') and hasattr(book, 'ask_sz'):
        # bid_sz/ask_sz are in BTC; convert to USD
        # we care about the side we enter. for now just use a conservative min of both
        depth_btc = min(book.best_bid_sz, book.best_ask_sz)
        depth_usd = depth_btc * entry_price # approx
        depth_limit = depth_usd * 0.50 # cap at 50% of available book depth
        if position_usd > depth_limit:
            log.info(f"DEPTH_LIMIT: Capping ${position_usd:.2f} -> ${depth_limit:.2f} (50% of book depth)")
            position_usd = depth_limit

    position_usd = min(position_usd, max_loss_usd)        # Never exceed risk budget
    position_usd = min(position_usd, Config.MAX_TRADE_USD) # Absolute per-trade cap

    log.debug(f"sizing info: posterior={posterior:.4f} px={entry_price:.4f} bal={balance:.2f} riskPct={risk_pct:.4f} streak={loss_streak} -> {position_usd:.2f} usd")
    if position_usd < Config.MIN_TRADE_USD:
        return None
    if balance < Config.MIN_TRADE_USD:
        return None
    return round(position_usd, 2)
