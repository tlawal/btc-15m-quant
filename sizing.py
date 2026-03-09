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
    edge: Optional[float] = None,
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

    # Never trade on negative edge — even for monster signals
    if edge is not None and edge < 0:
        log.info(f"sizing: negative edge ({edge:.4f}) — skipping position")
        return None

    # === LOW-BALANCE CONVICTION FLOOR ===
    # At small bankrolls, quarter-Kelly produces sub-minimum sizes even on good signals.
    # When balance < LOW_BALANCE_THRESHOLD_USD and the signal has meaningful positive edge
    # (>= REQUIRED_EDGE_LOW_BALANCE) and reasonable conviction (posterior >= 0.55),
    # force the minimum Polymarket-viable trade size so the bot can actually execute.
    # This is capped at 85% of balance to avoid risking the entire wallet on one trade.
    if (
        balance < Config.LOW_BALANCE_THRESHOLD_USD
        and edge is not None
        and edge >= Config.REQUIRED_EDGE_LOW_BALANCE
        and posterior >= 0.55
        and balance >= Config.MIN_TRADE_USD
    ):
        # Use MIN_TRADE_USD directly — the 85% cap created a deadlock at balances
        # between $5.75 and $6.76 where min(5.75, bal*0.85) < 5.75
        low_bal_floor = Config.MIN_TRADE_USD
        if balance >= Config.MIN_TRADE_USD:
            log.info(
                f"LOW_BAL_FLOOR: forced ${low_bal_floor:.2f} (bal=${balance:.2f} "
                f"edge={edge:.4f} post={posterior:.3f})"
            )
            return low_bal_floor

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
    if book and hasattr(book, 'total_bid_size') and hasattr(book, 'total_ask_size'):
        depth_usd = min(book.total_bid_size, book.total_ask_size)
        depth_limit = depth_usd * 0.50 # cap at 50% of available book depth
        if depth_limit > 0 and position_usd > depth_limit:
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
