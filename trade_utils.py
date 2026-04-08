def compute_auto_settle_outcome(
    *,
    entry_price: float,
    entry_size: float,
    partial_exits: list[dict],
    position_won: bool,
) -> dict:
    partial_recovered = sum(float(p.get("price", 0.0) or 0.0) * float(p.get("size", 0.0) or 0.0) for p in (partial_exits or []))
    partial_size = sum(float(p.get("size", 0.0) or 0.0) for p in (partial_exits or []))
    remaining_size = max(0.0, float(entry_size or 0.0) - partial_size)
    total_cost = float(entry_price or 0.0) * float(entry_size or 0.0) if entry_price and entry_size else 0.0
    settle_unit_px = 1.0 if position_won else 0.0
    settle_recovered = remaining_size * settle_unit_px
    total_recovered = partial_recovered + settle_recovered
    blended_exit_price = total_recovered / entry_size if entry_size else 0.0
    blended_pnl = (total_recovered / total_cost - 1.0) if total_cost > 0 else 0.0
    outcome = "WIN" if blended_pnl >= 0 else "LOSS"
    return {
        "partial_recovered": partial_recovered,
        "partial_size": partial_size,
        "remaining_size": remaining_size,
        "total_cost": total_cost,
        "settle_unit_px": settle_unit_px,
        "settle_recovered": settle_recovered,
        "total_recovered": total_recovered,
        "blended_exit_price": blended_exit_price,
        "blended_pnl": blended_pnl,
        "outcome": outcome,
        "exit_reason": "AUTO_SETTLE_WIN" if outcome == "WIN" else "AUTO_SETTLE_LOSS",
        "pnl_usd": total_recovered - total_cost,
    }
