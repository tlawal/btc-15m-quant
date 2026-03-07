import logging
from config import Config

log = logging.getLogger(__name__)

class StrategyOptimizer:
    def __init__(self, state):
        self.state = state
        self.score_offset = 0.0
        self.edge_offset = 0.0

    def detect_decay(self, window_size: int = 10):
        """
        Detect signal decay by looking at the last N trades.
        If win rate is below 40%, increase selectivity.
        """
        closed_trades = [t for t in self.state.trade_history if t.outcome in ("WIN", "LOSS")]
        if len(closed_trades) < window_size:
            return 0.0, 0.0

        recent = closed_trades[-window_size:]
        wins = sum(1 for t in recent if t.outcome == "WIN")
        win_rate = wins / window_size

        if win_rate < 0.40:
            log.warning(f"Signal decay detected: win_rate={win_rate*100:.1f}% over last {window_size} trades.")
            # Increase selectivity: +1 to min_score, +0.01 to required_edge
            self.score_offset += 1.0
            self.edge_offset += 0.010
            return self.score_offset, self.edge_offset
        
        # Slowly decay the offsets back to 0 if performance is good
        if win_rate > 0.60:
            self.score_offset = max(0.0, self.score_offset - 0.5)
            self.edge_offset = max(0.0, self.edge_offset - 0.005)

        return self.score_offset, self.edge_offset

    def get_adjusted_thresholds(self, base_score: float, base_edge: float):
        """Return thresholds adjusted by decay detection."""
        return base_score + self.score_offset, base_edge + self.edge_offset
