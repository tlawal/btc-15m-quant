import json
import os
import logging
import time
import pandas as pd
import joblib
from collections import defaultdict
from config import Config
from sklearn.ensemble import RandomForestClassifier

log = logging.getLogger("optimizer")

class SignalOptimizer:
    def __init__(self, state):
        self.state = state
        self.score_offset = 0.0
        self.edge_offset = 0.0
        self._kelly_multiplier = 1.0
        self._pnl_history = []
        self._last_optimize_ts = 0
        
        # Paths
        self.features_path = "/data/trade_features.jsonl" if os.path.exists("/data") else "trade_features.jsonl"
        self.model_path = "/data/optimizer_model.joblib" if os.path.exists("/data") else "optimizer_model.joblib"

    def log_trade(self, sig, outcome: str):
        """
        Logs a trade's features and its outcome (WIN/LOSS) to a JSONL file.
        outcome: "WIN" or "LOSS"
        """
        if not sig:
            return

        try:
            feats = sig.to_feature_dict()
            entry = {
                "ts": int(time.time()),
                "outcome": 1 if outcome == "WIN" else 0,
                "features": feats
            }
            with open(self.features_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            log.info(f"Logged trade features to {self.features_path}")
        except Exception as e:
            log.error(f"Failed to log trade features: {e}")

    def retrain_and_adjust(self):
        """
        Reloads trade history, trains a RandomForest, and adjusts offsets 
        based on feature importance and model precision.
        """
        if not os.path.exists(self.features_path):
            return

        try:
            data = []
            with open(self.features_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            
            if len(data) < 20: # Need a minimum sample size
                log.info(f"Not enough trade data for retraining: {len(data)}/20")
                return

            # Flatten features for DataFrame
            rows = []
            for entry in data:
                row = entry["features"].copy()
                row["target"] = entry["outcome"]
                rows.append(row)
            
            df = pd.DataFrame(rows)
            X = df.drop(columns=["target"])
            y = df["target"]

            # Train a shallow RF to find "what works"
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X, y)

            # Save model
            joblib.dump(model, self.model_path)

            # Evaluate precision
            preds = model.predict(X)
            # Simple heuristic: if overall precision is high, we can relax. If low, tighten.
            from sklearn.metrics import precision_score
            precision = precision_score(y, preds, zero_division=0)
            
            log.info(f"Retrained optimizer model. Training Precision: {precision:.2f}")

            # Adjust thresholds
            if precision < 0.55:
                # Tighten: increase required thresholds
                self.score_offset = min(self.score_offset + 0.5, 2.5)
                self.edge_offset = min(self.edge_offset + 0.005, 0.02)
                log.warning(f"Low precision ({precision:.2f}) -> Tightening thresholds (score_off={self.score_offset}, edge_off={self.edge_offset})")
            elif precision > 0.75:
                # Relax: allow more trades
                self.score_offset = max(self.score_offset - 0.25, -1.0)
                self.edge_offset = max(self.edge_offset - 0.002, -0.01)
                log.info(f"High precision ({precision:.2f}) -> Relaxing thresholds (score_off={self.score_offset}, edge_off={self.edge_offset})")

        except Exception as e:
            log.error(f"Retraining failed: {e}")

    def get_adjusted_thresholds(self, base_score: float, base_edge: float):
        """Return thresholds adjusted by learned offsets."""
        return base_score + self.score_offset, base_edge + self.edge_offset

    def record_trade_pnl(self, pnl_pct: float):
        """PnL history for Kelly scaling."""
        self._pnl_history.append(pnl_pct)
        if len(self._pnl_history) > 100:
            self._pnl_history = self._pnl_history[-100:]
        
        # Periodic recalibration
        if len(self._pnl_history) % 5 == 0:
            self._recalibrate_kelly()

    def _recalibrate_kelly(self):
        """Adjust Kelly fraction based on realized Sharpe."""
        if len(self._pnl_history) < 10:
            return

        pnls = self._pnl_history
        mean_pnl = sum(pnls) / len(pnls)
        std = pd.Series(pnls).std() or 1e-6
        sharpe = (mean_pnl / std) * (96 ** 0.5) # ~96 cycles per day approx

        if sharpe < 0:
            self._kelly_multiplier = 0.4
        elif sharpe < 0.5:
            self._kelly_multiplier = 0.7
        else:
            self._kelly_multiplier = 1.0
        
        log.info(f"Kelly multiplier adjusted to {self._kelly_multiplier:.2f} (Sharpe: {sharpe:.2f})")

    def get_kelly_multiplier(self) -> float:
        return self._kelly_multiplier

    def get_signal_accuracies(self):
        # Legacy compat for dashboard
        return {}

    def get_disabled_signals(self):
        # Legacy compat for dashboard
        return []
