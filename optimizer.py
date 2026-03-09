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
        self.exit_log_path = "/data/exit_outcomes.jsonl" if os.path.exists("/data") else "exit_outcomes.jsonl"

    # ── Phase A: Exit outcome logging ─────────────────────────────────────────

    def log_exit_attempt(
        self,
        *,
        exit_reason: str,
        held_side: str,
        entry_price: float,
        current_price: float,
        entry_posterior: float,
        current_posterior: float,
        minutes_remaining: float,
        hold_seconds: float,
        signed_score: float,
        entry_score: float,
        market_slug: str = "",
        window_ts: int = 0,
    ) -> str:
        """
        Log an exit event to exit_outcomes.jsonl.
        Returns the record ID so the caller can fill in settlement outcome later.
        The 'settlement_itm' field is written as None and filled at window roll.
        """
        unrealized_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        record_id = f"{int(time.time())}_{exit_reason}"
        record = {
            "id":                record_id,
            "ts":                int(time.time()),
            "window_ts":         window_ts,
            "market_slug":       market_slug,
            "exit_reason":       exit_reason,
            "held_side":         held_side,
            "entry_price":       round(entry_price, 4),
            "exit_price":        round(current_price, 4),
            "unrealized_pct":    round(unrealized_pct, 4),
            "entry_posterior":   round(entry_posterior, 4) if entry_posterior is not None else None,
            "exit_posterior":    round(current_posterior, 4) if current_posterior is not None else None,
            "minutes_remaining": round(minutes_remaining, 2),
            "hold_seconds":      round(hold_seconds, 1),
            "signed_score":      round(signed_score, 3),
            "entry_score":       round(entry_score, 3),
            "settlement_itm":    None,  # filled in at window roll
        }
        try:
            with open(self.exit_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            log.warning(f"log_exit_attempt write error: {e}")
        return record_id

    def fill_exit_settlement(self, window_ts: int, settled_itm: bool):
        """
        After window resolution, go through exit_outcomes.jsonl and fill
        settlement_itm=True/False for all records with matching window_ts
        that still have settlement_itm=None.
        Rewrites the file in place.
        """
        if not os.path.exists(self.exit_log_path):
            return
        try:
            records = []
            updated = 0
            with open(self.exit_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("window_ts") == window_ts and r.get("settlement_itm") is None:
                        r["settlement_itm"] = settled_itm
                        updated += 1
                    records.append(r)
            if updated:
                with open(self.exit_log_path, "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
                log.info(f"fill_exit_settlement: updated {updated} records for window_ts={window_ts} itm={settled_itm}")
        except Exception as e:
            log.warning(f"fill_exit_settlement error: {e}")

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
            
            if len(data) < 10: # Need a minimum sample size
                log.info(f"Not enough trade data for retraining: {len(data)}/10")
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

    def get_signal_accuracies(self) -> dict:
        """Read trade_features.jsonl and compute rolling 7-day accuracy per signal."""
        cutoff = time.time() - 7 * 86400
        signal_keys = ["ofi_score", "cvd_score", "flow_accel_score", "imbalance_score", "signed_score"]
        buckets: dict[str, list[int]] = {k: [] for k in signal_keys}
        try:
            if not os.path.exists(self.features_path):
                return {}
            with open(self.features_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if entry.get("ts", 0) < cutoff:
                        continue
                    outcome = entry.get("outcome", 0)
                    feats = entry.get("features", {})
                    for k in signal_keys:
                        if k in feats:
                            # A signal "contributed correctly" if its sign matches the outcome
                            val = feats[k]
                            if val is None:
                                continue
                            correct = 1 if (val > 0 and outcome == 1) or (val < 0 and outcome == 0) else 0
                            buckets[k].append(correct)
        except Exception as e:
            log.warning(f"get_signal_accuracies error: {e}")
            return {}
        return {k: round(sum(v) / len(v), 4) for k, v in buckets.items() if len(v) >= 5}

    def get_disabled_signals(self) -> list:
        """Return signal names with <45% accuracy over 20+ samples in last 7 days."""
        cutoff = time.time() - 7 * 86400
        signal_keys = ["ofi_score", "cvd_score", "flow_accel_score", "imbalance_score"]
        buckets: dict[str, list[int]] = {k: [] for k in signal_keys}
        try:
            if not os.path.exists(self.features_path):
                return []
            with open(self.features_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if entry.get("ts", 0) < cutoff:
                        continue
                    outcome = entry.get("outcome", 0)
                    feats = entry.get("features", {})
                    for k in signal_keys:
                        if k in feats and feats[k] is not None:
                            val = feats[k]
                            correct = 1 if (val > 0 and outcome == 1) or (val < 0 and outcome == 0) else 0
                            buckets[k].append(correct)
        except Exception as e:
            log.warning(f"get_disabled_signals error: {e}")
            return []
        disabled = []
        for k, v in buckets.items():
            if len(v) >= 20 and (sum(v) / len(v)) < 0.45:
                disabled.append(k)
                log.warning(f"AUTO-DISABLE signal {k}: accuracy {sum(v)/len(v):.2%} over {len(v)} samples")
        return disabled
