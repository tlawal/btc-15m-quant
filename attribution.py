import json
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import text
from state import StateManager

log = logging.getLogger(__name__)

class FeatureAttributor:
    def __init__(self, state_mgr: StateManager):
        self.state_mgr = state_mgr

    async def run_attribution(self, min_trades: int = 20):
        """Perform logistic regression on closed trades features to find signal importance."""
        async with self.state_mgr._session_factory() as session:
            # Query trades which have features logged (Phase 4+)
            result = await session.execute(text(
                "SELECT outcome_win, features FROM closed_trades WHERE features IS NOT NULL"
            ))
            rows = result.fetchall()

        if len(rows) < min_trades:
            log.info(f"Feature attribution skipped: only {len(rows)} trades with features (min {min_trades})")
            return None

        # Parse features
        data = []
        for row in rows:
            feats = json.loads(row.features)
            feats['target'] = row.outcome_win
            data.append(feats)

        df = pd.DataFrame(data)
        
        # Preprocess: drop target and handle Nones
        X = df.drop(columns=['target']).fillna(0)
        y = df['target']

        if len(y.unique()) < 2:
            log.warning("Feature attribution skipped: only one class present in outcome history")
            return None

        # Fit logistic regression
        model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
        model.fit(X, y)

        # Map coefficients to feature names
        coeffs = dict(zip(X.columns, model.coef_[0]))
        
        # Sort by absolute importance
        coeffs = dict(sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True))
        
        log.info(f"Feature attribution complete. Top signal: {next(iter(coeffs))}")
        return coeffs

    def suggest_threshold_caps(self, coeffs: dict, base_thresholds: dict) -> dict:
        """Suggest adjustments to thresholds based on signal strength (±20% max)."""
        adjustments = {}
        for feat, weight in coeffs.items():
            # Example: if 'rsi_score' has high positive coefficient, maybe lower its threshold?
            # Or if it has negative coefficient, definitely increase its threshold.
            pass
        return adjustments
