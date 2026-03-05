import os
import json
import joblib
import pandas as pd
import logging

log = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, model_path="model.joblib", feature_names_path="feature_names.json"):
        self.model = None
        self.feature_names = None
        
        # Look in /data first (Railway) then local
        alt_model = f"/data/{model_path}"
        alt_feats = f"/data/{feature_names_path}"
        
        target_model = alt_model if os.path.exists(alt_model) else model_path
        target_feats = alt_feats if os.path.exists(alt_feats) else feature_names_path

        if os.path.exists(target_model) and os.path.exists(target_feats):
            try:
                self.model = joblib.load(target_model)
                with open(target_feats, 'r') as f:
                    self.feature_names = json.load(f)
                log.info(f"ML model loaded: {target_model}")
            except Exception as e:
                log.warning(f"Failed to load ML model: {e}")
        else:
            log.info("No ML model found. Bot will run on Bayesian rules only.")

    def predict_up_prob(self, features: dict) -> float:
        """Returns the probability of the window ending UP (1) according to the model."""
        if self.model is None or self.feature_names is None:
            return None
        
        try:
            # Construct a DataFrame with exactly the same features and order
            # as used during training
            row = {}
            for name in self.feature_names:
                row[name] = features.get(name, 0.0) # Fallback to 0 if signal missing
            
            df = pd.DataFrame([row])
            # RandomForest.predict_proba returns [prob_0, prob_1]
            probs = self.model.predict_proba(df)[0]
            return float(probs[1]) 
        except Exception as e:
            log.debug(f"Inference error: {e}")
            return None
