"""
Tier 4 #17: SHAP-based sub-signal attribution on the retrained RF.

Institutional practice runs SHAP monthly on the production model so the
feature bag doesn't accumulate dead weight. Signals with persistently
negative mean |SHAP| contribution over a rolling 200-trade window are
flagged for disablement — a floor against the silent-signal-rot that
happens when markets change regime but the code doesn't.

Usage:
    from signal_shap import compute_shap_importance, persist_importance
    imp = compute_shap_importance()        # dict: feat -> {mean_abs, mean_signed, n}
    persist_importance(imp)                # writes logs/shap_importance.json

Dashboard integration: GET /api/shap reads the persisted file.

Dependencies:
- shap (optional). If unavailable, falls back to RF built-in feature_importances_
  (less principled — doesn't decompose by outcome direction — but keeps the
  monitor green on hosts without the extra dep).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

SHAP_LOG_PATH = Path(__file__).parent / "logs" / "shap_importance.json"
DEFAULT_MODEL_PATH = "optimizer_model.joblib"
DEFAULT_FEATURES_PATH = "trade_features.jsonl"
MIN_TRADES_FOR_SHAP = 50
ROLLING_WINDOW = 200
NEGATIVE_THRESHOLD_PCT = 0.15   # signal flagged if in bottom 15% by signed importance


def _load_model(model_path: str = DEFAULT_MODEL_PATH):
    try:
        import joblib
        if not Path(model_path).exists():
            # Accept either local or container-mounted path
            alt = f"/data/{Path(model_path).name}"
            if Path(alt).exists():
                model_path = alt
            else:
                return None
        return joblib.load(model_path)
    except Exception as e:
        log.debug("shap: load_model failed: %s", e)
        return None


def _load_features(features_path: str = DEFAULT_FEATURES_PATH, limit: int = ROLLING_WINDOW):
    p = Path(features_path)
    if not p.exists():
        alt = Path(f"/data/{p.name}")
        if alt.exists():
            p = alt
        else:
            return [], []
    try:
        with p.open("r") as f:
            lines = f.readlines()
    except Exception:
        return [], []
    lines = lines[-int(limit):] if limit else lines
    rows: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    feats = [r.get("features", r) for r in rows]
    outs  = [1 if (r.get("outcome") == "WIN" or r.get("outcome_win") == 1) else 0 for r in rows]
    return feats, outs


def compute_shap_importance(
    model_path: str = DEFAULT_MODEL_PATH,
    features_path: str = DEFAULT_FEATURES_PATH,
    window: int = ROLLING_WINDOW,
) -> dict:
    """
    Return a dict: { feature_name: {mean_abs, mean_signed, n}, ... }

    mean_abs:    rank-ordered feature importance (higher = more influential).
    mean_signed: positive = feature pushes toward WIN, negative = toward LOSS.
    n:           number of samples used.
    """
    model = _load_model(model_path)
    feats_list, outs = _load_features(features_path, limit=window)
    if model is None or not feats_list:
        return {
            "_meta": {
                "ts": int(time.time()),
                "status": "insufficient_data" if not feats_list else "no_model",
                "n": len(feats_list),
            },
            "features": {},
        }
    if len(feats_list) < MIN_TRADES_FOR_SHAP:
        return {
            "_meta": {
                "ts": int(time.time()),
                "status": "collecting",
                "n": len(feats_list),
                "need": MIN_TRADES_FOR_SHAP,
            },
            "features": {},
        }

    # Build the DataFrame in the same schema the model was trained on.
    try:
        import pandas as pd
        df = pd.DataFrame(feats_list).fillna(0.0)
    except Exception as e:
        log.debug("shap: dataframe build failed: %s", e)
        return {"_meta": {"ts": int(time.time()), "status": "df_failed"}, "features": {}}

    # Align to the model's expected features where possible.
    try:
        model_features = list(getattr(model, "feature_names_in_", []))
        if model_features:
            for col in model_features:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[model_features]
    except Exception:
        pass

    # Try SHAP first; fall back to RF built-in.
    result: dict[str, dict] = {}
    used_shap = False
    try:
        import shap   # type: ignore
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(df)
        # Binary classification: sv is a list [class0, class1]; use class1 (WIN).
        if isinstance(sv, list) and len(sv) >= 2:
            values = sv[1]
        else:
            values = sv
        import numpy as np
        mean_abs    = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)
        for i, col in enumerate(df.columns):
            result[col] = {
                "mean_abs":    float(mean_abs[i]),
                "mean_signed": float(mean_signed[i]),
                "n":           int(len(df)),
            }
        used_shap = True
    except Exception as e:
        log.debug("shap: shap library unavailable or failed: %s — falling back to RF importance", e)
        try:
            import numpy as np
            importances = getattr(model, "feature_importances_", None)
            if importances is None:
                return {"_meta": {"ts": int(time.time()), "status": "no_importances"}, "features": {}}
            for i, col in enumerate(df.columns):
                if i >= len(importances):
                    break
                result[col] = {
                    "mean_abs":    float(importances[i]),
                    "mean_signed": 0.0,    # RF built-in doesn't carry direction
                    "n":           int(len(df)),
                }
        except Exception as e2:
            log.debug("shap: RF fallback failed: %s", e2)
            return {"_meta": {"ts": int(time.time()), "status": "rf_fallback_failed"}, "features": {}}

    return {
        "_meta": {
            "ts":         int(time.time()),
            "status":     "ok",
            "n":          int(len(df)),
            "used_shap":  bool(used_shap),
        },
        "features": result,
    }


def flag_negative_signals(importance: dict, threshold_pct: float = NEGATIVE_THRESHOLD_PCT) -> list[str]:
    """
    Return the list of feature names whose signed contribution is in the bottom
    `threshold_pct` of the distribution. These are candidates for disabling.

    If SHAP wasn't used (mean_signed all zero), return [] — we can't direction.
    """
    feats = importance.get("features") or {}
    if not feats:
        return []
    signed = [(k, v.get("mean_signed", 0.0)) for k, v in feats.items()]
    signed_values = [s for _, s in signed]
    # If signed is a column of zeros we can't flag — RF fallback case.
    if not any(abs(s) > 1e-9 for s in signed_values):
        return []
    try:
        import numpy as np
        cutoff = float(np.quantile(signed_values, threshold_pct))
    except Exception:
        xs = sorted(signed_values)
        cutoff = xs[int(threshold_pct * (len(xs) - 1))] if xs else 0.0
    return sorted([k for k, s in signed if s <= cutoff and s < 0])


def persist_importance(importance: dict, path: Path = SHAP_LOG_PATH) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(importance, f, separators=(",", ":"))
        return True
    except Exception as e:
        log.debug("shap: persist failed: %s", e)
        return False


def load_persisted(path: Path = SHAP_LOG_PATH) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None
