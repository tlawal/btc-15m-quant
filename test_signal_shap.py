"""
Tier 4 #17: Tests for signal_shap.

No live shap dependency required — fallback path uses RF importances.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_log(monkeypatch, tmp_path):
    import signal_shap as ss
    monkeypatch.setattr(ss, "SHAP_LOG_PATH", tmp_path / "shap.json")
    yield


def _make_model_and_features(tmp_path, n=60):
    """Train a tiny RF on synthetic features for the fallback path."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    rng = np.random.RandomState(0)
    cols = ["ema_score", "vwap_score", "rsi_score", "macd_score"]
    X = pd.DataFrame(rng.randn(n, 4), columns=cols)
    # Make ema_score predictive of outcome.
    y = (X["ema_score"] > 0).astype(int).values
    m = RandomForestClassifier(n_estimators=10, random_state=42)
    m.fit(X, y)
    mp = tmp_path / "optimizer_model.joblib"
    joblib.dump(m, mp)

    fp = tmp_path / "trade_features.jsonl"
    lines = []
    for i in range(n):
        rec = {
            "features": {c: float(X.iloc[i][c]) for c in cols},
            "outcome":  "WIN" if y[i] == 1 else "LOSS",
        }
        lines.append(json.dumps(rec))
    fp.write_text("\n".join(lines))
    return str(mp), str(fp)


class TestComputeSHAP:
    def test_insufficient_data_returns_collecting(self, tmp_path):
        import signal_shap as ss
        fp = tmp_path / "features.jsonl"
        fp.write_text("")
        out = ss.compute_shap_importance(
            model_path=str(tmp_path / "no_model.joblib"),
            features_path=str(fp),
        )
        assert out["features"] == {}
        assert out["_meta"]["status"] in ("insufficient_data", "no_model")

    def test_fallback_to_rf_when_shap_missing(self, tmp_path, monkeypatch):
        """Even without shap, we should get RF importances back."""
        import signal_shap as ss

        mp, fp = _make_model_and_features(tmp_path)
        # Force shap import to fail by aliasing it to a broken module.
        import sys
        monkeypatch.setitem(sys.modules, "shap", None)

        out = ss.compute_shap_importance(model_path=mp, features_path=fp)
        assert out["_meta"]["status"] == "ok"
        assert out["_meta"]["used_shap"] is False
        feats = out["features"]
        assert "ema_score" in feats
        # ema_score was the predictive feature — should have the highest importance.
        top = max(feats.items(), key=lambda kv: kv[1]["mean_abs"])
        assert top[0] == "ema_score"

    def test_persist_and_load(self, tmp_path):
        import signal_shap as ss
        payload = {"_meta": {"status": "ok"}, "features": {"a": {"mean_abs": 0.5}}}
        assert ss.persist_importance(payload) is True
        loaded = ss.load_persisted()
        assert loaded == payload


class TestFlagNegative:
    def test_empty_returns_empty(self):
        import signal_shap as ss
        assert ss.flag_negative_signals({"features": {}}) == []

    def test_all_zero_signed_returns_empty(self):
        """RF fallback gives all signed=0 → we cannot direction, must not flag."""
        import signal_shap as ss
        imp = {
            "features": {
                "a": {"mean_abs": 0.2, "mean_signed": 0.0},
                "b": {"mean_abs": 0.1, "mean_signed": 0.0},
            }
        }
        assert ss.flag_negative_signals(imp) == []

    def test_flags_bottom_quantile(self):
        import signal_shap as ss
        imp = {
            "features": {
                "good":   {"mean_abs": 0.5, "mean_signed": +0.3},
                "ok":     {"mean_abs": 0.3, "mean_signed": +0.1},
                "bad":    {"mean_abs": 0.4, "mean_signed": -0.2},
                "worst":  {"mean_abs": 0.6, "mean_signed": -0.5},
            }
        }
        flagged = ss.flag_negative_signals(imp, threshold_pct=0.5)
        assert "worst" in flagged
        assert "good" not in flagged
