"""
Fix #5: Isotonic regression calibration for posterior probabilities.

Usage:
  Train:  python calibration.py                 # trains and saves model
  Test:   python calibration.py --dry-run       # trains, prints stats, doesn't save

The trained model is loaded at engine startup via load_calibration_model().
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "calibration_model.pkl"

# Module-level cached model
_model = None


def load_calibration_model(path: Optional[str] = None) -> bool:
    """Load a pre-trained isotonic regression model. Returns True if successful."""
    global _model
    p = Path(path) if path else _MODEL_PATH
    if not p.exists():
        log.info("calibration: no model at %s — skipping calibration", p)
        return False
    try:
        import joblib
        _model = joblib.load(p)
        log.info("calibration: loaded isotonic model from %s", p)
        return True
    except Exception as e:
        log.warning("calibration: failed to load model: %s", e)
        _model = None
        return False


def calibrate(p: float) -> float:
    """Apply isotonic calibration to a raw posterior probability.
    Returns the original value if no model is loaded."""
    if _model is None:
        return p
    try:
        import numpy as np
        result = _model.predict(np.array([[p]]))[0]
        return float(max(1e-6, min(1 - 1e-6, result)))
    except Exception:
        return p


def train_and_save(data_path: str = "remote_trade_features.jsonl",
                   output_path: Optional[str] = None,
                   dry_run: bool = False) -> dict:
    """Train isotonic regression on historical (posterior, outcome) pairs.

    Returns dict with training statistics.
    """
    import numpy as np

    posteriors = []
    outcomes = []
    data_file = Path(__file__).parent / data_path

    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found: {data_file}")

    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                feat = record.get("features", record)
                post = feat.get("posterior_final_up")
                outcome = record.get("outcome")
                if post is not None and outcome is not None:
                    posteriors.append(float(post))
                    outcomes.append(int(outcome))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    if len(posteriors) < 10:
        raise ValueError(f"Insufficient data: {len(posteriors)} samples (need >= 10)")

    X = np.array(posteriors).reshape(-1, 1)
    y = np.array(outcomes)

    # Brier score (before calibration)
    brier_before = float(np.mean((X.ravel() - y) ** 2))

    # Train isotonic regression
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(X.ravel(), y)

    # Brier score (after calibration)
    calibrated = ir.predict(X.ravel())
    brier_after = float(np.mean((calibrated - y) ** 2))

    # Calibration curve (5 bins)
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    cal_curve = []
    for i in range(n_bins):
        mask = (X.ravel() >= bin_edges[i]) & (X.ravel() < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_pred = float(np.mean(X.ravel()[mask]))
            mean_obs = float(np.mean(y[mask]))
            cal_curve.append({"bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                              "mean_predicted": round(mean_pred, 3),
                              "mean_observed": round(mean_obs, 3),
                              "count": int(mask.sum())})

    stats = {
        "n_samples": len(posteriors),
        "brier_before": round(brier_before, 4),
        "brier_after": round(brier_after, 4),
        "improvement_pct": round((brier_before - brier_after) / brier_before * 100, 1) if brier_before > 0 else 0,
        "calibration_curve": cal_curve,
        "win_rate": round(float(np.mean(y)), 3),
        "mean_posterior": round(float(np.mean(X)), 3),
    }

    if not dry_run:
        out = Path(output_path) if output_path else _MODEL_PATH
        import joblib
        joblib.dump(ir, out)
        log.info("calibration: saved model to %s", out)
        stats["model_path"] = str(out)

    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    dry = "--dry-run" in sys.argv
    try:
        stats = train_and_save(dry_run=dry)
        print(f"\n{'DRY RUN' if dry else 'TRAINED'} — Calibration Results:")
        print(f"  Samples:           {stats['n_samples']}")
        print(f"  Win Rate:          {stats['win_rate']:.1%}")
        print(f"  Mean Posterior:     {stats['mean_posterior']:.3f}")
        print(f"  Brier (before):    {stats['brier_before']:.4f}")
        print(f"  Brier (after):     {stats['brier_after']:.4f}")
        print(f"  Improvement:       {stats['improvement_pct']:.1f}%")
        print("\n  Calibration Curve:")
        for b in stats.get("calibration_curve", []):
            print(f"    {b['bin']:>9s}  pred={b['mean_predicted']:.3f}  obs={b['mean_observed']:.3f}  n={b['count']}")
        if not dry:
            print(f"\n  Model saved to: {stats.get('model_path', 'N/A')}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
