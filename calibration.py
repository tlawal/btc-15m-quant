"""
Tier 2 #9: Time-to-resolution bucketed isotonic calibration.

Backwards-compatible with the original single-model API:
- `calibrate(p)`                → uses global model (legacy)
- `calibrate(p, min_rem=2.5)`   → dispatches to the appropriate bucket model,
                                  falling back to the global model, then identity.

Buckets (minutes remaining at the time the posterior is evaluated):
- bucket_0_2    : < 2 min    (final-sprint, highest-accuracy regime)
- bucket_2_5    : 2–5 min    (entry sweet-spot)
- bucket_5_10   : 5–10 min   (mid-window)
- bucket_10plus : ≥ 10 min   (early)

Training:
  python calibration.py                   # trains all buckets + global, saves
  python calibration.py --dry-run         # reports stats without saving
  python calibration.py --source logs     # use logs/features.jsonl + logs/outcomes.jsonl
                                          # (default training source if jsonl absent)

The global model is preserved for legacy callers and as a fallback when a bucket
has too few samples (<100) to fit a reliable isotonic curve.

Empirically per Le (arXiv 2602.19520), prediction-market calibration is strongly
time-to-resolution-specific: final-minute posteriors are sharper than
mid-window posteriors on the same absolute-probability scale. A single global
calibrator systematically distorts both ends.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_GLOBAL_PATH = _HERE / "calibration_model.pkl"
_BUCKET_PATHS = {
    "bucket_0_2":    _HERE / "calibration_model_0_2.pkl",
    "bucket_2_5":    _HERE / "calibration_model_2_5.pkl",
    "bucket_5_10":   _HERE / "calibration_model_5_10.pkl",
    "bucket_10plus": _HERE / "calibration_model_10plus.pkl",
}

# Module-level cached models
_global_model = None
_bucket_models: Dict[str, object] = {}


# ───────────────────────── bucket dispatch ─────────────────────────

def _bucket_for(min_rem: Optional[float]) -> Optional[str]:
    if min_rem is None:
        return None
    try:
        m = float(min_rem)
    except (TypeError, ValueError):
        return None
    if m < 2.0:
        return "bucket_0_2"
    if m < 5.0:
        return "bucket_2_5"
    if m < 10.0:
        return "bucket_5_10"
    return "bucket_10plus"


# ───────────────────────── model loading ─────────────────────────

def load_calibration_model(path: Optional[str] = None) -> bool:
    """
    Load global + all bucketed models. Returns True iff at least one model
    loaded successfully. Missing bucket files are logged once at INFO.
    """
    global _global_model, _bucket_models
    _bucket_models = {}
    loaded_any = False
    try:
        import joblib
    except Exception as e:
        log.warning("calibration: joblib missing (%s) — calibration disabled", e)
        return False

    # Global (legacy)
    gp = Path(path) if path else _GLOBAL_PATH
    if gp.exists():
        try:
            _global_model = joblib.load(gp)
            log.info("calibration: loaded GLOBAL model from %s", gp)
            loaded_any = True
        except Exception as e:
            log.warning("calibration: failed to load global model: %s", e)
            _global_model = None
    else:
        log.info("calibration: no global model at %s", gp)

    # Buckets
    for name, p in _BUCKET_PATHS.items():
        if not p.exists():
            continue
        try:
            _bucket_models[name] = joblib.load(p)
            log.info("calibration: loaded %s from %s", name, p)
            loaded_any = True
        except Exception as e:
            log.warning("calibration: failed to load %s (%s)", name, e)

    if not loaded_any:
        log.info("calibration: no models loaded — calibration is identity")
    return loaded_any


def calibrate(p: float, min_rem: Optional[float] = None) -> float:
    """
    Calibrate a raw posterior. If `min_rem` is provided and the corresponding
    bucket model is loaded, uses it; otherwise falls back to the global model;
    otherwise returns `p` unchanged.
    """
    # Pick model
    model = None
    bucket = _bucket_for(min_rem)
    if bucket is not None:
        model = _bucket_models.get(bucket)
    if model is None:
        model = _global_model
    if model is None:
        return p

    try:
        import numpy as np
        result = model.predict(np.array([[float(p)]]))[0]
        return float(max(1e-6, min(1 - 1e-6, result)))
    except Exception:
        return p


# ───────────────────────── training ─────────────────────────

def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_training_pairs(source: str = "auto") -> List[Tuple[float, int, float]]:
    """
    Return list of (posterior, outcome, min_rem). Two sources:
    - 'jsonl' / 'auto' → remote_trade_features.jsonl (legacy format with 'outcome' field)
    - 'logs'           → logs/features.jsonl joined on window_start with logs/outcomes.jsonl
    """
    if source in ("auto", "jsonl"):
        fp = _HERE / "remote_trade_features.jsonl"
        if fp.exists():
            out = []
            for rec in _iter_jsonl(fp):
                feat = rec.get("features", rec)
                post = feat.get("posterior_final_up")
                outcome = rec.get("outcome")
                mr = feat.get("min_rem", feat.get("entry_min_rem"))
                if post is None or outcome is None:
                    continue
                try:
                    out.append((float(post), int(outcome), float(mr) if mr is not None else 7.0))
                except (ValueError, TypeError):
                    continue
            if out:
                log.info("calibration: loaded %d pairs from jsonl", len(out))
                return out

    # Fall through to logs
    if source in ("auto", "logs"):
        feat_p = _HERE / "logs" / "features.jsonl"
        out_p  = _HERE / "logs" / "outcomes.jsonl"
        outcomes = {}
        for rec in _iter_jsonl(out_p):
            ws = rec.get("window_start")
            bc = rec.get("btc_close")
            if ws is None or bc is None:
                continue
            outcomes[int(ws)] = float(bc)

        pairs: List[Tuple[float, int, float]] = []
        for rec in _iter_jsonl(feat_p):
            ws = rec.get("window") or rec.get("window_start")
            if ws is None:
                continue
            ws = int(ws)
            btc_close = outcomes.get(ws)
            if btc_close is None:
                continue
            strike = rec.get("strike")
            post = rec.get("posterior_final_up", rec.get("posterior_fair_up"))
            mr = rec.get("min_rem")
            if strike is None or post is None or mr is None:
                continue
            try:
                outcome = 1 if float(btc_close) > float(strike) else 0
                pairs.append((float(post), int(outcome), float(mr)))
            except (ValueError, TypeError):
                continue
        log.info("calibration: loaded %d pairs from logs/", len(pairs))
        return pairs

    return []


def _fit_isotonic(pairs: List[Tuple[float, int, float]]):
    """Fit + return an isotonic model and stats, or (None, stats) if too few samples."""
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    if len(pairs) < 100:
        return None, {"n_samples": len(pairs), "skipped": "too few samples"}

    X = np.array([p for p, _, _ in pairs])
    y = np.array([o for _, o, _ in pairs])

    brier_before = float(np.mean((X - y) ** 2))
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(X, y)
    pred = ir.predict(X)
    brier_after = float(np.mean((pred - y) ** 2))
    # Log-loss (avoid log(0))
    eps = 1e-9
    pc = np.clip(pred, eps, 1 - eps)
    log_loss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))

    return ir, {
        "n_samples": len(pairs),
        "brier_before": round(brier_before, 4),
        "brier_after": round(brier_after, 4),
        "improvement_pct": round((brier_before - brier_after) / max(1e-6, brier_before) * 100, 1),
        "log_loss": round(log_loss, 4),
        "win_rate": round(float(np.mean(y)), 3),
        "mean_posterior": round(float(np.mean(X)), 3),
    }


def train_and_save(
    data_path: str = "remote_trade_features.jsonl",
    output_path: Optional[str] = None,
    dry_run: bool = False,
    source: str = "auto",
) -> dict:
    """
    Train global + 4 bucketed isotonic models. Returns per-bucket stats.
    """
    pairs = _load_training_pairs(source=source)
    if len(pairs) < 100:
        raise ValueError(f"Insufficient training data: {len(pairs)} pairs (need ≥100)")

    # Global
    global_model, global_stats = _fit_isotonic(pairs)
    stats = {"global": global_stats, "buckets": {}}

    # Per-bucket
    by_bucket: Dict[str, List] = {k: [] for k in _BUCKET_PATHS}
    for p, o, mr in pairs:
        b = _bucket_for(mr)
        if b is not None:
            by_bucket[b].append((p, o, mr))

    bucket_models = {}
    for name, sub in by_bucket.items():
        model, bstats = _fit_isotonic(sub)
        if model is not None:
            bucket_models[name] = model
        stats["buckets"][name] = bstats

    # Save
    if not dry_run:
        import joblib
        if global_model is not None:
            out = Path(output_path) if output_path else _GLOBAL_PATH
            joblib.dump(global_model, out)
            log.info("calibration: saved GLOBAL → %s", out)
            stats["global"]["model_path"] = str(out)
        for name, model in bucket_models.items():
            p = _BUCKET_PATHS[name]
            joblib.dump(model, p)
            log.info("calibration: saved %s → %s", name, p)
            stats["buckets"][name]["model_path"] = str(p)

    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    dry = "--dry-run" in sys.argv
    source = "logs" if "--source" in sys.argv and "logs" in sys.argv else "auto"
    try:
        stats = train_and_save(dry_run=dry, source=source)
        print(f"\n{'DRY RUN' if dry else 'TRAINED'} — Calibration Results")
        print("=" * 60)
        g = stats["global"]
        if "skipped" in g:
            print(f"  GLOBAL: skipped ({g['skipped']}, n={g['n_samples']})")
        else:
            print(f"  GLOBAL  n={g['n_samples']:6d}  Brier {g['brier_before']:.4f} → {g['brier_after']:.4f}  "
                  f"({g['improvement_pct']:+.1f}%)  LogLoss={g['log_loss']:.4f}  WinRate={g['win_rate']:.3f}")
        print("  Buckets (time-to-resolution):")
        for name in ("bucket_0_2", "bucket_2_5", "bucket_5_10", "bucket_10plus"):
            bs = stats["buckets"].get(name, {})
            if "skipped" in bs:
                print(f"    {name:<16s}  skipped ({bs['skipped']}, n={bs.get('n_samples', 0)})")
            else:
                print(f"    {name:<16s}  n={bs['n_samples']:6d}  Brier {bs['brier_before']:.4f} → "
                      f"{bs['brier_after']:.4f}  ({bs['improvement_pct']:+.1f}%)  "
                      f"WinRate={bs['win_rate']:.3f}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
