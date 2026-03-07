"""
Phase 4: GBM Model Training with Walk-Forward Validation

Trains a LightGBM (or XGBoost fallback) classifier on signal features
to predict whether BTC closes above or below the 15m window strike.

Walk-forward: trains on data up to day N, validates on day N+1, repeating
across the entire dataset to simulate real production performance.
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import logging

log = logging.getLogger(__name__)

# Try LightGBM first, fall back to sklearn GBM
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss


def load_data(log_dir="./logs"):
    feats_path = f"{log_dir}/features.jsonl"
    outs_path = f"{log_dir}/outcomes.jsonl"

    if not os.path.exists(feats_path) or not os.path.exists(outs_path):
        print("Logs not found. Ensure bot has run for at least one full window.")
        return None, None

    feats = []
    with open(feats_path, 'r') as f:
        for line in f:
            try:
                feats.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df_feats = pd.DataFrame(feats)

    outs = []
    with open(outs_path, 'r') as f:
        for line in f:
            try:
                outs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df_outs = pd.DataFrame(outs)

    return df_feats, df_outs


def load_db_data(db_url: str = None):
    """Load training data from SQLite closed_trades table."""
    from sqlalchemy import create_engine, text
    url = db_url or os.getenv("DATABASE_URL", "sqlite:///./state.db")
    # Convert async URL to sync for training
    url = url.replace("sqlite+aiosqlite:", "sqlite:")
    engine = create_engine(url)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT features, outcome_win, regime, pnl_usd FROM closed_trades WHERE features IS NOT NULL"
        )).fetchall()

    if not rows:
        return None, None

    records = []
    targets = []
    for r in rows:
        try:
            feats = json.loads(r.features)
            records.append(feats)
            targets.append(r.outcome_win)
        except (json.JSONDecodeError, TypeError):
            continue

    return pd.DataFrame(records), pd.Series(targets, name="target")


def preprocess(df_feats, df_outs):
    df = pd.merge(df_feats, df_outs, left_on="window", right_on="window_start", how="inner")
    df['target'] = (df['btc_close'] > df['strike']).astype(int)

    drop_cols = ['ts', 'window', 'window_start', 'btc_close', 'ts_logged', 'target',
                 'strike', 'btc_price', 'min_rem']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['target']

    return X, y


def walk_forward_validate(X, y, n_splits=5):
    """Walk-forward cross-validation: train on past, test on future."""
    n = len(X)
    fold_size = n // (n_splits + 1)
    results = []

    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end:
            break

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        if len(y_train.unique()) < 2 or len(y_test) < 5:
            continue

        model = _build_model(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        results.append({"fold": i, "accuracy": acc, "log_loss": ll, "n_test": len(y_test)})
        print(f"  Fold {i}: acc={acc:.3f}  log_loss={ll:.3f}  n={len(y_test)}")

    return results


def _build_model(X_train, y_train):
    """Build a GBM model — LightGBM if available, else sklearn."""
    if HAS_LGBM:
        dtrain = lgb.Dataset(X_train, label=y_train)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 15,
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        return model

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train(use_db=True):
    log_dir = "/data" if os.path.exists("/data") else "./logs"

    X, y = None, None
    if use_db:
        X, y = load_db_data()

    if X is None:
        df_feats, df_outs = load_data(log_dir)
        if df_feats is None:
            return
        X, y = preprocess(df_feats, df_outs)

    if len(X) < 50:
        print(f"Dataset too small ({len(X)} rows). Need >= 50 samples.")
        return

    # Clean up features
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    feature_names = list(X.columns)

    print(f"Training on {len(X)} samples with {len(feature_names)} features...")
    print(f"Class balance: {y.value_counts().to_dict()}")
    print(f"Using {'LightGBM' if HAS_LGBM else 'sklearn GBM'}")

    # Walk-forward validation
    print("\n── Walk-Forward Validation ──")
    wf_results = walk_forward_validate(X, y)
    if wf_results:
        avg_acc = np.mean([r["accuracy"] for r in wf_results])
        avg_ll = np.mean([r["log_loss"] for r in wf_results])
        print(f"  Average: acc={avg_acc:.3f}  log_loss={avg_ll:.3f}")

    # Train final model on all data
    print("\n── Final Model ──")
    model = _build_model(X, y)

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        imp = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
        print("\nTop 10 Features:")
        for name, importance in imp[:10]:
            print(f"  {name:30s} {importance:.4f}")

    # Save
    model_path = f"{log_dir}/model.joblib"
    joblib.dump(model, model_path)
    with open(f"{log_dir}/feature_names.json", 'w') as f:
        json.dump(feature_names, f)

    # Save walk-forward results
    with open(f"{log_dir}/wf_results.json", 'w') as f:
        json.dump(wf_results, f)

    print(f"\nModel saved to {model_path}")
    print(f"Feature names saved ({len(feature_names)} features)")

    return model, feature_names


if __name__ == "__main__":
    train()
