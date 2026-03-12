"""
Phase 4: GBM Model Training with Walk-Forward Validation

Trains a LightGBM (or XGBoost fallback) classifier on signal features
to predict whether BTC closes above or below the 15m window strike.

Walk-forward: trains on data up to day N, validates on day N+1, repeating
across the entire dataset to simulate real production performance.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import logging

log = logging.getLogger(__name__)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss


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


def load_trade_features_jsonl(path: str):
    if not os.path.exists(path):
        return None, None

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            feats = item.get("features") or {}
            if not isinstance(feats, dict):
                continue
            outcome = item.get("outcome")
            if outcome is None:
                continue
            try:
                outcome = int(outcome)
            except Exception:
                continue
            row = feats.copy()
            row["target"] = outcome
            rows.append(row)

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    if "target" not in df.columns:
        return None, None
    X = df.drop(columns=["target"]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["target"].astype(int)
    return X, y


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
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    return model


def train(*, use_db: bool = True, prefer_trade_features: bool = True):
    log_dir = "/data" if os.path.exists("/data") else "."

    X, y = None, None

    if prefer_trade_features:
        trade_path = f"{log_dir}/trade_features.jsonl" if log_dir != "." else "trade_features.jsonl"
        X, y = load_trade_features_jsonl(trade_path)
        if X is None:
            log.info(f"No usable training rows found in {trade_path}")

    if X is None and use_db:
        X, y = load_db_data()

    if X is None:
        df_feats, df_outs = load_data("./logs")
        if df_feats is None:
            print("No training data available.")
            print("If you're on Railway and /data/trade_features.jsonl is empty, generate seed data with:")
            print("  python backtest_from_logs.py railway_logs.txt --seed-trade-features")
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
    print("Using RandomForestClassifier")

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
    model_path = f"{log_dir}/model.joblib" if log_dir != "." else "model.joblib"
    feats_path = f"{log_dir}/feature_names.json" if log_dir != "." else "feature_names.json"
    joblib.dump(model, model_path)
    with open(feats_path, 'w') as f:
        json.dump(feature_names, f)

    optimizer_model_path = f"{log_dir}/optimizer_model.joblib" if log_dir != "." else "optimizer_model.joblib"
    try:
        joblib.dump(model, optimizer_model_path)
    except Exception:
        pass

    # Save walk-forward results
    wf_path = f"{log_dir}/wf_results.json" if log_dir != "." else "wf_results.json"
    with open(wf_path, 'w') as f:
        json.dump(wf_results, f)

    print(f"\nModel saved to {model_path}")
    print(f"Feature names saved to {feats_path} ({len(feature_names)} features)")

    return model, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-db", action="store_true", help="Disable DB training source")
    parser.add_argument("--no-trade-features", action="store_true", help="Disable trade_features.jsonl training source")
    args = parser.parse_args()

    train(use_db=not args.no_db, prefer_trade_features=not args.no_trade_features)
