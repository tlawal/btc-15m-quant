import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_data(log_dir="./logs"):
    feats_path = f"{log_dir}/features.jsonl"
    outs_path = f"{log_dir}/outcomes.jsonl"
    
    if not os.path.exists(feats_path) or not os.path.exists(outs_path):
        print("Logs not found. Ensure bot has run for at least one full window.")
        return None, None

    # Load features
    feats = []
    with open(feats_path, 'r') as f:
        for line in f:
            feats.append(json.loads(line))
    df_feats = pd.DataFrame(feats)
    
    # Load outcomes
    outs = []
    with open(outs_path, 'r') as f:
        for line in f:
            outs.append(json.loads(line))
    df_outs = pd.DataFrame(outs)
    
    return df_feats, df_outs

def preprocess(df_feats, df_outs):
    # Join features with outcomes based on window_start
    # Each feature cycle belongs to exactly one window
    df = pd.merge(df_feats, df_outs, left_on="window", right_on="window_start", how="inner")
    
    # Label: 1 if outcome corresponds to direction, else 0
    # In features.jsonl we have btc_price (at cycle time) and strike
    # In outcomes.jsonl we have btc_close (at window end)
    
    # Target: 1 if btc_close > strike, 0 if btc_close <= strike
    df['target'] = (df['btc_close'] > df['strike']).astype(int)
    
    # Features to use
    drop_cols = ['ts', 'window', 'window_start', 'btc_close', 'ts_logged', 'target']
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['target']
    
    return X, y

def train():
    log_dir = "/data" if os.path.exists("/data") else "./logs"
    df_feats, df_outs = load_data(log_dir)
    if df_feats is None: return

    X, y = preprocess(df_feats, df_outs)
    if len(X) < 50:
        print(f"Dataset too small ({len(X)} rows). Need more data.")
        return

    print(f"Training on {len(X)} samples with {X.shape[1]} features...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_path = "model.joblib"
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save feature names for consistency
    with open("feature_names.json", 'w') as f:
        json.dump(list(X.columns), f)

if __name__ == "__main__":
    train()
