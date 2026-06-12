"""
predict_v2.py — Inference with optimal thresholds
Usage:
    python predict_v2.py --model xgboost
    python predict_v2.py --model random_forest --input transactions.csv
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

MODELS = {
    "logistic_regression": "models/logistic_regression.pkl",
    "random_forest":       "models/random_forest.pkl",
    "xgboost":             "models/xgboost.pkl",
    "isolation_forest":    "models/isolation_forest.pkl",
}


def predict(model_name, X):
    model      = joblib.load(MODELS[model_name])
    thresholds = joblib.load("models/optimal_thresholds.pkl")

    sc = StandardScaler()
    X_s = sc.fit_transform(X)

    # Map predict.py model name → results dict key
    key_map = {
        "logistic_regression": "Logistic Regression",
        "random_forest":       "Random Forest",
        "xgboost":             "XGBoost",
        "isolation_forest":    "Isolation Forest",
    }
    key   = key_map[model_name]
    thresh = thresholds.get(key, 0.5)

    if model_name == "isolation_forest":
        scores = -model.score_samples(X_s)
        preds  = (scores >= thresh).astype(int)
        return preds, scores, thresh

    probas = model.predict_proba(X_s)[:, 1]
    preds  = (probas >= thresh).astype(int)
    return preds, probas, thresh


def sample_transaction():
    cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    return pd.DataFrame(np.random.randn(1, 30), columns=cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=list(MODELS.keys()))
    parser.add_argument("--input", default=None)
    args = parser.parse_args()

    X = pd.read_csv(args.input) if args.input else sample_transaction()
    preds, scores, thresh = predict(args.model, X)

    print(f"\nModel: {args.model}  |  Optimal threshold: {thresh:.4f}\n")
    for i, (p, s) in enumerate(zip(preds, scores)):
        label = "FRAUD ⚠️ " if p == 1 else "LEGIT ✅ "
        print(f"  Transaction {i+1}: {label}  (score={s:.4f})")
