"""
Tune RF hyperparameters on the winning feature config.
Score: mean LOFO file-level accuracy (soft vote).
"""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import optimize as opt


def lofo_rf_file_acc(X, y, fl, files, **rf_params):
    accs = []
    for f in [1, 2, 3, 4]:
        tr, te = fl != f, fl == f
        rf = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=42)
        rf.fit(X[tr], y[tr])
        P = rf.predict_proba(X[te])
        df = pd.DataFrame(P, columns=rf.classes_)
        df["__f__"] = files[te]
        avg = df.groupby("__f__").mean().sort_index()
        pred = rf.classes_[avg.values.argmax(axis=1)]
        true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                       .groupby("f")["y"].first().loc[avg.index.values].values)
        accs.append(accuracy_score(true_file, pred))
    return float(np.mean(accs)), accs


if __name__ == "__main__":
    fparams = json.loads(sys.argv[1]) if len(sys.argv) > 1 else \
              dict(n_mfcc=13, n_fft=512, hop_length=128, n_psd_bins=64)
    print(f"fparams = {fparams}")
    hits = opt.build_hits()
    X, y, fl, files = opt.build_feature_matrix(hits, **fparams)
    print(f"X shape: {X.shape}")

    # Baseline RF (current default)
    print("\n--- Baseline RF ---")
    base = dict(n_estimators=400, min_samples_leaf=2, class_weight="balanced")
    acc, per = lofo_rf_file_acc(X, y, fl, files, **base)
    print(f"  baseline: {acc*100:.2f}  per-fold {[round(a*100) for a in per]}")

    # Random search
    print("\n--- RF random search ---")
    space = {
        "n_estimators": [200, 400, 600, 800, 1200, 2000],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_depth": [None, 6, 8, 12, 16, 24],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "criterion": ["gini", "entropy"],
        "bootstrap": [True, False],
    }
    rng = np.random.RandomState(0)
    n_iter = 40
    sampler = list(ParameterSampler(space, n_iter=n_iter, random_state=rng))
    rows = []
    for i, params in enumerate(sampler, 1):
        try:
            acc, per = lofo_rf_file_acc(X, y, fl, files, **params)
        except Exception as e:
            print(f"[{i:2d}/{n_iter}] ERROR: {e}")
            continue
        rows.append({**params, "file_acc": acc, "per_fold": per})
        marker = " ***" if acc > 0.8542 else ""
        print(f"[{i:2d}/{n_iter}] {acc*100:5.2f}  per-fold {[round(a*100) for a in per]}  {params}{marker}")

    df = pd.DataFrame(rows).sort_values("file_acc", ascending=False)
    print("\n=== Top 10 ===")
    print(df.head(10).to_string(index=False))
    df.to_csv("rf_tune_results.csv", index=False)
