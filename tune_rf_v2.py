"""RF random search on v1 + per_band_decay features (150 dims, 89.58 baseline)."""
import json, sys, numpy as np, pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import optimize as opt
from features_v2 import extract_features_v2


def lofo_rf_file_acc(X, y, fl, files, **rf_params):
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1, 2, 3, 4]:
        tr, te = fl != f, fl == f
        rf = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=42)
        rf.fit(X[tr], y[tr])
        P = rf.predict_proba(X[te])
        idx = [list(rf.classes_).index(c) for c in classes_ref]
        P = P[:, idx]
        df = pd.DataFrame(P, columns=classes_ref)
        df["__f__"] = files[te]
        avg = df.groupby("__f__").mean().sort_index()
        pred = classes_ref[avg.values.argmax(axis=1)]
        true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                       .groupby("f")["y"].first().loc[avg.index.values].values)
        accs.append(accuracy_score(true_file, pred))
    return float(np.mean(accs)), accs


hits = opt.build_hits()
X = np.array([extract_features_v2(h["signal"], h["sr"],
              use_logmel=False, use_bands=False, use_band_decay=True,
              use_contrast=False, use_attack=False, use_slope=False)
              for h in hits])
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
files = np.array([h["file_name"] for h in hits])
print(f"X shape: {X.shape}")

# baseline (current best)
print("\n--- baseline (current tuned RF) ---")
base = dict(n_estimators=600, min_samples_leaf=3, max_features=0.3, max_depth=12,
            criterion="gini", class_weight="balanced", bootstrap=True)
acc, per = lofo_rf_file_acc(X, y, fl, files, **base)
print(f"  {acc*100:.2f}  per-fold {[round(a*100) for a in per]}")

# random search
space = {
    "n_estimators": [300, 500, 800, 1200, 2000, 3000],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_depth": [None, 8, 12, 16, 24],
    "max_features": ["sqrt", "log2", 0.2, 0.3, 0.5],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "criterion": ["gini", "entropy"],
    "bootstrap": [True],
}
n_iter = 50
sampler = list(ParameterSampler(space, n_iter=n_iter, random_state=np.random.RandomState(0)))
print(f"\n--- {n_iter}-iter random search ---")
rows = []
best_acc = acc
for i, p in enumerate(sampler, 1):
    a, pf = lofo_rf_file_acc(X, y, fl, files, **p)
    rows.append({**p, "file_acc": a, "per_fold": pf})
    marker = " ***" if a > best_acc + 1e-9 else (" =new-best=" if a > best_acc - 1e-9 else "")
    if a > best_acc:
        best_acc = a
    print(f"[{i:2d}/{n_iter}] {a*100:5.2f}  per-fold {[round(x*100) for x in pf]}  {p}{marker}")
df = pd.DataFrame(rows).sort_values("file_acc", ascending=False)
print("\n=== Top 10 ===")
print(df.head(10).to_string(index=False))
df.to_csv("rf_tune_v2_results.csv", index=False)
