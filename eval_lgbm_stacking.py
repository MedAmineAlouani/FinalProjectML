"""
Try LightGBM, ExtraTrees, and stacking with logistic-regression meta-learner.
All on the v2 (per_band_decay) feature set.
"""
import json, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import optimize as opt
from features_v2 import extract_features_v2


def feat(sig, sr):
    return extract_features_v2(sig, sr,
        use_logmel=False, use_bands=False, use_band_decay=True,
        use_contrast=False, use_attack=False, use_slope=False)


def make_rf_tuned():
    return RandomForestClassifier(
        n_estimators=600, min_samples_leaf=3, max_features=0.3, max_depth=12,
        criterion="gini", class_weight="balanced", bootstrap=True,
        random_state=42, n_jobs=-1)


def make_et():
    return ExtraTreesClassifier(
        n_estimators=600, min_samples_leaf=2, max_features=0.3, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1)


def make_lgb(**overrides):
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
             min_child_samples=5, max_depth=-1, reg_alpha=0.0, reg_lambda=0.0,
             colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
             class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
    p.update(overrides)
    return lgb.LGBMClassifier(**p)


def make_lr():
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=3000, random_state=42))])


def lofo_file_acc(model_factory, X, y, fl, files):
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1, 2, 3, 4]:
        tr, te = fl != f, fl == f
        m = model_factory()
        m.fit(X[tr], y[tr])
        P = m.predict_proba(X[te])
        idx = [list(m.classes_).index(c) for c in classes_ref]
        P = P[:, idx]
        df = pd.DataFrame(P, columns=classes_ref)
        df["__f__"] = files[te]
        avg = df.groupby("__f__").mean().sort_index()
        pred = classes_ref[avg.values.argmax(axis=1)]
        true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                       .groupby("f")["y"].first().loc[avg.index.values].values)
        accs.append(accuracy_score(true_file, pred))
    return float(np.mean(accs)), accs


def lofo_ensemble(factories, X, y, fl, files, weights=None):
    classes_ref = np.array([0, 25, 50])
    accs = []
    if weights is None:
        weights = [1.0] * len(factories)
    for f in [1, 2, 3, 4]:
        tr, te = fl != f, fl == f
        proba_sum = None
        for w, fact in zip(weights, factories):
            m = fact()
            m.fit(X[tr], y[tr])
            P = m.predict_proba(X[te])
            idx = [list(m.classes_).index(c) for c in classes_ref]
            P = P[:, idx]
            df = pd.DataFrame(P, columns=classes_ref)
            df["__f__"] = files[te]
            avg = df.groupby("__f__").mean().sort_index()
            files_ref = avg.index.values
            P_w = w * avg.values
            proba_sum = P_w if proba_sum is None else proba_sum + P_w
        pred = classes_ref[proba_sum.argmax(axis=1)]
        true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                       .groupby("f")["y"].first().loc[files_ref].values)
        accs.append(accuracy_score(true_file, pred))
    return float(np.mean(accs)), accs


# Build features
hits = opt.build_hits()
X = np.array([feat(h["signal"], h["sr"]) for h in hits])
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
files = np.array([h["file_name"] for h in hits])
print(f"X shape: {X.shape}\n")

# Baseline: tuned RF
print("--- single models (LOFO file-level, soft vote) ---")
for name, fact in [("RF (tuned)", make_rf_tuned),
                   ("ExtraTrees", make_et),
                   ("LightGBM (default)", make_lgb)]:
    t0 = time.time()
    a, p = lofo_file_acc(fact, X, y, fl, files)
    print(f"  {name:25s} {a*100:5.2f}  per-fold {[round(x*100) for x in p]}  [{time.time()-t0:.1f}s]")

# LightGBM tuning - small grid
print("\n--- LightGBM hyperparameter sweep ---")
lgb_configs = [
    dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5),
    dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=5),
    dict(n_estimators=800, learning_rate=0.03, num_leaves=15, min_child_samples=5),
    dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=3),
    dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10),
    dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=5),
    dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10, reg_lambda=0.1),
    dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10, reg_lambda=1.0),
    dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=5, max_depth=6),
    dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=5, colsample_bytree=0.5),
]
best_lgb = None
best_lgb_acc = 0
for cfg in lgb_configs:
    f = lambda c=cfg: make_lgb(**c)
    a, p = lofo_file_acc(f, X, y, fl, files)
    if a > best_lgb_acc:
        best_lgb_acc, best_lgb = a, cfg
    print(f"  {cfg}  ->  {a*100:.2f}  per-fold {[round(x*100) for x in p]}")

print(f"\nBest LightGBM: {best_lgb_acc*100:.2f} with {best_lgb}")

# Ensembles
def make_lgb_best():
    return make_lgb(**best_lgb)

print("\n--- soft-vote ensembles ---")
ensembles = [
    ("RF + LGB",          [make_rf_tuned, make_lgb_best]),
    ("RF + ET",           [make_rf_tuned, make_et]),
    ("RF + ET + LGB",     [make_rf_tuned, make_et, make_lgb_best]),
    ("RF + LR + LGB",     [make_rf_tuned, make_lr, make_lgb_best]),
    ("RF + LR",           [make_rf_tuned, make_lr]),
    ("RF + LGB (RF=2x)",  [make_rf_tuned, make_lgb_best]),  # weighted
]
for name, facts in ensembles:
    weights = [2.0, 1.0] if "RF=2x" in name else None
    a, p = lofo_ensemble(facts, X, y, fl, files, weights=weights)
    print(f"  {name:25s} {a*100:5.2f}  per-fold {[round(x*100) for x in p]}")
