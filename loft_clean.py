"""
CLEAN evaluation: per-flange centering + feature selection are computed
ONLY on the training fold, never with test-fold data. No leakage.

Best candidate from preliminary results:
  LR + per-flange centering + (optionally) torque-discriminative
  feature selection.

We evaluate on three independent test schemes:
  - LOFO  (leave one flange out)            - "new flange" generalization
  - LOFT  (leave one (flange,torque) out)   - "new torque on known flange"
  - LOAO  (leave one area out)              - "new strike position"
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score

import optimize as opt
from features_v2 import extract_features_v2


def feat(s, sr):
    return extract_features_v2(s, sr, use_logmel=False, use_bands=False,
        use_band_decay=True, use_contrast=False, use_attack=False, use_slope=False)


def make_lr(C=1.0):
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=3000, C=C, random_state=42))])


def make_rf():
    return RandomForestClassifier(n_estimators=600, min_samples_leaf=3, max_features=0.3,
        max_depth=12, criterion="gini", class_weight="balanced", bootstrap=True,
        random_state=42, n_jobs=-1)


def fit_per_flange_means(X_train, fl_train):
    """Compute per-flange MEANS from TRAINING data only."""
    means = {}
    for f in np.unique(fl_train):
        means[f] = X_train[fl_train == f].mean(axis=0)
    return means


def apply_per_flange_centering(X, fl, means):
    """Subtract per-flange mean. For unseen flanges, fall back to global mean
    of provided means (which doesn't happen if all flanges are in train)."""
    Xc = X.astype(float).copy()
    if not means:
        return Xc
    fallback = np.mean(list(means.values()), axis=0)
    for i in range(len(X)):
        Xc[i] -= means.get(fl[i], fallback)
    return Xc


def fit_feature_selection(X_train, y_train, fl_train, n_keep):
    """Select top-K features by torque/flange F-ratio, training data only."""
    f_torque, _ = f_classif(X_train, y_train)
    f_flange, _ = f_classif(X_train, fl_train)
    score = np.nan_to_num(f_torque) / (np.nan_to_num(f_flange) + 1.0)
    return np.argsort(score)[::-1][:n_keep]


def file_pred(P, classes_ref, files_te, y_te):
    df = pd.DataFrame(P, columns=classes_ref); df["__f__"] = files_te
    avg = df.groupby("__f__").mean().sort_index()
    pred = classes_ref[avg.values.argmax(axis=1)]
    true_file = (pd.DataFrame({"f": files_te, "y": y_te})
                   .groupby("f")["y"].first().loc[avg.index.values].values)
    return pred, true_file


def evaluate(X, y, fl, files, get_test_indices, fold_iter, model_factory,
             use_centering=False, n_select=None, name=""):
    """
    fold_iter: list of fold descriptions (for printing per-fold)
    get_test_indices(fold_desc) returns boolean mask for test
    """
    classes = np.array([0, 25, 50])
    accs = []
    for desc in fold_iter:
        te = get_test_indices(desc)
        tr = ~te
        X_tr, y_tr, fl_tr = X[tr], y[tr], fl[tr]
        X_te = X[te]
        # Per-fold feature selection
        if n_select is not None:
            keep = fit_feature_selection(X_tr, y_tr, fl_tr, n_select)
            X_tr = X_tr[:, keep]; X_te = X_te[:, keep]
        # Per-fold centering
        if use_centering:
            means = fit_per_flange_means(X_tr, fl_tr)
            X_tr = apply_per_flange_centering(X_tr, fl_tr, means)
            X_te = apply_per_flange_centering(X_te, fl[te], means)
        m = model_factory()
        m.fit(X_tr, y_tr)
        P = m.predict_proba(X_te)
        idx = [list(m.classes_).index(c) for c in classes]; P = P[:, idx]
        pred, true = file_pred(P, classes, files[te], y[te])
        accs.append(accuracy_score(true, pred))
    return float(np.mean(accs)), accs


# ----- main -----
hits = opt.build_hits()
X = np.array([feat(h["signal"], h["sr"]) for h in hits])
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
ar = np.array([h["area_id"] for h in hits])
files = np.array([h["file_name"] for h in hits])
print(f"X: {X.shape}\n")


# Test schemes
def lofo_folds(): return [1,2,3,4]
def lofo_test(f): return fl == f

def loft_folds(): return [(f, t) for f in [1,2,3,4] for t in [0,25,50]]
def loft_test(ft): return (fl == ft[0]) & (y == ft[1])

def loao_folds(): return [1,2,3,4]
def loao_test(a): return ar == a


def benchmark(model_factory, use_centering, n_select, name):
    print(f"\n{name}")
    for scheme_name, folds, get_test in [
        ("LOFO", lofo_folds(), lofo_test),
        ("LOFT", loft_folds(), loft_test),
        ("LOAO", loao_folds(), loao_test),
    ]:
        m, per = evaluate(X, y, fl, files, get_test, folds, model_factory,
                          use_centering=use_centering, n_select=n_select)
        print(f"  {scheme_name:5s}  {m*100:5.2f}%   ({len(per)} folds, std {np.std(per)*100:.1f})")
    return None


# ===== Run all candidates =====
print("=" * 80)
print("Clean evaluation (per-fold centering & feature selection, no leakage)")
print("=" * 80)

# Old champion
benchmark(make_rf, False, None, "RF (raw, no centering, all features) - old champion")

# New candidates
benchmark(make_lr, False, None, "LR (raw, no centering, all features)")
benchmark(make_lr, True,  None, "LR + per-flange centering, all features")
benchmark(lambda: make_lr(C=10.0), True, None, "LR C=10 + per-flange centering")
benchmark(lambda: make_lr(C=0.1), True, None, "LR C=0.1 + per-flange centering")

for k in [50, 80, 100]:
    benchmark(make_lr, True, k, f"LR + per-flange centering + top-{k}")

# Hybrid: RF + per-flange centering
benchmark(make_rf, True, None, "RF + per-flange centering")
benchmark(make_rf, True, 100, "RF + per-flange centering + top-100")
