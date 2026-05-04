"""
Approaches that haven't been tried:

  G. Multi-seed RF bag — average soft-vote probabilities across N RFs
     trained with different random seeds. Reduces variance from RF's
     stochasticity.
  H. Per-area feature centering — subtract median per area_id (instead
     of per flange_id). Removes per-strike-position bias.
  I. Combined per-flange + per-area centering.
  J. Top-k hit confidence selection — drop the lowest-confidence
     hits per file before averaging.
  K. Trimmed mean of probabilities (drop outlier hits).
  L. Use median proba (more robust than mean) for soft vote.
  M. Combine multi-seed + per-flange centering.
  N. Train on TRIMMED training set (drop hits whose RF prediction
     disagrees with their file's majority — denoise the labels).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import optimize as opt
from features_v2 import extract_features_v2


def feat(sig, sr):
    return extract_features_v2(sig, sr,
        use_logmel=False, use_bands=False, use_band_decay=True,
        use_contrast=False, use_attack=False, use_slope=False)


def make_rf_tuned(seed=42):
    return RandomForestClassifier(
        n_estimators=600, min_samples_leaf=3, max_features=0.3, max_depth=12,
        criterion="gini", class_weight="balanced", bootstrap=True,
        random_state=seed, n_jobs=-1)


def file_level_from_proba(P, classes_ref, files_te, y_te):
    df = pd.DataFrame(P, columns=classes_ref); df["__f__"] = files_te
    avg = df.groupby("__f__").mean().sort_index()
    pred = classes_ref[avg.values.argmax(axis=1)]
    true_file = (pd.DataFrame({"f": files_te, "y": y_te})
                   .groupby("f")["y"].first().loc[avg.index.values].values)
    return pred, true_file, avg


def lofo_baseline(X, y, fl, files, name="baseline"):
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        m = make_rf_tuned()
        m.fit(X[tr], y[tr])
        P = m.predict_proba(X[te])
        idx = [list(m.classes_).index(c) for c in classes_ref]
        P = P[:, idx]
        pred, true_file, _ = file_level_from_proba(P, classes_ref, files[te], y[te])
        accs.append(accuracy_score(true_file, pred))
    print(f"  {name:50s} {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


# G. Multi-seed RF bag
def lofo_multiseed(X, y, fl, files, n_seeds=10, name=None):
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        proba_sum = None
        for seed in range(n_seeds):
            m = make_rf_tuned(seed=seed)
            m.fit(X[tr], y[tr])
            P = m.predict_proba(X[te])
            idx = [list(m.classes_).index(c) for c in classes_ref]
            P = P[:, idx]
            proba_sum = P if proba_sum is None else proba_sum + P
        P = proba_sum / n_seeds
        pred, true_file, _ = file_level_from_proba(P, classes_ref, files[te], y[te])
        accs.append(accuracy_score(true_file, pred))
    label = name or f"multi-seed RF bag (n={n_seeds})"
    print(f"  {label:50s} {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


# H. Per-area centering
def per_area_centered(X, area_ids):
    Xc = X.astype(float).copy()
    for a in np.unique(area_ids):
        m = area_ids == a
        Xc[m] -= np.median(X[m], axis=0)
    return Xc


# I. Per-flange + per-area
def per_flange_centered(X, fl_ids):
    Xc = X.astype(float).copy()
    for f in np.unique(fl_ids):
        m = fl_ids == f
        Xc[m] -= np.median(X[m], axis=0)
    return Xc


# J/K. Top-k hit confidence + trimmed mean
def lofo_topk(X, y, fl, files, top_k=15, name=None):
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        m = make_rf_tuned()
        m.fit(X[tr], y[tr])
        P = m.predict_proba(X[te])
        idx = [list(m.classes_).index(c) for c in classes_ref]
        P = P[:, idx]
        # For each test hit, confidence = max(proba)
        conf = P.max(axis=1)
        df = pd.DataFrame(P, columns=classes_ref)
        df["__f__"] = files[te]
        df["__c__"] = conf
        # Per file, keep top_k by confidence
        result = []
        for fname, sub in df.groupby("__f__"):
            top = sub.nlargest(min(top_k, len(sub)), "__c__")
            result.append({"f": fname, "p": top[classes_ref].mean().values})
        avg = pd.DataFrame({"f": [r["f"] for r in result],
                            **{c: [r["p"][i] for r in result] for i, c in enumerate(classes_ref)}})
        avg = avg.sort_values("f")
        pred = classes_ref[avg[classes_ref].values.argmax(axis=1)]
        true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                       .groupby("f")["y"].first().loc[avg["f"].values].values)
        accs.append(accuracy_score(true_file, pred))
    label = name or f"top-k confident hits (k={top_k})"
    print(f"  {label:50s} {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


# L. Median soft vote
def lofo_median_vote(X, y, fl, files):
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        m = make_rf_tuned()
        m.fit(X[tr], y[tr])
        P = m.predict_proba(X[te])
        idx = [list(m.classes_).index(c) for c in classes_ref]
        P = P[:, idx]
        df = pd.DataFrame(P, columns=classes_ref); df["__f__"] = files[te]
        avg = df.groupby("__f__").median().sort_index()
        pred = classes_ref[avg.values.argmax(axis=1)]
        true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                       .groupby("f")["y"].first().loc[avg.index.values].values)
        accs.append(accuracy_score(true_file, pred))
    print(f"  {'median soft vote per file':50s} {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


# N. Label denoising: drop training hits that disagree with their file's majority
def lofo_denoised_train(X, y, fl, files, area_ids):
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        # Use a quick RF to get hit-level predictions on train, drop disagreers
        Xtr, ytr = X[tr], y[tr]
        files_tr = files[tr]
        m_quick = make_rf_tuned()
        m_quick.fit(Xtr, ytr)
        yp = m_quick.predict(Xtr)
        # Per file, find majority predicted class
        df = pd.DataFrame({"f": files_tr, "p": yp, "true": ytr})
        keep = []
        for fname, sub in df.groupby("f"):
            file_pred = sub["p"].value_counts().idxmax()
            keep_mask = sub["p"] == file_pred
            keep.extend(sub.index[keep_mask].tolist())
        keep = np.array(keep)
        # Reset index alignment
        Xtr_d = Xtr[keep]
        ytr_d = ytr[keep]
        # Refit on cleaned train
        m_final = make_rf_tuned()
        m_final.fit(Xtr_d, ytr_d)
        P = m_final.predict_proba(X[te])
        idx = [list(m_final.classes_).index(c) for c in classes_ref]
        P = P[:, idx]
        pred, true_file, _ = file_level_from_proba(P, classes_ref, files[te], y[te])
        accs.append(accuracy_score(true_file, pred))
    print(f"  {'label denoising (drop disagreeing train hits)':50s} {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


hits = opt.build_hits()
X = np.array([feat(h["signal"], h["sr"]) for h in hits])
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
ar = np.array([h["area_id"] for h in hits])
files = np.array([h["file_name"] for h in hits])
print(f"X shape: {X.shape}\n")

print("--- baseline ---")
lofo_baseline(X, y, fl, files, name="tuned RF (current best)")

print("\n--- multi-seed RF bag ---")
for n in [3, 5, 10, 20]:
    lofo_multiseed(X, y, fl, files, n_seeds=n)

print("\n--- per-area centered features ---")
Xc_a = per_area_centered(X, ar)
lofo_baseline(Xc_a, y, fl, files, name="per-area centered")
print("\n--- per-flange centered features ---")
Xc_f = per_flange_centered(X, fl)
lofo_baseline(Xc_f, y, fl, files, name="per-flange centered")
print("\n--- per-area + per-flange centered ---")
Xc_af = per_flange_centered(per_area_centered(X, ar), fl)
lofo_baseline(Xc_af, y, fl, files, name="per-area + per-flange centered")

print("\n--- multi-seed on per-area centered ---")
lofo_multiseed(Xc_a, y, fl, files, n_seeds=10, name="multi-seed (n=10) on per-area")

print("\n--- top-k confident hits per file ---")
for k in [5, 10, 15, 20]:
    lofo_topk(X, y, fl, files, top_k=k)

print("\n--- median vote ---")
lofo_median_vote(X, y, fl, files)

print("\n--- label denoising on training ---")
lofo_denoised_train(X, y, fl, files, ar)

print("\n--- multi-seed + per-area + top-k ---")
lofo_multiseed(per_area_centered(X, ar), y, fl, files, n_seeds=20,
               name="multi-seed (20) + per-area centered")
