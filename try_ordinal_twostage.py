"""
Approaches that target the specific failure modes we identified:

  3 of 5 errors involve the middle class (25) being confused with adjacent
  classes (0 or 50). 0 and 50 are never confused with each other.

Strategies:
  A. Ordinal regression — treat torque as ordered. Predict a real-valued
     score and snap to nearest of {0, 25, 50}. Penalizes 0->50 mistakes
     more than 25->50.
  B. Two-stage with stage-1 doing 0 vs (25 or 50), stage-2 only between
     25 and 50.
  C. Two-stage with stage-1 doing (0 or 25) vs 50, stage-2 only between
     0 and 25.
  D. Three pairwise binary classifiers (0 vs 25, 0 vs 50, 25 vs 50) with
     soft-vote.
  E. Mid-class boost: weight class 25 higher in RF.
  F. Confidence-conditional refinement: when RF is uncertain (margin
     between top two < threshold), defer to a stage-2 model trained on
     adjacent classes only.
"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import optimize as opt
from features_v2 import extract_features_v2


def feat(sig, sr):
    return extract_features_v2(sig, sr,
        use_logmel=False, use_bands=False, use_band_decay=True,
        use_contrast=False, use_attack=False, use_slope=False)


def make_rf_tuned(class_weight="balanced"):
    return RandomForestClassifier(
        n_estimators=600, min_samples_leaf=3, max_features=0.3, max_depth=12,
        criterion="gini", class_weight=class_weight, bootstrap=True,
        random_state=42, n_jobs=-1)


def file_level_from_proba(P, classes_ref, files_te, y_te):
    df = pd.DataFrame(P, columns=classes_ref); df["__f__"] = files_te
    avg = df.groupby("__f__").mean().sort_index()
    pred = classes_ref[avg.values.argmax(axis=1)]
    true_file = (pd.DataFrame({"f": files_te, "y": y_te})
                   .groupby("f")["y"].first().loc[avg.index.values].values)
    return pred, true_file, avg


def lofo(model_factory, X, y, fl, files, name="model"):
    """Standard classifier LOFO."""
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        m = model_factory()
        m.fit(X[tr], y[tr])
        P = m.predict_proba(X[te])
        idx = [list(m.classes_).index(c) for c in classes_ref]
        P = P[:, idx]
        pred, true_file, _ = file_level_from_proba(P, classes_ref, files[te], y[te])
        accs.append(accuracy_score(true_file, pred))
    print(f"  {name:40s}  {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


# A. Ordinal regression: regress to torque value, snap to nearest class
def lofo_ordinal_regressor(X, y, fl, files, name="ordinal regressor"):
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        m = RandomForestRegressor(n_estimators=600, min_samples_leaf=3,
                                  max_features=0.3, max_depth=12,
                                  random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr].astype(float))
        # Hit-level prediction; aggregate by file mean, then snap
        yp_hit = m.predict(X[te])
        df = pd.DataFrame({"f": files[te], "p": yp_hit})
        means = df.groupby("f")["p"].mean()
        # snap each file mean to nearest of {0, 25, 50}
        pred = np.array([classes_ref[np.argmin(np.abs(classes_ref - v))] for v in means.values])
        true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                       .groupby("f")["y"].first().loc[means.index.values].values)
        accs.append(accuracy_score(true_file, pred))
    print(f"  {name:40s}  {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


# B/C. Two-stage classifiers
def lofo_two_stage(X, y, fl, files, stage1_split="0 vs loose"):
    """
    Two-stage classifier.

    stage1_split == "0 vs loose":   stage1 = 0 vs (25/50). stage2 picks 25/50 only when stage1=loose.
    stage1_split == "tight vs 50":  stage1 = (0/25) vs 50. stage2 picks 0/25 when stage1=tight.
    """
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        Xtr, ytr, Xte = X[tr], y[tr], X[te]

        if stage1_split == "0 vs loose":
            y1 = (ytr != 0).astype(int)        # 0=tight, 1=loose
            stage2_mask = ytr != 0
            stage2_classes = np.array([25, 50])
        else:
            y1 = (ytr == 50).astype(int)       # 0=not50, 1=is50
            stage2_mask = ytr != 50
            stage2_classes = np.array([0, 25])

        m1 = make_rf_tuned()
        m1.fit(Xtr, y1)
        # stage 2
        m2 = make_rf_tuned()
        m2.fit(Xtr[stage2_mask], ytr[stage2_mask])

        # Hit-level: combine probabilities into a 3-class proba
        P1 = m1.predict_proba(Xte)  # shape (n, 2), m1.classes_ are [0,1]
        P2 = m2.predict_proba(Xte)  # shape (n, 2), m2.classes_ are stage2_classes
        i1_one = list(m1.classes_).index(1)
        i1_zero = list(m1.classes_).index(0)
        p1_class1 = P1[:, i1_one]
        p1_class0 = P1[:, i1_zero]

        # 3-class combined proba per hit
        P3 = np.zeros((len(Xte), 3))
        if stage1_split == "0 vs loose":
            # class 0 = 0; class 1 = 25; class 2 = 50
            P3[:, 0] = p1_class0
            i25 = list(m2.classes_).index(25)
            i50 = list(m2.classes_).index(50)
            P3[:, 1] = p1_class1 * P2[:, i25]
            P3[:, 2] = p1_class1 * P2[:, i50]
        else:
            # stage1: 1 = is50, 0 = not50
            i0 = list(m2.classes_).index(0)
            i25 = list(m2.classes_).index(25)
            P3[:, 0] = p1_class0 * P2[:, i0]
            P3[:, 1] = p1_class0 * P2[:, i25]
            P3[:, 2] = p1_class1   # is 50
        # normalize
        P3 = P3 / (P3.sum(axis=1, keepdims=True) + 1e-12)

        pred, true_file, _ = file_level_from_proba(P3, classes_ref, files[te], y[te])
        accs.append(accuracy_score(true_file, pred))
    print(f"  {'two-stage ' + stage1_split:40s}  {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


# D. Pairwise binary classifiers + soft vote
def lofo_pairwise(X, y, fl, files):
    classes_ref = np.array([0, 25, 50])
    pairs = [(0, 25), (0, 50), (25, 50)]
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        Xtr, ytr, Xte = X[tr], y[tr], X[te]
        proba_per_class = np.zeros((len(Xte), 3))
        for a, b in pairs:
            m = (ytr == a) | (ytr == b)
            mdl = make_rf_tuned()
            mdl.fit(Xtr[m], ytr[m])
            P = mdl.predict_proba(Xte)
            ia = list(mdl.classes_).index(a)
            ib = list(mdl.classes_).index(b)
            ja = list(classes_ref).index(a)
            jb = list(classes_ref).index(b)
            proba_per_class[:, ja] += P[:, ia]
            proba_per_class[:, jb] += P[:, ib]
        proba_per_class /= 2.0  # each class voted by 2 pairs
        pred, true_file, _ = file_level_from_proba(proba_per_class, classes_ref, files[te], y[te])
        accs.append(accuracy_score(true_file, pred))
    print(f"  {'pairwise binary classifiers':40s}  {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


# E. Boost class 25 with custom weight
def lofo_class_weighted(X, y, fl, files, weight_25=1.5):
    classes_ref = np.array([0, 25, 50])
    cw = {0: 1.0, 25: weight_25, 50: 1.0}
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        m = RandomForestClassifier(
            n_estimators=600, min_samples_leaf=3, max_features=0.3, max_depth=12,
            criterion="gini", class_weight=cw, bootstrap=True,
            random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr])
        P = m.predict_proba(X[te])
        idx = [list(m.classes_).index(c) for c in classes_ref]
        P = P[:, idx]
        pred, true_file, _ = file_level_from_proba(P, classes_ref, files[te], y[te])
        accs.append(accuracy_score(true_file, pred))
    print(f"  {'class25_weight=' + str(weight_25):40s}  {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


# F. Confidence-conditional refinement
def lofo_conditional_refine(X, y, fl, files, margin_thresh=0.20):
    """
    If RF's top-two margin < threshold for a hit, replace its proba with
    that of a binary classifier trained on the top-two adjacent classes.
    """
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1,2,3,4]:
        tr, te = fl != f, fl == f
        # main 3-class
        m_main = make_rf_tuned()
        m_main.fit(X[tr], y[tr])
        P = m_main.predict_proba(X[te])
        idx = [list(m_main.classes_).index(c) for c in classes_ref]
        P = P[:, idx]

        # binary 0-vs-25 and 25-vs-50 trained on the corresponding class pairs
        m_low = make_rf_tuned()
        mask_low = (y[tr] == 0) | (y[tr] == 25)
        m_low.fit(X[tr][mask_low], y[tr][mask_low])

        m_high = make_rf_tuned()
        mask_high = (y[tr] == 25) | (y[tr] == 50)
        m_high.fit(X[tr][mask_high], y[tr][mask_high])

        # For each hit with low margin, refine
        refined = P.copy()
        for i, prob in enumerate(P):
            top_two = np.argsort(prob)[-2:]
            margin = prob[top_two[1]] - prob[top_two[0]]
            if margin >= margin_thresh:
                continue
            top = set(classes_ref[top_two])
            if top == {0, 25}:
                Pi = m_low.predict_proba(X[te][i:i+1])[0]
                ic0 = list(m_low.classes_).index(0)
                ic25 = list(m_low.classes_).index(25)
                refined[i, 0] = Pi[ic0]
                refined[i, 1] = Pi[ic25]
                refined[i, 2] = 0.0
            elif top == {25, 50}:
                Pi = m_high.predict_proba(X[te][i:i+1])[0]
                ic25 = list(m_high.classes_).index(25)
                ic50 = list(m_high.classes_).index(50)
                refined[i, 1] = Pi[ic25]
                refined[i, 2] = Pi[ic50]
                refined[i, 0] = 0.0
            # if {0,50}, leave alone (we know that doesn't happen in errors)
        refined = refined / (refined.sum(axis=1, keepdims=True) + 1e-12)
        pred, true_file, _ = file_level_from_proba(refined, classes_ref, files[te], y[te])
        accs.append(accuracy_score(true_file, pred))
    print(f"  {'conditional refine, margin<' + str(margin_thresh):40s}  {np.mean(accs)*100:5.2f}  per-fold {[round(a*100) for a in accs]}")
    return float(np.mean(accs))


# Build features
hits = opt.build_hits()
X = np.array([feat(h["signal"], h["sr"]) for h in hits])
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
files = np.array([h["file_name"] for h in hits])
print(f"X shape: {X.shape}\n")

print("--- baseline ---")
lofo(make_rf_tuned, X, y, fl, files, name="tuned RF (current best)")

print("\n--- ordinal regression ---")
lofo_ordinal_regressor(X, y, fl, files)

print("\n--- two-stage ---")
lofo_two_stage(X, y, fl, files, stage1_split="0 vs loose")
lofo_two_stage(X, y, fl, files, stage1_split="tight vs 50")

print("\n--- pairwise binary ---")
lofo_pairwise(X, y, fl, files)

print("\n--- class weight on 25 ---")
for w in [1.0, 1.25, 1.5, 2.0, 3.0]:
    lofo_class_weighted(X, y, fl, files, weight_25=w)

print("\n--- conditional refinement (top-2 margin threshold) ---")
for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
    lofo_conditional_refine(X, y, fl, files, margin_thresh=thr)
