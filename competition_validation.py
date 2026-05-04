"""
The validation strategy that matches the actual competition test scenario.

Competition setup (from Final_Project_Competition_2026-1.docx):
  "We will tighten these four pairs of bolted flanges under different
   preload levels, each pair of flange will be tightened with a
   specific torque value (0/25/50 ft-lbs) uniformly. Please come to lab
   and collect four independent test sets from these four pairs of
   bolted flanges, respectively."

So:
  - Test flanges = SAME 4 physical flanges as training (NOT new flanges)
  - Each test flange has ONE unknown torque
  - All 4 areas of each flange get recorded

The validation strategy that mimics this is:
  Leave-one-(flange,torque)-cell-out
  - Hold out 1 flange + 1 torque + all 4 areas of that combination
  - Train on the rest
  - 12 folds (4 flanges x 3 torques)

This is between LOFO and the leaky 70/30 split in difficulty.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import lightgbm as lgb

import optimize as opt
from features_v2 import extract_features_v2


def feat(s, sr):
    return extract_features_v2(s, sr, use_logmel=False, use_bands=False,
        use_band_decay=True, use_contrast=False, use_attack=False, use_slope=False)


def make_rf():
    return RandomForestClassifier(n_estimators=600, min_samples_leaf=3, max_features=0.3,
        max_depth=12, criterion="gini", class_weight="balanced", bootstrap=True,
        random_state=42, n_jobs=-1)


def make_lgb():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
        min_child_samples=5, class_weight="balanced", random_state=42,
        n_jobs=-1, verbose=-1)


def make_lr():
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=3000, C=1.0, random_state=42))])


def make_svm():
    return Pipeline([("sc", StandardScaler()),
                     ("m", SVC(kernel="rbf", C=10, gamma="scale",
                               probability=True, random_state=42))])


def loft_eval(X, y, fl, ar, files, model_factory, name):
    """
    Leave-One-Flange-Torque-out.

    For each (flange, torque) cell, hold it out (4 files / ~80 hits),
    train on the rest, predict.

    NOTE: by holding out ALL hits at (flange=F, torque=T), we test the
    EXACT scenario the competition uses: we know what torque T sounds like
    on OTHER flanges, and we know what other torques sound like on flange F,
    but we've never heard torque T on flange F. The test files will likewise
    have an unknown torque on each known flange.
    """
    classes_ref = np.array([0, 25, 50])
    accs_file = []
    accs_hit = []
    per_fold_results = []
    for f in [1, 2, 3, 4]:
        for t in [0, 25, 50]:
            tr = ~((fl == f) & (y == t))
            te = (fl == f) & (y == t)
            if not te.any():
                continue
            m = model_factory()
            m.fit(X[tr], y[tr])
            yp = m.predict(X[te])
            accs_hit.append(accuracy_score(y[te], yp))
            P = m.predict_proba(X[te])
            idx = [list(m.classes_).index(c) for c in classes_ref]
            P = P[:, idx]
            df = pd.DataFrame(P, columns=classes_ref); df["__f__"] = files[te]
            avg = df.groupby("__f__").mean().sort_index()
            pred = classes_ref[avg.values.argmax(axis=1)]
            true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                           .groupby("f")["y"].first().loc[avg.index.values].values)
            file_acc = accuracy_score(true_file, pred)
            accs_file.append(file_acc)
            per_fold_results.append((f, t, file_acc))
    print(f"  {name:30s}  hit={np.mean(accs_hit)*100:5.2f}  "
          f"file={np.mean(accs_file)*100:5.2f}  N folds={len(accs_file)}")
    # Print per-fold details
    print(f"     per-fold (flange,torque,file-acc):")
    for f, t, a in per_fold_results:
        marker = " <- WRONG" if a < 1.0 else ""
        print(f"        F{f} T{t:>2}: {a*100:6.2f}%{marker}")
    return float(np.mean(accs_file))


# ----- main -----
hits = opt.build_hits()
X = np.array([feat(h["signal"], h["sr"]) for h in hits])
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
ar = np.array([h["area_id"] for h in hits])
files = np.array([h["file_name"] for h in hits])
print(f"X: {X.shape}\n")

print("=" * 80)
print("Leave-One-(Flange,Torque)-out — matches the actual competition test setup")
print("=" * 80)
print()
loft_eval(X, y, fl, ar, files, make_rf, "tuned RF")
print()
loft_eval(X, y, fl, ar, files, make_lgb, "LightGBM")
print()
loft_eval(X, y, fl, ar, files, make_lr, "LR")
print()
loft_eval(X, y, fl, ar, files, make_svm, "SVM")
