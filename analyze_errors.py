"""
Identify which specific files (and their true torque) the tuned RF
misclassifies. Print confusion matrices per LOFO fold.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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


hits = opt.build_hits()
X = np.array([feat(h["signal"], h["sr"]) for h in hits])
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
files = np.array([h["file_name"] for h in hits])

classes_ref = np.array([0, 25, 50])
all_errors = []
for f in [1, 2, 3, 4]:
    print(f"\n===== Held-out flange {f} =====")
    tr, te = fl != f, fl == f
    m = make_rf_tuned()
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
    cm = confusion_matrix(true_file, pred, labels=classes_ref)
    print("Confusion matrix (rows=true, cols=pred, classes [0,25,50]):")
    print(cm)
    # List wrong files with their probabilities
    for fname, t, p, probs in zip(avg.index.values, true_file, pred, avg.values):
        if t != p:
            print(f"  WRONG: {fname:15s}  true={t:2d}  pred={p:2d}  "
                  f"probs(0,25,50)=({probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f})")
            all_errors.append((fname, int(t), int(p), tuple(probs)))

print(f"\n=== Total errors: {len(all_errors)} ===")
print("\nError summary by direction:")
import collections
dirs = collections.Counter((e[1], e[2]) for e in all_errors)
for (t, p), c in sorted(dirs.items()):
    print(f"  true={t:2d} -> pred={p:2d}:  {c} times")

print("\nAll errors (file, true, pred, soft-vote margins):")
for f, t, p, prob in all_errors:
    margin = prob[list(classes_ref).index(p)] - prob[list(classes_ref).index(t)]
    print(f"  {f:15s}  true={t:2d}  pred={p:2d}  margin={margin:+.3f}")
