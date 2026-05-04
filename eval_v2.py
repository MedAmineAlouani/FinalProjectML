"""
Evaluate features_v2 with various ablations + tuned RF.
Compare against the v1 baseline (135 dims, 87.50% LOFO file-level).
"""
import sys, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import optimize as opt
from features_v2 import extract_features_v2


def make_rf_tuned():
    return RandomForestClassifier(
        n_estimators=600, min_samples_leaf=3, max_features=0.3, max_depth=12,
        criterion="gini", class_weight="balanced", bootstrap=True,
        random_state=42, n_jobs=-1)


def lofo_file(X, y, fl, files, model_factory):
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


def build(hits, **kw):
    X = np.array([extract_features_v2(h["signal"], h["sr"], **kw) for h in hits])
    y = np.array([h["torque"] for h in hits])
    fl = np.array([h["flange_id"] for h in hits])
    files = np.array([h["file_name"] for h in hits])
    return X, y, fl, files


if __name__ == "__main__":
    hits = opt.build_hits()
    print(f"Loaded {len(hits)} hits.")

    # Baseline: v1 only (no extras)
    print("\n--- Baseline v1 (135 dims, no extras) ---")
    X, y, fl, files = build(hits, use_logmel=False, use_bands=False,
                            use_band_decay=False, use_contrast=False,
                            use_attack=False, use_slope=False)
    print(f"  shape: {X.shape}")
    acc, per = lofo_file(X, y, fl, files, make_rf_tuned)
    print(f"  tuned RF: {acc*100:.2f}  per-fold {[round(a*100) for a in per]}")

    # Each addition alone, on top of v1
    additions = [
        ("logmel_quantiles", dict(use_logmel=True)),
        ("band_energies",     dict(use_bands=True)),
        ("per_band_decay",    dict(use_band_decay=True)),
        ("spectral_contrast", dict(use_contrast=True)),
        ("attack",            dict(use_attack=True)),
        ("spectral_slope",    dict(use_slope=True)),
    ]
    print("\n--- Each new feature alone (added to v1) ---")
    base = dict(use_logmel=False, use_bands=False, use_band_decay=False,
                use_contrast=False, use_attack=False, use_slope=False)
    individual_results = {}
    for name, flag in additions:
        kw = {**base, **flag}
        t0 = time.time()
        X, y, fl, files = build(hits, **kw)
        acc, per = lofo_file(X, y, fl, files, make_rf_tuned)
        individual_results[name] = acc
        print(f"  +{name:18s}  shape={X.shape[1]:3d}  acc={acc*100:5.2f}  "
              f"per-fold {[round(a*100) for a in per]}  [{time.time()-t0:.1f}s]")

    # All combined
    print("\n--- All new features combined ---")
    X, y, fl, files = build(hits)  # all defaults True
    print(f"  shape: {X.shape}")
    acc, per = lofo_file(X, y, fl, files, make_rf_tuned)
    print(f"  tuned RF: {acc*100:.2f}  per-fold {[round(a*100) for a in per]}")

    # Greedy: only the helpful additions (those that beat baseline alone)
    print("\n--- Greedy: only additions that beat baseline alone ---")
    helpful = [name for name, _ in additions if individual_results[name] > 0.875 + 1e-9]
    print(f"  helpful additions: {helpful}")
    if helpful:
        kw = {**base}
        for h in helpful:
            for name, flag in additions:
                if name == h:
                    kw.update(flag)
        X, y, fl, files = build(hits, **kw)
        print(f"  shape: {X.shape}")
        acc, per = lofo_file(X, y, fl, files, make_rf_tuned)
        print(f"  tuned RF: {acc*100:.2f}  per-fold {[round(a*100) for a in per]}")
