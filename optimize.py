"""
Hyperparameter optimization for the torque-classification pipeline.

Stages:
  1) Grid search over feature hyperparameters (MFCC/FFT/HOP/PSD) using the
     fast models (LR, RF, GB). Score = mean LOFO file-level accuracy.
  2) With the best feature settings, evaluate every model + ensembles +
     per-flange-centering trick.
  3) Random-search hyperparameters of the top model.
"""

import os
import re
import glob
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings("ignore")

import librosa
from scipy.signal import find_peaks, welch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HITS_CACHE = os.path.join(DATA_DIR, "_hits_cache.pkl")

IGNORE_START_SEC = 0.15
ENVELOPE_WIN_SEC = 0.01
MIN_PEAK_DISTANCE_SEC = 0.30
PEAK_HEIGHT_FACTOR = 2.5
PRE_HIT_SEC = 0.02
POST_HIT_SEC = 0.15

RANDOM_STATE = 42


# ---------------------------------------------------------------- IO

def parse_labeled(path):
    name = os.path.basename(path)
    m = re.match(r"^(0|25|50)ftlbF(\d+)A(\d+)\.m4a$", name)
    if not m:
        return None
    return dict(file_name=name, file_path=path,
                torque=int(m.group(1)),
                flange_id=int(m.group(2)),
                area_id=int(m.group(3)))


def collect_labeled():
    rows = [parse_labeled(p) for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.m4a")))]
    return [r for r in rows if r is not None]


def normalize_audio(s):
    m = np.max(np.abs(s))
    return s if m == 0 else s / m


def split_into_hits(signal, sr):
    start = int(IGNORE_START_SEC * sr)
    s = signal[start:]
    win = max(1, int(ENVELOPE_WIN_SEC * sr))
    env = np.convolve(np.abs(s), np.ones(win) / win, mode="same")
    thr = env.mean() + PEAK_HEIGHT_FACTOR * env.std()
    peaks, _ = find_peaks(env, height=thr, distance=int(MIN_PEAK_DISTANCE_SEC * sr))
    pre = int(PRE_HIT_SEC * sr)
    post = int(POST_HIT_SEC * sr)
    hits = []
    for p in peaks:
        a, b = max(0, p - pre), min(len(s), p + post)
        if b > a:
            hits.append(s[a:b])
    return hits


def build_hits():
    if os.path.exists(HITS_CACHE):
        with open(HITS_CACHE, "rb") as f:
            return pickle.load(f)
    print("Loading + segmenting audio (one-time)...")
    files = collect_labeled()
    out = []
    for r in files:
        sig, sr = librosa.load(r["file_path"], sr=None, mono=True)
        sig = normalize_audio(sig)
        for hid, h in enumerate(split_into_hits(sig, sr), 1):
            out.append({**r, "sr": sr, "hit_id": hid, "signal": h})
    with open(HITS_CACHE, "wb") as f:
        pickle.dump(out, f)
    print(f"Cached {len(out)} hits to {HITS_CACHE}")
    return out


# ---------------------------------------------------------------- FEATURES

def safe_mean_std(M):
    M = np.asarray(M)
    return np.concatenate([M.mean(axis=1), M.std(axis=1)])


def extract_features(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512,
                     n_psd_bins=128, nperseg=1024):
    n = len(signal)
    n_fft_eff = min(n_fft, n)
    hop_eff = max(1, min(hop_length, n // 2))
    nperseg_eff = min(nperseg, n)

    freqs, psd = welch(signal, fs=sr, nperseg=nperseg_eff)
    psd_log = np.log10(psd + 1e-12)
    if len(psd_log) >= n_psd_bins:
        psd_feat = psd_log[:n_psd_bins]
    else:
        psd_feat = np.pad(psd_log, (0, n_psd_bins - len(psd_log)))

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft_eff, hop_length=hop_eff)
    mfcc_d = librosa.feature.delta(mfcc)
    mfcc_feat = safe_mean_std(mfcc)
    delta_feat = safe_mean_std(mfcc_d)

    cent = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft_eff, hop_length=hop_eff)
    bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=n_fft_eff, hop_length=hop_eff)
    roll = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=n_fft_eff,
                                            hop_length=hop_eff, roll_percent=0.85)
    flat = librosa.feature.spectral_flatness(y=signal, n_fft=n_fft_eff, hop_length=hop_eff)
    zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=n_fft_eff, hop_length=hop_eff)
    rms = librosa.feature.rms(y=signal, frame_length=n_fft_eff, hop_length=hop_eff)
    spec_feat = np.concatenate([safe_mean_std(x) for x in [cent, bw, roll, flat, zcr, rms]])

    psd_n = psd / psd.sum()
    dom = freqs[np.argmax(psd)]
    ent = -np.sum(psd_n * np.log2(psd_n + 1e-12))
    fs_feat = np.array([dom, ent])

    abs_s = np.abs(signal)
    if abs_s.max() == 0:
        decay_feat = np.zeros(5)
    else:
        win = max(1, int(0.002 * sr))
        env = np.convolve(abs_s, np.ones(win) / win, mode="same")
        pi = int(np.argmax(env))
        peak = env[pi] + 1e-12
        after = env[pi:]
        b50 = np.where(after <= 0.5 * peak)[0]
        b10 = np.where(after <= 0.1 * peak)[0]
        d50 = b50[0] / sr if len(b50) else len(after) / sr
        d10 = b10[0] / sr if len(b10) else len(after) / sr
        t = np.arange(len(after)) / sr
        slope = np.polyfit(t, np.log(after + 1e-8), 1)[0] if len(t) > 5 else 0.0
        decay_feat = np.array([peak, pi / sr, d50, d10, slope])

    return np.concatenate([psd_feat, mfcc_feat, delta_feat, spec_feat, fs_feat, decay_feat])


def build_feature_matrix(hits, **fparams):
    X = np.array([extract_features(h["signal"], h["sr"], **fparams) for h in hits])
    y = np.array([h["torque"] for h in hits])
    fl = np.array([h["flange_id"] for h in hits])
    files = np.array([h["file_name"] for h in hits])
    return X, y, fl, files


# ---------------------------------------------------------------- MODELS

def make_model(name):
    if name == "LR":
        return Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=3000, C=1.0,
                                                  random_state=RANDOM_STATE))])
    if name == "SVM":
        # use LinearSVC + calibration for probability fast, OR rbf w/ probability
        return Pipeline([("sc", StandardScaler()),
                         ("m", SVC(kernel="rbf", C=10, gamma="scale",
                                   probability=True, random_state=RANDOM_STATE))])
    if name == "SVM_lin":
        # Calibrated linear SVM — much faster than RBF + probability
        base = LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)
        return Pipeline([("sc", StandardScaler()),
                         ("m", CalibratedClassifierCV(base, cv=3, method="sigmoid"))])
    if name == "RF":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1)
    if name == "ET":
        return ExtraTreesClassifier(n_estimators=600, min_samples_leaf=1,
                                     class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1)
    if name == "GB":
        return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=3, random_state=RANDOM_STATE)
    if name == "HGB":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              max_depth=None, random_state=RANDOM_STATE)
    raise ValueError(name)


# ---------------------------------------------------------------- EVAL

def per_flange_centered(X, fl_ids):
    Xc = X.copy().astype(float)
    for f in np.unique(fl_ids):
        m = (fl_ids == f)
        Xc[m] -= np.median(X[m], axis=0)
    return Xc


def lofo_eval(X, y, fl, files, model_names, soft=True):
    """Single LOFO pass for given models. Returns hit-level + file-level dict."""
    classes_ref = np.array([0, 25, 50])
    hit_acc = {n: [] for n in model_names}
    file_acc = {n: [] for n in model_names}
    for test_f in [1, 2, 3, 4]:
        tr, te = fl != test_f, fl == test_f
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        files_te = files[te]
        true_file = pd.DataFrame({"f": files_te, "y": yte}).groupby("f")["y"].first()

        for name in model_names:
            mdl = make_model(name)
            mdl.fit(Xtr, ytr)
            yp = mdl.predict(Xte)
            hit_acc[name].append(accuracy_score(yte, yp))

            if soft and hasattr(mdl, "predict_proba"):
                P = mdl.predict_proba(Xte)
                # Reorder to canonical class order
                idx = [list(mdl.classes_).index(c) for c in classes_ref]
                P = P[:, idx]
                df = pd.DataFrame(P, columns=classes_ref)
                df["__file__"] = files_te
                avg = df.groupby("__file__").mean().sort_index()
                pred = classes_ref[avg.values.argmax(axis=1)]
                files_pred = avg.index.values
            else:
                df = pd.DataFrame({"f": files_te, "p": yp})
                grp = df.groupby("f")["p"].agg(lambda s: s.value_counts().idxmax())
                files_pred = grp.index.values
                pred = grp.values
            true_aligned = true_file.loc[files_pred].values
            file_acc[name].append(accuracy_score(true_aligned, pred))
    return ({k: float(np.mean(v)) for k, v in hit_acc.items()},
            {k: float(np.mean(v)) for k, v in file_acc.items()},
            {k: [float(x) for x in v] for k, v in file_acc.items()})


def lofo_ensemble(X, y, fl, files, member_names):
    """Soft-vote ensemble of trained members per LOFO fold."""
    classes_ref = np.array([0, 25, 50])
    file_acc = []
    per_fold = []
    for test_f in [1, 2, 3, 4]:
        tr, te = fl != test_f, fl == test_f
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        files_te = files[te]
        true_file = pd.DataFrame({"f": files_te, "y": yte}).groupby("f")["y"].first()
        proba_sum = None
        for name in member_names:
            mdl = make_model(name)
            mdl.fit(Xtr, ytr)
            P = mdl.predict_proba(Xte)
            idx = [list(mdl.classes_).index(c) for c in classes_ref]
            P = P[:, idx]
            df = pd.DataFrame(P, columns=classes_ref)
            df["__file__"] = files_te
            avg = df.groupby("__file__").mean().sort_index()
            if proba_sum is None:
                proba_sum = avg.values.copy()
                files_ref = avg.index.values
            else:
                proba_sum += avg.values
        pred = classes_ref[proba_sum.argmax(axis=1)]
        true_aligned = true_file.loc[files_ref].values
        acc = accuracy_score(true_aligned, pred)
        file_acc.append(acc)
        per_fold.append(acc)
    return float(np.mean(file_acc)), per_fold


# ---------------------------------------------------------------- GRID SEARCH

def grid_search(hits, models=("LR", "RF")):
    # Focused grid around the area where combo 1 won (small fft/hop, small psd).
    # 36 combos × ~15s each ≈ 9 minutes
    grid = {
        "n_mfcc":     [13, 20, 32],
        "n_fft":      [512, 1024, 2048],
        "hop_length": [64, 128, 256],
        "n_psd_bins": [32, 64, 128],
    }
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    print(f"Grid size: {len(combos)} combos, models = {models}")

    rows = []
    for i, vals in enumerate(combos, 1):
        fparams = dict(zip(keys, vals))
        try:
            X, y, fl, files = build_feature_matrix(hits, **fparams)
            hit, fil, _ = lofo_eval(X, y, fl, files, models)
            best = max(fil, key=fil.get)
            row = {**fparams,
                   **{f"file_{k}": fil[k] for k in fil},
                   "best_model": best, "best_file": fil[best]}
            rows.append(row)
            print(f"[{i:3d}/{len(combos)}] mfcc={fparams['n_mfcc']:2d} fft={fparams['n_fft']:4d} "
                  f"hop={fparams['hop_length']:3d} psd={fparams['n_psd_bins']:3d} | "
                  + " ".join(f"{k}={v*100:5.2f}" for k, v in fil.items())
                  + f" | best {best}={fil[best]*100:.2f}")
        except Exception as e:
            print(f"[{i}] {fparams} ERROR: {e}")

    df = pd.DataFrame(rows).sort_values("best_file", ascending=False)
    df.to_csv(os.path.join(DATA_DIR, "grid_results.csv"), index=False)
    print("\n=== Top 15 ===")
    cols = ["n_mfcc", "n_fft", "hop_length", "n_psd_bins"] + \
           [c for c in df.columns if c.startswith("file_")] + ["best_model", "best_file"]
    print(df[cols].head(15).to_string(index=False))
    return df


def evaluate_at_params(hits, fparams, full=True):
    X, y, fl, files = build_feature_matrix(hits, **fparams)
    print(f"Feature matrix shape: {X.shape}")
    models = ["LR", "SVM_lin", "RF", "ET", "HGB", "GB"] if full else ["LR", "RF", "HGB"]
    print(f"\n--- LOFO (raw features) ---")
    hit, fil, per = lofo_eval(X, y, fl, files, models)
    for m in models:
        print(f"  {m:8s} hit={hit[m]*100:5.2f}  file={fil[m]*100:5.2f}  per-fold={[round(x*100) for x in per[m]]}")

    Xc = per_flange_centered(X, fl)
    print(f"\n--- LOFO (per-flange centered features) ---")
    hit_c, fil_c, per_c = lofo_eval(Xc, y, fl, files, models)
    for m in models:
        print(f"  {m:8s} hit={hit_c[m]*100:5.2f}  file={fil_c[m]*100:5.2f}  per-fold={[round(x*100) for x in per_c[m]]}")

    print(f"\n--- Ensembles (raw, soft-vote) ---")
    for ens in [("RF", "HGB"), ("LR", "RF", "HGB"), ("LR", "RF", "GB"),
                ("LR", "SVM_lin", "RF", "HGB"), ("RF", "ET", "HGB"),
                ("LR", "RF", "ET", "HGB", "GB")]:
        avg, per = lofo_ensemble(X, y, fl, files, ens)
        print(f"  {ens} -> {avg*100:.2f}  per-fold {[round(x*100) for x in per]}")

    print(f"\n--- Ensembles (centered, soft-vote) ---")
    for ens in [("RF", "HGB"), ("LR", "RF", "HGB"), ("LR", "RF", "ET", "HGB", "GB")]:
        avg, per = lofo_ensemble(Xc, y, fl, files, ens)
        print(f"  {ens} -> {avg*100:.2f}  per-fold {[round(x*100) for x in per]}")
    return X, y, fl, files


if __name__ == "__main__":
    import sys
    hits = build_hits()
    print(f"Hits: {len(hits)}")
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluate at one set of params
        fparams = json.loads(sys.argv[2])
        evaluate_at_params(hits, fparams, full=True)
    else:
        df = grid_search(hits, models=("LR", "RF"))
        # take top params and run full eval
        best = df.iloc[0]
        print("\nBest params:", best[["n_mfcc", "n_fft", "hop_length", "n_psd_bins"]].to_dict())
        fparams = {k: int(best[k]) for k in ["n_mfcc", "n_fft", "hop_length", "n_psd_bins"]}
        evaluate_at_params(hits, fparams, full=True)
