"""
Test hit-level data augmentation on the v2 (per_band_decay) feature set.

Each augmentation is applied to TRAIN hits only (test hits stay original).
We try several augmentation budgets and combinations.
"""
import json, sys, time
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import optimize as opt
from features_v2 import extract_features_v2

RNG = np.random.RandomState(42)


def aug_gain(sig, db_range=3.0):
    g = 10 ** (RNG.uniform(-db_range, db_range) / 20.0)
    return np.clip(sig * g, -1.0, 1.0)


def aug_noise(sig, snr_db_range=(25, 40)):
    snr = RNG.uniform(*snr_db_range)
    p_sig = (sig ** 2).mean() + 1e-12
    p_noise = p_sig / (10 ** (snr / 10))
    n = RNG.normal(0, np.sqrt(p_noise), size=sig.shape)
    return np.clip(sig + n, -1.0, 1.0)


def aug_time_stretch(sig, sr, rate_range=(0.95, 1.05)):
    rate = RNG.uniform(*rate_range)
    out = librosa.effects.time_stretch(sig, rate=rate)
    # keep length similar to original (truncate or pad)
    if len(out) >= len(sig):
        return out[:len(sig)]
    return np.pad(out, (0, len(sig) - len(out)))


def aug_pitch(sig, sr, semitone_range=0.5):
    s = RNG.uniform(-semitone_range, semitone_range)
    return librosa.effects.pitch_shift(sig, sr=sr, n_steps=s)


def aug_shift(sig, max_shift_sec=0.005, sr=48000):
    n = int(RNG.uniform(-max_shift_sec, max_shift_sec) * sr)
    if n == 0: return sig
    if n > 0: return np.concatenate([np.zeros(n), sig[:-n]])
    return np.concatenate([sig[-n:], np.zeros(-n)])


AUGS = {
    "gain":   lambda sig, sr: aug_gain(sig),
    "noise":  lambda sig, sr: aug_noise(sig),
    "stretch": aug_time_stretch,
    "pitch":  aug_pitch,
    "shift":  lambda sig, sr: aug_shift(sig, sr=sr),
}


def feat(sig, sr):
    return extract_features_v2(sig, sr,
        use_logmel=False, use_bands=False, use_band_decay=True,
        use_contrast=False, use_attack=False, use_slope=False)


def make_rf_tuned():
    return RandomForestClassifier(
        n_estimators=600, min_samples_leaf=3, max_features=0.3, max_depth=12,
        criterion="gini", class_weight="balanced", bootstrap=True,
        random_state=42, n_jobs=-1)


def lofo_with_aug(hits, aug_names, n_copies=1):
    """For each LOFO fold, augment TRAIN hits only and add `n_copies` of each."""
    classes_ref = np.array([0, 25, 50])
    accs = []
    for f in [1, 2, 3, 4]:
        train_hits = [h for h in hits if h["flange_id"] != f]
        test_hits  = [h for h in hits if h["flange_id"] == f]

        # Build train set (original + augmented)
        X_train, y_train = [], []
        for h in train_hits:
            X_train.append(feat(h["signal"], h["sr"]))
            y_train.append(h["torque"])
        # augment
        for h in train_hits:
            for _ in range(n_copies):
                sig = h["signal"].copy()
                for name in aug_names:
                    sig = AUGS[name](sig, h["sr"])
                X_train.append(feat(sig, h["sr"]))
                y_train.append(h["torque"])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Test set (original only)
        X_test = np.array([feat(h["signal"], h["sr"]) for h in test_hits])
        y_test = np.array([h["torque"] for h in test_hits])
        files_te = np.array([h["file_name"] for h in test_hits])

        m = make_rf_tuned()
        m.fit(X_train, y_train)
        P = m.predict_proba(X_test)
        idx = [list(m.classes_).index(c) for c in classes_ref]
        P = P[:, idx]
        df = pd.DataFrame(P, columns=classes_ref)
        df["__f__"] = files_te
        avg = df.groupby("__f__").mean().sort_index()
        pred = classes_ref[avg.values.argmax(axis=1)]
        true_file = (pd.DataFrame({"f": files_te, "y": y_test})
                       .groupby("f")["y"].first().loc[avg.index.values].values)
        accs.append(accuracy_score(true_file, pred))
    return float(np.mean(accs)), accs


hits = opt.build_hits()

# baseline (no augmentation)
print("--- baseline (no augmentation) ---")
acc, per = lofo_with_aug(hits, [], n_copies=0)
print(f"  {acc*100:.2f}  per-fold {[round(a*100) for a in per]}")

# Each augmentation alone, 1 copy
print("\n--- single augmentation, 1 extra copy per train hit ---")
for name in AUGS:
    t0 = time.time()
    a, p = lofo_with_aug(hits, [name], n_copies=1)
    print(f"  {name:8s} {a*100:5.2f}  per-fold {[round(x*100) for x in p]}  [{time.time()-t0:.1f}s]")

# Combinations of cheap augs
print("\n--- combined augmentations ---")
combos = [
    ["gain", "noise"],
    ["gain", "noise", "shift"],
    ["gain", "noise", "stretch"],
    ["gain", "noise", "shift", "stretch"],
]
for c in combos:
    t0 = time.time()
    a, p = lofo_with_aug(hits, c, n_copies=1)
    print(f"  {'+'.join(c):30s} {a*100:5.2f}  per-fold {[round(x*100) for x in p]}  [{time.time()-t0:.1f}s]")

# 2 copies of best single
print("\n--- 2 extra copies (gain+noise) ---")
a, p = lofo_with_aug(hits, ["gain", "noise"], n_copies=2)
print(f"  {a*100:.2f}  per-fold {[round(x*100) for x in p]}")
