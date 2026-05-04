"""
Try YAMNet pretrained audio embeddings as features.

YAMNet is a CNN trained on AudioSet (~2M YouTube clips). It outputs
1024-dim embeddings every 0.48 seconds. We:
  1. Resample each hit to 16 kHz (YAMNet's required rate)
  2. Pad/loop to >= 0.96 seconds (YAMNet needs at least one frame)
  3. Average the per-frame embeddings to get one 1024-dim vector per hit
  4. Try LR/RF/LightGBM on these embeddings, alone and with v2 concat
  5. Also try LOAO (leave-one-area-out) as alternative validation
"""
import os, time, pickle
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import librosa
import tensorflow as tf
import tensorflow_hub as hub

import optimize as opt
from features_v2 import extract_features_v2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import lightgbm as lgb

CACHE = "_yamnet_emb_cache.npz"


def make_rf():
    return RandomForestClassifier(n_estimators=600, min_samples_leaf=3,
        max_features=0.3, max_depth=12, criterion="gini",
        class_weight="balanced", bootstrap=True, random_state=42, n_jobs=-1)


def make_lr():
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=3000, C=1.0,
                                              random_state=42))])


def make_lgb():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
        num_leaves=15, min_child_samples=5, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1)


def yamnet_embeddings(hits):
    """Extract YAMNet embeddings for all hits. Cached to disk."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["X"], z["y"], z["fl"], z["ar"], z["files"]

    print("Loading YAMNet from TF Hub...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    print("YAMNet loaded.")

    X, y, fl, ar, files = [], [], [], [], []
    t0 = time.time()
    for i, h in enumerate(hits):
        sig = h["signal"].astype(np.float32)
        sr = h["sr"]
        # Resample to 16 kHz (YAMNet requirement)
        if sr != 16000:
            sig16 = librosa.resample(sig, orig_sr=sr, target_sr=16000)
        else:
            sig16 = sig
        # YAMNet needs >= 15600 samples (~0.975s); pad by looping
        target = 16000
        if len(sig16) < target:
            reps = target // len(sig16) + 1
            sig16 = np.tile(sig16, reps)[:target]

        scores, embeddings, spectrogram = yamnet(sig16)
        emb = embeddings.numpy().mean(axis=0)   # (1024,)
        X.append(emb)
        y.append(h["torque"])
        fl.append(h["flange_id"])
        ar.append(h["area_id"])
        files.append(h["file_name"])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(hits)}  ({time.time()-t0:.1f}s elapsed)")

    X = np.array(X, dtype=np.float32)
    y = np.array(y); fl = np.array(fl); ar = np.array(ar)
    files = np.array(files)
    np.savez(CACHE, X=X, y=y, fl=fl, ar=ar, files=files)
    print(f"YAMNet embeddings: {X.shape}, cached to {CACHE}")
    return X, y, fl, ar, files


def lofo_eval(X, y, fl, files, model_factory, name):
    classes_ref = np.array([0, 25, 50])
    accs_file, accs_hit = [], []
    for f in [1, 2, 3, 4]:
        tr, te = fl != f, fl == f
        m = model_factory()
        m.fit(X[tr], y[tr])
        yp = m.predict(X[te])
        accs_hit.append(accuracy_score(y[te], yp))
        if hasattr(m, "predict_proba"):
            P = m.predict_proba(X[te])
            idx = [list(m.classes_).index(c) for c in classes_ref]
            P = P[:, idx]
            df = pd.DataFrame(P, columns=classes_ref); df["__f__"] = files[te]
            avg = df.groupby("__f__").mean().sort_index()
            pred = classes_ref[avg.values.argmax(axis=1)]
        else:
            df = pd.DataFrame({"f": files[te], "p": yp})
            grp = df.groupby("f")["p"].agg(lambda s: s.value_counts().idxmax())
            pred = grp.values; avg = pd.DataFrame(index=grp.index)
        true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                       .groupby("f")["y"].first().loc[avg.index.values].values)
        accs_file.append(accuracy_score(true_file, pred))
    print(f"  {name:35s}  hit={np.mean(accs_hit)*100:5.2f}  "
          f"file={np.mean(accs_file)*100:5.2f}  "
          f"per-fold {[round(a*100) for a in accs_file]}")
    return float(np.mean(accs_file))


def loao_eval(X, y, ar, files, model_factory, name):
    """Leave-one-area-out: 4 folds, each is a different area held out across
    all flanges. Also stratified differently than LOFO."""
    classes_ref = np.array([0, 25, 50])
    accs_file = []
    for a in [1, 2, 3, 4]:
        tr, te = ar != a, ar == a
        m = model_factory()
        m.fit(X[tr], y[tr])
        if hasattr(m, "predict_proba"):
            P = m.predict_proba(X[te])
            idx = [list(m.classes_).index(c) for c in classes_ref]
            P = P[:, idx]
            df = pd.DataFrame(P, columns=classes_ref); df["__f__"] = files[te]
            avg = df.groupby("__f__").mean().sort_index()
            pred = classes_ref[avg.values.argmax(axis=1)]
        else:
            yp = m.predict(X[te])
            df = pd.DataFrame({"f": files[te], "p": yp})
            grp = df.groupby("f")["p"].agg(lambda s: s.value_counts().idxmax())
            pred = grp.values; avg = pd.DataFrame(index=grp.index)
        true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                       .groupby("f")["y"].first().loc[avg.index.values].values)
        accs_file.append(accuracy_score(true_file, pred))
    print(f"  {name:35s}  LOAO file={np.mean(accs_file)*100:5.2f}  "
          f"per-fold {[round(a*100) for a in accs_file]}")
    return float(np.mean(accs_file))


# ----- main -----
hits = opt.build_hits()
print(f"Hits: {len(hits)}")

X_yam, y, fl, ar, files = yamnet_embeddings(hits)
print(f"YAMNet shape: {X_yam.shape}")

# Build v2 features for concat
print("\nBuilding v2 features...")
X_v2 = np.array([extract_features_v2(h["signal"], h["sr"],
                                     use_logmel=False, use_bands=False,
                                     use_band_decay=True, use_contrast=False,
                                     use_attack=False, use_slope=False)
                 for h in hits], dtype=np.float32)
print(f"v2 shape: {X_v2.shape}")

X_concat = np.concatenate([X_v2, X_yam], axis=1)
print(f"concat shape: {X_concat.shape}")

print("\n=== LOFO file-level (the main metric) ===")
print("[YAMNet only]")
lofo_eval(X_yam, y, fl, files, make_lr, "YAMNet + LR")
lofo_eval(X_yam, y, fl, files, make_rf, "YAMNet + RF")
lofo_eval(X_yam, y, fl, files, make_lgb, "YAMNet + LGB")

print("\n[v2 only - baseline]")
lofo_eval(X_v2, y, fl, files, make_rf, "v2 + RF (current best)")

print("\n[v2 + YAMNet concat]")
lofo_eval(X_concat, y, fl, files, make_lr, "v2+YAMNet + LR")
lofo_eval(X_concat, y, fl, files, make_rf, "v2+YAMNet + RF")
lofo_eval(X_concat, y, fl, files, make_lgb, "v2+YAMNet + LGB")

print("\n=== LOAO file-level (alternative validation) ===")
print("[YAMNet only]")
loao_eval(X_yam, y, ar, files, make_lr, "YAMNet + LR")
loao_eval(X_yam, y, ar, files, make_rf, "YAMNet + RF")

print("\n[v2 only]")
loao_eval(X_v2, y, ar, files, make_rf, "v2 + RF")

print("\n[v2 + YAMNet concat]")
loao_eval(X_concat, y, ar, files, make_rf, "v2+YAMNet + RF")
loao_eval(X_concat, y, ar, files, make_lr, "v2+YAMNet + LR")
