"""
Train the Flange-Invariant Logistic Regression pipeline from scratch on the
labeled audio files in the repository root, then serialize all artifacts so the
FastAPI inference server can use them at request time.

Reproduces the training procedure from the notebook
`Final_Project_ML_Second_Attempt (6).ipynb` — specifically the cells that
build the final flange-invariant LR + per-class isotonic calibration.

Outputs (written to backend/model/):
    flange_invariant_lr.pkl       sklearn Pipeline (StandardScaler + LR)
    selected_feature_indices.pkl  ndarray of 100 feature indices
    flange_means.pkl              {flange_id: mean_vector_of_selected_feats}
    calibrators.pkl               {class: IsotonicRegression}
    metadata.json                 small human-readable summary
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make backend importable when running this script directly from the repo root.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from feature_extraction import extract_hybrid_features  # noqa: E402
from segmentation import normalize_audio, split_into_hits  # noqa: E402

import librosa  # noqa: E402  (heavy import, kept after path setup)

RANDOM_STATE = 42
CLASSES_REF = np.array([0, 25, 50])
N_KEEP_FEATURES = 100
LABELED_PATTERN = re.compile(r"^(0|25|50)ftlbf(\d+)a(\d+)$")


def collect_labeled_files(data_dir: str) -> list[dict]:
    """Find every labeled audio file and parse torque/flange/area from the name."""
    rows = []
    for ext in ("*.m4a", "*.wav", "*.mp3"):
        for path in sorted(glob.glob(os.path.join(data_dir, ext))):
            name = os.path.splitext(os.path.basename(path))[0].replace(" ", "").lower()
            m = LABELED_PATTERN.match(name)
            if m is None:
                continue
            rows.append({
                "file_path": path,
                "file_name": os.path.basename(path),
                "torque": int(m.group(1)),
                "flange_id": int(m.group(2)),
                "area_id": int(m.group(3)),
            })
    return rows


def build_hit_dataset(files: list[dict]) -> list[dict]:
    """Load every labeled file, normalize, segment into per-hit signals."""
    hits = []
    for row in files:
        signal, sr = librosa.load(row["file_path"], sr=None, mono=True)
        signal = normalize_audio(signal)
        seg = split_into_hits(signal, sr)
        for hit_id, hit_signal in enumerate(seg["hits"], start=1):
            hits.append({
                **row,
                "sr": sr,
                "hit_id": hit_id,
                "signal": hit_signal,
            })
    return hits


def fit_per_flange_means(X: np.ndarray, fl: np.ndarray) -> dict:
    return {int(f): X[fl == f].mean(axis=0) for f in np.unique(fl)}


def apply_per_flange_centering(X: np.ndarray, fl: np.ndarray, means: dict) -> np.ndarray:
    Xc = X.astype(float).copy()
    if not means:
        return Xc
    fallback = np.mean(list(means.values()), axis=0)
    for i in range(len(X)):
        Xc[i] -= means.get(int(fl[i]), fallback)
    return Xc


def fit_torque_discriminative_features(
    X: np.ndarray, y: np.ndarray, fl: np.ndarray, n_keep: int = N_KEEP_FEATURES,
) -> np.ndarray:
    f_torque, _ = f_classif(X, y)
    f_flange, _ = f_classif(X, fl)
    score = np.nan_to_num(f_torque) / (np.nan_to_num(f_flange) + 1.0)
    return np.argsort(score)[::-1][:n_keep]


def make_lr_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, C=C, random_state=RANDOM_STATE)),
    ])


def fit_lofo_isotonic_calibrators(
    X: np.ndarray, y: np.ndarray, fl: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Fit a per-class isotonic calibrator on LOFO out-of-fold LR probabilities."""
    oof = np.zeros((len(y), len(CLASSES_REF)), dtype=float)
    for held in np.unique(fl):
        tr = fl != held
        te = fl == held
        keep_h = fit_torque_discriminative_features(X[tr], y[tr], fl[tr])
        means_h = fit_per_flange_means(X[tr][:, keep_h], fl[tr])
        Xtr = apply_per_flange_centering(X[tr][:, keep_h], fl[tr], means_h)
        Xte = apply_per_flange_centering(X[te][:, keep_h], fl[te], means_h)
        m = make_lr_pipeline()
        m.fit(Xtr, y[tr])
        proba = m.predict_proba(Xte)
        idx = [list(m.classes_).index(c) for c in CLASSES_REF]
        oof[te] = proba[:, idx]

    calibrators = {}
    for ci, c in enumerate(CLASSES_REF):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(oof[:, ci], (y == c).astype(float))
        calibrators[int(c)] = iso
    return calibrators, oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(THIS_DIR.parent),
                        help="Folder containing the labeled .m4a audio files.")
    parser.add_argument("--out-dir", default=str(THIS_DIR / "model"),
                        help="Where to write the serialized model artifacts.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Scanning labeled files in {args.data_dir} ...")
    files = collect_labeled_files(args.data_dir)
    if not files:
        raise SystemExit(f"No labeled files found in {args.data_dir}")
    print(f"[train] Found {len(files)} labeled audio files.")

    print("[train] Segmenting into single-hit samples ...")
    hits = build_hit_dataset(files)
    print(f"[train] Total single-hit samples: {len(hits)}")

    print("[train] Extracting 150-D feature vectors ...")
    X = np.stack([extract_hybrid_features(h["signal"], h["sr"]) for h in hits])
    y = np.array([h["torque"] for h in hits])
    fl = np.array([h["flange_id"] for h in hits])
    print(f"[train] Feature matrix: {X.shape}  classes: {sorted(set(y))}  flanges: {sorted(set(fl))}")

    print("[train] Selecting top-100 torque-discriminative features ...")
    keep_idx = fit_torque_discriminative_features(X, y, fl)
    flange_means = fit_per_flange_means(X[:, keep_idx], fl)

    print("[train] Fitting Flange-Invariant LR ...")
    X_lr = apply_per_flange_centering(X[:, keep_idx], fl, flange_means)
    model = make_lr_pipeline()
    model.fit(X_lr, y)
    train_acc = model.score(X_lr, y)
    print(f"[train] In-sample LR accuracy (sanity only): {train_acc * 100:.2f}%")

    print("[train] Fitting per-class isotonic calibrators on LOFO out-of-fold ...")
    calibrators, oof = fit_lofo_isotonic_calibrators(X, y, fl)

    raw_pred = CLASSES_REF[oof.argmax(axis=1)]
    print(f"[train] LOFO raw hit-level accuracy: {(raw_pred == y).mean() * 100:.2f}%")

    cal = np.column_stack([
        calibrators[int(c)].predict(np.clip(oof[:, ci], 0, 1))
        for ci, c in enumerate(CLASSES_REF)
    ])
    cal = np.clip(cal, 1e-9, None)
    cal = cal / cal.sum(axis=1, keepdims=True)
    cal_pred = CLASSES_REF[cal.argmax(axis=1)]
    print(f"[train] LOFO calibrated hit-level accuracy: {(cal_pred == y).mean() * 100:.2f}%")

    # ----- Confusion matrices + per-class metrics -----
    def confusion(y_true, y_pred):
        m = np.zeros((len(CLASSES_REF), len(CLASSES_REF)), dtype=int)
        for ti, t in enumerate(CLASSES_REF):
            for pi, p in enumerate(CLASSES_REF):
                m[ti, pi] = int(((y_true == t) & (y_pred == p)).sum())
        return m

    cm_raw = confusion(y, raw_pred)
    cm_cal = confusion(y, cal_pred)

    # File-level (soft-vote) accuracy from LOFO OOF
    files_arr = np.array([h["file_name"] for h in hits])
    file_pred_rows = []
    file_true_rows = []
    for fname in np.unique(files_arr):
        mask = files_arr == fname
        avg = cal[mask].mean(axis=0)
        file_pred_rows.append(int(CLASSES_REF[int(np.argmax(avg))]))
        file_true_rows.append(int(y[mask][0]))
    file_pred = np.array(file_pred_rows)
    file_true = np.array(file_true_rows)
    cm_file = confusion(file_true, file_pred)
    file_acc = float((file_pred == file_true).mean())

    def per_class_recall(cm):
        rec = {}
        for ci, c in enumerate(CLASSES_REF):
            denom = cm[ci, :].sum()
            rec[int(c)] = float(cm[ci, ci] / denom) if denom > 0 else 0.0
        return rec

    print(f"[train] LOFO file-level accuracy (soft-vote): {file_acc * 100:.2f}%")

    print(f"[train] Writing artifacts to {out_dir} ...")
    with open(out_dir / "flange_invariant_lr.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(out_dir / "selected_feature_indices.pkl", "wb") as f:
        pickle.dump(keep_idx.astype(int), f)
    with open(out_dir / "flange_means.pkl", "wb") as f:
        pickle.dump({int(k): v for k, v in flange_means.items()}, f)
    with open(out_dir / "calibrators.pkl", "wb") as f:
        pickle.dump(calibrators, f)

    metadata = {
        "model": "Flange-Invariant Logistic Regression",
        "classes": CLASSES_REF.tolist(),
        "n_total_features": int(X.shape[1]),
        "n_selected_features": int(len(keep_idx)),
        "flanges_seen_in_training": sorted(int(f) for f in flange_means.keys()),
        "n_training_hits": int(X.shape[0]),
        "n_training_files": int(len(files)),
        "lofo_raw_hit_accuracy": float((raw_pred == y).mean()),
        "lofo_calibrated_hit_accuracy": float((cal_pred == y).mean()),
        "lofo_file_accuracy_softvote": file_acc,
        "in_sample_hit_accuracy": float(train_acc),
        "feature_extraction": {
            "n_mfcc": 13, "n_fft": 512, "hop_length": 128, "n_psd_bins": 64,
        },
        "segmentation": {
            "ignore_start_sec": 0.15, "envelope_win_sec": 0.01,
            "min_peak_distance_sec": 0.30, "peak_height_factor": 2.5,
            "pre_hit_sec": 0.02, "post_hit_sec": 0.15,
        },
        "confusion_matrices": {
            "labels": [int(c) for c in CLASSES_REF],
            "hit_level_raw":        cm_raw.tolist(),
            "hit_level_calibrated": cm_cal.tolist(),
            "file_level":           cm_file.tolist(),
        },
        "per_class_recall": {
            "hit_level_raw":        per_class_recall(cm_raw),
            "hit_level_calibrated": per_class_recall(cm_cal),
            "file_level":           per_class_recall(cm_file),
        },
        "n_lofo_files": int(len(file_pred)),
        "training_data": {
            "hits_per_class": {int(c): int((y == c).sum()) for c in CLASSES_REF},
            "hits_per_flange": {int(f): int((fl == f).sum()) for f in np.unique(fl)},
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("[train] Done.")


if __name__ == "__main__":
    main()
