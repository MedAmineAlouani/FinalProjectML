"""
Flange-Invariant Logistic Regression inference for the bolt-looseness detector.

Loads the artifacts produced by `train_and_export.py` and exposes a single
`predict_from_audio` function that the FastAPI route calls.
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import librosa
import numpy as np

from feature_extraction import extract_hybrid_features
from segmentation import normalize_audio, split_into_hits

CLASSES_REF = np.array([0, 25, 50])

# Heuristics for user-friendly warnings.
MIN_DURATION_SEC = 0.30
MIN_HITS_FOR_STABLE_PREDICTION = 2
LOW_CONFIDENCE_THRESHOLD = 0.50
MEDIUM_CONFIDENCE_THRESHOLD = 0.70


class FlangeInvariantLR:
    """Loads the trained Flange-Invariant LR pipeline and runs inference."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.model_dir = model_dir

        with open(model_dir / "flange_invariant_lr.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(model_dir / "selected_feature_indices.pkl", "rb") as f:
            self.keep_idx = np.asarray(pickle.load(f), dtype=int)
        with open(model_dir / "flange_means.pkl", "rb") as f:
            self.flange_means = {int(k): np.asarray(v) for k, v in pickle.load(f).items()}
        with open(model_dir / "calibrators.pkl", "rb") as f:
            self.calibrators = pickle.load(f)
        with open(model_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        # Pre-compute the column order used inside the trained LR.
        self._class_index = [list(self.model.classes_).index(int(c)) for c in CLASSES_REF]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _center_for_flange(self, X_selected: np.ndarray, flange_id: int) -> np.ndarray:
        """Subtract the per-flange training mean from each row of the selected features."""
        if flange_id in self.flange_means:
            mean_vec = self.flange_means[flange_id]
        else:
            mean_vec = np.mean(list(self.flange_means.values()), axis=0)
        return X_selected.astype(float) - mean_vec

    def _calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply per-class isotonic calibration in CLASSES_REF order, then renormalise."""
        cal = np.column_stack([
            self.calibrators[int(c)].predict(np.clip(raw_proba[:, ci], 0.0, 1.0))
            for ci, c in enumerate(CLASSES_REF)
        ])
        cal = np.clip(cal, 1e-9, None)
        return cal / cal.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_from_audio(
        self,
        audio_bytes: bytes,
        flange_id: int,
        file_extension: str = ".m4a",
    ) -> dict:
        """
        Run the full Flange-Invariant LR pipeline on a raw audio byte buffer.

        Returns a JSON-serialisable dict containing the waveform, envelope,
        detected hits, per-hit probabilities, soft-vote average, and the
        final torque prediction.
        """
        # ------------------------- Load audio -------------------------- #
        try:
            signal, sr = librosa.load(
                io.BytesIO(audio_bytes), sr=None, mono=True,
            )
        except Exception:
            # Fall back to passing a temp file (some browser blob types confuse soundfile).
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                signal, sr = librosa.load(tmp.name, sr=None, mono=True)

        signal = normalize_audio(signal)
        duration_sec = float(len(signal) / sr) if sr else 0.0

        # ------------------------- Segment hits ------------------------ #
        seg = split_into_hits(signal, sr)
        hits = seg["hits"]
        peaks_in_trimmed = seg["peaks"].tolist()
        # Convert peaks back to original-signal seconds for visualisation.
        peak_times_sec = [
            float((p + seg["start_offset"]) / sr) for p in peaks_in_trimmed
        ]

        warnings: list[str] = []

        if duration_sec < MIN_DURATION_SEC:
            warnings.append("Audio too short or too noisy. Try recording a longer sample.")

        if len(hits) == 0:
            warnings.append("No clear hit detected. Try striking the flange again.")
            return self._empty_response(
                signal=signal, envelope=seg["envelope"], sr=sr,
                start_offset=seg["start_offset"], duration_sec=duration_sec,
                peak_times_sec=peak_times_sec, warnings=warnings,
                flange_id=flange_id,
            )

        if len(hits) < MIN_HITS_FOR_STABLE_PREDICTION:
            warnings.append("Only one hit detected. Prediction may be less stable.")

        # ------------------------- Extract features -------------------- #
        X = np.stack([extract_hybrid_features(h, sr) for h in hits])
        X_selected = X[:, self.keep_idx]
        X_centered = self._center_for_flange(X_selected, int(flange_id))

        # ------------------------- Predict ----------------------------- #
        raw_proba = self.model.predict_proba(X_centered)
        raw_proba = raw_proba[:, self._class_index]
        per_hit_proba = self._calibrate(raw_proba)
        avg_proba = per_hit_proba.mean(axis=0)
        final_class_idx = int(np.argmax(avg_proba))
        final_class = int(CLASSES_REF[final_class_idx])
        final_confidence = float(avg_proba[final_class_idx])

        if final_confidence < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(
                "Low confidence prediction. Try recording multiple clean hammer strikes."
            )

        # ------------------------- Build response ---------------------- #
        per_hit = []
        for i, (hit_signal, hit_proba) in enumerate(zip(hits, per_hit_proba)):
            cls_idx = int(np.argmax(hit_proba))
            per_hit.append({
                "hit_id": i + 1,
                "time_sec": peak_times_sec[i] if i < len(peak_times_sec) else None,
                "duration_sec": float(len(hit_signal) / sr),
                "waveform": _downsample_for_plot(hit_signal, target=600),
                "probabilities": {
                    str(int(c)): float(hit_proba[ci]) for ci, c in enumerate(CLASSES_REF)
                },
                "predicted_torque": int(CLASSES_REF[cls_idx]),
                "confidence": float(hit_proba[cls_idx]),
            })

        return {
            "ok": True,
            "flange_id": int(flange_id),
            "sample_rate": int(sr),
            "duration_sec": duration_sec,
            "n_hits": len(hits),
            "hit_times_sec": peak_times_sec,
            "waveform": _downsample_for_plot(signal, target=2000),
            "envelope": _downsample_for_plot(
                seg["envelope"], target=2000, offset_sec=seg["start_offset"] / sr,
            ),
            "per_hit": per_hit,
            "averaged_probabilities": {
                str(int(c)): float(avg_proba[ci]) for ci, c in enumerate(CLASSES_REF)
            },
            "final_prediction": {
                "torque_ftlbs": final_class,
                "confidence": final_confidence,
                "confidence_level": _confidence_level(final_confidence),
            },
            "warnings": warnings,
            "model_metadata": {
                "name": self.metadata.get("model"),
                "lofo_calibrated_hit_accuracy": self.metadata.get("lofo_calibrated_hit_accuracy"),
                "n_training_hits": self.metadata.get("n_training_hits"),
            },
        }

    def _empty_response(self, signal, envelope, sr, start_offset, duration_sec,
                        peak_times_sec, warnings, flange_id) -> dict:
        return {
            "ok": False,
            "flange_id": int(flange_id),
            "sample_rate": int(sr),
            "duration_sec": float(duration_sec),
            "n_hits": 0,
            "hit_times_sec": peak_times_sec,
            "waveform": _downsample_for_plot(signal, target=2000),
            "envelope": _downsample_for_plot(
                envelope, target=2000, offset_sec=start_offset / sr if sr else 0.0,
            ),
            "per_hit": [],
            "averaged_probabilities": None,
            "final_prediction": None,
            "warnings": warnings,
            "model_metadata": {
                "name": self.metadata.get("model"),
            },
        }


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _confidence_level(p: float) -> str:
    if p >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "high"
    if p >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _downsample_for_plot(arr: np.ndarray, target: int = 2000, offset_sec: float = 0.0) -> dict:
    """
    Compress a long array into a short list of (x, y) pairs suitable for plotting in the browser.

    Uses min/max bucketing so the visual envelope of the waveform is preserved.
    """
    arr = np.asarray(arr, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return {"values": [], "offset_sec": float(offset_sec), "length": 0}

    if n <= target:
        return {"values": arr.tolist(), "offset_sec": float(offset_sec), "length": n}

    bucket = max(1, n // target)
    n_buckets = n // bucket
    trimmed = arr[: n_buckets * bucket].reshape(n_buckets, bucket)
    # Interleave min and max so positive/negative excursions both show up.
    mins = trimmed.min(axis=1)
    maxs = trimmed.max(axis=1)
    out = np.empty(n_buckets * 2, dtype=np.float32)
    out[0::2] = mins
    out[1::2] = maxs
    return {
        "values": out.tolist(),
        "offset_sec": float(offset_sec),
        "length": n,
        "bucket_size": int(bucket),
    }
