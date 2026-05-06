"""
Hammer-hit segmentation for the Flange-Invariant Acoustic Bolt-Looseness Detector.

Mirrors the segmentation logic used in the training notebook
(Final_Project_ML_Second_Attempt (6).ipynb) so inference matches training exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

# Same hyperparameters as the notebook.
IGNORE_START_SEC = 0.15
ENVELOPE_WIN_SEC = 0.01
MIN_PEAK_DISTANCE_SEC = 0.30
PEAK_HEIGHT_FACTOR = 2.5
PRE_HIT_SEC = 0.02
POST_HIT_SEC = 0.15


def normalize_audio(signal: np.ndarray) -> np.ndarray:
    """Normalize the audio signal to the range [-1, 1]."""
    signal = np.asarray(signal, dtype=np.float32)
    max_value = float(np.max(np.abs(signal))) if signal.size else 0.0
    if max_value == 0.0:
        return signal
    return signal / max_value


def split_into_hits(signal: np.ndarray, sr: int) -> dict:
    """
    Split a multi-hit recording into individual hammer-hit segments.

    Returns a dict with the trimmed signal, smoothed envelope,
    detected peak indices (in trimmed-signal coordinates), and
    the list of single-hit segments.
    """
    signal = np.asarray(signal, dtype=np.float32)

    start_idx = int(IGNORE_START_SEC * sr)
    signal_trimmed = signal[start_idx:]

    if signal_trimmed.size == 0:
        return {
            "hits": [],
            "peaks": np.array([], dtype=int),
            "envelope": np.array([], dtype=np.float32),
            "signal_trimmed": signal_trimmed,
            "start_offset": start_idx,
            "sr": sr,
        }

    envelope_win = max(1, int(ENVELOPE_WIN_SEC * sr))
    kernel = np.ones(envelope_win, dtype=np.float32) / envelope_win
    envelope = np.convolve(np.abs(signal_trimmed), kernel, mode="same")

    threshold = float(np.mean(envelope) + PEAK_HEIGHT_FACTOR * np.std(envelope))
    min_distance = int(MIN_PEAK_DISTANCE_SEC * sr)

    peaks, _ = find_peaks(envelope, height=threshold, distance=min_distance)

    pre_samples = int(PRE_HIT_SEC * sr)
    post_samples = int(POST_HIT_SEC * sr)

    hits = []
    for peak in peaks:
        start = max(0, peak - pre_samples)
        end = min(len(signal_trimmed), peak + post_samples)
        segment = signal_trimmed[start:end]
        if segment.size > 0:
            hits.append(segment.astype(np.float32))

    return {
        "hits": hits,
        "peaks": peaks.astype(int),
        "envelope": envelope.astype(np.float32),
        "signal_trimmed": signal_trimmed,
        "start_offset": start_idx,
        "sr": sr,
    }
