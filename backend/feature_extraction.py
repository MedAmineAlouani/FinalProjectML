"""
150-D feature extraction for the Flange-Invariant Acoustic Bolt-Looseness Detector.

This is a faithful port of the `extract_hybrid_features` pipeline used in the
training notebook. The output dimensionality MUST match what the trained model
expects (150 features), so do not modify hyperparameters here without retraining.
"""
from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import welch

# Tuned hyperparameters from the notebook (LOFO-tuned).
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 128
N_PSD_BINS = 64

PER_BAND_DECAY_BANDS = (
    (100, 500),
    (500, 2000),
    (2000, 6000),
    (6000, 12000),
    (12000, 24000),
)


def safe_mean_std(feature_matrix: np.ndarray) -> np.ndarray:
    """Concatenate (mean, std) along the time axis of a (n_features, n_frames) matrix."""
    feature_matrix = np.asarray(feature_matrix)
    return np.concatenate([feature_matrix.mean(axis=1), feature_matrix.std(axis=1)])


def extract_psd_features(signal: np.ndarray, sr: int, n_bins: int = N_PSD_BINS) -> np.ndarray:
    """Fixed-length log-PSD via Welch."""
    freqs, psd = welch(signal, fs=sr, nperseg=min(1024, len(signal)))
    psd_log = np.log10(psd + 1e-12)
    if len(psd_log) >= n_bins:
        return psd_log[:n_bins]
    return np.pad(psd_log, (0, n_bins - len(psd_log)), mode="constant")


def extract_mfcc_features(signal: np.ndarray, sr: int) -> np.ndarray:
    """MFCC mean/std + delta MFCC mean/std."""
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=N_MFCC,
        n_fft=min(N_FFT, len(signal)),
        hop_length=min(HOP_LENGTH, max(1, len(signal) // 2)),
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    return np.concatenate([safe_mean_std(mfcc), safe_mean_std(mfcc_delta)])


def extract_spectral_summary_features(signal: np.ndarray, sr: int) -> np.ndarray:
    """Centroid / bandwidth / rolloff / flatness / ZCR / RMS, mean+std each."""
    n_fft = min(N_FFT, len(signal))
    hop_length = min(HOP_LENGTH, max(1, len(signal) // 2))
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(y=signal, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop_length)
    return np.concatenate([safe_mean_std(x) for x in [centroid, bandwidth, rolloff, flatness, zcr, rms]])


def extract_frequency_shape_features(signal: np.ndarray, sr: int) -> np.ndarray:
    """Dominant frequency + spectral entropy."""
    freqs, psd = welch(signal, fs=sr, nperseg=min(1024, len(signal)))
    psd = psd + 1e-12
    dominant_freq = freqs[np.argmax(psd)]
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return np.array([dominant_freq, spectral_entropy])


def extract_decay_features(signal: np.ndarray, sr: int) -> np.ndarray:
    """Time-domain decay summaries: peak value, peak time, 50%/10% decay times, log-slope."""
    abs_signal = np.abs(signal)
    if np.max(abs_signal) == 0:
        return np.zeros(5)
    win = max(1, int(0.002 * sr))
    envelope = np.convolve(abs_signal, np.ones(win) / win, mode="same")
    peak_idx = int(np.argmax(envelope))
    peak_value = envelope[peak_idx] + 1e-12
    after = envelope[peak_idx:]
    below_50 = np.where(after <= 0.50 * peak_value)[0]
    below_10 = np.where(after <= 0.10 * peak_value)[0]
    decay_50 = below_50[0] / sr if len(below_50) > 0 else len(after) / sr
    decay_10 = below_10[0] / sr if len(below_10) > 0 else len(after) / sr
    t = np.arange(len(after)) / sr
    log_env = np.log(after + 1e-8)
    slope = np.polyfit(t, log_env, 1)[0] if len(t) > 5 else 0.0
    return np.array([peak_value, peak_idx / sr, decay_50, decay_10, slope])


def extract_per_band_decay_features(
    signal: np.ndarray,
    sr: int,
    bands: tuple = PER_BAND_DECAY_BANDS,
) -> np.ndarray:
    """
    Per-band T60-style decay features. For each band:
      - log-envelope slope after the peak
      - half-life in frames
      - peak-to-mean ratio
    """
    n = len(signal)
    n_fft = min(512, n)
    hop = max(1, n // 64)
    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    out = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        if not m.any() or S.shape[1] < 4:
            out.extend([0.0, 0.0, 0.0])
            continue
        env = S[m].mean(axis=0) + 1e-12
        peak_t = int(np.argmax(env))
        if peak_t >= len(env) - 3:
            out.extend([0.0, 0.0, 0.0])
            continue
        after = env[peak_t:]
        log_env = np.log(after + 1e-9)
        t = np.arange(len(after))
        slope = np.polyfit(t, log_env, 1)[0]
        below_50 = np.where(after <= 0.5 * after[0])[0]
        half_life = below_50[0] if len(below_50) > 0 else len(after)
        ptm = after[0] / (after.mean() + 1e-12)
        out.extend([slope, half_life / hop, ptm])
    return np.array(out)


def extract_hybrid_features(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Full 150-D feature vector used by the Flange-Invariant LR.

    Layout:
        Welch log-PSD                      -> 64
        MFCC mean/std + delta MFCC m/std   -> 52
        Spectral summary (6 feats * 2)     -> 12
        Dominant freq + spectral entropy   ->  2
        Global decay (5)                   ->  5
        Per-band T60-style decay (5 * 3)   -> 15
        TOTAL                              -> 150
    """
    signal = np.asarray(signal, dtype=np.float32)
    return np.concatenate([
        extract_psd_features(signal, sr),
        extract_mfcc_features(signal, sr),
        extract_spectral_summary_features(signal, sr),
        extract_frequency_shape_features(signal, sr),
        extract_decay_features(signal, sr),
        extract_per_band_decay_features(signal, sr),
    ]).astype(np.float64)
