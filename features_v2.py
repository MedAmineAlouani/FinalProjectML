"""
Richer feature set: adds log-mel quantiles, per-band energies, per-band
T60 decay, spectral contrast, attack/onset shape, spectral slope.

These extend (do not replace) the original `extract_hybrid_features`.
The combined vector goes from 135 dims to ~250-300 dims depending on
which extras are enabled.
"""
import numpy as np
import librosa
from scipy.signal import welch

from optimize import (extract_features as _legacy_features,
                      safe_mean_std)


# ------------------------------------------------- log-mel quantiles
def logmel_quantile_features(signal, sr, n_mels=40, n_fft=512, hop_length=128,
                             quantiles=(0.1, 0.5, 0.9)):
    n = len(signal)
    n_fft_eff = min(n_fft, n)
    hop_eff = max(1, min(hop_length, n // 2))
    M = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft_eff, hop_length=hop_eff,
                                       power=2.0)
    L = librosa.power_to_db(M + 1e-12)        # (n_mels, T)
    feats = [L.mean(axis=1), L.std(axis=1)]
    for q in quantiles:
        feats.append(np.quantile(L, q, axis=1))
    # also the temporal slope per band (energy evolution)
    if L.shape[1] >= 4:
        t = np.arange(L.shape[1])
        slopes = np.array([np.polyfit(t, row, 1)[0] for row in L])
    else:
        slopes = np.zeros(n_mels)
    feats.append(slopes)
    return np.concatenate(feats)


# ------------------------------------------------- per-band energy ratios
def band_energy_features(signal, sr,
                         bands=((0, 250), (250, 500), (500, 1000), (1000, 2000),
                                (2000, 4000), (4000, 8000), (8000, 16000), (16000, 24000))):
    freqs, psd = welch(signal, fs=sr, nperseg=min(1024, len(signal)))
    total = psd.sum() + 1e-12
    ratios = []
    log_ratios = []
    centroids = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            ratios.append(0.0); log_ratios.append(-9.0); centroids.append(0.0)
            continue
        e = psd[m].sum()
        r = e / total
        ratios.append(r)
        log_ratios.append(np.log10(r + 1e-9))
        centroids.append((freqs[m] * psd[m]).sum() / (e + 1e-12))
    return np.concatenate([ratios, log_ratios, centroids])


# ------------------------------------------------- per-band T60-style decay
def per_band_decay_features(signal, sr,
                            bands=((100, 500), (500, 2000), (2000, 6000),
                                   (6000, 12000), (12000, 24000))):
    n = len(signal)
    n_fft = min(512, n)
    hop = max(1, n // 64)
    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop))  # (F, T)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    out = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        if not m.any() or S.shape[1] < 4:
            out.extend([0.0, 0.0, 0.0])
            continue
        env = S[m].mean(axis=0) + 1e-12       # (T,)
        peak_t = int(np.argmax(env))
        if peak_t >= len(env) - 3:
            out.extend([0.0, 0.0, 0.0])
            continue
        after = env[peak_t:]
        log_env = np.log(after + 1e-9)
        t = np.arange(len(after))
        slope = np.polyfit(t, log_env, 1)[0]   # decay rate (negative is decay)
        # half-life (samples to reach 50% of peak)
        below_50 = np.where(after <= 0.5 * after[0])[0]
        half_life = below_50[0] if len(below_50) > 0 else len(after)
        # peak-to-mean ratio (transient sharpness in this band)
        ptm = after[0] / (after.mean() + 1e-12)
        out.extend([slope, half_life / hop, ptm])
    return np.array(out)


# ------------------------------------------------- spectral contrast
def spectral_contrast_features(signal, sr, n_bands=6, n_fft=512, hop_length=128):
    n = len(signal)
    n_fft_eff = min(n_fft, n)
    hop_eff = max(1, min(hop_length, n // 2))
    try:
        sc = librosa.feature.spectral_contrast(y=signal, sr=sr, n_bands=n_bands,
                                               n_fft=n_fft_eff, hop_length=hop_eff)
        return safe_mean_std(sc)
    except Exception:
        return np.zeros(2 * (n_bands + 1))


# ------------------------------------------------- attack/onset shape
def attack_features(signal, sr):
    abs_s = np.abs(signal)
    if abs_s.max() == 0:
        return np.zeros(7)
    win = max(1, int(0.001 * sr))   # 1 ms smoothing for sharp attack
    env = np.convolve(abs_s, np.ones(win) / win, mode="same")
    peak_idx = int(np.argmax(env))
    peak_val = env[peak_idx] + 1e-12

    # Rise time: 10% -> 90% of peak (samples before peak)
    rising = env[:peak_idx + 1]
    if len(rising) < 2:
        rise_10_90 = 0.0
    else:
        i10 = np.where(rising >= 0.1 * peak_val)[0]
        i90 = np.where(rising >= 0.9 * peak_val)[0]
        rise_10_90 = (i90[0] - i10[0]) / sr if (len(i10) and len(i90)) else 0.0

    # Crest factor (peak / RMS)
    rms = np.sqrt(np.mean(signal ** 2)) + 1e-12
    crest = peak_val / rms

    # Rate of attack: max derivative of envelope before peak
    deriv = np.diff(env)
    pre_peak_deriv = deriv[:peak_idx] if peak_idx > 0 else deriv
    max_attack_rate = float(pre_peak_deriv.max()) if len(pre_peak_deriv) else 0.0

    # Energy distribution (cumulative)
    e_cum = np.cumsum(signal ** 2)
    e_cum /= (e_cum[-1] + 1e-12)
    t10 = np.searchsorted(e_cum, 0.1) / sr
    t50 = np.searchsorted(e_cum, 0.5) / sr
    t90 = np.searchsorted(e_cum, 0.9) / sr

    return np.array([rise_10_90, crest, max_attack_rate,
                     t10, t50, t90, peak_idx / sr])


# ------------------------------------------------- spectral slope
def spectral_slope_features(signal, sr):
    freqs, psd = welch(signal, fs=sr, nperseg=min(1024, len(signal)))
    psd = psd + 1e-12
    log_psd = np.log10(psd)
    log_f = np.log10(freqs + 1.0)
    # Linear fit on log-log scale
    slope, intercept = np.polyfit(log_f, log_psd, 1)
    # Quadratic (curvature) — rolloff vs slope
    quad = np.polyfit(log_f, log_psd, 2)[0]
    return np.array([slope, intercept, quad])


# ------------------------------------------------- combined extractor
def extract_features_v2(signal, sr,
                        n_mfcc=13, n_fft=512, hop_length=128, n_psd_bins=64,
                        use_logmel=True, use_bands=True, use_band_decay=True,
                        use_contrast=True, use_attack=True, use_slope=True):
    """
    Backwards-compatible: starts with the original 135-dim hybrid vector,
    then appends optional new feature blocks.
    """
    parts = [_legacy_features(signal, sr, n_mfcc=n_mfcc, n_fft=n_fft,
                              hop_length=hop_length, n_psd_bins=n_psd_bins)]
    if use_logmel:
        parts.append(logmel_quantile_features(signal, sr, n_mels=40,
                                              n_fft=n_fft, hop_length=hop_length))
    if use_bands:
        parts.append(band_energy_features(signal, sr))
    if use_band_decay:
        parts.append(per_band_decay_features(signal, sr))
    if use_contrast:
        parts.append(spectral_contrast_features(signal, sr,
                                                n_fft=n_fft, hop_length=hop_length))
    if use_attack:
        parts.append(attack_features(signal, sr))
    if use_slope:
        parts.append(spectral_slope_features(signal, sr))
    return np.concatenate(parts)
