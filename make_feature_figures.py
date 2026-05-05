"""
Visual figures for the Methods section of the poster.

Outputs (in ./poster_figures/):
  fig_logmel_by_torque.png      - log-mel spectrograms at 0/25/50 ft-lbs
  fig_envelope_decay.png        - log-envelope decay curves at 0/25/50 ft-lbs
  fig_mfcc_by_torque.png        - MFCC sequences at 0/25/50 ft-lbs
  fig_psd_by_torque.png         - Welch log-PSD at 0/25/50 ft-lbs
"""
import os, glob, re
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import librosa, librosa.display
from scipy.signal import welch
import matplotlib.pyplot as plt
import optimize as opt

OUTDIR = "poster_figures"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

# Pick one representative hit from the same flange/area at each torque so the
# comparison isolates torque (other variables held fixed).
FLANGE = 1
AREA = 1

print(f"Loading hits ...")
hits = opt.build_hits()
print(f"  {len(hits)} hits loaded")

# Find one hit per torque from F{FLANGE}A{AREA}
representative = {}
for h in hits:
    if h["flange_id"] == FLANGE and h["area_id"] == AREA and h["torque"] not in representative:
        representative[h["torque"]] = h
    if len(representative) == 3:
        break
for t in [0, 25, 50]:
    assert t in representative, f"missing torque {t}"
    print(f"  torque {t}: F{FLANGE}A{AREA}, {len(representative[t]['signal'])} samples @ {representative[t]['sr']} Hz")

# === Figure 1: Log-Mel Spectrograms ===
print("\nFigure: log-mel spectrograms")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), sharey=True)
for ax, t in zip(axes, [0, 25, 50]):
    h = representative[t]
    sig, sr = h["signal"], h["sr"]
    # Resample to 16 kHz so the spectrogram is comparable across torques
    sig16 = librosa.resample(sig.astype(np.float32), orig_sr=sr, target_sr=16000)
    M = librosa.feature.melspectrogram(y=sig16, sr=16000, n_mels=64,
                                        n_fft=1024, hop_length=64, power=2.0)
    L = librosa.power_to_db(M + 1e-12)
    img = librosa.display.specshow(L, sr=16000, hop_length=64,
                                     x_axis="time", y_axis="mel",
                                     ax=ax, fmax=8000, cmap="magma")
    ax.set_title(f"{t} ft-lbs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
fig.colorbar(img, ax=axes, format="%+2.0f dB", aspect=20, pad=0.02, label="dB")
fig.suptitle("Log-Mel Spectrogram of a Single Hit at 0, 25, 50 ft-lbs (F1A1)",
              fontsize=12)
plt.savefig(f"{OUTDIR}/fig_logmel_by_torque.png", bbox_inches="tight")
plt.close()

# === Figure 2: Log-Envelope Decay ===
print("Figure: log-envelope decay")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
colors = {0: "#4C72B0", 25: "#DD8452", 50: "#55A868"}
for t in [0, 25, 50]:
    h = representative[t]
    sig, sr = h["signal"], h["sr"]
    a = np.abs(sig)
    win = max(1, int(0.002 * sr))
    e = np.convolve(a, np.ones(win)/win, mode="same")
    peak = np.argmax(e)
    after = e[peak:]
    t_axis = np.arange(len(after)) / sr * 1000   # ms
    ax.plot(t_axis, 20*np.log10(after / (after[0] + 1e-12) + 1e-12),
             label=f"{t} ft-lbs", color=colors[t], linewidth=2)
ax.set_xlabel("Time after peak (ms)")
ax.set_ylabel("Log envelope (dB, peak-normalized)")
ax.set_title("Hit Ring-Down Decay at Different Torques (F1A1)\n"
              "Lower torque → longer late-tail decay", fontsize=11)
ax.set_ylim(-60, 5)
ax.set_xlim(0, 200)
ax.legend(loc="upper right")
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_envelope_decay.png", bbox_inches="tight")
plt.close()

# === Figure 3: MFCC Sequence ===
print("Figure: MFCC sequence")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), sharey=True)
for ax, t in zip(axes, [0, 25, 50]):
    h = representative[t]
    sig, sr = h["signal"], h["sr"]
    M = librosa.feature.mfcc(y=sig.astype(np.float32), sr=sr, n_mfcc=13,
                              n_fft=512, hop_length=128)
    img = ax.imshow(M, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_title(f"{t} ft-lbs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time frame")
    ax.set_ylabel("MFCC index")
fig.colorbar(img, ax=axes, label="MFCC value", aspect=20, pad=0.02)
fig.suptitle("MFCC (13 coefficients) of a Single Hit at 0, 25, 50 ft-lbs (F1A1)",
              fontsize=12)
plt.savefig(f"{OUTDIR}/fig_mfcc_by_torque.png", bbox_inches="tight")
plt.close()

# === Figure 4: Welch Log-PSD ===
print("Figure: Welch log-PSD")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for t in [0, 25, 50]:
    h = representative[t]
    sig, sr = h["signal"], h["sr"]
    f, p = welch(sig, fs=sr, nperseg=min(1024, len(sig)))
    ax.semilogx(f, 10*np.log10(p + 1e-14), label=f"{t} ft-lbs",
                 color=colors[t], linewidth=2)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power spectral density (dB)")
ax.set_title("Frequency Spectrum of a Single Hit at Different Torques (F1A1)\n"
              "Spectral shape shifts with bolt tightness", fontsize=11)
ax.legend(loc="upper right")
ax.grid(linestyle="--", alpha=0.4)
ax.set_xlim(50, 24000)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_psd_by_torque.png", bbox_inches="tight")
plt.close()

print(f"\nAll feature figures saved to {OUTDIR}/")
for fn in sorted(os.listdir(OUTDIR)):
    if "by_torque" in fn or "decay" in fn:
        sz = os.path.getsize(os.path.join(OUTDIR, fn)) // 1024
        print(f"  {fn}  ({sz} KB)")
