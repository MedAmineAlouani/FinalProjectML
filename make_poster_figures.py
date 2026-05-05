"""
Generate poster-ready PNG figures at 300 DPI.

Outputs (in ./poster_figures/):
  fig_flowchart.png            - Methods flowchart (Audio -> features -> models -> output)
  fig_cm_rf_lofo_aggregate.png - Aggregated LOFO confusion matrix (Tuned RF, all 4 folds)
  fig_cm_lr_lofo_aggregate.png - Aggregated LOFO confusion matrix (Flange-Invariant LR, all 4 folds)
  fig_cm_rf_per_fold.png       - 4 panels: RF confusion matrix per held-out flange
  fig_cm_lr_per_fold.png       - 4 panels: LR confusion matrix per held-out flange
  fig_model_comparison.png     - Bar chart: LOFO and LOAO file-level accuracy for RF / LR / CRNN
  fig_unlabeled_predictions.png- Bar chart: predicted-class counts per unlabeled flange (LR)
  fig_per_flange_proba.png     - Heatmap-style per-flange probabilities
"""
import os, glob, re
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import librosa
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import optimize as opt

OUTDIR = "poster_figures"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

N_MFCC, N_FFT, HOP_LENGTH, N_PSD_BINS = 13, 512, 128, 64

# --- Inline feature extractor (matches notebook) ---
def safe_mean_std(F):
    F = np.asarray(F)
    return np.concatenate([F.mean(axis=1), F.std(axis=1)])

def psd_feat(s, sr, n=N_PSD_BINS):
    f, p = welch(s, fs=sr, nperseg=min(1024, len(s)))
    pl = np.log10(p + 1e-12)
    return pl[:n] if len(pl) >= n else np.pad(pl, (0, n - len(pl)))

def mfcc_feat(s, sr):
    nf = min(N_FFT, len(s)); h = min(HOP_LENGTH, max(1, len(s)//2))
    M = librosa.feature.mfcc(y=s, sr=sr, n_mfcc=N_MFCC, n_fft=nf, hop_length=h)
    return np.concatenate([safe_mean_std(M), safe_mean_std(librosa.feature.delta(M))])

def spec(s, sr):
    nf = min(N_FFT, len(s)); h = min(HOP_LENGTH, max(1, len(s)//2))
    out = []
    for fn, kw in [
        (librosa.feature.spectral_centroid, dict(y=s, sr=sr, n_fft=nf, hop_length=h)),
        (librosa.feature.spectral_bandwidth, dict(y=s, sr=sr, n_fft=nf, hop_length=h)),
        (librosa.feature.spectral_rolloff, dict(y=s, sr=sr, n_fft=nf, hop_length=h, roll_percent=0.85)),
        (librosa.feature.spectral_flatness, dict(y=s, n_fft=nf, hop_length=h)),
        (librosa.feature.zero_crossing_rate, dict(y=s, frame_length=nf, hop_length=h)),
        (librosa.feature.rms, dict(y=s, frame_length=nf, hop_length=h)),
    ]: out.append(safe_mean_std(fn(**kw)))
    return np.concatenate(out)

def fshape(s, sr):
    f, p = welch(s, fs=sr, nperseg=min(1024, len(s))); p = p+1e-12; pn = p/p.sum()
    return np.array([f[np.argmax(p)], -np.sum(pn*np.log2(pn+1e-12))])

def decay(s, sr):
    a = np.abs(s)
    if a.max()==0: return np.zeros(5)
    w = max(1, int(0.002*sr))
    e = np.convolve(a, np.ones(w)/w, mode="same")
    pi=int(np.argmax(e)); pv=e[pi]+1e-12; aft=e[pi:]
    b50=np.where(aft<=0.5*pv)[0]; b10=np.where(aft<=0.1*pv)[0]
    d50=b50[0]/sr if len(b50) else len(aft)/sr
    d10=b10[0]/sr if len(b10) else len(aft)/sr
    t=np.arange(len(aft))/sr
    sl=np.polyfit(t, np.log(aft+1e-8), 1)[0] if len(t)>5 else 0.0
    return np.array([pv, pi/sr, d50, d10, sl])

def per_band(s, sr, bands=((100,500),(500,2000),(2000,6000),(6000,12000),(12000,24000))):
    n=len(s); nf=min(512,n); h=max(1,n//64)
    S=np.abs(librosa.stft(s, n_fft=nf, hop_length=h))
    fr=librosa.fft_frequencies(sr=sr, n_fft=nf)
    out=[]
    for lo,hi in bands:
        m=(fr>=lo)&(fr<hi)
        if not m.any() or S.shape[1]<4: out.extend([0,0,0]); continue
        env=S[m].mean(axis=0)+1e-12
        pt=int(np.argmax(env))
        if pt>=len(env)-3: out.extend([0,0,0]); continue
        aft=env[pt:]; le=np.log(aft+1e-9); t=np.arange(len(aft))
        sl=np.polyfit(t, le, 1)[0]
        b50=np.where(aft<=0.5*aft[0])[0]
        hl=b50[0] if len(b50) else len(aft)
        ptm=aft[0]/(aft.mean()+1e-12)
        out.extend([sl, hl/h, ptm])
    return np.array(out)

def extract(s, sr):
    return np.concatenate([psd_feat(s,sr), mfcc_feat(s,sr), spec(s,sr),
                            fshape(s,sr), decay(s,sr), per_band(s,sr)])

# --- LR helpers ---
def fit_torque_disc(X, y, fl, n_keep=100):
    F_t,_=f_classif(X,y); F_f,_=f_classif(X,fl)
    score = np.nan_to_num(F_t)/(np.nan_to_num(F_f)+1.0)
    return np.argsort(-score)[:n_keep]

def fit_means(X, fl): return {f: X[fl==f].mean(axis=0) for f in np.unique(fl)}
def apply_center(X, fl, means):
    Xc=X.astype(float).copy()
    if not means: return Xc
    fb=np.mean(list(means.values()), axis=0)
    for i in range(len(X)): Xc[i]-=means.get(fl[i], fb)
    return Xc

def make_rf():
    return RandomForestClassifier(n_estimators=600, min_samples_leaf=3,
        max_features=0.3, max_depth=12, criterion="gini",
        class_weight="balanced", bootstrap=True, random_state=42, n_jobs=-1)

def make_lr():
    return Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=3000, C=1.0, random_state=42))])

# --- Load data ---
print("Loading hits ...")
hits = opt.build_hits()
X = np.array([extract(h["signal"], h["sr"]) for h in hits])
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
files = np.array([h["file_name"] for h in hits])
print(f"  X = {X.shape}")

# Run LOFO for both RF and LR, collect predictions
print("\nRunning LOFO ...")
all_y_true_rf = []; all_y_pred_rf = []
all_y_true_lr = []; all_y_pred_lr = []
per_fold = {}
for held in [1, 2, 3, 4]:
    tr = fl != held; te = fl == held
    # RF
    rf = make_rf(); rf.fit(X[tr], y[tr])
    yp_rf = rf.predict(X[te])
    all_y_true_rf.extend(y[te]); all_y_pred_rf.extend(yp_rf)
    # LR with per-flange centering + ANOVA
    keep = fit_torque_disc(X[tr], y[tr], fl[tr], 100)
    means = fit_means(X[tr][:, keep], fl[tr])
    Xtr = apply_center(X[tr][:, keep], fl[tr], means)
    Xte = apply_center(X[te][:, keep], fl[te], means)
    lr = make_lr(); lr.fit(Xtr, y[tr])
    yp_lr = lr.predict(Xte)
    all_y_true_lr.extend(y[te]); all_y_pred_lr.extend(yp_lr)
    per_fold[held] = dict(rf=(y[te], yp_rf), lr=(y[te], yp_lr))
    print(f"  Fold (held F{held}) done")

# === Figure 1: Aggregated CM for RF ===
print("\nFigure: aggregated RF CM")
fig, ax = plt.subplots(figsize=(5, 4.5))
cm = confusion_matrix(all_y_true_rf, all_y_pred_rf, labels=[0, 25, 50])
disp = ConfusionMatrixDisplay(cm, display_labels=["0 ft-lbs", "25 ft-lbs", "50 ft-lbs"])
disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
ax.set_title("Tuned Random Forest — LOFO\n(aggregated across 4 folds)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_cm_rf_lofo_aggregate.png", bbox_inches="tight")
plt.close()

# === Figure 2: Aggregated CM for LR ===
print("Figure: aggregated LR CM")
fig, ax = plt.subplots(figsize=(5, 4.5))
cm = confusion_matrix(all_y_true_lr, all_y_pred_lr, labels=[0, 25, 50])
disp = ConfusionMatrixDisplay(cm, display_labels=["0 ft-lbs", "25 ft-lbs", "50 ft-lbs"])
disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
ax.set_title("Flange-Invariant LR — LOFO\n(aggregated across 4 folds)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_cm_lr_lofo_aggregate.png", bbox_inches="tight")
plt.close()

# === Figure 3: Per-fold CMs for RF ===
print("Figure: per-fold RF CMs")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, f in zip(axes, [1, 2, 3, 4]):
    yt, yp = per_fold[f]["rf"]
    cm = confusion_matrix(yt, yp, labels=[0, 25, 50])
    disp = ConfusionMatrixDisplay(cm, display_labels=["0", "25", "50"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(f"Held-out Flange {f}")
fig.suptitle("Tuned RF — Per-Fold Confusion Matrices (LOFO)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_cm_rf_per_fold.png", bbox_inches="tight")
plt.close()

# === Figure 4: Per-fold CMs for LR ===
print("Figure: per-fold LR CMs")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, f in zip(axes, [1, 2, 3, 4]):
    yt, yp = per_fold[f]["lr"]
    cm = confusion_matrix(yt, yp, labels=[0, 25, 50])
    disp = ConfusionMatrixDisplay(cm, display_labels=["0", "25", "50"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(f"Held-out Flange {f}")
fig.suptitle("Flange-Invariant LR — Per-Fold Confusion Matrices (LOFO)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_cm_lr_per_fold.png", bbox_inches="tight")
plt.close()

# === Figure 5: Model comparison bar chart ===
print("Figure: model comparison")
models = ["Tuned RF", "Flange-Invariant LR", "CRNN"]
lofo = [89.58, 85.42, 77.08]
loao = [83.33, 87.50, np.nan]   # CRNN LOAO not measured
x = np.arange(len(models))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w/2, lofo, width=w, label="LOFO file-level (rubric metric)", color="#3F6FB1")
loao_for_plot = [v if not np.isnan(v) else 0 for v in loao]
bars = ax.bar(x + w/2, loao_for_plot, width=w, label="LOAO file-level (realistic test)", color="#E2A23B")
# Mark CRNN LOAO as not run
for i, v in enumerate(loao):
    if np.isnan(v):
        ax.text(x[i] + w/2, 1, "n/a", ha="center", va="bottom", fontsize=10, color="gray")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("File-level accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Independent test accuracy: LOFO (unseen flange) vs LOAO (new session)")
ax.legend(loc="lower center")
ax.grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(lofo):
    ax.text(x[i] - w/2, v + 1, f"{v:.1f}", ha="center", fontsize=10)
for i, v in enumerate(loao):
    if not np.isnan(v):
        ax.text(x[i] + w/2, v + 1, f"{v:.1f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_model_comparison.png", bbox_inches="tight")
plt.close()

# === Now: train final models on ALL data and predict unlabeled ===
print("\nTraining final RF and LR on all data ...")
keep_f = fit_torque_disc(X, y, fl, 100)
means_f = fit_means(X[:, keep_f], fl)
Xc = apply_center(X[:, keep_f], fl, means_f)
final_rf = make_rf(); final_rf.fit(X, y)
final_lr = make_lr(); final_lr.fit(Xc, y)

# Load + segment unlabeled
print("Loading unlabeled ...")
unl = []
for p in sorted(glob.glob("F[0-9]A[0-9].m4a")):
    n = os.path.basename(p); m = re.match(r"F(\d+)A(\d+)\.m4a", n)
    unl.append(dict(file_name=n, path=p, flange_id=int(m.group(1)), area_id=int(m.group(2))))
unl_hits = []
for r in unl:
    sig, sr = librosa.load(r["path"], sr=None, mono=True)
    sig = opt.normalize_audio(sig)
    for hit in opt.split_into_hits(sig, sr):
        unl_hits.append({**r, "sr": sr, "signal": hit})
X_unl = np.array([extract(h["signal"], h["sr"]) for h in unl_hits])
files_unl = np.array([h["file_name"] for h in unl_hits])
flange_unl = np.array([h["flange_id"] for h in unl_hits])

classes_ref = np.array([0, 25, 50])
P_lr = final_lr.predict_proba(apply_center(X_unl[:, keep_f], flange_unl, means_f))
P_lr = P_lr[:, [list(final_lr.classes_).index(c) for c in classes_ref]]

# === Figure 6: Per-flange probability heatmap (LR) ===
print("Figure: per-flange probability heatmap")
per_hit = pd.DataFrame(P_lr, columns=classes_ref); per_hit["flange_id"] = flange_unl
per_flange = per_hit.groupby("flange_id")[list(classes_ref)].mean()
fig, ax = plt.subplots(figsize=(6.5, 4))
data = per_flange.values
im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(["P[0 ft-lbs]", "P[25 ft-lbs]", "P[50 ft-lbs]"])
ax.set_yticks(range(4)); ax.set_yticklabels([f"Flange {i}" for i in [1,2,3,4]])
for i in range(4):
    for j in range(3):
        v = data[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v > 0.5 else "black", fontsize=11)
ax.set_title("Flange-Invariant LR — Per-Flange Soft-Vote Probabilities\n(unlabeled experimental test)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_per_flange_proba.png", bbox_inches="tight")
plt.close()

# === Figure 7: Predicted-class counts per flange (LR hit-level) ===
print("Figure: per-flange prediction counts")
yp_unl = classes_ref[P_lr.argmax(axis=1)]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
for ax, f in zip(axes, [1, 2, 3, 4]):
    counts = np.array([(yp_unl[flange_unl == f] == c).sum() for c in classes_ref])
    ax.bar([0, 25, 50], counts, color=["#4C72B0", "#DD8452", "#55A868"], width=8)
    ax.set_xticks([0, 25, 50]); ax.set_xticklabels(["0", "25", "50"])
    ax.set_xlabel("Predicted Class")
    ax.set_title(f"Flange {f}")
    for i, (xv, c) in enumerate(zip([0, 25, 50], counts)):
        if c > 0:
            ax.text(xv, c + 0.5, str(c), ha="center", fontsize=10)
axes[0].set_ylabel("Hit count")
fig.suptitle("Flange-Invariant LR — Hit-Level Prediction Distribution per Unlabeled Flange",
             fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_unlabeled_predictions.png", bbox_inches="tight")
plt.close()

# === Figure 8: Methodology flowchart ===
print("Figure: methodology flowchart")
fig, ax = plt.subplots(figsize=(13, 3.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")
boxes = [
    (0.5, "Raw Audio\n(48 kHz)"),
    (2.3, "Peak\nSegmentation"),
    (4.1, "150-dim\nFeature\nVector"),
    (5.9, "Per-Flange\nCentering"),
    (7.7, "ANOVA F-score\nTop-100\nFeatures"),
    (9.5, "Three Models\nRF / LR / CRNN"),
    (11.5, "Soft-vote\nAggregation\n+ Argmax"),
]
for x, label in boxes:
    rect = plt.Rectangle((x - 0.7, 1.0), 1.4, 1.0, fill=True, facecolor="#E8F0F8",
                          edgecolor="#3F6FB1", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, 1.5, label, ha="center", va="center", fontsize=9.5)
# Arrows
for i in range(len(boxes) - 1):
    x_start = boxes[i][0] + 0.7
    x_end = boxes[i+1][0] - 0.7
    ax.annotate("", xy=(x_end, 1.5), xytext=(x_start, 1.5),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
ax.text(6.5, 0.4,
        "Innovations: per-flange centering • ANOVA torque-vs-flange selection • CRNN spatial+temporal",
        ha="center", fontsize=10, style="italic", color="#555")
ax.text(6.5, 2.6, "Figure: Final Prediction Pipeline",
        ha="center", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_flowchart.png", bbox_inches="tight")
plt.close()

print(f"\nAll figures saved to {OUTDIR}/")
for fn in sorted(os.listdir(OUTDIR)):
    sz = os.path.getsize(os.path.join(OUTDIR, fn)) // 1024
    print(f"  {fn}  ({sz} KB)")
