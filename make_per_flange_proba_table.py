"""Render the per-flange probability heatmap as a 3-row x 4-column
table (torques as rows, flanges as columns), highlighting the largest
probability in each column."""
import os, glob, re
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import librosa
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import optimize as opt

OUTDIR = "poster_figures"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

N_MFCC, N_FFT, HOP_LENGTH, N_PSD_BINS = 13, 512, 128, 64

# Inline feature extractor (matches notebook)
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
def make_lr():
    return Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=3000, C=1.0, random_state=42))])

# Load labeled data + train final LR
print("Loading labeled hits ...")
hits = opt.build_hits()
X = np.array([extract(h["signal"], h["sr"]) for h in hits])
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
print(f"  X = {X.shape}")

keep_f = fit_torque_disc(X, y, fl, 100)
means_f = fit_means(X[:, keep_f], fl)
Xc = apply_center(X[:, keep_f], fl, means_f)
lr_final = make_lr(); lr_final.fit(Xc, y)
print("LR trained.")

# Load unlabeled, get per-flange probabilities
print("Loading unlabeled ...")
unl_hits = []
for path in sorted(glob.glob("F[0-9]A[0-9].m4a")):
    name = os.path.basename(path); m = re.match(r"F(\d+)A(\d+)\.m4a", name)
    sig, sr = librosa.load(path, sr=None, mono=True)
    sig = opt.normalize_audio(sig)
    for hit in opt.split_into_hits(sig, sr):
        unl_hits.append({"flange_id": int(m.group(1)), "sr": sr, "signal": hit})
X_u = np.array([extract(h["signal"], h["sr"]) for h in unl_hits])
fl_u = np.array([h["flange_id"] for h in unl_hits])

classes_ref = np.array([0, 25, 50])
P = lr_final.predict_proba(apply_center(X_u[:, keep_f], fl_u, means_f))
P = P[:, [list(lr_final.classes_).index(c) for c in classes_ref]]

per_hit = pd.DataFrame(P, columns=classes_ref); per_hit["flange_id"] = fl_u
per_flange = per_hit.groupby("flange_id")[list(classes_ref)].mean()
print("\nPer-flange probabilities:")
print(per_flange.round(3).to_string())

# Build the transposed table: torques as rows, flanges as columns
data = per_flange.T   # rows=[0,25,50], cols=[1,2,3,4]
data.index = ["0 ft-lbs", "25 ft-lbs", "50 ft-lbs"]
data.columns = [f"Flange {f}" for f in data.columns]


# === Render as a clean PNG table with column-wise highlighting ===
def render_columnmax(df, fname, title, header_color="#222222"):
    rows = df.shape[0] + 1
    cols = df.shape[1] + 1
    fig, ax = plt.subplots(figsize=(2.2 * cols, 0.7 * rows + 0.6))
    ax.axis("off")

    table_data = [[""] + list(df.columns)]
    for idx, row in df.iterrows():
        table_data.append([str(idx)] + [f"{v:.3f}" for v in row.values])

    table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                     colWidths=[1.0/cols] * cols)
    table.auto_set_font_size(False); table.set_fontsize(13); table.scale(1, 1.6)

    # Header row
    for j in range(cols):
        c = table[0, j]; c.set_facecolor(header_color)
        c.set_text_props(color="white", fontweight="bold")
    # Row index column ("Torque" labels) — distinct background
    for i in range(1, rows):
        c = table[i, 0]; c.set_facecolor("#E8E8E8"); c.set_text_props(fontweight="bold")
    # Highlight the highest probability in each column (= each flange's argmax)
    for col_j in range(1, cols):
        col_vals = df.iloc[:, col_j - 1].values
        top_row = int(np.argmax(col_vals)) + 1
        cell = table[top_row, col_j]
        cell.set_facecolor("#FFE08A")
        cell.set_text_props(fontweight="bold", color="#000")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved {fname}")


print("\nRendering table ...")
render_columnmax(data, f"{OUTDIR}/fig_table_per_flange_probabilities.png",
                  "Flange-Invariant LR — Per-Flange Soft-Vote Probabilities\n"
                  "(highest probability per flange highlighted)")
print("Done.")
