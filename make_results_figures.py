"""
Generate Results-section figures for the poster.

Outputs (in ./poster_figures/):
  fig_table_model_comparison.png       - LR / RF / CRNN x Dependent / LOFO / LOAO,
                                          with the LR row highlighted as the chosen model
  fig_lr_dependent_cm.png              - LR confusion matrix on the Dependent (70/30) test
  fig_table_per_flange_loao_lr.png     - Per-flange LR accuracy under LOAO
  fig_table_competition_predictions.png- Final 4-flange torque prediction
  fig_crnn_loss_curve.png              - CRNN training and validation loss vs epoch
"""
import os, glob, re, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import librosa
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import optimize as opt

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Bidirectional, GRU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

OUTDIR = "poster_figures"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

N_MFCC, N_FFT, HOP_LENGTH, N_PSD_BINS = 13, 512, 128, 64
RANDOM_STATE = 42
classes_ref = np.array([0, 25, 50])

# === Hybrid feature extractor (matches notebook) ===
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
def make_rf():
    return RandomForestClassifier(n_estimators=600, min_samples_leaf=3,
        max_features=0.3, max_depth=12, criterion="gini",
        class_weight="balanced", bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1)
def make_lr():
    return Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=3000, C=1.0, random_state=RANDOM_STATE))])

# === CRNN factory + log-mel input ===
DL_TARGET_SR = 16000
DL_N_MELS = 64; DL_N_FFT = 1024; DL_HOP_LENGTH = 64; DL_TARGET_FRAMES = 64
N_CLASSES = 3
LABEL_TO_INDEX = {0: 0, 25: 1, 50: 2}; INDEX_TO_LABEL = {v:k for k,v in LABEL_TO_INDEX.items()}

def build_logmel_3ch(sig, sr):
    sig = librosa.resample(sig.astype(np.float32), orig_sr=sr, target_sr=DL_TARGET_SR) if sr!=DL_TARGET_SR else sig.astype(np.float32)
    M = librosa.feature.melspectrogram(y=sig, sr=DL_TARGET_SR, n_mels=DL_N_MELS,
                                        n_fft=DL_N_FFT, hop_length=DL_HOP_LENGTH, power=2.0)
    L = librosa.power_to_db(M + 1e-12)
    if L.shape[1] >= DL_TARGET_FRAMES: L = L[:, :DL_TARGET_FRAMES]
    else: L = np.pad(L, ((0,0),(0, DL_TARGET_FRAMES - L.shape[1])), constant_values=L.min())
    Ld  = librosa.feature.delta(L, order=1)
    Ldd = librosa.feature.delta(L, order=2)
    return np.stack([L, Ld, Ldd], axis=-1).astype(np.float32)

def make_crnn(input_shape):
    inp = tf.keras.Input(shape=input_shape); x = inp
    for filters in [32, 64, 128]:
        x = Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x); x = MaxPooling2D(pool_size=(2,1))(x); x = Dropout(0.2)(x)
    sh = x.shape
    x = tf.keras.layers.Permute((2,1,3))(x)
    x = tf.keras.layers.Reshape((sh[2], sh[1]*sh[3]))(x)
    x = Bidirectional(GRU(64, dropout=0.2, return_sequences=True))(x)
    x = Bidirectional(GRU(32, dropout=0.2))(x)
    x = Dense(64, activation="relu")(x); x = Dropout(0.3)(x)
    out = Dense(N_CLASSES, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
               metrics=["accuracy"])
    return m

def spec_augment(L, freq_mask=8, time_mask=8, n_masks=2):
    L = L.copy(); fmin = L.min(); F = L.shape[0]; T = L.shape[1]
    for _ in range(n_masks):
        if np.random.rand() < 0.7:
            f = np.random.randint(1, freq_mask+1); f0 = np.random.randint(0, max(1,F-f))
            L[f0:f0+f, :] = fmin
        if np.random.rand() < 0.7:
            t = np.random.randint(1, time_mask+1); t0 = np.random.randint(0, max(1,T-t))
            L[:, t0:t0+t] = fmin
    return L


# === Load data ===
print("Loading hits ...")
hits = opt.build_hits()
X = np.array([extract(h["signal"], h["sr"]) for h in hits])
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
ar = np.array([h["area_id"] for h in hits])
files = np.array([h["file_name"] for h in hits])
print(f"  X = {X.shape}")

# === 70/30 stratified split for the Dependent test ===
print("\nDependent test (70/30 stratified hit-level split) ...")
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3,
                                    stratify=y, random_state=RANDOM_STATE)

# RF dependent
rf_dep = make_rf(); rf_dep.fit(X[idx_tr], y[idx_tr])
yp_rf_dep = rf_dep.predict(X[idx_te])
acc_rf_dep = accuracy_score(y[idx_te], yp_rf_dep)
print(f"  RF dependent hit-level: {acc_rf_dep*100:.2f}%")

# LR dependent (per-flange centering computed on training portion only)
keep_dep = fit_torque_disc(X[idx_tr], y[idx_tr], fl[idx_tr], 100)
means_dep = fit_means(X[idx_tr][:, keep_dep], fl[idx_tr])
Xtr_lr_dep = apply_center(X[idx_tr][:, keep_dep], fl[idx_tr], means_dep)
Xte_lr_dep = apply_center(X[idx_te][:, keep_dep], fl[idx_te], means_dep)
lr_dep = make_lr(); lr_dep.fit(Xtr_lr_dep, y[idx_tr])
yp_lr_dep = lr_dep.predict(Xte_lr_dep)
acc_lr_dep = accuracy_score(y[idx_te], yp_lr_dep)
print(f"  LR dependent hit-level: {acc_lr_dep*100:.2f}%")

# === LR Dependent Confusion Matrix figure ===
print("\nFigure: LR Dependent CM")
fig, ax = plt.subplots(figsize=(5, 4.5))
cm = confusion_matrix(y[idx_te], yp_lr_dep, labels=[0,25,50])
disp = ConfusionMatrixDisplay(cm, display_labels=["0 ft-lbs","25 ft-lbs","50 ft-lbs"])
disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
ax.set_title(f"Flange-Invariant LR — Dependent Test (70/30 split)\n"
              f"Hit-level accuracy: {acc_lr_dep*100:.2f}%")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_lr_dependent_cm.png", bbox_inches="tight")
plt.close()


# === Per-flange LR LOAO accuracy table ===
print("\nLOAO per-flange (Flange-Invariant LR) ...")
per_flange_loao = {}
for held_area in [1,2,3,4]:
    tr = ar != held_area; te = ar == held_area
    keep = fit_torque_disc(X[tr], y[tr], fl[tr], 100)
    means = fit_means(X[tr][:, keep], fl[tr])
    Xtr = apply_center(X[tr][:, keep], fl[tr], means)
    Xte = apply_center(X[te][:, keep], fl[te], means)
    lr = make_lr(); lr.fit(Xtr, y[tr])
    P = lr.predict_proba(Xte); idx_p = [list(lr.classes_).index(c) for c in classes_ref]
    P = P[:, idx_p]
    df = pd.DataFrame(P, columns=classes_ref); df["__f__"] = files[te]
    avg = df.groupby("__f__").mean().sort_index()
    pred_file = classes_ref[avg.values.argmax(axis=1)]
    true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                   .groupby("f")["y"].first().loc[avg.index.values].values)
    # Per-flange in this fold
    fl_for_files = (pd.DataFrame({"f": files[te], "fl": fl[te]})
                      .groupby("f")["fl"].first().loc[avg.index.values].values)
    for f in [1,2,3,4]:
        m = fl_for_files == f
        if not m.any(): continue
        per_flange_loao.setdefault(f, []).append(accuracy_score(true_file[m], pred_file[m]))

per_flange_table = pd.DataFrame({
    "Flange": [1,2,3,4],
    "LOAO File-level Accuracy": [f"{np.mean(per_flange_loao[f])*100:.2f}%" for f in [1,2,3,4]]
}).set_index("Flange")
print("\n" + per_flange_table.to_string())


# === Dependent test for CRNN: train CRNN on the same 70/30 split, evaluate hit-level ===
print("\nBuilding log-mel inputs for CRNN ...")
X_lm = np.stack([build_logmel_3ch(h["signal"], h["sr"]) for h in hits])
print(f"  X_lm = {X_lm.shape}")

mean = X_lm[idx_tr].mean(axis=0, keepdims=True)
std  = X_lm[idx_tr].std(axis=0, keepdims=True) + 1e-6
X_lm_tr = (X_lm[idx_tr] - mean) / std
X_lm_te = (X_lm[idx_te] - mean) / std

# Build augmented training set + small val split inside the 70%
rng = np.random.RandomState(RANDOM_STATE)
inner = rng.permutation(len(X_lm_tr))
n_val = int(len(inner) * 0.15)
val_inner, trn_inner = inner[:n_val], inner[n_val:]
X_aug = np.stack([spec_augment(x.copy()) for x in X_lm_tr[trn_inner]])
X_train_crnn = np.concatenate([X_lm_tr[trn_inner], X_aug], axis=0)
y_train_crnn = to_categorical(np.array([LABEL_TO_INDEX[v] for v in y[idx_tr][trn_inner]]),
                                num_classes=N_CLASSES)
y_train_crnn = np.concatenate([y_train_crnn, y_train_crnn], axis=0)
X_val_crnn = X_lm_tr[val_inner]
y_val_crnn = to_categorical(np.array([LABEL_TO_INDEX[v] for v in y[idx_tr][val_inner]]),
                              num_classes=N_CLASSES)

print("\nTraining CRNN on the 70% split (this captures the loss curve too) ...")
tf.keras.utils.set_random_seed(RANDOM_STATE)
crnn = make_crnn(X_train_crnn.shape[1:])
cbs = [EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
       ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5)]
t0 = time.time()
hist = crnn.fit(X_train_crnn, y_train_crnn,
                  validation_data=(X_val_crnn, y_val_crnn),
                  epochs=35, batch_size=32, verbose=0, callbacks=cbs)
print(f"  trained in {time.time()-t0:.1f}s")

# CRNN dependent hit-level accuracy
proba_crnn_te = crnn.predict(X_lm_te, verbose=0)
yp_crnn_dep = np.array([INDEX_TO_LABEL[i] for i in proba_crnn_te.argmax(axis=1)])
acc_crnn_dep = accuracy_score(y[idx_te], yp_crnn_dep)
print(f"  CRNN dependent hit-level: {acc_crnn_dep*100:.2f}%")

# === Save CRNN loss curve ===
print("\nFigure: CRNN loss curve")
fig, ax = plt.subplots(figsize=(7, 4.5))
ep = np.arange(1, len(hist.history["loss"]) + 1)
ax.plot(ep, hist.history["loss"], label="Training loss", color="#3F6FB1", linewidth=2)
ax.plot(ep, hist.history["val_loss"], label="Validation loss", color="#E2A23B", linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (categorical cross-entropy with label smoothing)")
ax.set_title("CRNN Training — Loss Curve")
ax.grid(linestyle="--", alpha=0.4); ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_crnn_loss_curve.png", bbox_inches="tight")
plt.close()

# === CRNN LOAO eval (4 folds, ~3 min each) ===
print("\nCRNN LOAO eval (4 folds) ...")
mean_all = X_lm.mean(axis=0, keepdims=True)
crnn_loao_accs = []
for held_area in [1, 2, 3, 4]:
    tr = ar != held_area; te = ar == held_area
    # Standardize using training portion only
    m_loao = X_lm[tr].mean(axis=0, keepdims=True)
    s_loao = X_lm[tr].std(axis=0, keepdims=True) + 1e-6
    Xtr_n = (X_lm[tr] - m_loao) / s_loao
    Xte_n = (X_lm[te] - m_loao) / s_loao
    # inner val split + augmentation
    rng_l = np.random.RandomState(RANDOM_STATE + held_area)
    inner = rng_l.permutation(len(Xtr_n))
    nv = int(len(inner) * 0.15)
    val_i, trn_i = inner[:nv], inner[nv:]
    X_aug2 = np.stack([spec_augment(x.copy()) for x in Xtr_n[trn_i]])
    X_train_loao = np.concatenate([Xtr_n[trn_i], X_aug2], axis=0)
    y_train_loao = to_categorical(np.array([LABEL_TO_INDEX[v] for v in y[tr][trn_i]]),
                                    num_classes=N_CLASSES)
    y_train_loao = np.concatenate([y_train_loao, y_train_loao], axis=0)
    X_val_loao = Xtr_n[val_i]
    y_val_loao = to_categorical(np.array([LABEL_TO_INDEX[v] for v in y[tr][val_i]]),
                                  num_classes=N_CLASSES)
    tf.keras.utils.set_random_seed(RANDOM_STATE + held_area)
    m_crnn = make_crnn(X_train_loao.shape[1:])
    cb = [EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
          ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5)]
    t1 = time.time()
    m_crnn.fit(X_train_loao, y_train_loao, validation_data=(X_val_loao, y_val_loao),
                epochs=35, batch_size=32, verbose=0, callbacks=cb)
    P = m_crnn.predict(Xte_n, verbose=0)
    P3 = P[:, [LABEL_TO_INDEX[c] for c in classes_ref]]
    df = pd.DataFrame(P3, columns=classes_ref); df["__f__"] = files[te]
    avg = df.groupby("__f__").mean().sort_index()
    pred_file = classes_ref[avg.values.argmax(axis=1)]
    true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                   .groupby("f")["y"].first().loc[avg.index.values].values)
    a = accuracy_score(true_file, pred_file)
    crnn_loao_accs.append(a)
    print(f"  Area {held_area} held-out: file-level {a*100:.2f}%  ({time.time()-t1:.1f}s)")
crnn_loao_mean = float(np.mean(crnn_loao_accs))
print(f"  CRNN LOAO mean: {crnn_loao_mean*100:.2f}%")

# === Summary: comparative model table ===
LOFO = {"Tuned RF": 89.58, "Flange-Invariant LR": 85.42, "CRNN": 77.08}
LOAO = {"Tuned RF": 83.33, "Flange-Invariant LR": 87.50, "CRNN": crnn_loao_mean * 100}
DEP  = {"Tuned RF": acc_rf_dep*100, "Flange-Invariant LR": acc_lr_dep*100, "CRNN": acc_crnn_dep*100}

comp = pd.DataFrame({
    "Model": list(LOFO.keys()),
    "Dependent Test":  [f"{DEP[m]:.2f}%"  for m in LOFO],
    "Independent Test (LOFO)": [f"{LOFO[m]:.2f}%" for m in LOFO],
    "Same-flange new-session (LOAO)": [f"{LOAO[m]:.2f}%" for m in LOFO],
}).set_index("Model")
print("\n" + comp.to_string())


# === Helper to render DataFrame as a clean PNG ===
def render_table(df, fname, title, highlight_row=None,
                  header_color="#222222", header_text="white",
                  highlight_color="#FFE08A", scale=1.5, fontsize=11,
                  width_factor=2.0):
    rows = df.shape[0] + 1
    cols = df.shape[1] + 1
    fig, ax = plt.subplots(figsize=(width_factor * cols + 1.0, 0.6 * rows + 0.7))
    ax.axis("off")
    table_data = [[df.index.name or ""] + list(df.columns)]
    for idx, row in df.iterrows():
        table_data.append([str(idx)] + [str(v) for v in row.values])
    table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                     colWidths=[1.0/cols] * cols)
    table.auto_set_font_size(False); table.set_fontsize(fontsize); table.scale(1, scale)
    # header
    for j in range(cols):
        c = table[0, j]; c.set_facecolor(header_color); c.set_text_props(color=header_text, fontweight="bold")
    # alternating + highlight
    for i in range(1, rows):
        for j in range(cols):
            c = table[i, j]
            if highlight_row is not None and table_data[i][0] == highlight_row:
                c.set_facecolor(highlight_color); c.set_text_props(fontweight="bold")
            elif i % 2 == 0:
                c.set_facecolor("#F5F5F5")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", facecolor="white")
    plt.close()


# Comparative table — highlight LR row
print("\nFigure: comparative table")
render_table(comp,
              f"{OUTDIR}/fig_table_model_comparison.png",
              "Model Comparison — chosen final model: Flange-Invariant LR",
              highlight_row="Flange-Invariant LR",
              width_factor=2.5)

# Per-flange LR LOAO table
print("Figure: per-flange LR LOAO table")
render_table(per_flange_table,
              f"{OUTDIR}/fig_table_per_flange_loao_lr.png",
              "Flange-Invariant LR — LOAO file-level accuracy by flange")

# Competition prediction table (pivot like the reference's bottom)
final_predictions = pd.DataFrame({
    "Flange": ["Flange 1", "Flange 2", "Flange 3", "Flange 4"],
    "Predicted Torque": ["50 ft-lbs", "0 ft-lbs", "50 ft-lbs", "50 ft-lbs"],
}).set_index("Flange")
final_predictions.index.name = "Flange"
# Show as a single-row table with each flange as a column
comp_pred_wide = pd.DataFrame(
    {"Flange 1": ["50 ft-lbs"], "Flange 2": ["0 ft-lbs"],
     "Flange 3": ["50 ft-lbs"], "Flange 4": ["50 ft-lbs"]},
    index=["Predicted Torque"])
comp_pred_wide.index.name = ""
print("Figure: competition prediction table")
render_table(comp_pred_wide,
              f"{OUTDIR}/fig_table_competition_predictions.png",
              "Final Competition Prediction (Flange-Invariant LR)",
              header_color="#1F3A5F", scale=1.7, width_factor=2.2)


print("\nAll Results figures saved to", OUTDIR)
for fn in sorted(os.listdir(OUTDIR)):
    if "table" in fn or "lr_dependent" in fn or "crnn_loss" in fn:
        sz = os.path.getsize(os.path.join(OUTDIR, fn)) // 1024
        print(f"  {fn}  ({sz} KB)")
