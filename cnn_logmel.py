"""
Small CNN on log-mel spectrograms, trained end-to-end per LOFO fold.

For each hit we compute a (n_mels, T) log-mel spectrogram and feed it
to a tiny CNN.  Random horizontal time-shift + Gaussian noise + SpecAugment
(time/frequency masking) for augmentation.  Trained for a small number
of epochs per fold, averaging hit-level probabilities to get file-level
predictions.

This is end-to-end deep learning -- different feature space from the
hand-crafted v2 features.  If it can capture flange-invariant info that
shallow models can't, it might break the 89.58% ceiling.
"""
import os, time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import optimize as opt
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

N_MELS = 64
N_FFT = 512
HOP = 128
TARGET_FRAMES = 64        # ~ 0.17 sec at hop 128, sr 48k
TARGET_SR = 16000          # downsample for CNN -- typical for audio CNN
RANDOM_STATE = 42


def to_mel(sig, sr):
    """Resample to TARGET_SR and compute log-mel spectrogram."""
    if sr != TARGET_SR:
        sig = librosa.resample(sig.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    M = librosa.feature.melspectrogram(y=sig, sr=TARGET_SR,
                                       n_mels=N_MELS, n_fft=N_FFT,
                                       hop_length=HOP, power=2.0)
    L = librosa.power_to_db(M + 1e-12)
    # Pad/crop time axis to TARGET_FRAMES
    if L.shape[1] >= TARGET_FRAMES:
        L = L[:, :TARGET_FRAMES]
    else:
        L = np.pad(L, ((0, 0), (0, TARGET_FRAMES - L.shape[1])), constant_values=L.min())
    return L.astype(np.float32)


class HitDataset(Dataset):
    def __init__(self, mels, labels, augment=False):
        self.mels = mels
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.mels)

    def _spec_augment(self, x):
        """Apply small SpecAugment: random time/freq mask."""
        x = x.copy()
        # frequency mask
        if np.random.rand() < 0.5:
            f = np.random.randint(0, 8)
            f0 = np.random.randint(0, max(1, x.shape[0] - f))
            x[f0:f0+f, :] = x.min()
        # time mask
        if np.random.rand() < 0.5:
            t = np.random.randint(0, 6)
            t0 = np.random.randint(0, max(1, x.shape[1] - t))
            x[:, t0:t0+t] = x.min()
        # additive noise
        if np.random.rand() < 0.5:
            x = x + np.random.normal(0, 0.5, x.shape).astype(np.float32)
        return x

    def __getitem__(self, idx):
        x = self.mels[idx]
        if self.augment:
            x = self._spec_augment(x)
        # standardize per-spectrogram
        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(x).unsqueeze(0), int(self.labels[idx])


class TinyCNN(nn.Module):
    def __init__(self, n_classes=3, dropout=0.3):
        super().__init__()
        # input: (1, 64, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                                    # 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                                    # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                                    # 8x8
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                            # 64x1x1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)


def train_one_fold(train_mels, train_labels, val_mels, val_labels,
                   epochs=30, batch_size=64, lr=1e-3):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    model = TinyCNN().to(DEVICE)
    opt_ = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    train_ds = HitDataset(train_mels, train_labels, augment=True)
    val_ds = HitDataset(val_mels, val_labels, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x = x.to(DEVICE); y = y.to(DEVICE)
            opt_.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt_.step()
        sched.step()

    # Get probabilities on val
    model.eval()
    all_probs = []
    with torch.no_grad():
        for x, _ in val_dl:
            x = x.to(DEVICE)
            logits = model(x)
            p = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(p)
    return np.concatenate(all_probs)


def lofo_cnn(mels, y, fl, files, epochs=30):
    label_map = {0: 0, 25: 1, 50: 2}
    inv_map = {v: k for k, v in label_map.items()}
    y_idx = np.array([label_map[v] for v in y])

    classes_ref = np.array([0, 25, 50])
    file_accs = []
    hit_accs = []
    for f in [1, 2, 3, 4]:
        tr, te = fl != f, fl == f
        t0 = time.time()
        probs = train_one_fold(mels[tr], y_idx[tr], mels[te], y_idx[te], epochs=epochs)
        # hit-level
        pred_idx = probs.argmax(axis=1)
        hit_acc = accuracy_score(y_idx[te], pred_idx)
        hit_accs.append(hit_acc)
        # file-level via soft vote
        df = pd.DataFrame(probs, columns=[0, 25, 50])
        df["__f__"] = files[te]
        avg = df.groupby("__f__").mean().sort_index()
        pred = classes_ref[avg.values.argmax(axis=1)]
        true_file = (pd.DataFrame({"f": files[te], "y": y[te]})
                       .groupby("f")["y"].first().loc[avg.index.values].values)
        file_acc = accuracy_score(true_file, pred)
        file_accs.append(file_acc)
        print(f"  Fold {f}:  hit={hit_acc*100:5.2f}  file={file_acc*100:5.2f}  "
              f"({time.time()-t0:.1f}s)")
    print(f"  MEAN:    hit={np.mean(hit_accs)*100:5.2f}  file={np.mean(file_accs)*100:5.2f}  "
          f"per-fold {[round(a*100) for a in file_accs]}")
    return float(np.mean(file_accs))


# ----- main -----
if __name__ == "__main__":
    hits = opt.build_hits()
    print(f"Hits: {len(hits)}")

    print("\nComputing log-mel spectrograms...")
    t0 = time.time()
    mels = np.stack([to_mel(h["signal"], h["sr"]) for h in hits])
    print(f"mel shape: {mels.shape}  ({time.time()-t0:.1f}s)")

    y = np.array([h["torque"] for h in hits])
    fl = np.array([h["flange_id"] for h in hits])
    files = np.array([h["file_name"] for h in hits])

    print("\n=== Tiny CNN on log-mel (LOFO) ===")
    lofo_cnn(mels, y, fl, files, epochs=30)
