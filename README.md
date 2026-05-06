# Flange-Invariant Acoustic Bolt-Looseness Detector

A polished web demo for the **University of Houston Machine Learning
Competition 2026** — *Robust Acoustic Bolt-Looseness Detection Across Recording
Sessions*. Strike a bolted flange with a steel hammer, record the percussion
audio, and the model classifies the bolt preload as **0**, **25**, or
**50 ft-lbs** by listening to the flange's acoustic ring-down.

The model is the **Flange-Invariant Logistic Regression** pipeline from
`Final_Project_ML_Second_Attempt (6).ipynb`. The web app is a faithful
re-implementation of that pipeline; no other model is used.

---

## Pipeline

```
Raw Audio  →  Hit Detection  →  150-D Features  →  Flange-Invariant LR  →  Torque Prediction
```

1. Load + normalize raw audio (mono, peak-normalized to ±1).
2. Trim the first 0.15 s, build a smoothed amplitude envelope, and run
   `scipy.signal.find_peaks` with threshold `mean + 2.5·std` and minimum
   spacing 0.30 s.
3. Cut a window of `[-0.02 s, +0.15 s]` around each detected impact.
4. For each hit, extract a 150-D feature vector:
   - 64 log-PSD bins (Welch)
   - 26 + 26 MFCC mean/std and delta MFCC mean/std (`n_mfcc=13`)
   - 12 spectral statistics (centroid, bandwidth, rolloff, flatness, ZCR, RMS — mean + std)
   - 2 frequency-shape features (dominant frequency, spectral entropy)
   - 5 global decay features (peak, decay-50, decay-10, log-slope, …)
   - 15 per-band T60-style decay features (5 bands × {slope, half-life, peak-to-mean})
5. Apply the **Flange-Invariant LR** preprocessing:
   - select the same top-100 feature indices (highest `F(torque) / (F(flange)+1)`)
   - subtract the per-flange training mean for the chosen flange ID
   - scale with the trained `StandardScaler`
   - run the trained `LogisticRegression`
   - apply per-class isotonic calibration (LOFO out-of-fold) and renormalise
6. Soft-vote across all detected hits (mean of per-hit calibrated probabilities).
7. The final torque class is the argmax of the averaged probabilities.

---

## How to run locally

### 0. Prerequisites

* Python 3.10+
* `ffmpeg` (needed by `librosa` to read `.m4a` files)
  * macOS: `brew install ffmpeg`
  * Ubuntu/Debian: `sudo apt-get install ffmpeg`

### 1. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Train and export model artifacts

The repository ships with the labeled audio files (e.g. `25ftlbF1A1.m4a`) at the
project root. Train the Flange-Invariant LR and export everything the inference
server needs:

```bash
python backend/train_and_export.py
```

This produces:

```
backend/model/
  flange_invariant_lr.pkl       sklearn Pipeline (StandardScaler + LR)
  selected_feature_indices.pkl  ndarray of 100 feature indices
  flange_means.pkl              {flange_id: mean_vector_of_selected_feats}
  calibrators.pkl               per-class isotonic calibrators
  metadata.json                 human-readable summary + LOFO accuracy
```

### 3. Run the web app

```bash
python backend/app.py
# or:
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser. The single-page UI is served
from `frontend/index.html`.

### Quick smoke test from the CLI

```bash
curl -s -X POST http://localhost:8000/api/predict \
  -F "audio=@50ftlbF1A1.m4a" \
  -F "flange_id=1" | python -m json.tool | head
```

---

## Model files required

The inference server expects these files inside `backend/model/`:

| File                               | Purpose                                                    |
|------------------------------------|------------------------------------------------------------|
| `flange_invariant_lr.pkl`          | Trained `StandardScaler + LogisticRegression` pipeline     |
| `selected_feature_indices.pkl`     | The 100 ANOVA-selected feature column indices              |
| `flange_means.pkl`                 | Per-flange mean vectors for the selected features          |
| `calibrators.pkl`                  | Per-class isotonic calibrators fit on LOFO out-of-fold     |
| `metadata.json`                    | Human-readable training summary (LOFO accuracy, etc.)      |

If any file is missing, `/api/predict` returns `503` with a clear error message;
just re-run `python backend/train_and_export.py`.

---

## API

### `GET /api/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": { "...": "metadata.json contents" }
}
```

### `POST /api/predict`

Multipart form:

| Field       | Type                | Notes                            |
|-------------|---------------------|----------------------------------|
| `audio`     | file (audio/*)      | `.m4a`, `.wav`, `.mp3`, `.webm`  |
| `flange_id` | int                 | One of `1`, `2`, `3`, `4`        |

Response shape (truncated):

```json
{
  "ok": true,
  "flange_id": 1,
  "sample_rate": 48000,
  "duration_sec": 18.36,
  "n_hits": 20,
  "hit_times_sec": [1.11, 1.95, ...],
  "waveform": { "values": [...], "offset_sec": 0.0, "length": 881600 },
  "envelope": { "values": [...], "offset_sec": 0.15, "length": 874400 },
  "per_hit": [
    {
      "hit_id": 1,
      "time_sec": 1.11,
      "probabilities": { "0": 1e-09, "25": 0.04, "50": 0.96 },
      "predicted_torque": 50,
      "confidence": 0.96
    }
  ],
  "averaged_probabilities": { "0": 1e-09, "25": 0.03, "50": 0.97 },
  "final_prediction": {
    "torque_ftlbs": 50,
    "confidence": 0.97,
    "confidence_level": "high"
  },
  "warnings": []
}
```

User-facing warnings the server may surface:

* "No clear hit detected. Try striking the flange again."
* "Only one hit detected. Prediction may be less stable."
* "Low confidence prediction. Try recording multiple clean hammer strikes."
* "Audio too short or too noisy."

---

## Project structure

```
backend/
  app.py                  FastAPI server + static-file serving
  inference.py            Loads artifacts, runs Flange-Invariant LR
  feature_extraction.py   150-D feature pipeline
  segmentation.py         Envelope + peak hit detection
  train_and_export.py     Trains the model from the labeled audio files
  requirements.txt
  model/                  Generated by train_and_export.py
frontend/
  index.html              Polished dark dashboard UI (Tailwind via CDN)
  app.js                  Recording, upload, waveform canvas, results
README.md
```

---

## Why "Flange-Invariant"?

Each physical flange has its own resonant fingerprint — geometry, mounting,
microphone position. A naïve model can latch onto the *flange identity* instead
of the bolt looseness, which then collapses on a held-out flange. The
Flange-Invariant LR subtracts each flange's typical acoustic signature before
classification, forcing the model to focus on *torque-driven* changes in the
ring-down. The 100 ANOVA-selected features maximise `F(torque) / (F(flange)+1)`,
i.e. variance explained by torque relative to variance explained by flange.

The notebook also explored a Random Forest and deep models (CNN, RNN), and an
ensemble plus a two-stage classifier. For this competition's realistic
same-flange-new-session regime, the Flange-Invariant LR was the chosen final
model, and that is what the web app implements end-to-end.
