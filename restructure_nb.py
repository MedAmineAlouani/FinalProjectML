"""
Streamline the notebook so it goes straight from setup -> good approach,
structured around the three competition tasks.

Plan:
  Keep cells 0-31 (setup, hit segmentation, features, hyperparams)
  REPLACE cells 32-55 (the original baseline + improvements + champion intro)
    with a clean structure:
      - Build feature matrix
      - Define the two shallow models (tuned RF + LR-centered champion)
      - Task 1: Dependent Test (70/30) with both models, w/ confusion matrices
      - Task 2: Independent Test (LOFO) with both models, w/ per-fold CMs and
                2-class breakdown (matches Table 2 of competition doc)
      - Train both models on ALL labeled data for final use
  Keep cells 56-63 (prediction on unlabeled + constrained prediction)
"""
import json

with open("Final_Project_ML_Second_Attempt.ipynb") as f:
    nb = json.load(f)

print(f"Starting cell count: {len(nb['cells'])}")

# Find boundary indices ----------------------------------------------------
# The setup ends after cell 31 (apply hyperparameters + build feature matrix).
# The prediction section starts at cell 56 ("# Prediction on unlabeled data").
def find_md(text):
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'markdown': continue
        s = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        if s.strip().startswith(text):
            return i
    return None

i_pred = find_md("# Prediction on unlabeled data")
print(f"Prediction section starts at cell {i_pred}")

# Boundary: keep cells 0..31, drop 32..(i_pred-1), keep i_pred..end
SETUP_END = 32   # everything strictly before this index is setup we keep
preserved_setup = nb['cells'][:SETUP_END]
preserved_pred  = nb['cells'][i_pred:]
print(f"Preserved setup cells: {len(preserved_setup)}")
print(f"Preserved prediction cells: {len(preserved_pred)}")


# Build the new middle section --------------------------------------------
def md(s):
    return {"cell_type": "markdown", "metadata": {},
            "source": s.splitlines(keepends=True)}
def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(keepends=True)}


MODELS_INTRO = """# Shallow learning models

We use **two complementary shallow models**:

1. **Tuned Random Forest** — best performer on the original Independent Test
   (Table 2 / leave-one-flange-out).  Hyperparameters were chosen by random
   search with LOFO file-level accuracy as the score:
   `n_estimators=600, min_samples_leaf=3, max_features=0.3, max_depth=12,
    criterion="gini", class_weight="balanced", bootstrap=True`.

2. **Flange-invariant Logistic Regression** — a more robust model for the
   actual experimental test, where the test flanges are the same physical
   flanges as training but at unknown new torques.  It uses
   per-flange feature centering + top-100 torque-discriminative features
   (selected by `f_classif(X, y) / f_classif(X, flange)` ratio).  This
   forces the model to ignore flange-specific resonance patterns and
   focus on torque-driven variation.

Both models use **soft voting** to aggregate the ~20 hits in a file into
one file-level prediction (average `predict_proba`, then `argmax`).
"""

MODELS_CODE = """# ============================================================
# Build the labeled feature matrix once (used by all tasks)
# ============================================================
X_features   = np.array([extract_hybrid_features(h["signal"], h["sr"]) for h in hits_data])
y_labels     = np.array([h["torque"]      for h in hits_data])
flange_ids   = np.array([h["flange_id"]   for h in hits_data])
area_ids     = np.array([h["area_id"]     for h in hits_data])
source_files = np.array([h["file_name"]   for h in hits_data])

print("Feature matrix:", X_features.shape,
      "| classes:", np.unique(y_labels),
      "| flanges:", np.unique(flange_ids))
"""

MODELS_DEFS = """# ============================================================
# Model 1: Tuned Random Forest (used directly on the v2 features)
# ============================================================
def make_rf_tuned():
    return RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=3,
        max_features=0.3,
        max_depth=12,
        criterion="gini",
        class_weight="balanced",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# Model 2: Flange-invariant Logistic Regression
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif


def fit_per_flange_means(X_train, fl_train):
    \"\"\"Compute per-flange feature means from TRAINING data only.\"\"\"
    return {f: X_train[fl_train == f].mean(axis=0) for f in np.unique(fl_train)}


def apply_per_flange_centering(X, fl_ids, means):
    \"\"\"Subtract the per-flange mean from each row.\"\"\"
    Xc = X.astype(float).copy()
    if not means:
        return Xc
    fallback = np.mean(list(means.values()), axis=0)
    for i in range(len(X)):
        Xc[i] -= means.get(fl_ids[i], fallback)
    return Xc


def fit_torque_discriminative_features(X_train, y_train, fl_train, n_keep=100):
    \"\"\"Pick the n_keep features with the highest torque/flange F-ratio.\"\"\"
    f_torque, _ = f_classif(X_train, y_train)
    f_flange, _ = f_classif(X_train, fl_train)
    score = np.nan_to_num(f_torque) / (np.nan_to_num(f_flange) + 1.0)
    return np.argsort(score)[::-1][:n_keep]


def make_lr_champion(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=3000, C=C, random_state=RANDOM_STATE)),
    ])
"""

SOFT_VOTE_HELPERS = """# ============================================================
# Soft-vote helper: aggregate hits in a file into one prediction
# ============================================================

def soft_vote_per_file(model, X_test, files_test, classes_ref=np.array([0, 25, 50])):
    \"\"\"Average predict_proba across hits in the same file, return arrays of
    (file_names, predicted_class).\"\"\"
    proba = model.predict_proba(X_test)
    idx = [list(model.classes_).index(c) for c in classes_ref]
    proba = proba[:, idx]
    df = pd.DataFrame(proba, columns=classes_ref)
    df["__file__"] = files_test
    avg = df.groupby("__file__").mean().sort_index()
    pred = classes_ref[avg.values.argmax(axis=1)]
    return avg.index.values, pred


def file_truth(files_test, y_test):
    \"\"\"True torque per file (assumes all hits in a file share the label).\"\"\"
    return (pd.DataFrame({"f": files_test, "y": y_test})
              .groupby("f")["y"].first()
              .sort_index())
"""

TASK1_INTRO = """# Task 1: Dependent Test (70/30 split)

Per the competition spec, we combine all 4 datasets into one big dataset,
split into 70% train / 30% test, train each model and report classification
accuracy + confusion matrix.

> **NOTE on data leakage.**  Because the same multi-hit recording produces
> ~20 nearly-identical hits with the same label, a random hit-level split
> puts hits from the same recording into both train and test.  This
> drives all models to **near 100%** accuracy on the dependent test, so
> the dependent test is largely uninformative — it does **not** measure
> generalization.  We report it as required and use the **Independent
> Test (Task 2)** for honest comparison.
"""

TASK1_CODE = """# ============================================================
# Task 1: Dependent Test - 70/30 stratified hit-level split
# ============================================================
X_train_dep, X_test_dep, y_train_dep, y_test_dep, files_train_dep, files_test_dep = (
    train_test_split(X_features, y_labels, source_files,
                     test_size=0.30, random_state=RANDOM_STATE,
                     stratify=y_labels))
print(f"Train: {X_train_dep.shape}, Test: {X_test_dep.shape}")
CLASS_NAMES = ["0 ft-lbs", "25 ft-lbs", "50 ft-lbs"]
classes_ref = np.array([0, 25, 50])


# ----- Model 1: Tuned RF -----
print("=" * 70)
print("Tuned Random Forest")
print("=" * 70)
rf = make_rf_tuned()
rf.fit(X_train_dep, y_train_dep)
yp_rf = rf.predict(X_test_dep)
acc_rf_dep = accuracy_score(y_test_dep, yp_rf)
print(f"Accuracy: {acc_rf_dep * 100:.2f}%")
print(classification_report(y_test_dep, yp_rf, target_names=CLASS_NAMES, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test_dep, yp_rf,
                                        display_labels=CLASS_NAMES)
plt.title("Tuned RF - Dependent Test 3-class CM"); plt.show()


# ----- Model 2: Flange-invariant LR -----
print("=" * 70)
print("Flange-invariant Logistic Regression")
print("=" * 70)
# Fit per-flange means + feature selection on the TRAIN portion only
fl_train_dep = np.array([flange_ids[np.where(source_files == f)[0][0]]
                          for f in files_train_dep])
fl_test_dep  = np.array([flange_ids[np.where(source_files == f)[0][0]]
                          for f in files_test_dep])
keep_idx_dep = fit_torque_discriminative_features(X_train_dep, y_train_dep,
                                                    fl_train_dep, n_keep=100)
means_dep = fit_per_flange_means(X_train_dep[:, keep_idx_dep], fl_train_dep)

X_tr_lr = apply_per_flange_centering(X_train_dep[:, keep_idx_dep], fl_train_dep, means_dep)
X_te_lr = apply_per_flange_centering(X_test_dep[:, keep_idx_dep],  fl_test_dep,  means_dep)

lr = make_lr_champion()
lr.fit(X_tr_lr, y_train_dep)
yp_lr = lr.predict(X_te_lr)
acc_lr_dep = accuracy_score(y_test_dep, yp_lr)
print(f"Accuracy: {acc_lr_dep * 100:.2f}%")
print(classification_report(y_test_dep, yp_lr, target_names=CLASS_NAMES, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test_dep, yp_lr,
                                        display_labels=CLASS_NAMES)
plt.title("Flange-invariant LR - Dependent Test 3-class CM"); plt.show()


dependent_summary = pd.DataFrame([
    {"Model": "Tuned Random Forest",                  "Dependent Test Accuracy (%)": acc_rf_dep * 100},
    {"Model": "Flange-invariant Logistic Regression", "Dependent Test Accuracy (%)": acc_lr_dep * 100},
])
display(dependent_summary)
"""

TASK2_INTRO = """# Task 2: Independent Test (Table 2 / leave-one-flange-out)

For each held-out flange we train on the other 3 datasets and test on the
held-out one.  This is the **honest measure** of model generalization since
no hits from the test flange leak into training.

For each fold we report:
  - 3-class accuracy + confusion matrix (0 / 25 / 50 ft-lbs)
  - 2-class accuracy + confusion matrix (0 vs 25-and-50 ft-lbs)
  - File-level accuracy (~12 files per fold) using soft-vote aggregation
"""

TASK2_CODE = """# ============================================================
# Task 2: Independent Test (LOFO, leave-one-flange-out)
# ============================================================
classes_ref = np.array([0, 25, 50])
TWO_CLASS_NAMES = ["0 ft-lbs (tight)", "25/50 ft-lbs (loose)"]


def loose_binarize(y):
    \"\"\"Convert torque labels to 0=tight (0 ft-lbs) vs 1=loose (25 or 50).\"\"\"
    return (y != 0).astype(int)


def lofo_3class_2class_eval(model_factory, name, use_centering_and_select=False):
    print(f"\\n{'=' * 78}\\n{name}\\n{'=' * 78}")
    rows = []
    for test_flange in [1, 2, 3, 4]:
        tr = flange_ids != test_flange
        te = flange_ids == test_flange

        Xtr = X_features[tr]; ytr = y_labels[tr]; fl_tr = flange_ids[tr]
        Xte = X_features[te]; yte = y_labels[te]; fl_te = flange_ids[te]
        files_te = source_files[te]

        # If LR-style: per-flange centering + feature selection (TRAIN-only stats)
        if use_centering_and_select:
            keep_idx = fit_torque_discriminative_features(Xtr, ytr, fl_tr, n_keep=100)
            Xtr = Xtr[:, keep_idx]
            Xte = Xte[:, keep_idx]
            means = fit_per_flange_means(Xtr, fl_tr)
            Xtr = apply_per_flange_centering(Xtr, fl_tr, means)
            Xte = apply_per_flange_centering(Xte, fl_te, means)

        m = model_factory()
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)

        acc_3 = accuracy_score(yte, yp)
        acc_2 = accuracy_score(loose_binarize(yte), loose_binarize(yp))

        # File-level via soft vote
        files_pred, pred_file = soft_vote_per_file(m, Xte, files_te, classes_ref)
        true_file_arr = file_truth(files_te, yte).loc[files_pred].values
        acc_file = accuracy_score(true_file_arr, pred_file)

        cm_3 = confusion_matrix(yte, yp,           labels=classes_ref)
        cm_2 = confusion_matrix(loose_binarize(yte), loose_binarize(yp), labels=[0, 1])

        print(f"\\n--- Held-out Flange {test_flange} ---")
        print(f"  3-class hit-level accuracy : {acc_3 * 100:5.2f}%")
        print(f"  2-class hit-level accuracy : {acc_2 * 100:5.2f}%   (0 vs 25/50)")
        print(f"  File-level accuracy        : {acc_file * 100:5.2f}%   ({len(true_file_arr)} files)")
        print("  3-class confusion matrix (rows=true, cols=pred):")
        print("    " + "       ".join(f"{c:>3} ft-lbs" for c in classes_ref))
        for r, true_c in enumerate(classes_ref):
            print(f"    {true_c:>3} ft-lbs " + " ".join(f"{cm_3[r, c]:>10d}" for c in range(3)))
        print("  2-class confusion matrix:")
        print(f"    {'tight':>15} {'loose':>10}")
        print(f"    tight {cm_2[0, 0]:>10d} {cm_2[0, 1]:>10d}")
        print(f"    loose {cm_2[1, 0]:>10d} {cm_2[1, 1]:>10d}")

        rows.append({"Test Flange": test_flange,
                     "3-class hit %": round(acc_3 * 100, 2),
                     "2-class hit %": round(acc_2 * 100, 2),
                     "File-level %":  round(acc_file * 100, 2)})

    summary = pd.DataFrame(rows)
    summary.loc["mean"] = ["mean",
                            summary["3-class hit %"].mean(),
                            summary["2-class hit %"].mean(),
                            summary["File-level %"].mean()]
    print(f"\\nSummary for {name}:")
    display(summary)
    return summary


rf_summary = lofo_3class_2class_eval(make_rf_tuned,
                                       "Tuned Random Forest",
                                       use_centering_and_select=False)
lr_summary = lofo_3class_2class_eval(make_lr_champion,
                                       "Flange-invariant Logistic Regression",
                                       use_centering_and_select=True)


# Combined comparison
combined = pd.DataFrame({
    "Model":         ["Tuned RF", "Flange-invariant LR"],
    "3-class hit %": [rf_summary.loc["mean", "3-class hit %"],
                       lr_summary.loc["mean", "3-class hit %"]],
    "2-class hit %": [rf_summary.loc["mean", "2-class hit %"],
                       lr_summary.loc["mean", "2-class hit %"]],
    "File-level %":  [rf_summary.loc["mean", "File-level %"],
                       lr_summary.loc["mean", "File-level %"]],
})
print("\\n=== Independent Test (LOFO) - mean across 4 folds ===")
display(combined)
"""

TRAIN_FINAL_INTRO = """## Train both models on ALL labeled data

For the actual experimental test (Task 3) we train each model on the full
48-file labeled dataset.  No held-out portion — we want every available
hit informing the final predictions.
"""

TRAIN_FINAL_CODE = """# ============================================================
# Final tuned RF (trained on all labeled hits)
# ============================================================
final_rf = make_rf_tuned()
final_rf.fit(X_features, y_labels)
print("Tuned RF trained on", X_features.shape[0], "hits.")


# ============================================================
# Final flange-invariant LR (trained on all labeled hits)
# ============================================================
keep_idx_final     = fit_torque_discriminative_features(X_features, y_labels,
                                                          flange_ids, n_keep=100)
flange_means_final = fit_per_flange_means(X_features[:, keep_idx_final], flange_ids)
X_lr_final         = apply_per_flange_centering(X_features[:, keep_idx_final],
                                                 flange_ids, flange_means_final)
champion_model = make_lr_champion()
champion_model.fit(X_lr_final, y_labels)
print(f"Flange-invariant LR trained on {X_features.shape[0]} hits "
      f"with {X_lr_final.shape[1]} torque-discriminative features.")
"""

new_middle = [
    md(MODELS_INTRO),
    code(MODELS_CODE),
    code(MODELS_DEFS),
    code(SOFT_VOTE_HELPERS),
    md(TASK1_INTRO),
    code(TASK1_CODE),
    md(TASK2_INTRO),
    code(TASK2_CODE),
    md(TRAIN_FINAL_INTRO),
    code(TRAIN_FINAL_CODE),
]

# Update the cell 2 (imports) to include train_test_split + classification_report
# + ConfusionMatrixDisplay if not already there
imp_cell = preserved_setup[2]
imp_src = ''.join(imp_cell['source']) if isinstance(imp_cell['source'], list) else imp_cell['source']
needed = [
    "from sklearn.model_selection import train_test_split",
    "from sklearn.preprocessing import StandardScaler",
    "from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix)",
    "from sklearn.ensemble import RandomForestClassifier",
    "from sklearn.pipeline import Pipeline",
]
# The original imports already include most of these, so just verify quickly
for n in needed:
    if n.split()[1] not in imp_src and 'sklearn' not in imp_src:
        print(f"WARNING: import missing: {n}")
# We don't rewrite the imports cell; the original already has all sklearn pieces.

new_cells = preserved_setup + new_middle + preserved_pred
nb['cells'] = new_cells

# Clear all outputs since we're going to re-execute
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

with open("Final_Project_ML_Second_Attempt.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal cell count: {len(nb['cells'])}")
print("Streamlined notebook written.")
EOF