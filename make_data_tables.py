"""Generate Table 1 (labeled distribution) and Table 2 (unlabeled distribution) as PNG."""
import os, glob, re
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import librosa
import matplotlib.pyplot as plt
import optimize as opt

OUTDIR = "poster_figures"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# Labeled
print("Loading labeled hits ...")
hits = opt.build_hits()
y = np.array([h["torque"] for h in hits])
fl = np.array([h["flange_id"] for h in hits])
print(f"  {len(hits)} labeled single-hit samples")

# Unlabeled
print("Loading unlabeled hits ...")
unl_hits = []
for path in sorted(glob.glob("F[0-9]A[0-9].m4a")):
    name = os.path.basename(path); m = re.match(r"F(\d+)A(\d+)\.m4a", name)
    sig, sr = librosa.load(path, sr=None, mono=True)
    sig = opt.normalize_audio(sig)
    for hit in opt.split_into_hits(sig, sr):
        unl_hits.append({"flange_id": int(m.group(1)), "signal": hit})
fl_un = np.array([h["flange_id"] for h in unl_hits])
print(f"  {len(unl_hits)} unlabeled single-hit samples")

# --- Build Table 1 dataframe ---
t1 = (pd.crosstab(
        pd.Categorical(fl, categories=[1,2,3,4]),
        pd.Categorical(y,  categories=[0,25,50]),
        rownames=["Flange"], colnames=["Torque"])
        .rename(columns={0:"Torque 0", 25:"Torque 25", 50:"Torque 50"}))
t1["Total"] = t1.sum(axis=1)
t1.loc["Total"] = t1.sum(axis=0)
print("\nTable 1 contents:")
print(t1.to_string())

# --- Build Table 2 dataframe ---
t2 = (pd.Series(fl_un).value_counts()
        .reindex([1,2,3,4]).astype(int)
        .rename("Samples Collected").to_frame())
t2.index.name = "Flange"
t2.loc["Total"] = t2.sum(axis=0)
print("\nTable 2 contents:")
print(t2.to_string())


def render_table(df, fname, title, header_color="#222222", header_text="white"):
    """Render a pandas DataFrame as a clean poster-style table PNG."""
    rows = df.shape[0] + 1   # +1 for header
    cols = df.shape[1] + 1   # +1 for index
    fig, ax = plt.subplots(figsize=(1.2 * cols + 1.5, 0.55 * rows + 0.5))
    ax.axis("off")

    # Build cell texts
    table_data = [[df.index.name or ""] + list(df.columns)]
    for idx, row in df.iterrows():
        table_data.append([str(idx)] + [str(v) for v in row.values])

    table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                     colWidths=[1.0/cols] * cols)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # Style header row
    for j in range(cols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text, fontweight="bold")

    # Style "Total" row (last row) - light gray
    last_row_idx = rows - 1
    for j in range(cols):
        cell = table[last_row_idx, j]
        cell.set_facecolor("#D0D0D0")
        cell.set_text_props(fontweight="bold")

    # Alternate row colors for readability
    for i in range(1, rows - 1):
        for j in range(cols):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#F5F5F5")

    # Title above
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved {fname}")


print("\nRendering tables ...")
render_table(t1, f"{OUTDIR}/table1_labeled_distribution.png",
              "Table 1: Task 1 & 2 data distribution (labeled single-hit samples)")
render_table(t2, f"{OUTDIR}/table2_unlabeled_distribution.png",
              "Table 2: Task 3 data distribution (unlabeled experimental-test samples)")

print("\nDone. Files in poster_figures/:")
for fn in sorted(os.listdir(OUTDIR)):
    print(f"  {fn}")
