"""
Better methodology flowchart: branching diagram showing three model tracks.

Layout:
  Audio -> Segment -> [3 parallel branches with model-specific preprocessing]
                                  -> Per-flange soft-vote -> Final torque
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

OUTDIR = "poster_figures"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# Colors per branch
COLOR_RF   = "#4C72B0"   # blue
COLOR_LR   = "#DD8452"   # orange
COLOR_CRNN = "#55A868"   # green
COLOR_SHARED = "#5D6D7E" # slate
COLOR_INNOV  = "#C0392B" # red border for innovations
COLOR_OUTPUT = "#2E2E2E" # dark gray

fig, ax = plt.subplots(figsize=(14, 6.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")

def box(x, y, w, h, label, fill, edge=None, edgewidth=1.5, fontsize=10,
        textcolor="white", italic=False):
    if edge is None:
        edge = fill
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                            facecolor=fill, edgecolor=edge, linewidth=edgewidth)
    ax.add_patch(patch)
    style = "italic" if italic else "normal"
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
             fontsize=fontsize, color=textcolor, fontweight="bold", style=style,
             linespacing=1.15)

def arrow(x1, y1, x2, y2, color="#444"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle="-|>", mutation_scale=14,
                         color=color, linewidth=1.7)
    ax.add_patch(a)

# === TITLE ===
ax.text(7, 6.6, "Final Prediction Pipeline",
         ha="center", fontsize=14, fontweight="bold")

# === Shared input (left side) ===
box(0.3, 4.6, 1.6, 0.8, "Raw Audio\n(48 kHz)", COLOR_SHARED)
box(2.4, 4.6, 1.6, 0.8, "Peak\nSegmentation\n(~20 hits/file)", COLOR_SHARED)
arrow(1.9, 5.0, 2.4, 5.0, COLOR_SHARED)

# === Branch labels (well above the box rows so they don't collide) ===
ax.text(4.3, 5.95, "Tuned RF", ha="left", fontsize=11, fontweight="bold", color=COLOR_RF)
ax.text(4.3, 4.15, "Flange-Invariant LR", ha="left", fontsize=11, fontweight="bold", color=COLOR_LR)
ax.text(4.3, 1.75, "CRNN (Conv2D + Bi-GRU)", ha="left", fontsize=11, fontweight="bold", color=COLOR_CRNN)

# Top branch: RF
box(4.5, 5.1, 1.9, 0.7, "150-dim\nHybrid Features", COLOR_RF, fontsize=9)
box(7.0, 5.1, 1.7, 0.7, "Tuned RF\n(600 trees)", COLOR_RF, fontsize=9)

# Middle branch: LR
box(4.5, 3.3, 1.9, 0.7, "150-dim\nHybrid Features", COLOR_LR, fontsize=9)
# Innovation: per-flange centering (red border)
box(6.6, 3.3, 1.9, 0.7, "Per-Flange\nCentering", "#FFE4D6", edge=COLOR_INNOV,
    edgewidth=2, fontsize=9, textcolor=COLOR_INNOV)
# Innovation: ANOVA top-100
box(8.7, 3.3, 1.9, 0.7, "ANOVA Top-100\nF(torque)/F(flange)", "#FFE4D6",
    edge=COLOR_INNOV, edgewidth=2, fontsize=9, textcolor=COLOR_INNOV)
box(10.8, 3.3, 1.5, 0.7, "LogReg\n(L2)", COLOR_LR, fontsize=9)

# Bottom branch: CRNN
# Innovation: 3-channel input
box(4.5, 0.9, 2.1, 0.7, "Log-mel + Δ + ΔΔ\n(3-channel)", "#D8EEDC",
    edge=COLOR_INNOV, edgewidth=2, fontsize=9, textcolor=COLOR_INNOV)
box(6.8, 0.9, 1.9, 0.7, "Conv2D\n(freq-only pool)", COLOR_CRNN, fontsize=9)
box(8.9, 0.9, 1.6, 0.7, "Bi-GRU\nstack", COLOR_CRNN, fontsize=9)
box(10.7, 0.9, 1.6, 0.7, "Dense\nhead", COLOR_CRNN, fontsize=9)

# Branching arrows from "Peak Segmentation"
arrow(4.0, 5.0, 4.5, 5.45, "#444")
arrow(4.0, 5.0, 4.5, 3.65, "#444")
arrow(4.0, 5.0, 4.5, 1.25, "#444")

# Within-branch arrows
# RF
arrow(6.4, 5.45, 7.0, 5.45, COLOR_RF)
# LR
arrow(6.4, 3.65, 6.6, 3.65, COLOR_LR)
arrow(8.5, 3.65, 8.7, 3.65, COLOR_LR)
arrow(10.6, 3.65, 10.8, 3.65, COLOR_LR)
# CRNN
arrow(6.6, 1.25, 6.8, 1.25, COLOR_CRNN)
arrow(8.7, 1.25, 8.9, 1.25, COLOR_CRNN)
arrow(10.5, 1.25, 10.7, 1.25, COLOR_CRNN)

# === Right side: aggregation + final ===
# Per-hit probabilities label
ax.text(11.45, 5.85, "P[0,25,50]", ha="center", fontsize=8, color=COLOR_RF, style="italic")
ax.text(11.85, 4.15, "P[0,25,50]", ha="center", fontsize=8, color=COLOR_LR, style="italic")
ax.text(11.85, 1.75, "P[0,25,50]", ha="center", fontsize=8, color=COLOR_CRNN, style="italic")

# Soft-vote box
box(12.7, 3.0, 1.1, 1.4, "Soft-vote\nper flange\n+\nArgmax", COLOR_OUTPUT, fontsize=9)

# Arrows from each branch end into soft-vote
arrow(8.7, 5.45, 12.85, 4.4, COLOR_RF)
arrow(12.3, 3.65, 12.7, 3.65, COLOR_LR)
arrow(12.3, 1.25, 12.85, 3.0, COLOR_CRNN)

# Final output below soft-vote
box(12.5, 1.6, 1.5, 0.7, "Final Torque\nprediction", "#2C3E50", fontsize=10)
arrow(13.25, 3.0, 13.25, 2.3, "#2C3E50")

# === Innovations legend ===
legend_y = 0.05
ax.add_patch(mpatches.FancyBboxPatch(
    (0.3, legend_y), 13.4, 0.5,
    boxstyle="round,pad=0.02",
    facecolor="#FFFFFF", edgecolor="#CCCCCC", linewidth=1))
ax.text(7, legend_y + 0.25,
         "Innovations beyond course content (red borders): "
         "per-flange centering • ANOVA F(torque)/F(flange) selection • "
         "3-channel log-mel + Δ + ΔΔ CRNN input",
         ha="center", fontsize=9.5, color="#444", style="italic")

plt.savefig(f"{OUTDIR}/fig_flowchart.png", bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {OUTDIR}/fig_flowchart.png")
