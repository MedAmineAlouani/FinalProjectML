"""Standalone "Final Torque prediction" arrow + box, for placing
separately on the poster (not part of the main flowchart)."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTDIR = "poster_figures"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

fig, ax = plt.subplots(figsize=(4, 1.2))
ax.set_xlim(0, 4); ax.set_ylim(0, 1.2); ax.axis("off")

# Arrow (left side)
a = FancyArrowPatch((0.1, 0.6), (1.4, 0.6),
                     arrowstyle="-|>", mutation_scale=22,
                     color="#2C3E50", linewidth=2.4)
ax.add_patch(a)

# Box (right side)
patch = FancyBboxPatch((1.5, 0.15), 2.4, 0.9,
                        boxstyle="round,pad=0.02,rounding_size=0.05",
                        facecolor="#2C3E50", edgecolor="#2C3E50", linewidth=1.5)
ax.add_patch(patch)
ax.text(2.7, 0.6, "Final Torque\nprediction",
         ha="center", va="center", fontsize=14, color="white",
         fontweight="bold", linespacing=1.2)

plt.savefig(f"{OUTDIR}/fig_final_output.png", bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {OUTDIR}/fig_final_output.png")
