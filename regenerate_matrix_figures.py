"""
regenerate_matrix_figures.py
============================
Regenerates Figures 5 and 6 from saved CSV outputs.
Now includes RPD=1.0 mean-predictor reference line on the colorbar.

Run from SOLUM root folder:
    python regenerate_matrix_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

PLSR_CSV = os.path.join("outputs", "transferability_matrix_plsr_rpd.csv")
RF_CSV   = os.path.join("outputs", "transferability_matrix_rf_rpd.csv")
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

RPD_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "rpd_cmap",
    [(0.0, "#C0392B"), (0.4, "#E67E22"), (1.0, "#0d7a7a")]
)

LU_NAMES = {
    "A": "Cropland (Arable)",
    "B": "Cropland (Permanent)",
    "C": "Woodland",
    "D": "Shrubland",
    "E": "Grassland",
    "F": "Bare Land",
}


def plot_matrix(rpd_matrix: pd.DataFrame, model_label: str, fig_num: int, out_path: str):
    lu_codes = list(rpd_matrix.index)
    n = len(lu_codes)
    rpd_vals = rpd_matrix.values.astype(float)
    vmin, vmax = 0.5, 3.5

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rpd_vals, vmin=vmin, vmax=vmax, cmap=RPD_COLORMAP, aspect="auto")

    for i in range(n):
        for j in range(n):
            val = rpd_vals[i, j]
            text = "N/A" if np.isnan(val) else f"{val:.2f}"
            brightness = (val - vmin) / (vmax - vmin)
            text_color = "white" if brightness < 0.35 or brightness > 0.78 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    for k in range(n):
        rect = plt.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=2.2
        )
        ax.add_patch(rect)

    xlabels = [f"{lu}\n({LU_NAMES.get(lu, lu)})" for lu in lu_codes]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(xlabels, fontsize=8.5, rotation=30, ha="right")
    ax.set_yticklabels(xlabels, fontsize=8.5)
    ax.set_xlabel("Target Land Use Class (test set)", fontsize=10, labelpad=8)
    ax.set_ylabel("Source Land Use Class (training set)", fontsize=10, labelpad=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("RPD", fontsize=9)
    # RPD threshold lines on colorbar
    cbar.ax.axhline(y=(1.0 - vmin)/(vmax - vmin), color="#C0392B",
                    linewidth=1.5, linestyle="-")   # mean predictor
    cbar.ax.axhline(y=(1.4 - vmin)/(vmax - vmin), color="black",
                    linewidth=1.0, linestyle="--")  # moderate threshold
    cbar.ax.axhline(y=(2.0 - vmin)/(vmax - vmin), color="black",
                    linewidth=1.0, linestyle="--")  # good threshold
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"Figure {fig_num}: Transferability Matrix ({model_label} RPD)\n"
        "Diagonal = in-domain; off-diagonal = cross-domain transfer. "
        "Dashed lines at RPD=1.4 and 2.0; red line at RPD=1.0 (mean predictor).",
        fontsize=8.5, loc="left", pad=10
    )

    legend_elements = [
        Patch(facecolor="#C0392B", label="Poor (RPD < 1.4)"),
        Patch(facecolor="#E67E22", label="Moderate (1.4 \u2264 RPD < 2.0)"),
        Patch(facecolor="#0d7a7a", label="Good (RPD \u2265 2.0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=8.5,
              framealpha=0.9, frameon=True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for path in [PLSR_CSV, RF_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find {path}\n"
                "Run from the SOLUM root folder with outputs/ already populated."
            )

    plsr_matrix = pd.read_csv(PLSR_CSV, index_col=0)
    rf_matrix   = pd.read_csv(RF_CSV,   index_col=0)

    print("PLSR RPD matrix:")
    print(plsr_matrix.to_string(float_format="{:.2f}".format))
    print("\nRF RPD matrix:")
    print(rf_matrix.to_string(float_format="{:.2f}".format))

    plot_matrix(plsr_matrix, "PLSR", 5,
                os.path.join(FIG_DIR, "fig5_transferability_plsr.png"))
    plot_matrix(rf_matrix, "RF", 6,
                os.path.join(FIG_DIR, "fig6_transferability_rf.png"))

    print("\nDone.")