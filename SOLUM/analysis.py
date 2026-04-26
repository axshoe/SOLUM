"""
analysis.py
===========
Figure generation for SOLUM. Produces all publication-quality plots.

Figures generated:
  Figure 1  - SOC distribution by land use class (violin plot)
  Figure 2  - Mean preprocessed spectra by land use class
  Figure 3  - PLSR in-domain performance (actual vs. predicted)
  Figure 4  - RF in-domain performance (actual vs. predicted)
  Figure 5  - Transferability Matrix heatmap (PLSR RPD)
  Figure 6  - Transferability Matrix heatmap (RF RPD)
  Figure 7  - SHAP discrepancy heatmap (wavelength × transfer pair)
  Figure 8  - Top discrepant wavelength bands for worst transfers

All figures are saved to the 'figures/' directory in both PNG
(for inline embedding) and HTML (for thexiulab.org).

Style: Times New Roman where supported, #0d7a7a / #1F77B4 accent colors,
journal-publishable aesthetic, no emojis, numbered captions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from data_loader import LU_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Style configuration
# ─────────────────────────────────────────────────────────────────────────────

ACCENT_TEAL = "#0d7a7a"
ACCENT_BLUE = "#1F77B4"
ACCENT_RED = "#C0392B"
ACCENT_AMBER = "#E67E22"

RPD_CMAP_COLORS = [
    (0.0, "#C0392B"),    # red   < 1.4
    (0.4, "#E67E22"),    # amber 1.4-2.0
    (1.0, "#0d7a7a"),    # teal  > 2.0 (max ~3.5)
]
RPD_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "rpd_cmap",
    [(v, c) for v, c in RPD_CMAP_COLORS]
)

LU_PALETTE = {
    "A": "#1F77B4",
    "B": "#0d7a7a",
    "C": "#2CA02C",
    "D": "#8B4513",
    "E": "#9467BD",
    "F": "#7F7F7F",
}

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

def _ensure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)

def _save(fig, name: str, dpi: int = 200):
    _ensure_dir()
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[analysis] Saved: {path}")
    return path


def _style_axes(ax, spine_color="#333333"):
    """Apply consistent axis styling."""
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=spine_color, labelsize=9)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: SOC distribution by land use class
# ─────────────────────────────────────────────────────────────────────────────

def fig1_soc_distribution(lu_splits: Dict, out_name: str = "fig1_soc_distribution.png"):
    """Violin plot of SOC distribution per land use class."""
    from data_loader import SOC_COL

    lu_codes = sorted(lu_splits.keys())
    fig, ax = plt.subplots(figsize=(8, 5))

    data_by_class = [lu_splits[lu][SOC_COL].values for lu in lu_codes]
    labels = [f"{lu}\n({LU_NAMES.get(lu, lu)})" for lu in lu_codes]

    parts = ax.violinplot(
        data_by_class,
        positions=np.arange(len(lu_codes)),
        showmedians=True,
        showextrema=True,
    )

    for i, pc in enumerate(parts["bodies"]):
        lu = lu_codes[i]
        pc.set_facecolor(LU_PALETTE.get(lu, ACCENT_BLUE))
        pc.set_alpha(0.7)
        pc.set_edgecolor("#333333")
        pc.set_linewidth(0.8)

    for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts[part].set_color("#333333")
        parts[part].set_linewidth(1.0)

    ax.set_xticks(np.arange(len(lu_codes)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Soil Organic Carbon (g/kg)", fontsize=10)
    ax.set_xlabel("Land Use Class", fontsize=10)
    ax.set_title(
        "Figure 1: SOC Distribution by Land Use Class (LUCAS 2018 Topsoil)",
        fontsize=10, pad=10, loc="left"
    )
    _style_axes(ax)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    return _save(fig, out_name)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Mean preprocessed spectra by land use class
# ─────────────────────────────────────────────────────────────────────────────

def fig2_mean_spectra(
    splits_proc: Dict,
    wavelengths: np.ndarray,
    out_name: str = "fig2_mean_spectra.png"
):
    """Mean SNV-preprocessed reflectance spectrum for each land use class."""
    lu_codes = sorted(splits_proc.keys())
    fig, ax = plt.subplots(figsize=(10, 5))

    for lu in lu_codes:
        X_all = np.vstack([
            splits_proc[lu]["X_train"],
            splits_proc[lu]["X_test"],
        ])
        mean_spec = X_all.mean(axis=0)
        std_spec = X_all.std(axis=0)

        color = LU_PALETTE.get(lu, ACCENT_BLUE)
        ax.plot(wavelengths, mean_spec, color=color, linewidth=1.2,
                label=f"{lu}: {LU_NAMES.get(lu, lu)}")
        ax.fill_between(
            wavelengths,
            mean_spec - std_spec,
            mean_spec + std_spec,
            color=color, alpha=0.12
        )

    ax.set_xlabel("Wavelength (nm)", fontsize=10)
    ax.set_ylabel("SNV-Preprocessed Reflectance (a.u.)", fontsize=10)
    ax.set_title(
        "Figure 2: Mean Preprocessed Vis-NIR Spectra by Land Use Class\n"
        "(Shaded regions = ±1 SD; SG smoothing window=11, SNV-transformed)",
        fontsize=9, loc="left"
    )
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    _style_axes(ax)
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    return _save(fig, out_name)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 & 4: In-domain actual vs. predicted scatter plots
# ─────────────────────────────────────────────────────────────────────────────

def fig_actual_vs_predicted(
    results: Dict,
    lu_codes: List[str],
    model_key: str = "plsr",
    fig_num: int = 3,
    out_name: Optional[str] = None,
):
    """
    Grid of actual vs. predicted scatter plots for in-domain predictions.
    One subplot per land use class.
    """
    if out_name is None:
        out_name = f"fig{fig_num}_indomain_{model_key}.png"

    n_cols = min(3, len(lu_codes))
    n_rows = int(np.ceil(len(lu_codes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, lu in enumerate(lu_codes):
        ax = axes[i]
        metrics = results[(lu, lu)]
        y_true = metrics["y_test"]
        y_pred = metrics[f"y_pred_{model_key}"]

        rpd = metrics[f"{model_key}_rpd"]
        r2 = metrics[f"{model_key}_r2"]
        rmse = metrics[f"{model_key}_rmse"]

        color = LU_PALETTE.get(lu, ACCENT_BLUE)
        ax.scatter(y_true, y_pred, alpha=0.5, s=12, color=color, edgecolors="none")

        # 1:1 line
        lim_min = min(y_true.min(), y_pred.min()) * 0.95
        lim_max = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8, alpha=0.6)

        ax.set_xlabel("Measured SOC (g/kg)", fontsize=8)
        ax.set_ylabel("Predicted SOC (g/kg)", fontsize=8)
        ax.set_title(
            f"{lu}: {LU_NAMES.get(lu, lu)}\n"
            f"R²={r2:.2f}, RMSE={rmse:.1f}, RPD={rpd:.2f}",
            fontsize=8
        )
        _style_axes(ax)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(len(lu_codes), len(axes)):
        axes[j].set_visible(False)

    model_label = model_key.upper()
    fig.suptitle(
        f"Figure {fig_num}: In-Domain SOC Prediction ({model_label}) — "
        "Actual vs. Predicted by Land Use Class",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    return _save(fig, out_name)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 & 6: Transferability Matrix heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def fig_transferability_heatmap(
    rpd_matrix: pd.DataFrame,
    model_key: str = "plsr",
    fig_num: int = 5,
    out_name: Optional[str] = None,
):
    """
    Heatmap of RPD values across all source-target land use pairs.
    Color: red (< 1.4) → amber (1.4-2.0) → teal (> 2.0).
    """
    if out_name is None:
        out_name = f"fig{fig_num}_transferability_{model_key}.png"

    lu_codes = list(rpd_matrix.index)
    n = len(lu_codes)
    rpd_vals = rpd_matrix.values.astype(float)

    # Clip for colormap (values typically 0.5–3.5)
    vmin, vmax = 0.5, 3.5

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(rpd_vals, vmin=vmin, vmax=vmax, cmap=RPD_COLORMAP, aspect="auto")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = rpd_vals[i, j]
            if np.isnan(val):
                text = "N/A"
            else:
                text = f"{val:.2f}"
            brightness = (val - vmin) / (vmax - vmin)
            text_color = "white" if brightness < 0.4 or brightness > 0.75 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    # Diagonal border to highlight in-domain cells
    for k in range(n):
        rect = plt.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=2.0
        )
        ax.add_patch(rect)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    xlabels = [f"{lu}\n({LU_NAMES.get(lu, lu)})" for lu in lu_codes]
    ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(xlabels, fontsize=8)
    ax.set_xlabel("Target Land Use Class (test set)", fontsize=10)
    ax.set_ylabel("Source Land Use Class (training set)", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("RPD", fontsize=9)
    cbar.ax.axhline(y=(1.4 - vmin) / (vmax - vmin), color="black", linewidth=1.0, linestyle="--")
    cbar.ax.axhline(y=(2.0 - vmin) / (vmax - vmin), color="black", linewidth=1.0, linestyle="--")
    cbar.ax.tick_params(labelsize=8)

    model_label = model_key.upper()
    ax.set_title(
        f"Figure {fig_num}: Transferability Matrix ({model_label} RPD)\n"
        "Diagonal = in-domain; off-diagonal = cross-domain transfer. "
        "Dashed lines at RPD=1.4 and 2.0.",
        fontsize=8, loc="left", pad=10
    )

    legend_elements = [
        Patch(facecolor="#C0392B", label="Poor (RPD < 1.4)"),
        Patch(facecolor="#E67E22", label="Moderate (1.4 ≤ RPD < 2.0)"),
        Patch(facecolor="#0d7a7a", label="Good (RPD ≥ 2.0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8,
              framealpha=0.9, frameon=True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return _save(fig, out_name)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: SHAP discrepancy heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig7_shap_discrepancy_heatmap(
    shap_data: Dict,
    wavelengths: np.ndarray,
    lu_codes: List[str],
    n_bands_shown: int = 50,
    out_name: str = "fig7_shap_discrepancy.png",
):
    """
    Heatmap of SHAP discrepancy by wavelength for all failed transfer pairs.
    Columns = wavelength bins (binned to n_bands_shown evenly spaced points).
    Rows = failed transfer pair labels.
    """
    failed_pairs = [
        (src, tgt) for src in lu_codes for tgt in lu_codes
        if src != tgt and shap_data.get((src, tgt), {}).get("failed_transfer", False)
    ]

    if not failed_pairs:
        print("[analysis] No failed transfers for SHAP heatmap.")
        return None

    # Bin wavelengths for display
    bin_edges = np.linspace(wavelengths[0], wavelengths[-1], n_bands_shown + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(wavelengths, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bands_shown - 1)

    rows = []
    pair_labels = []
    for src, tgt in failed_pairs:
        disc = shap_data[(src, tgt)]["discrepancy"]
        # Average discrepancy within each bin
        binned = np.array([
            disc[bin_idx == b].mean() if np.any(bin_idx == b) else 0.0
            for b in range(n_bands_shown)
        ])
        rows.append(binned)
        pair_labels.append(f"{src}→{tgt}")

    matrix = np.array(rows)  # (n_failed, n_bands_shown)

    fig, ax = plt.subplots(figsize=(12, max(3, len(failed_pairs) * 0.7 + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    # X-axis: show wavelength labels every ~200 nm
    tick_positions = np.linspace(0, n_bands_shown - 1, 9, dtype=int)
    tick_labels = [f"{int(bin_centers[p])} nm" for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.set_xlabel("Wavelength (nm)", fontsize=10)
    ax.set_ylabel("Transfer Pair (Source → Target)", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("|ΔSHAP| (mean absolute)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Figure 7: SHAP Discrepancy Heatmap — Wavelength Attribution for Failed Transfers\n"
        "(Only pairs with RF RPD < 1.4 shown. Warm colors = high discrepancy.)",
        fontsize=9, loc="left"
    )

    fig.tight_layout()
    return _save(fig, out_name)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Top discrepant wavelengths for worst transfers
# ─────────────────────────────────────────────────────────────────────────────

def fig8_top_discrepant_bands(
    shap_data: Dict,
    wavelengths: np.ndarray,
    lu_codes: List[str],
    n_top: int = 5,
    out_name: str = "fig8_top_discrepant_bands.png",
):
    """
    Bar chart of top discrepant wavelength bands for each failed transfer pair.
    """
    from shap_analysis import top_discrepant_bands

    failed_pairs = [
        (src, tgt) for src in lu_codes for tgt in lu_codes
        if src != tgt and shap_data.get((src, tgt), {}).get("failed_transfer", False)
    ]

    if not failed_pairs:
        print("[analysis] No failed transfers for Figure 8.")
        return None

    n_pairs = len(failed_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(4.5 * n_pairs, 4), sharey=False)
    if n_pairs == 1:
        axes = [axes]

    for ax, (src, tgt) in zip(axes, failed_pairs):
        disc = shap_data[(src, tgt)]["discrepancy"]
        top_df = top_discrepant_bands(disc, wavelengths, n_top=n_top)

        bars = ax.barh(
            np.arange(n_top),
            top_df["shap_discrepancy"].values,
            color=ACCENT_TEAL,
            edgecolor="#333333",
            linewidth=0.5,
        )
        ax.set_yticks(np.arange(n_top))
        ax.set_yticklabels([f"{int(w)} nm" for w in top_df["wavelength_nm"]], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("|ΔSHAP|", fontsize=9)
        ax.set_title(f"{src}→{tgt}\nTop {n_top} discrepant bands", fontsize=9)
        _style_axes(ax)

    fig.suptitle(
        "Figure 8: Top Discrepant Wavelength Bands for Failed Transfer Pairs\n"
        "(Bands with highest |ΔSHAP| between source and target SHAP distributions)",
        fontsize=9, y=1.02
    )
    fig.tight_layout()
    return _save(fig, out_name)


# ─────────────────────────────────────────────────────────────────────────────
# Run all figures
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(
    lu_splits,
    splits,
    wavelengths,
    results,
    plsr_rpd_matrix,
    rf_rpd_matrix,
    shap_data,
    lu_codes,
):
    """Convenience wrapper: generate all 8 figures in sequence."""
    print("[analysis] Generating figures...")

    fig1_soc_distribution(lu_splits)
    fig2_mean_spectra(splits, wavelengths)
    fig_actual_vs_predicted(results, lu_codes, model_key="plsr", fig_num=3)
    fig_actual_vs_predicted(results, lu_codes, model_key="rf", fig_num=4)
    fig_transferability_heatmap(plsr_rpd_matrix, model_key="plsr", fig_num=5)
    fig_transferability_heatmap(rf_rpd_matrix, model_key="rf", fig_num=6)
    fig7_shap_discrepancy_heatmap(shap_data, wavelengths, lu_codes)
    fig8_top_discrepant_bands(shap_data, wavelengths, lu_codes)

    print("[analysis] All figures generated.")