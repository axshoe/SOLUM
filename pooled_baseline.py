"""
pooled_baseline.py
==================
Two additions to the SOLUM pipeline that strengthen publishability:

1. POOLED BASELINE MODEL
   Trains one RF and one PLSR on all land use classes combined (80% of
   total data), then evaluates on each class's individual test set.
   Compares per-class RPD to the diagonal of the transferability matrix.
   This demonstrates empirically what the field's standard practice of
   pooling classes actually does to per-class performance.

2. REPEATED HOLDOUTS
   Runs the full transferability matrix 5 times with different random seeds.
   Reports mean ± SD RPD per cell. Addresses single-split reproducibility
   concerns before a reviewer raises them.

Usage (run from SOLUM root folder):
    python pooled_baseline.py \
        --chemistry data/LUCAS_Topsoil_2015_20200323.csv \
        --spectra   data/spectra/ \
        --n-repeats 5 \
        --verbose

Outputs written to outputs/:
    pooled_baseline_results.csv        per-class RPD for pooled vs. per-class models
    transferability_matrix_rf_mean.csv  mean RPD across repeats
    transferability_matrix_rf_sd.csv    SD of RPD across repeats
    transferability_matrix_plsr_mean.csv
    transferability_matrix_plsr_sd.csv

Figures written to figures/:
    fig9_pooled_baseline.png
    fig10_repeated_holdouts_rf.png
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ── add solum/ to path so we can import existing modules ─────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solum"))

from data_loader import load_lucas, get_X_y, LU_NAMES, SOC_COL
from spectral_preprocessing import preprocess_spectra
from plsr import NIPALS_PLSR, select_n_components
from rf_model import RFModel
from transferability_matrix import (
    compute_r2, compute_rmse, compute_rpd, classify_rpd, stratified_split
)

# ── style constants (match analysis.py) ──────────────────────────────────────
ACCENT_TEAL  = "#0d7a7a"
ACCENT_RED   = "#C0392B"
ACCENT_AMBER = "#E67E22"
ACCENT_BLUE  = "#1F77B4"

RPD_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "rpd_cmap",
    [(0.0, "#C0392B"), (0.4, "#E67E22"), (1.0, "#0d7a7a")]
)

FIG_DIR = "figures"
OUT_DIR = "outputs"


# ─────────────────────────────────────────────────────────────────────────────
# 1. POOLED BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pooled_baseline(
    lu_splits: dict,
    spectral_cols: list,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train one RF and one PLSR on all land use classes combined,
    evaluate on each class's individual test set.

    Returns a DataFrame comparing pooled RPD to per-class (diagonal) RPD.
    """
    lu_codes = sorted(lu_splits.keys())

    # ── build combined training set ───────────────────────────────────────────
    # For each class, split 80/20. Pool the training halves together.
    # Keep each class's test set separate for fair comparison.
    train_X_list, train_y_list = [], []
    class_test = {}   # {lu: (X_test, y_test)}

    for lu in lu_codes:
        X_train, X_test, y_train, y_test = stratified_split(
            lu_splits[lu], spectral_cols, random_state=random_state
        )
        X_train_proc = preprocess_spectra(X_train)
        X_test_proc  = preprocess_spectra(X_test)
        train_X_list.append(X_train_proc)
        train_y_list.append(y_train)
        class_test[lu] = (X_test_proc, y_test)

    X_pool = np.vstack(train_X_list)
    y_pool = np.concatenate(train_y_list)

    if verbose:
        print(f"[pooled] Combined training set: {X_pool.shape[0]} samples")

    # ── train pooled RF ───────────────────────────────────────────────────────
    if verbose:
        print("[pooled] Training pooled RF...")
    rf_pool = RandomForestRegressor(
        n_estimators=200, max_features=0.1,
        random_state=random_state, n_jobs=-1
    )
    rf_pool.fit(X_pool, y_pool)

    # ── train pooled PLSR ─────────────────────────────────────────────────────
    if verbose:
        print("[pooled] Selecting PLSR components for pooled model...")
    n_comp = select_n_components(X_pool, y_pool, max_components=20, verbose=False)
    plsr_pool = NIPALS_PLSR(n_components=n_comp)
    plsr_pool.fit(X_pool, y_pool)
    if verbose:
        print(f"[pooled] Pooled PLSR fitted with n_components={n_comp}")

    # ── also train per-class models on same splits ────────────────────────────
    per_class_rf_rpd   = {}
    per_class_plsr_rpd = {}
    pooled_rf_rpd      = {}
    pooled_plsr_rpd    = {}

    for lu in lu_codes:
        X_test, y_test = class_test[lu]

        # per-class RF (trained on same split as pooled, for fair comparison)
        X_tr_proc = train_X_list[lu_codes.index(lu)]
        y_tr      = train_y_list[lu_codes.index(lu)]
        rf_cls    = RandomForestRegressor(
            n_estimators=200, max_features=0.1,
            random_state=random_state, n_jobs=-1
        )
        rf_cls.fit(X_tr_proc, y_tr)
        per_class_rf_rpd[lu] = compute_rpd(y_test, rf_cls.predict(X_test))

        # per-class PLSR
        nc = select_n_components(X_tr_proc, y_tr, max_components=20, verbose=False)
        plsr_cls = NIPALS_PLSR(n_components=nc)
        plsr_cls.fit(X_tr_proc, y_tr)
        per_class_plsr_rpd[lu] = compute_rpd(y_test, plsr_cls.predict(X_test))

        # pooled evaluations
        pooled_rf_rpd[lu]   = compute_rpd(y_test, rf_pool.predict(X_test))
        pooled_plsr_rpd[lu] = compute_rpd(y_test, plsr_pool.predict(X_test))

        if verbose:
            print(
                f"  {lu} ({LU_NAMES.get(lu, lu)}): "
                f"Per-class RF={per_class_rf_rpd[lu]:.2f}, "
                f"Pooled RF={pooled_rf_rpd[lu]:.2f} | "
                f"Per-class PLSR={per_class_plsr_rpd[lu]:.2f}, "
                f"Pooled PLSR={pooled_plsr_rpd[lu]:.2f}"
            )

    rows = []
    for lu in lu_codes:
        rows.append({
            "class": lu,
            "class_name": LU_NAMES.get(lu, lu),
            "n_test": len(class_test[lu][1]),
            "per_class_rf_rpd":   per_class_rf_rpd[lu],
            "pooled_rf_rpd":      pooled_rf_rpd[lu],
            "per_class_plsr_rpd": per_class_plsr_rpd[lu],
            "pooled_plsr_rpd":    pooled_plsr_rpd[lu],
            "rf_degradation":     per_class_rf_rpd[lu] - pooled_rf_rpd[lu],
            "plsr_degradation":   per_class_plsr_rpd[lu] - pooled_plsr_rpd[lu],
        })

    return pd.DataFrame(rows)


def fig_pooled_baseline(df: pd.DataFrame, out_path: str):
    """
    Grouped bar chart: per-class RF RPD vs pooled RF RPD per land use class.
    Includes RPD=1.0 reference line (mean predictor baseline).
    """
    lu_codes = df["class"].tolist()
    x = np.arange(len(lu_codes))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width/2, df["per_class_rf_rpd"], width,
                   color=ACCENT_TEAL, alpha=0.88, label="Per-class RF (in-domain)",
                   edgecolor="#333", linewidth=0.6)
    bars2 = ax.bar(x + width/2, df["pooled_rf_rpd"], width,
                   color=ACCENT_AMBER, alpha=0.88, label="Pooled RF (all classes combined)",
                   edgecolor="#333", linewidth=0.6)

    # RPD reference lines
    ax.axhline(y=2.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axhline(y=1.4, color="#333333", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axhline(y=1.0, color=ACCENT_RED, linewidth=1.2, linestyle="-", alpha=0.8)

    ax.text(len(lu_codes) - 0.1, 2.05, "RPD=2.0 (Good)", fontsize=8, color="#555555", ha="right")
    ax.text(len(lu_codes) - 0.1, 1.45, "RPD=1.4 (Moderate)", fontsize=8, color="#555555", ha="right")
    ax.text(len(lu_codes) - 0.1, 1.05, "RPD=1.0 (Mean predictor)", fontsize=8.5,
            color=ACCENT_RED, ha="right", fontweight="bold")

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8, color="#333333")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_xticks(x)
    xticklabels = [f"{row['class']}\n({row['class_name']})" for _, row in df.iterrows()]
    ax.set_xticklabels(xticklabels, fontsize=8.5)
    ax.set_ylabel("RPD", fontsize=10)
    ax.set_xlabel("Land Use Class", fontsize=10)
    ax.set_ylim(0, max(df["per_class_rf_rpd"].max(), df["pooled_rf_rpd"].max()) * 1.18)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title(
        "Figure 9: Per-class vs. Pooled RF Model Performance by Land Use Class\n"
        "Red line = mean predictor baseline (RPD=1.0). Values below this line are actively harmful.",
        fontsize=9, loc="left"
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. REPEATED HOLDOUTS
# ─────────────────────────────────────────────────────────────────────────────

def run_repeated_holdouts(
    lu_splits: dict,
    spectral_cols: list,
    n_repeats: int = 5,
    verbose: bool = True,
) -> tuple:
    """
    Run the full RF transferability matrix n_repeats times with different
    random seeds. Returns mean and SD matrices.

    Uses RF only (not PLSR) for speed — PLSR with LOO-CV on 21k samples
    per repeat would take many hours. RF with fixed hyperparameters
    (from the main run) takes ~20 min per repeat.
    """
    lu_codes = sorted(lu_splits.keys())
    n = len(lu_codes)
    seeds = [42, 123, 7, 999, 2025][:n_repeats]

    all_rf_matrices = []

    for rep_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\n[repeats] === Repeat {rep_idx+1}/{n_repeats} (seed={seed}) ===")

        # train per-class RF models
        models = {}
        splits = {}
        for lu in lu_codes:
            X_train, X_test, y_train, y_test = stratified_split(
                lu_splits[lu], spectral_cols, random_state=seed
            )
            X_train_proc = preprocess_spectra(X_train)
            X_test_proc  = preprocess_spectra(X_test)
            splits[lu] = (X_train_proc, X_test_proc, y_train, y_test)

            rf = RandomForestRegressor(
                n_estimators=200, max_features=0.1,
                random_state=seed, n_jobs=-1
            )
            rf.fit(X_train_proc, y_train)
            models[lu] = rf
            if verbose:
                print(f"  [repeats] Trained RF on {lu}")

        # evaluate all pairs
        rf_matrix = pd.DataFrame(index=lu_codes, columns=lu_codes, dtype=float)
        for src in lu_codes:
            for tgt in lu_codes:
                _, X_test, _, y_test = splits[tgt]
                y_pred = models[src].predict(X_test)
                rpd = compute_rpd(y_test, y_pred)
                rf_matrix.loc[src, tgt] = rpd
                if verbose:
                    domain = " [IN-DOMAIN]" if src == tgt else ""
                    print(f"    {src}→{tgt}{domain}: RPD={rpd:.2f}")

        all_rf_matrices.append(rf_matrix.values.astype(float))

    stack = np.stack(all_rf_matrices, axis=0)  # (n_repeats, n, n)
    mean_vals = np.mean(stack, axis=0)
    sd_vals   = np.std(stack, axis=0, ddof=1)

    mean_df = pd.DataFrame(mean_vals, index=lu_codes, columns=lu_codes)
    sd_df   = pd.DataFrame(sd_vals,   index=lu_codes, columns=lu_codes)

    return mean_df, sd_df, all_rf_matrices


def fig_repeated_holdouts(
    mean_df: pd.DataFrame,
    sd_df: pd.DataFrame,
    out_path: str,
):
    """
    Heatmap showing mean RF RPD with SD annotated in each cell.
    Same colormap as main transferability figures.
    """
    lu_codes = list(mean_df.index)
    n = len(lu_codes)
    mean_vals = mean_df.values.astype(float)
    sd_vals   = sd_df.values.astype(float)
    vmin, vmax = 0.5, 3.5

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_vals, vmin=vmin, vmax=vmax, cmap=RPD_COLORMAP, aspect="auto")

    for i in range(n):
        for j in range(n):
            m = mean_vals[i, j]
            s = sd_vals[i, j]
            brightness = (m - vmin) / (vmax - vmin)
            text_color = "white" if brightness < 0.35 or brightness > 0.78 else "black"
            ax.text(j, i, f"{m:.2f}\n±{s:.2f}",
                    ha="center", va="center", fontsize=8.5,
                    color=text_color, fontweight="bold", linespacing=1.4)

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
    cbar.set_label("Mean RPD", fontsize=9)
    cbar.ax.axhline(y=(1.4 - vmin)/(vmax - vmin), color="black", linewidth=1.0, linestyle="--")
    cbar.ax.axhline(y=(2.0 - vmin)/(vmax - vmin), color="black", linewidth=1.0, linestyle="--")
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Figure 10: RF Transferability Matrix — Mean RPD ± SD across 5 repeated holdouts\n"
        "Bold borders = in-domain. Dashed colorbar lines at RPD=1.4 and 2.0.",
        fontsize=8.5, loc="left", pad=10
    )

    legend_elements = [
        Patch(facecolor="#C0392B", label="Poor (RPD < 1.4)"),
        Patch(facecolor="#E67E22", label="Moderate (1.4 ≤ RPD < 2.0)"),
        Patch(facecolor="#0d7a7a", label="Good (RPD ≥ 2.0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=8.5,
              framealpha=0.9, frameon=True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SOLUM: Pooled baseline + repeated holdouts")
    parser.add_argument("--chemistry", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--n-repeats", type=int, default=5,
                        help="Number of repeated holdouts (default: 5)")
    parser.add_argument("--skip-repeats", action="store_true",
                        help="Only run pooled baseline, skip repeated holdouts (~30 min saved per repeat)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("SOLUM: Pooled Baseline + Repeated Holdouts")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[Step 1] Loading dataset...")
    from data_loader import load_lucas
    df_clean, spectral_cols, wavelengths, lu_splits = load_lucas(
        chemistry_csv=args.chemistry,
        spectra_dir=args.spectra,
        verbose=True,
    )
    lu_codes = sorted(lu_splits.keys())
    print(f"[Step 1] Classes: {lu_codes}, Bands: {len(spectral_cols)}")

    # ── Pooled baseline ───────────────────────────────────────────────────────
    print("\n[Step 2] Running pooled baseline analysis...")
    pooled_df = run_pooled_baseline(
        lu_splits, spectral_cols,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    print("\n[Step 2] Results:")
    print(pooled_df[["class", "class_name", "per_class_rf_rpd",
                      "pooled_rf_rpd", "rf_degradation"]].to_string(index=False,
                      float_format="{:.2f}".format))

    pooled_df.to_csv(os.path.join(OUT_DIR, "pooled_baseline_results.csv"), index=False)
    print(f"[Step 2] Saved: {OUT_DIR}/pooled_baseline_results.csv")

    fig_pooled_baseline(pooled_df, os.path.join(FIG_DIR, "fig9_pooled_baseline.png"))

    # ── Repeated holdouts ─────────────────────────────────────────────────────
    if not args.skip_repeats:
        print(f"\n[Step 3] Running {args.n_repeats} repeated holdouts (RF only)...")
        print("         Estimated time: ~20-40 min per repeat on CPU.")
        print("         Use --skip-repeats to skip this step if time is short.\n")

        mean_df, sd_df, all_matrices = run_repeated_holdouts(
            lu_splits, spectral_cols,
            n_repeats=args.n_repeats,
            verbose=args.verbose,
        )

        print("\n[Step 3] Mean RF RPD across repeats:")
        print(mean_df.to_string(float_format="{:.2f}".format))
        print("\n[Step 3] SD of RF RPD across repeats:")
        print(sd_df.to_string(float_format="{:.2f}".format))

        mean_df.to_csv(os.path.join(OUT_DIR, "transferability_matrix_rf_mean.csv"))
        sd_df.to_csv(os.path.join(OUT_DIR, "transferability_matrix_rf_sd.csv"))
        print(f"[Step 3] Saved mean/SD matrices to {OUT_DIR}/")

        fig_repeated_holdouts(mean_df, sd_df,
                              os.path.join(FIG_DIR, "fig10_repeated_holdouts_rf.png"))
    else:
        print("\n[Step 3] Skipped repeated holdouts (--skip-repeats).")

    print("\n" + "=" * 60)
    print("Done. Check outputs/ and figures/ for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
