"""
covariate_analysis.py
=====================
Implements Gomez's suggestion: test whether ancillary soil covariates
(clay, pH, sand, silt, CaCO3-proxy) explain the visible-range SHAP
discrepancy in the failed cross-land-use transfers.

LUCAS 2015 does not include iron content or Munsell color, but it does
include clay, sand, silt, pH(CaCl2), pH(H2O), EC, and coarse fragments.
Clay content is a proxy for MAOM stabilization capacity; pH co-varies
with carbonate content and iron oxide speciation.

Approach:
1. For each failed transfer pair (RF RPD < 1.4), take the target class
   test samples
2. Compute per-sample visible-range (560-640 nm) spectral difference
   from the source class mean spectrum (a proxy for where SHAP discrepancy
   concentrates)
3. Run PCA on the ancillary covariates (clay, sand, silt, pH, EC)
4. Correlate covariate PCA scores with the visible-range spectral
   difference to see which soil property best explains the discrepancy
5. Also directly correlate each covariate with visible-range difference

Usage:
    python covariate_analysis.py \
        --chemistry data/LUCAS_Topsoil_2015_20200323.csv \
        --spectra   data/spectra/ \
        --verbose

Outputs:
    outputs/covariate_correlations.csv    -- per-covariate correlation with vis-range signal
    outputs/covariate_pca_loadings.csv    -- PCA loadings
    figures/fig15_covariate_analysis.png  -- correlation + PCA biplot
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solum"))

from data_loader import load_lucas, LU_NAMES, SOC_COL
from spectral_preprocessing import preprocess_spectra
from transferability_matrix import stratified_split

OUT  = "outputs"
FIGS = "figures"

TEAL  = "#0d7a7a"
RED   = "#C0392B"
AMBER = "#E67E22"

# Failed transfer pairs (RF RPD < 1.4) from the main analysis
FAILED_PAIRS = [("C", "B"), ("D", "B"), ("E", "B"), ("F", "B"), ("F", "D")]

# Visible range where SHAP discrepancy concentrated
VIS_RANGE = (560, 640)

# Candidate ancillary covariates in LUCAS 2015 chemistry file
COVARIATE_CANDIDATES = ["Clay", "Sand", "Silt", "pH(CaCl2)", "pH(H2O)", "EC", "Coarse"]


def find_covariate_columns(chemistry_df, verbose=True):
    """Identify which ancillary covariate columns are present."""
    available = []
    for col in COVARIATE_CANDIDATES:
        if col in chemistry_df.columns:
            available.append(col)
    if verbose:
        print(f"[covar] Available covariates: {available}")
        missing = [c for c in COVARIATE_CANDIDATES if c not in available]
        if missing:
            print(f"[covar] Not in dataset: {missing}")
    return available


def get_vis_range_indices(wavelengths, vis_range=VIS_RANGE):
    """Get spectral band indices within the visible range."""
    return np.where((wavelengths >= vis_range[0]) & (wavelengths <= vis_range[1]))[0]


def run_covariate_analysis(chemistry_df, lu_splits, spectral_cols, wavelengths,
                            covariates, random_state=42, verbose=True):
    """
    For each failed pair, compute per-sample visible-range spectral
    deviation from source mean, then correlate with covariates.
    """
    vis_idx = get_vis_range_indices(wavelengths)
    if verbose:
        print(f"[covar] Visible range {VIS_RANGE[0]}-{VIS_RANGE[1]} nm = {len(vis_idx)} bands")

    all_records = []

    # Precompute each source class mean spectrum once
    src_mean_spectra = {}
    for src in set(s for s, _ in FAILED_PAIRS):
        if src in lu_splits:
            X_src = preprocess_spectra(lu_splits[src][spectral_cols].values.astype(float))
            src_mean_spectra[src] = X_src.mean(axis=0)

    # IMPORTANT: deduplicate by target class. B is the target in 4 of 5 failed
    # pairs; without dedup, cropland samples get counted 4x and inflate n.
    # For each unique target class, we compute vis deviation against the mean
    # of ALL source classes that fail into it, averaged. This gives one row
    # per physical soil sample.
    targets_seen = {}
    for src, tgt in FAILED_PAIRS:
        if src not in lu_splits or tgt not in lu_splits:
            continue
        targets_seen.setdefault(tgt, []).append(src)

    for tgt, srcs in targets_seen.items():
        tgt_df = lu_splits[tgt]
        X_tgt = preprocess_spectra(tgt_df[spectral_cols].values.astype(float))

        # Average visible-range deviation across all failing source classes
        devs = []
        for src in srcs:
            dev = np.abs(X_tgt[:, vis_idx] - src_mean_spectra[src][vis_idx]).mean(axis=1)
            devs.append(dev)
        vis_deviation = np.mean(devs, axis=0)

        for i in range(len(tgt_df)):
            rec = {
                "target": tgt,
                "failing_sources": ",".join(srcs),
                "vis_deviation": vis_deviation[i],
                "soc": tgt_df.iloc[i][SOC_COL],
            }
            for cov in covariates:
                rec[cov] = tgt_df.iloc[i][cov]
            all_records.append(rec)

    records_df = pd.DataFrame(all_records)
    if verbose:
        print(f"[covar] {len(records_df)} unique target samples (deduplicated by class)")

    # ── Correlate each covariate with visible-range deviation ────────────
    corr_results = []
    for cov in covariates:
        sub = records_df[["vis_deviation", cov]].dropna()
        if len(sub) < 10:
            continue
        # Convert covariate to numeric if needed
        try:
            cov_vals = pd.to_numeric(sub[cov], errors="coerce")
            valid = ~cov_vals.isna()
            if valid.sum() < 10:
                continue
            r_pearson, p_pearson = pearsonr(sub["vis_deviation"][valid], cov_vals[valid])
            r_spearman, p_spearman = spearmanr(sub["vis_deviation"][valid], cov_vals[valid])
            corr_results.append({
                "covariate": cov,
                "pearson_r": r_pearson,
                "pearson_p": p_pearson,
                "spearman_r": r_spearman,
                "spearman_p": p_spearman,
                "n": int(valid.sum()),
            })
            if verbose:
                print(f"[covar] {cov}: Pearson r={r_pearson:+.3f} (p={p_pearson:.3g}), "
                      f"Spearman r={r_spearman:+.3f}")
        except Exception as e:
            if verbose:
                print(f"[covar] Skipping {cov}: {e}")

    corr_df = pd.DataFrame(corr_results)

    return records_df, corr_df


def run_pca_covariates(records_df, covariates, verbose=True):
    """
    PCA on the covariates, then correlate PC scores with visible-range deviation.
    Shows which combination of soil properties aligns with the discrepancy.
    """
    # Build numeric covariate matrix
    cov_data = records_df[covariates].apply(pd.to_numeric, errors="coerce")
    valid_mask = ~cov_data.isna().any(axis=1)
    cov_valid = cov_data[valid_mask].values
    vis_valid = records_df["vis_deviation"][valid_mask].values

    if len(cov_valid) < 10:
        print("[pca] Not enough complete covariate rows for PCA.")
        return None, None

    # Standardize
    cov_mean = cov_valid.mean(axis=0)
    cov_std  = cov_valid.std(axis=0)
    cov_std[cov_std == 0] = 1.0
    cov_z = (cov_valid - cov_mean) / cov_std

    # PCA via SVD
    U, S, Vt = np.linalg.svd(cov_z, full_matrices=False)
    scores = U * S
    loadings = Vt.T
    explained_var = (S ** 2) / (S ** 2).sum()

    if verbose:
        print(f"\n[pca] Explained variance: "
              f"{', '.join(f'PC{i+1}={v:.1%}' for i, v in enumerate(explained_var[:3]))}")

    # Correlate each PC with visible-range deviation
    pc_corr = []
    for pc_idx in range(min(3, scores.shape[1])):
        r, p = pearsonr(scores[:, pc_idx], vis_valid)
        pc_corr.append({"PC": f"PC{pc_idx+1}", "r_with_vis": r, "p": p,
                        "explained_var": explained_var[pc_idx]})
        if verbose:
            print(f"[pca] PC{pc_idx+1} vs vis-deviation: r={r:+.3f} (p={p:.3g})")

    loadings_df = pd.DataFrame(
        loadings[:, :3],
        index=covariates,
        columns=[f"PC{i+1}" for i in range(3)]
    )
    if verbose:
        print("\n[pca] Loadings:")
        print(loadings_df.to_string(float_format="{:+.3f}".format))

    return loadings_df, pd.DataFrame(pc_corr)


def fig_covariate_analysis(corr_df, records_df, covariates, out_path):
    """Bar chart of covariate correlations + scatter of top covariate."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: correlation bars ────────────────────────────────────────
    ax = axes[0]
    corr_sorted = corr_df.reindex(corr_df["pearson_r"].abs().sort_values(ascending=True).index)
    colors = [RED if r < 0 else TEAL for r in corr_sorted["pearson_r"]]
    ax.barh(corr_sorted["covariate"], corr_sorted["pearson_r"], color=colors, alpha=0.8,
            edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with visible-range deviation", fontsize=10)
    ax.set_title("Which soil covariate explains\nthe visible-range discrepancy?", fontsize=10)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    # Annotate significance
    for i, (_, row) in enumerate(corr_sorted.iterrows()):
        sig = "*" if row["pearson_p"] < 0.05 else ""
        ax.text(row["pearson_r"], i, f" {row['pearson_r']:+.2f}{sig}",
                va="center", ha="left" if row["pearson_r"] >= 0 else "right",
                fontsize=8.5)

    # ── Panel 2: scatter of top covariate ────────────────────────────────
    ax2 = axes[1]
    top_cov = corr_df.reindex(corr_df["pearson_r"].abs().sort_values(ascending=False).index).iloc[0]["covariate"]
    sub = records_df[["vis_deviation", top_cov, "target"]].copy()
    sub[top_cov] = pd.to_numeric(sub[top_cov], errors="coerce")
    sub = sub.dropna()

    targets = sub["target"].unique()
    target_colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))
    for target, color in zip(targets, target_colors):
        tdata = sub[sub["target"] == target]
        ax2.scatter(tdata[top_cov], tdata["vis_deviation"],
                    alpha=0.5, s=20, color=color,
                    label=f"target {target}", edgecolors="none")
    ax2.set_xlabel(f"{top_cov}", fontsize=10)
    ax2.set_ylabel("Visible-range spectral deviation", fontsize=10)
    ax2.set_title(f"Top covariate: {top_cov}", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9, title="Transfer pair")
    ax2.grid(alpha=0.3, linestyle="--")

    fig.suptitle(
        "Figure 15: Do ancillary soil covariates explain the visible-range SHAP discrepancy?\n"
        "Gomez suggestion: clay content and pH as proxies for iron oxide / MAOM stabilization.",
        fontsize=9, y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chemistry", required=True)
    p.add_argument("--spectra", required=True)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    print("=" * 60)
    print("SOLUM: Covariate Analysis (Gomez suggestion)")
    print("=" * 60)

    print("\n[Step 1] Loading dataset...")
    from data_loader import load_lucas
    df_clean, spectral_cols, wavelengths, lu_splits = load_lucas(
        chemistry_csv=args.chemistry,
        spectra_dir=args.spectra,
        verbose=True,
    )

    # Reload raw chemistry to find covariate columns
    chemistry_df = pd.read_csv(args.chemistry, low_memory=False)
    covariates = find_covariate_columns(chemistry_df, verbose=True)

    if not covariates:
        print("ERROR: No ancillary covariates found in chemistry file.")
        return

    # Check which covariates made it into lu_splits
    sample_split = list(lu_splits.values())[0]
    covariates_present = [c for c in covariates if c in sample_split.columns]
    if args.verbose:
        print(f"[covar] Covariates present in split data: {covariates_present}")

    if not covariates_present:
        print("\nNOTE: Covariates are in the chemistry file but not carried into lu_splits.")
        print("The data_loader may drop them. Checking columns in split data:")
        print(list(sample_split.columns[:30]))
        print("\nWill attempt to merge covariates back from chemistry file...")
        # Attempt merge by index/point_id
        return

    print("\n[Step 2] Running covariate correlation analysis...")
    records_df, corr_df = run_covariate_analysis(
        chemistry_df, lu_splits, spectral_cols, wavelengths,
        covariates_present, random_state=args.random_state, verbose=args.verbose
    )
    records_df.to_csv(os.path.join(OUT, "covariate_records.csv"), index=False)
    corr_df.to_csv(os.path.join(OUT, "covariate_correlations.csv"), index=False)

    print("\n[Step 2] Correlation summary (sorted by |Pearson r|):")
    corr_sorted = corr_df.reindex(corr_df["pearson_r"].abs().sort_values(ascending=False).index)
    print(corr_sorted.to_string(index=False, float_format="{:.3f}".format))

    print("\n[Step 3] Running PCA on covariates...")
    loadings_df, pc_corr_df = run_pca_covariates(records_df, covariates_present, verbose=args.verbose)
    if loadings_df is not None:
        loadings_df.to_csv(os.path.join(OUT, "covariate_pca_loadings.csv"))

    print("\n[Step 4] Generating figure...")
    fig_covariate_analysis(corr_df, records_df, covariates_present,
                           os.path.join(FIGS, "fig15_covariate_analysis.png"))

    print("\n" + "=" * 60)
    print("Done. Check outputs/ and figures/.")
    print("=" * 60)


if __name__ == "__main__":
    main()