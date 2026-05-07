"""
representativeness_analysis.py
================================
Implements Sanderman's two methodological suggestions:

1. REPRESENTATIVENESS ANALYSIS
   Projects each target class test set into the source model's PLS score
   space and computes a quantitative representativeness metric (Mahalanobis
   distance). This determines whether cross-domain failures are driven by
   extrapolation (target samples outside source training distribution) vs.
   genuine compositional domain shift within the spectral space.

   If cropland consistently falls outside the training distribution of every
   other class, the primary finding becomes "cropland is a spectral outlier"
   rather than a SHAP attribution story.

2. LOG1P TRANSFORMATION
   Retrains PLSR models with natural log (SOC + 1) as the response variable
   and back-transforms predictions before computing RPD. Given the severe
   right skew in woodland SOC (mean 93 g/kg, tail to 500+), this should
   substantially improve PLSR in-domain performance for high-SOC classes.
   RF is unaffected (handles skew natively) but included for comparison.

Usage (run from SOLUM root):
    python representativeness_analysis.py \
        --chemistry data/LUCAS_Topsoil_2015_20200323.csv \
        --spectra   data/spectra/ \
        --verbose

Outputs:
    outputs/representativeness_mahal.csv   -- per (source, target) pair Mahal. distance stats
    outputs/transferability_plsr_log1p.csv -- RPD matrix with log1p PLSR
    figures/fig11_representativeness.png   -- PLS score plot + distance summary
    figures/fig12_log1p_comparison.png     -- before/after log1p PLSR RPD
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import chi2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solum"))

from data_loader import load_lucas, LU_NAMES, SOC_COL
from spectral_preprocessing import preprocess_spectra
from plsr import NIPALS_PLSR, select_n_components
from transferability_matrix import (
    compute_rpd, compute_rmse, compute_r2, classify_rpd, stratified_split
)

TEAL  = "#0d7a7a"
RED   = "#C0392B"
AMBER = "#E67E22"
BLUE  = "#1F77B4"
BLACK = "#1a1a1a"
OUT   = "outputs"
FIGS  = "figures"


# ─────────────────────────────────────────────────────────────────────────────
# 1. REPRESENTATIVENESS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def mahalanobis_representativeness(
    T_train: np.ndarray,   # (n_train, n_components) — source training scores
    T_target: np.ndarray,  # (n_test, n_components)  — target test scores
    alpha: float = 0.95,   # chi-squared threshold for "within distribution"
) -> dict:
    """
    Compute Mahalanobis distance from each target sample to the source
    training set score distribution.

    A sample is "representative" if its Mahalanobis distance falls within
    the chi-squared threshold at the given alpha level (alpha=0.95 means
    the 95th percentile of the training distribution).

    Parameters
    ----------
    T_train : (n_train, n_comp)
    T_target : (n_test, n_comp)
    alpha : float

    Returns
    -------
    dict with keys:
        mahal_distances : (n_test,) — per-sample Mahalanobis distance
        threshold : float — chi2 threshold
        pct_within : float — % of target samples within threshold
        mean_mahal : float
        median_mahal : float
    """
    n_comp = T_train.shape[1]
    mu = T_train.mean(axis=0)
    cov = np.cov(T_train.T)

    # Regularize covariance slightly for numerical stability
    cov = cov + np.eye(n_comp) * 1e-8

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # Mahalanobis distance for each target sample
    diffs = T_target - mu  # (n_test, n_comp)
    mahal = np.array([
        np.sqrt(d @ cov_inv @ d) for d in diffs
    ])

    # Chi-squared threshold: sqrt(chi2.ppf(alpha, df=n_comp))
    threshold = np.sqrt(chi2.ppf(alpha, df=n_comp))
    pct_within = 100.0 * np.mean(mahal <= threshold)

    return {
        "mahal_distances": mahal,
        "threshold": threshold,
        "pct_within": pct_within,
        "mean_mahal": float(np.mean(mahal)),
        "median_mahal": float(np.median(mahal)),
    }


def run_representativeness_analysis(
    lu_splits: dict,
    spectral_cols: list,
    plsr_max_components: int = 20,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For each (source, target) pair:
    1. Fit PLSR on source training set
    2. Project source training scores (T_train) — defines the spectral space
    3. Project target test samples into that same space (T_target)
    4. Compute Mahalanobis distance and % within training distribution

    Returns a DataFrame with one row per (source, target) pair.
    """
    lu_codes = sorted(lu_splits.keys())
    results = []

    # ── Preprocess and split all classes ───────────────────────────────────
    splits = {}
    for lu in lu_codes:
        X_tr, X_te, y_tr, y_te = stratified_split(
            lu_splits[lu], spectral_cols, random_state=random_state
        )
        splits[lu] = {
            "X_train": preprocess_spectra(X_tr),
            "X_test":  preprocess_spectra(X_te),
            "y_train": y_tr,
            "y_test":  y_te,
        }

    # ── Fit source PLSR and compute scores ─────────────────────────────────
    source_models = {}
    for src in lu_codes:
        X_tr = splits[src]["X_train"]
        y_tr = splits[src]["y_train"]

        if verbose:
            print(f"[repr] Fitting PLSR on source: {src} ({LU_NAMES.get(src,src)})")

        n_comp = select_n_components(X_tr, y_tr, max_components=plsr_max_components, verbose=False)
        model = NIPALS_PLSR(n_components=n_comp)
        model.fit(X_tr, y_tr)
        source_models[src] = model

        if verbose:
            print(f"[repr]   n_components={n_comp}")

    # ── Compute representativeness for all pairs ────────────────────────────
    for src in lu_codes:
        model = source_models[src]
        X_tr  = splits[src]["X_train"]
        T_train = model.get_scores(X_tr)   # (n_train, n_comp)

        for tgt in lu_codes:
            X_te  = splits[tgt]["X_test"]
            y_te  = splits[tgt]["y_test"]
            T_target = model.get_scores(X_te)

            # Representativeness metric
            repr_result = mahalanobis_representativeness(T_train, T_target)

            # Also compute RPD (for reference)
            y_pred = model.predict(X_te)
            rpd = compute_rpd(y_te, y_pred)

            results.append({
                "source": src,
                "source_name": LU_NAMES.get(src, src),
                "target": tgt,
                "target_name": LU_NAMES.get(tgt, tgt),
                "is_in_domain": src == tgt,
                "plsr_rpd": rpd,
                "pct_within_dist": repr_result["pct_within"],
                "mean_mahal": repr_result["mean_mahal"],
                "median_mahal": repr_result["median_mahal"],
                "mahal_threshold": repr_result["threshold"],
            })

            if verbose:
                flag = "[IN-DOMAIN]" if src == tgt else ""
                print(
                    f"  {src}>{tgt} {flag}: "
                    f"RPD={rpd:.2f}, "
                    f"% within dist={repr_result['pct_within']:.1f}%, "
                    f"mean Mahal={repr_result['mean_mahal']:.2f}"
                )

    return pd.DataFrame(results)


def fig_representativeness(repr_df: pd.DataFrame, out_path: str):
    """
    Two-panel figure:
    Left:  % of target samples within source distribution (heatmap)
    Right: mean Mahalanobis distance (heatmap)
    Both with same row/col structure as transferability matrix.
    """
    lu_codes = sorted(repr_df["source"].unique())
    n = len(lu_codes)

    # Pivot
    pct_mat  = repr_df.pivot(index="source", columns="target", values="pct_within_dist")
    mahal_mat = repr_df.pivot(index="source", columns="target", values="mean_mahal")
    rpd_mat   = repr_df.pivot(index="source", columns="target", values="plsr_rpd")
    pct_mat   = pct_mat.loc[lu_codes, lu_codes]
    mahal_mat = mahal_mat.loc[lu_codes, lu_codes]
    rpd_mat   = rpd_mat.loc[lu_codes, lu_codes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, mat, title, cmap, vmin, vmax, fmt in [
        (axes[0], pct_mat,  "% Target Samples Within Source Spectral Distribution",
         "RdYlGn", 0, 100, "{:.0f}%"),
        (axes[1], mahal_mat, "Mean Mahalanobis Distance (Target to Source PLS Space)",
         "YlOrRd", 0, None, "{:.2f}"),
    ]:
        vals = mat.values.astype(float)
        if vmax is None:
            vmax = np.nanpercentile(vals, 95) * 1.1

        im = ax.imshow(vals, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")

        for i in range(n):
            for j in range(n):
                v = vals[i, j]
                rpd_v = rpd_mat.values[i, j]
                # Color text based on background brightness
                bg_norm = (v - vmin) / (vmax - vmin + 1e-9)
                tc = "white" if bg_norm > 0.65 else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=8.5, color=tc, fontweight="bold")

        # Highlight diagonal
        for k in range(n):
            rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                  fill=False, edgecolor=BLACK, linewidth=2.2)
            ax.add_patch(rect)

        xlabels = [f"{lu}\n({LU_NAMES.get(lu,lu)})" for lu in lu_codes]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha="right")
        ax.set_yticklabels(xlabels, fontsize=8)
        ax.set_xlabel("Target class (test)", fontsize=9, labelpad=4)
        ax.set_ylabel("Source class (training)", fontsize=9, labelpad=4)
        ax.set_title(title, fontsize=9, pad=8)

        cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Figure 11: Representativeness Analysis — Does the target class fall inside the source model's spectral space?\n"
        "Low % within distribution or high Mahalanobis distance indicates extrapolation, not interpolation.",
        fontsize=9, y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOG1P PLSR RETRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_log1p_transferability(
    lu_splits: dict,
    spectral_cols: list,
    plsr_max_components: int = 20,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Retrain PLSR with log1p(SOC) as the response variable.
    Back-transform predictions: y_pred_soc = exp(y_pred_log) - 1
    Compute RPD on original SOC scale.

    Returns transferability matrix DataFrame.
    """
    lu_codes = sorted(lu_splits.keys())

    # ── Split and preprocess ───────────────────────────────────────────────
    splits = {}
    for lu in lu_codes:
        X_tr, X_te, y_tr, y_te = stratified_split(
            lu_splits[lu], spectral_cols, random_state=random_state
        )
        splits[lu] = {
            "X_train": preprocess_spectra(X_tr),
            "X_test":  preprocess_spectra(X_te),
            "y_train": y_tr,
            "y_test":  y_te,
            "y_train_log": np.log1p(y_tr),
        }

    # ── Train log1p PLSR per source ────────────────────────────────────────
    models = {}
    for src in lu_codes:
        X_tr   = splits[src]["X_train"]
        y_log  = splits[src]["y_train_log"]

        if verbose:
            print(f"[log1p] Fitting on source: {src}")

        n_comp = select_n_components(X_tr, y_log, max_components=plsr_max_components, verbose=False)
        model  = NIPALS_PLSR(n_components=n_comp)
        model.fit(X_tr, y_log)
        models[src] = model

        if verbose:
            print(f"[log1p]   n_components={n_comp}")

    # ── Evaluate all pairs ────────────────────────────────────────────────
    results = []
    rpd_matrix = pd.DataFrame(index=lu_codes, columns=lu_codes, dtype=float)

    for src in lu_codes:
        for tgt in lu_codes:
            X_te  = splits[tgt]["X_test"]
            y_te  = splits[tgt]["y_test"]   # original SOC scale

            # Predict in log space, back-transform
            y_pred_log = models[src].predict(X_te)
            y_pred_soc = np.expm1(y_pred_log)    # exp(x) - 1
            # Clip negatives (shouldn't happen often with log1p but just in case)
            y_pred_soc = np.maximum(y_pred_soc, 0)

            rpd = compute_rpd(y_te, y_pred_soc)
            rpd_matrix.loc[src, tgt] = rpd

            results.append({
                "source": src,
                "target": tgt,
                "is_in_domain": src == tgt,
                "plsr_log1p_rpd": rpd,
                "plsr_log1p_rmse": compute_rmse(y_te, y_pred_soc),
                "plsr_log1p_r2": compute_r2(y_te, y_pred_soc),
                "tier": classify_rpd(rpd),
            })

            if verbose:
                flag = "[IN-DOMAIN]" if src == tgt else ""
                print(f"  {src}>{tgt} {flag}: RPD={rpd:.2f} ({classify_rpd(rpd)})")

    return pd.DataFrame(results), rpd_matrix


def fig_log1p_comparison(
    raw_rpd_csv: str,
    log1p_df: pd.DataFrame,
    lu_codes: list,
    out_path: str,
):
    """
    Side-by-side bar chart comparing in-domain PLSR RPD:
    raw SOC training vs. log1p SOC training.
    """
    # Load original PLSR matrix
    original = pd.read_csv(raw_rpd_csv, index_col=0)
    original_diag = {lu: float(original.loc[lu, lu]) for lu in lu_codes}

    log1p_diag = {
        row["source"]: row["plsr_log1p_rpd"]
        for _, row in log1p_df.iterrows()
        if row["source"] == row["target"]
    }

    x = np.arange(len(lu_codes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - w/2,
                   [original_diag.get(lu, 0) for lu in lu_codes],
                   w, label="PLSR (raw SOC)", color=TEAL, alpha=0.85,
                   edgecolor=BLACK, linewidth=0.6)
    bars2 = ax.bar(x + w/2,
                   [log1p_diag.get(lu, 0) for lu in lu_codes],
                   w, label="PLSR (log1p SOC)", color=AMBER, alpha=0.85,
                   edgecolor=BLACK, linewidth=0.6)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=BLACK)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=BLACK)

    ax.axhline(y=2.0, color="#333", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.axhline(y=1.4, color="#333", linewidth=0.8, linestyle=":",  alpha=0.5)

    ax.set_xticks(x)
    xticklabels = [f"{lu}\n({LU_NAMES.get(lu, lu)})" for lu in lu_codes]
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylabel("In-Domain RPD", fontsize=10)
    ax.set_xlabel("Land Use Class", fontsize=10)
    ax.set_ylim(0, max(list(original_diag.values()) + list(log1p_diag.values())) * 1.2)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_title(
        "Figure 12: Effect of log1p SOC transformation on in-domain PLSR performance\n"
        "Log1p transformation compresses the right tail of the SOC distribution "
        "(woodland mean 93 g/kg, max >500 g/kg), which should improve PLSR fit.",
        fontsize=9, loc="left"
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SOLUM: Representativeness analysis + log1p PLSR"
    )
    parser.add_argument("--chemistry", required=True)
    parser.add_argument("--spectra",   required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-repr",  action="store_true",
                        help="Skip representativeness analysis (faster)")
    parser.add_argument("--skip-log1p", action="store_true",
                        help="Skip log1p retraining")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    print("=" * 60)
    print("SOLUM: Representativeness + log1p Analysis")
    print("=" * 60)

    print("\n[Step 1] Loading dataset...")
    from data_loader import load_lucas
    df_clean, spectral_cols, wavelengths, lu_splits = load_lucas(
        chemistry_csv=args.chemistry,
        spectra_dir=args.spectra,
        verbose=True,
    )
    lu_codes = sorted(lu_splits.keys())
    print(f"[Step 1] Classes: {lu_codes}")

    # ── Representativeness ────────────────────────────────────────────────
    if not args.skip_repr:
        print("\n[Step 2] Running representativeness analysis...")
        repr_df = run_representativeness_analysis(
            lu_splits, spectral_cols,
            random_state=args.random_state,
            verbose=args.verbose,
        )
        repr_df.to_csv(os.path.join(OUT, "representativeness_mahal.csv"), index=False)
        print(f"\n[Step 2] Saved: {OUT}/representativeness_mahal.csv")

        print("\n[Step 2] Summary — % target samples within source spectral distribution:")
        pct_pivot = repr_df.pivot(index="source", columns="target", values="pct_within_dist")
        print(pct_pivot.loc[lu_codes, lu_codes].to_string(float_format="{:.1f}".format))

        print("\n[Step 2] Key finding: cropland (B) as target:")
        b_rows = repr_df[repr_df["target"] == "B"].sort_values("source")
        for _, row in b_rows.iterrows():
            print(f"  {row['source']}->B: {row['pct_within_dist']:.1f}% within dist, "
                  f"mean Mahal={row['mean_mahal']:.2f}, PLSR RPD={row['plsr_rpd']:.2f}")

        fig_representativeness(
            repr_df,
            os.path.join(FIGS, "fig11_representativeness.png")
        )
    else:
        print("\n[Step 2] Skipped representativeness (--skip-repr).")

    # ── Log1p retraining ──────────────────────────────────────────────────
    if not args.skip_log1p:
        print("\n[Step 3] Retraining PLSR with log1p(SOC) transformation...")
        log1p_df, log1p_rpd_matrix = run_log1p_transferability(
            lu_splits, spectral_cols,
            random_state=args.random_state,
            verbose=args.verbose,
        )
        log1p_df.to_csv(os.path.join(OUT, "transferability_plsr_log1p.csv"), index=False)
        log1p_rpd_matrix.to_csv(os.path.join(OUT, "transferability_matrix_plsr_log1p_rpd.csv"))
        print(f"\n[Step 3] log1p PLSR Transferability Matrix:")
        print(log1p_rpd_matrix.to_string(float_format="{:.2f}".format))

        raw_csv = os.path.join(OUT, "transferability_matrix_plsr_rpd.csv")
        if os.path.exists(raw_csv):
            fig_log1p_comparison(
                raw_csv, log1p_df, lu_codes,
                os.path.join(FIGS, "fig12_log1p_comparison.png")
            )
        else:
            print("[Step 3] Note: raw PLSR RPD CSV not found; skipping comparison figure.")
    else:
        print("\n[Step 3] Skipped log1p retraining (--skip-log1p).")

    print("\n" + "=" * 60)
    print("Done. Check outputs/ and figures/.")
    print("=" * 60)


if __name__ == "__main__":
    main()
