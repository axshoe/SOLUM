"""
log1p_evaluation.py
===================
Properly evaluates the log1p PLSR models by computing fit metrics
on BOTH the log scale (where the model was trained) and the raw SOC
scale (back-transformed). This separates the transformation benefit
from the back-transformation distortion that was hurting RPD.

Also adds a direct comparison: for the failed transfers (B-column),
does log1p PLSR improve or worsen performance relative to raw PLSR?

Usage:
    python log1p_evaluation.py \
        --chemistry data/LUCAS_Topsoil_2015_20200323.csv \
        --spectra   data/spectra/ \
        --verbose

Outputs:
    outputs/log1p_evaluation_full.csv   -- per-pair metrics on both scales
    outputs/log1p_vs_raw_comparison.csv -- side-by-side for all pairs
    figures/fig13_log1p_scales.png      -- comparison figure
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solum"))

from data_loader import load_lucas, LU_NAMES, SOC_COL
from spectral_preprocessing import preprocess_spectra
from plsr import NIPALS_PLSR, select_n_components
from transferability_matrix import (
    compute_r2, compute_rmse, compute_rpd, classify_rpd, stratified_split
)

TEAL  = "#0d7a7a"
RED   = "#C0392B"
AMBER = "#E67E22"
OUT   = "outputs"
FIGS  = "figures"


def rmse_log(y_true_raw, y_pred_log):
    """RMSE on log scale: compare log1p(y_true) vs y_pred_log directly."""
    y_true_log = np.log1p(y_true_raw)
    return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))

def r2_log(y_true_raw, y_pred_log):
    """R2 on log scale."""
    y_true_log = np.log1p(y_true_raw)
    ss_res = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot = np.sum((y_true_log - y_true_log.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def rpd_log(y_true_raw, y_pred_log):
    """RPD on log scale: SD(log1p(y_true)) / RMSE_log."""
    y_true_log = np.log1p(y_true_raw)
    rmse = rmse_log(y_true_raw, y_pred_log)
    sd   = float(np.std(y_true_log, ddof=1))
    return sd / rmse if rmse > 0 else float("inf")


def run_dual_evaluation(lu_splits, spectral_cols, random_state=42, verbose=True):
    """
    Train PLSR twice per source class:
    - Once on raw SOC
    - Once on log1p(SOC)

    Evaluate both on all target classes.
    For each pair, report metrics on both log scale and raw scale.
    """
    lu_codes = sorted(lu_splits.keys())
    results  = []

    # ── Split and preprocess all classes ─────────────────────────────────
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

    # ── Train both model variants per source ─────────────────────────────
    raw_models  = {}
    log_models  = {}

    for src in lu_codes:
        X_tr      = splits[src]["X_train"]
        y_tr      = splits[src]["y_train"]
        y_tr_log  = splits[src]["y_train_log"]

        if verbose:
            print(f"[log1p_eval] Training source: {src}")

        # Raw PLSR
        nc_raw = select_n_components(X_tr, y_tr, max_components=20, verbose=False)
        m_raw  = NIPALS_PLSR(n_components=nc_raw)
        m_raw.fit(X_tr, y_tr)
        raw_models[src] = m_raw

        # Log1p PLSR
        nc_log = select_n_components(X_tr, y_tr_log, max_components=20, verbose=False)
        m_log  = NIPALS_PLSR(n_components=nc_log)
        m_log.fit(X_tr, y_tr_log)
        log_models[src] = m_log

        if verbose:
            print(f"  raw ncomp={nc_raw}, log1p ncomp={nc_log}")

    # ── Evaluate all pairs ────────────────────────────────────────────────
    for src in lu_codes:
        for tgt in lu_codes:
            X_te = splits[tgt]["X_test"]
            y_te = splits[tgt]["y_test"]   # raw SOC

            # ── Raw PLSR predictions ──────────────────────────────────────
            y_pred_raw  = raw_models[src].predict(X_te)
            raw_rpd     = compute_rpd(y_te, y_pred_raw)
            raw_rmse    = compute_rmse(y_te, y_pred_raw)
            raw_r2      = compute_r2(y_te, y_pred_raw)

            # ── Log1p PLSR predictions ────────────────────────────────────
            y_pred_log  = log_models[src].predict(X_te)  # in log space

            # Metrics on LOG SCALE (no back-transform)
            log_rpd_logscale  = rpd_log(y_te, y_pred_log)
            log_rmse_logscale = rmse_log(y_te, y_pred_log)
            log_r2_logscale   = r2_log(y_te, y_pred_log)

            # Metrics on RAW SCALE (back-transform via expm1)
            y_pred_backtrans  = np.maximum(np.expm1(y_pred_log), 0)
            log_rpd_rawscale  = compute_rpd(y_te, y_pred_backtrans)
            log_rmse_rawscale = compute_rmse(y_te, y_pred_backtrans)
            log_r2_rawscale   = compute_r2(y_te, y_pred_backtrans)

            in_domain = src == tgt
            flag      = "[IN-DOMAIN]" if in_domain else ""

            if verbose:
                print(
                    f"  {src}>{tgt} {flag}: "
                    f"raw RPD={raw_rpd:.2f} | "
                    f"log RPD (log scale)={log_rpd_logscale:.2f} | "
                    f"log RPD (raw scale)={log_rpd_rawscale:.2f}"
                )

            results.append({
                "source":         src,
                "target":         tgt,
                "source_name":    LU_NAMES.get(src, src),
                "target_name":    LU_NAMES.get(tgt, tgt),
                "is_in_domain":   in_domain,
                # Raw PLSR
                "raw_rpd":        raw_rpd,
                "raw_rmse":       raw_rmse,
                "raw_r2":         raw_r2,
                # Log1p PLSR — log scale (model's native space)
                "log_rpd_logscale":  log_rpd_logscale,
                "log_rmse_logscale": log_rmse_logscale,
                "log_r2_logscale":   log_r2_logscale,
                # Log1p PLSR — raw scale (back-transformed, what original paper used)
                "log_rpd_rawscale":  log_rpd_rawscale,
                "log_rmse_rawscale": log_rmse_rawscale,
                "log_r2_rawscale":   log_r2_rawscale,
                # Delta: log (log scale) vs raw
                "delta_rpd_log_vs_raw": log_rpd_logscale - raw_rpd,
            })

    return pd.DataFrame(results)


def fig_log1p_scales(df, out_path):
    """
    3-panel figure showing per-class in-domain RPD:
      Panel 1: Raw PLSR vs log1p PLSR (log scale) vs log1p PLSR (raw scale)
      Panel 2: For B-column failures, all three metrics
      Panel 3: Delta (log scale - raw PLSR) for all pairs as scatter
    """
    lu_codes = sorted(df["source"].unique())
    in_dom   = df[df["is_in_domain"]].set_index("source")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: in-domain comparison ────────────────────────────────────
    ax = axes[0]
    x  = np.arange(len(lu_codes))
    w  = 0.25
    ax.bar(x - w,   [in_dom.loc[lu, "raw_rpd"]         for lu in lu_codes],
           w, label="Raw PLSR",                color=TEAL,  alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.bar(x,       [in_dom.loc[lu, "log_rpd_logscale"] for lu in lu_codes],
           w, label="log1p PLSR (log scale)",  color=AMBER, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.bar(x + w,   [in_dom.loc[lu, "log_rpd_rawscale"] for lu in lu_codes],
           w, label="log1p PLSR (raw scale)",  color=RED,   alpha=0.60, edgecolor="k", linewidth=0.5)
    ax.axhline(2.0, color="#555", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(1.4, color="#555", linestyle=":",  linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lu}\n({LU_NAMES.get(lu,lu)})" for lu in lu_codes], fontsize=8)
    ax.set_ylabel("In-Domain RPD", fontsize=10)
    ax.set_title("In-Domain Performance\n(all three metrics)", fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Panel 2: B-column (cropland target) all three metrics ────────────
    ax2   = axes[1]
    b_col = df[df["target"] == "B"].set_index("source")
    srcs  = [lu for lu in lu_codes if lu != "B"] + ["B"]
    x2    = np.arange(len(srcs))
    ax2.bar(x2 - w,  [b_col.loc[s, "raw_rpd"]         for s in srcs],
            w, label="Raw PLSR",               color=TEAL,  alpha=0.85, edgecolor="k", linewidth=0.5)
    ax2.bar(x2,      [b_col.loc[s, "log_rpd_logscale"] for s in srcs],
            w, label="log1p (log scale)",      color=AMBER, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax2.bar(x2 + w,  [b_col.loc[s, "log_rpd_rawscale"] for s in srcs],
            w, label="log1p (raw scale)",      color=RED,   alpha=0.60, edgecolor="k", linewidth=0.5)
    ax2.axhline(1.0, color=RED, linestyle="-",  linewidth=1.2, alpha=0.8, label="Mean predictor (1.0)")
    ax2.axhline(1.4, color="#555", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"{s}->B" for s in srcs], fontsize=8.5)
    ax2.set_ylabel("RPD", fontsize=10)
    ax2.set_title("Cropland (B) as Target\n(all source classes)", fontsize=9)
    ax2.legend(fontsize=7.5, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Panel 3: delta scatter (log scale - raw) vs raw RPD ──────────────
    ax3   = axes[2]
    off   = df[~df["is_in_domain"]]
    sc    = ax3.scatter(off["raw_rpd"], off["delta_rpd_log_vs_raw"],
                        c=off["raw_rpd"], cmap="RdYlGn", vmin=0.4, vmax=2.5,
                        s=45, alpha=0.75, edgecolors="k", linewidths=0.3)
    ax3.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax3.set_xlabel("Raw PLSR RPD", fontsize=10)
    ax3.set_ylabel("log1p RPD (log scale) minus Raw PLSR RPD", fontsize=9)
    ax3.set_title("Does log1p help?\n(positive = log1p better on log scale)", fontsize=9)
    plt.colorbar(sc, ax=ax3, label="Raw PLSR RPD", shrink=0.8)
    ax3.grid(alpha=0.3, linestyle="--")

    fig.suptitle(
        "Figure 13: Log1p PLSR Evaluation — Separating Transformation Benefit from Back-Transform Distortion\n"
        "Key question: does log1p improve model fit in its native space, even if back-transformed RPD looks worse?",
        fontsize=9, y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chemistry", required=True)
    p.add_argument("--spectra",   required=True)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    print("=" * 60)
    print("SOLUM: Log1p Dual-Scale Evaluation")
    print("=" * 60)

    print("\n[Step 1] Loading dataset...")
    from data_loader import load_lucas
    df_clean, spectral_cols, wavelengths, lu_splits = load_lucas(
        chemistry_csv=args.chemistry,
        spectra_dir=args.spectra,
        verbose=True,
    )
    lu_codes = sorted(lu_splits.keys())

    print("\n[Step 2] Running dual evaluation (raw + log1p, both scales)...")
    results_df = run_dual_evaluation(
        lu_splits, spectral_cols,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    results_df.to_csv(os.path.join(OUT, "log1p_evaluation_full.csv"), index=False)

    print("\n" + "=" * 60)
    print("IN-DOMAIN SUMMARY")
    print("=" * 60)
    in_dom = results_df[results_df["is_in_domain"]][
        ["source", "raw_rpd", "log_rpd_logscale", "log_rpd_rawscale"]
    ].set_index("source")
    in_dom.columns = ["Raw PLSR RPD", "log1p RPD (log scale)", "log1p RPD (raw scale)"]
    print(in_dom.to_string(float_format="{:.3f}".format))

    print("\n" + "=" * 60)
    print("B-COLUMN (CROPLAND TARGET) SUMMARY")
    print("=" * 60)
    b_col = results_df[results_df["target"] == "B"][
        ["source", "raw_rpd", "log_rpd_logscale", "log_rpd_rawscale"]
    ].set_index("source")
    b_col.columns = ["Raw PLSR RPD", "log1p RPD (log scale)", "log1p RPD (raw scale)"]
    print(b_col.to_string(float_format="{:.3f}".format))

    print("\n" + "=" * 60)
    print("KEY QUESTION: Does log1p improve model fit in its native (log) space?")
    print("Delta = log1p RPD (log scale) - Raw PLSR RPD")
    print("Positive = log1p better; Negative = log1p worse")
    print("=" * 60)
    results_df["delta"] = results_df["log_rpd_logscale"] - results_df["raw_rpd"]
    pivot = results_df.pivot(index="source", columns="target", values="delta")
    pivot = pivot.loc[lu_codes, lu_codes]
    print(pivot.to_string(float_format="{:+.2f}".format))

    fig_log1p_scales(results_df, os.path.join(FIGS, "fig13_log1p_scales.png"))

    print(f"\nSaved: {OUT}/log1p_evaluation_full.csv")
    print("Done.")


if __name__ == "__main__":
    main()
