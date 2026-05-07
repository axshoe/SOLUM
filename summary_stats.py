"""
summary_stats.py
================
Computes the additional statistics Safanelli requested:
  - Mean, median, SD, IQR of observed SOC values for each class
    (on raw and log1p scale)
  - Mean error (bias) for all 25 transfer pairs, both model variants
    (raw PLSR and log1p PLSR), on both raw and log scales

Also checks for outliers in the cropland in-domain log1p model
(the one with negative back-transformed R2) using a simple approach:
  - Flag predictions beyond 3 SD from the mean error as potential outliers
  - Report how many there are and their SOC range

Usage:
    python summary_stats.py \
        --chemistry data/LUCAS_Topsoil_2015_20200323.csv \
        --spectra   data/spectra/ \
        --verbose

Outputs:
    outputs/soc_summary_stats.csv         -- per-class SOC distribution stats
    outputs/bias_table_all_pairs.csv      -- mean error for all pairs
    outputs/outlier_check_cropland.csv    -- outlier flagging for B in-domain
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solum"))

from data_loader import load_lucas, LU_NAMES, SOC_COL
from spectral_preprocessing import preprocess_spectra
from plsr import NIPALS_PLSR, select_n_components
from transferability_matrix import stratified_split

OUT = "outputs"


def iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def summary_stats_table(lu_splits):
    """Per-class SOC distribution on raw and log1p scale."""
    rows = []
    for lu, df in sorted(lu_splits.items()):
        y = df[SOC_COL].values.astype(float)
        y_log = np.log1p(y)
        rows.append({
            "class": lu,
            "class_name": LU_NAMES.get(lu, lu),
            "n": len(y),
            # Raw scale
            "raw_mean":   float(np.mean(y)),
            "raw_median": float(np.median(y)),
            "raw_sd":     float(np.std(y, ddof=1)),
            "raw_iqr":    iqr(y),
            "raw_min":    float(np.min(y)),
            "raw_max":    float(np.max(y)),
            # Log1p scale
            "log_mean":   float(np.mean(y_log)),
            "log_median": float(np.median(y_log)),
            "log_sd":     float(np.std(y_log, ddof=1)),
            "log_iqr":    iqr(y_log),
            "log_min":    float(np.min(y_log)),
            "log_max":    float(np.max(y_log)),
        })
    return pd.DataFrame(rows)


def run_bias_analysis(lu_splits, spectral_cols, random_state=42, verbose=True):
    """
    Train raw and log1p PLSR per source class.
    For each (source, target) pair compute:
      - Mean error (bias) = mean(y_pred - y_true)
      - On raw scale for raw PLSR
      - On log scale for log1p PLSR (native)
      - On raw scale for log1p PLSR (back-transformed)
    Also return per-pair predictions for outlier analysis.
    """
    lu_codes = sorted(lu_splits.keys())

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

    raw_models = {}
    log_models = {}

    for src in lu_codes:
        X_tr     = splits[src]["X_train"]
        y_tr     = splits[src]["y_train"]
        y_tr_log = splits[src]["y_train_log"]

        if verbose:
            print(f"[bias] Training source: {src}")

        nc_raw = select_n_components(X_tr, y_tr, max_components=20, verbose=False)
        m_raw  = NIPALS_PLSR(n_components=nc_raw)
        m_raw.fit(X_tr, y_tr)
        raw_models[src] = m_raw

        nc_log = select_n_components(X_tr, y_tr_log, max_components=20, verbose=False)
        m_log  = NIPALS_PLSR(n_components=nc_log)
        m_log.fit(X_tr, y_tr_log)
        log_models[src] = m_log

    rows = []
    all_preds = {}  # for outlier checking

    for src in lu_codes:
        for tgt in lu_codes:
            X_te = splits[tgt]["X_test"]
            y_te = splits[tgt]["y_test"]

            # Raw PLSR
            y_pred_raw  = raw_models[src].predict(X_te)
            err_raw     = y_pred_raw - y_te
            bias_raw    = float(np.mean(err_raw))

            # Log1p PLSR
            y_pred_log  = log_models[src].predict(X_te)
            err_log_native = y_pred_log - np.log1p(y_te)
            bias_log_native = float(np.mean(err_log_native))

            y_pred_bt   = np.maximum(np.expm1(y_pred_log), 0)
            err_bt      = y_pred_bt - y_te
            bias_bt     = float(np.mean(err_bt))

            rows.append({
                "source":           src,
                "target":           tgt,
                "source_name":      LU_NAMES.get(src, src),
                "target_name":      LU_NAMES.get(tgt, tgt),
                "is_in_domain":     src == tgt,
                # Raw PLSR bias (raw scale)
                "bias_raw_plsr":    bias_raw,
                "mae_raw_plsr":     float(np.mean(np.abs(err_raw))),
                # log1p PLSR bias on log scale (native)
                "bias_log_logscale":  bias_log_native,
                "mae_log_logscale":   float(np.mean(np.abs(err_log_native))),
                # log1p PLSR bias on raw scale (back-transformed)
                "bias_log_rawscale":  bias_bt,
                "mae_log_rawscale":   float(np.mean(np.abs(err_bt))),
                # n
                "n_test":           len(y_te),
                "y_test_mean":      float(np.mean(y_te)),
                "y_test_sd":        float(np.std(y_te, ddof=1)),
            })

            all_preds[(src, tgt)] = {
                "y_true":     y_te,
                "y_pred_raw": y_pred_raw,
                "y_pred_log": y_pred_log,
                "y_pred_bt":  y_pred_bt,
            }

            if verbose:
                flag = "[IN-DOMAIN]" if src == tgt else ""
                print(f"  {src}>{tgt} {flag}: "
                      f"bias_raw={bias_raw:+.2f}, "
                      f"bias_log_native={bias_log_native:+.3f}, "
                      f"bias_log_bt={bias_bt:+.2f}")

    return pd.DataFrame(rows), all_preds


def outlier_check(all_preds, src="B", tgt="B", threshold_sd=3.0):
    """
    For the specified (src, tgt) pair, flag outliers in the log1p model
    (back-transformed) as predictions where error > threshold_sd * SD(error).

    Returns a DataFrame of flagged samples.
    """
    preds = all_preds[(src, tgt)]
    y_true   = preds["y_true"]
    y_pred_bt = preds["y_pred_bt"]
    err      = y_pred_bt - y_true

    err_sd   = np.std(err, ddof=1)
    err_mean = np.mean(err)
    flagged  = np.abs(err - err_mean) > threshold_sd * err_sd

    df = pd.DataFrame({
        "y_true_soc":      y_true[flagged],
        "y_pred_bt_soc":   y_pred_bt[flagged],
        "y_pred_log":      preds["y_pred_log"][flagged],
        "error_raw":       err[flagged],
        "error_z":         (err[flagged] - err_mean) / err_sd,
    })
    return df.sort_values("error_z", key=abs, ascending=False).reset_index(drop=True)


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

    print("=" * 60)
    print("SOLUM: Summary Statistics + Bias Analysis")
    print("=" * 60)

    print("\n[Step 1] Loading dataset...")
    from data_loader import load_lucas
    df_clean, spectral_cols, wavelengths, lu_splits = load_lucas(
        chemistry_csv=args.chemistry,
        spectra_dir=args.spectra,
        verbose=True,
    )

    # ── SOC summary stats ─────────────────────────────────────────────────
    print("\n[Step 2] Computing SOC summary statistics per class...")
    stats_df = summary_stats_table(lu_splits)
    stats_df.to_csv(os.path.join(OUT, "soc_summary_stats.csv"), index=False)

    print("\nSOC DISTRIBUTION SUMMARY (raw scale, g/kg)")
    print("-" * 70)
    raw_cols = ["class", "class_name", "n", "raw_mean", "raw_median", "raw_sd", "raw_iqr", "raw_min", "raw_max"]
    print(stats_df[raw_cols].to_string(index=False, float_format="{:.2f}".format))

    print("\nSOC DISTRIBUTION SUMMARY (log1p scale)")
    print("-" * 70)
    log_cols = ["class", "class_name", "n", "log_mean", "log_median", "log_sd", "log_iqr"]
    print(stats_df[log_cols].to_string(index=False, float_format="{:.3f}".format))

    # ── Bias analysis ─────────────────────────────────────────────────────
    print("\n[Step 3] Running bias analysis for all transfer pairs...")
    bias_df, all_preds = run_bias_analysis(
        lu_splits, spectral_cols,
        random_state=args.random_state,
        verbose=args.verbose,
    )
    bias_df.to_csv(os.path.join(OUT, "bias_table_all_pairs.csv"), index=False)

    print("\nBIAS (MEAN ERROR) SUMMARY — RAW PLSR (raw scale)")
    print("Positive = overprediction, Negative = underprediction")
    bias_pivot = bias_df.pivot(index="source", columns="target", values="bias_raw_plsr")
    lu_codes = sorted(lu_splits.keys())
    print(bias_pivot.loc[lu_codes, lu_codes].to_string(float_format="{:+.2f}".format))

    print("\nBIAS (MEAN ERROR) SUMMARY — log1p PLSR (log scale, native)")
    bias_log_pivot = bias_df.pivot(index="source", columns="target", values="bias_log_logscale")
    print(bias_log_pivot.loc[lu_codes, lu_codes].to_string(float_format="{:+.3f}".format))

    print("\nBIAS (MEAN ERROR) SUMMARY — log1p PLSR (raw scale, back-transformed)")
    bias_bt_pivot = bias_df.pivot(index="source", columns="target", values="bias_log_rawscale")
    print(bias_bt_pivot.loc[lu_codes, lu_codes].to_string(float_format="{:+.2f}".format))

    # ── Outlier check on B->B log1p model ────────────────────────────────
    print("\n[Step 4] Outlier check: B->B log1p model (back-transformed, negative R2)")
    outliers = outlier_check(all_preds, src="B", tgt="B", threshold_sd=3.0)
    print(f"  Flagged {len(outliers)} outlier(s) at >3 SD from mean error")
    if len(outliers) > 0:
        print(outliers[["y_true_soc", "y_pred_bt_soc", "error_raw", "error_z"]].head(20).to_string(index=False, float_format="{:.2f}".format))
        outliers.to_csv(os.path.join(OUT, "outlier_check_cropland.csv"), index=False)

    # Also check range of predictions
    bb_preds = all_preds[("B", "B")]
    print(f"\n  B->B log1p back-transformed prediction range:")
    print(f"  y_true: min={bb_preds['y_true'].min():.1f}, max={bb_preds['y_true'].max():.1f}, mean={bb_preds['y_true'].mean():.1f}")
    print(f"  y_pred (bt): min={bb_preds['y_pred_bt'].min():.1f}, max={bb_preds['y_pred_bt'].max():.1f}, mean={bb_preds['y_pred_bt'].mean():.1f}")
    print(f"  y_pred (log native): min={bb_preds['y_pred_log'].min():.3f}, max={bb_preds['y_pred_log'].max():.3f}")

    print(f"\nSaved to {OUT}/")
    print("Done.")


if __name__ == "__main__":
    main()
