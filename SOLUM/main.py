"""
main.py
=======
SOLUM: Soil Organic Carbon Transferability Benchmark
Updated for LUCAS 2015 two-file format (chemistry CSV + spectra folder).

Usage
-----
    python solum/main.py \
        --chemistry data/LUCAS_Topsoil_2015_20200323.csv \
        --spectra   data/spectra/ \
        --verbose

Optional flags:
    --no-grid-search   Skip RF hyperparameter grid search (faster)
    --no-shap          Skip SHAP computation (faster)
    --output-dir       Directory for CSV outputs (default: ./outputs)
    --random-state     Random seed (default: 42)
    --sample-n N       Subsample N samples per class (for quick testing)
    --verbose          Verbose output
"""

import argparse
import os
import time
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="SOLUM: SOC Transferability Benchmark")
    parser.add_argument(
        "--chemistry",
        required=True,
        help="Path to LUCAS_Topsoil_2015_20200323.csv (soil chemistry file).",
    )
    parser.add_argument(
        "--spectra",
        required=True,
        help="Path to folder containing spectra_AT.csv, spectra_BE.csv, etc.",
    )
    parser.add_argument("--no-grid-search", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Subsample N samples per class for quick testing.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("=" * 60)
    print("SOLUM: Soil Organic Carbon Transferability Benchmark")
    print("=" * 60)

    # ── Step 1: Load and merge LUCAS 2015 ────────────────────────────────────
    print("\n[Step 1] Loading LUCAS 2015 Topsoil dataset...")
    from data_loader import load_lucas

    df_clean, spectral_cols, wavelengths, lu_splits = load_lucas(
        chemistry_csv=args.chemistry,
        spectra_dir=args.spectra,
        verbose=True,
    )

    if args.sample_n is not None:
        print(f"\n[Step 1] Subsampling to {args.sample_n} samples per class...")
        for lu in list(lu_splits.keys()):
            df_lu = lu_splits[lu]
            if len(df_lu) > args.sample_n:
                lu_splits[lu] = df_lu.sample(n=args.sample_n, random_state=args.random_state)

    lu_codes = sorted(lu_splits.keys())
    print(f"\n[Step 1] Classes: {lu_codes}")
    print(f"[Step 1] Spectral bands: {len(spectral_cols)}")

    # ── Step 2: Transferability Matrix ───────────────────────────────────────
    print("\n[Step 2] Building transferability matrix...")
    from transferability_matrix import build_transferability_matrix, results_to_dataframe

    plsr_rpd_matrix, rf_rpd_matrix, results, plsr_models, rf_models, splits = \
        build_transferability_matrix(
            lu_splits=lu_splits,
            spectral_cols=spectral_cols,
            rf_grid_search=not args.no_grid_search,
            random_state=args.random_state,
            verbose=args.verbose,
        )

    print("\n[Step 2] PLSR Transferability Matrix (RPD):")
    print(plsr_rpd_matrix.to_string(float_format="{:.2f}".format))
    print("\n[Step 2] RF Transferability Matrix (RPD):")
    print(rf_rpd_matrix.to_string(float_format="{:.2f}".format))

    plsr_rpd_matrix.to_csv(os.path.join(args.output_dir, "transferability_matrix_plsr_rpd.csv"))
    rf_rpd_matrix.to_csv(os.path.join(args.output_dir, "transferability_matrix_rf_rpd.csv"))
    results_to_dataframe(results, lu_codes).to_csv(
        os.path.join(args.output_dir, "results_all_pairs.csv"), index=False
    )
    print(f"[Step 2] Results exported to {args.output_dir}/")

    # ── Step 3: SHAP ─────────────────────────────────────────────────────────
    shap_data = {}
    if not args.no_shap:
        print("\n[Step 3] Running SHAP attribution...")
        from shap_analysis import run_shap_analysis
        shap_data, discrepancy_df = run_shap_analysis(
            rf_models=rf_models,
            splits=splits,
            results=results,
            wavelengths=wavelengths,
            lu_codes=lu_codes,
            verbose=args.verbose,
        )
        if not discrepancy_df.empty:
            discrepancy_df.to_csv(os.path.join(args.output_dir, "shap_discrepancy.csv"))
    else:
        print("[Step 3] Skipping SHAP.")

    # ── Step 4: Figures ───────────────────────────────────────────────────────
    print("\n[Step 4] Generating figures...")
    from analysis import generate_all_figures
    generate_all_figures(
        lu_splits=lu_splits,
        splits=splits,
        wavelengths=wavelengths,
        results=results,
        plsr_rpd_matrix=plsr_rpd_matrix,
        rf_rpd_matrix=rf_rpd_matrix,
        shap_data=shap_data,
        lu_codes=lu_codes,
    )

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"SOLUM complete. Elapsed: {elapsed/60:.1f} min")
    print(f"Outputs: {args.output_dir}/   Figures: figures/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()