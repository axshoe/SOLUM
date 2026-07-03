"""
ossl_mir_analysis_v2.py
=======================
FIXED VERSION. The OSSL LUCAS data is self-contained: it has MIR spectra,
SOC values, and land use codes all in its own tables. No join to the
LUCAS 2015 file is needed (and would fail anyway, since the MIR scans are
from the 2009-2012 archive, not 2015).

This version:
1. Downloads OSSL MIR, soil lab, and site tables (cached)
2. Merges them on id.layer_local_c
3. Pulls SOC from oc_iso.10694_w.pct (converts w.pct to g/kg: pct * 10)
4. Pulls land use from site.landuse2009_src_code / landuse2012_src_code
5. Maps LUCAS land use codes to B/C/D/E/F
6. Runs the MIR PLSR transferability matrix
7. Compares to the saved VNIR matrix

Usage:
    python ossl_mir_analysis_v2.py --verbose

Outputs:
    outputs/ossl_lucas_mir_merged.csv
    outputs/transferability_matrix_mir_plsr_rpd.csv
    figures/fig14_mir_vs_vnir_comparison.png
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solum"))

from spectral_preprocessing import preprocess_spectra
from plsr import NIPALS_PLSR, select_n_components
from transferability_matrix import compute_rpd, classify_rpd

OUT  = "outputs"
FIGS = "figures"
CACHE_DIR = os.path.join("data", "ossl_cache")

LU_NAMES = {
    "B": "Permanent Cropland", "C": "Woodland", "D": "Shrubland",
    "E": "Grassland", "F": "Bare Land",
}

RPD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rpd", [(0.0, "#C0392B"), (0.38, "#E67E22"), (1.0, "#0d7a7a")]
)

OSSL_URLS = {
    "mir":  "https://storage.googleapis.com/soilspec4gg-public/datasets/LUCAS/ossl_mir_v1.3.csv.gz",
    "soil": "https://storage.googleapis.com/soilspec4gg-public/datasets/LUCAS/ossl_soillab_v1.3.csv.gz",
    "site": "https://storage.googleapis.com/soilspec4gg-public/datasets/LUCAS/ossl_soilsite_v1.3.csv.gz",
}


def load_ossl(verbose=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    dfs = {}
    for key, url in OSSL_URLS.items():
        cache_path = os.path.join(CACHE_DIR, f"ossl_{key}.csv.gz")
        src = cache_path if os.path.exists(cache_path) else url
        if verbose:
            print(f"[ossl] Loading {key} from {'cache' if os.path.exists(cache_path) else 'web'}")
        df = pd.read_csv(src, compression="gzip", low_memory=False)
        if not os.path.exists(cache_path):
            df.to_csv(cache_path, index=False, compression="gzip")
        dfs[key] = df
        if verbose:
            print(f"[ossl]   {key}: {df.shape[0]} rows, {df.shape[1]} cols")
    return dfs


# LUCAS land cover code -> SOLUM class mapping
# LUCAS LC1 codes: B=cropland(permanent woody: fruit trees, olive, vineyard),
# C=woodland, D=shrubland, E=grassland, F=bareland
# The OSSL landuse codes may be full LUCAS LC1 codes (e.g., "B71", "C10")
def map_lucas_lu(code):
    if pd.isna(code):
        return None
    s = str(code).strip().upper()
    if not s or s in ("NA", "NAN", ""):
        return None
    first = s[0]
    # B = permanent crops (the LUCAS "B" cropland codes)
    if first == "B":
        return "B"
    if first == "C":
        return "C"   # woodland
    if first == "D":
        return "D"   # shrubland
    if first == "E":
        return "E"   # grassland
    if first == "F":
        return "F"   # bare land
    if first == "A":
        return "A"   # arable (will be dropped)
    return None


def merge_ossl(dfs, verbose=True):
    mir  = dfs["mir"]
    soil = dfs["soil"]
    site = dfs["site"]

    id_col = "id.layer_local_c"

    # SOC column: oc_iso.10694_w.pct is organic carbon in weight percent
    soc_col = None
    for c in ["oc_iso.10694_w.pct", "oc_usda.c729_w.pct", "oc_iso.10694_w.pct"]:
        if c in soil.columns:
            soc_col = c
            break
    if soc_col is None:
        # find any column with 'oc' and 'pct'
        oc_cols = [c for c in soil.columns if c.startswith("oc") and "pct" in c]
        soc_col = oc_cols[0] if oc_cols else None
    if verbose:
        print(f"[merge] SOC column: {soc_col}")

    # Land use columns from site table
    lu_cols = [c for c in site.columns if "landuse" in c.lower()]
    if verbose:
        print(f"[merge] Land use columns: {lu_cols}")

    # Ancillary covariates in soil table
    covar_map = {}
    for target, patterns in [
        ("clay", ["clay.tot"]), ("sand", ["sand.tot"]), ("silt", ["silt.tot"]),
        ("ph_h2o", ["ph.h2o"]), ("ph_cacl2", ["ph.cacl2"]), ("caco3", ["caco3"]),
        ("cf", ["cf_iso", "cf."]),
    ]:
        for c in soil.columns:
            if any(c.startswith(p) for p in patterns):
                covar_map[target] = c
                break

    # Merge soil + site on id
    keep_soil = [id_col, soc_col] + [v for v in covar_map.values()]
    keep_soil = [c for c in keep_soil if c in soil.columns]
    merged = soil[keep_soil].merge(
        site[[id_col] + lu_cols], on=id_col, how="inner"
    )
    if verbose:
        print(f"[merge] soil+site merged: {merged.shape[0]} rows")

    # Merge with MIR
    merged = mir.merge(merged, on=id_col, how="inner")
    if verbose:
        print(f"[merge] +MIR: {merged.shape[0]} rows")

    # Determine land use: prefer 2015 if present, else 2012, else 2009
    def resolve_lu(row):
        for col in ["site.landuse2015_src_code", "site.landuse2012_src_code",
                    "site.landuse2009_src_code"] + lu_cols:
            if col in row.index and pd.notna(row[col]):
                mapped = map_lucas_lu(row[col])
                if mapped:
                    return mapped
        return None

    merged["lu_class"] = merged.apply(resolve_lu, axis=1)

    # SOC to g/kg (w.pct * 10)
    merged["soc_g_kg"] = pd.to_numeric(merged[soc_col], errors="coerce") * 10.0

    if verbose:
        print(f"\n[merge] Land use distribution (raw codes):")
        for col in lu_cols:
            print(f"  {col}:")
            print(merged[col].value_counts().head(10).to_string())
        print(f"\n[merge] Mapped LU class distribution:")
        print(merged["lu_class"].value_counts().to_string())
        print(f"\n[merge] SOC range: {merged['soc_g_kg'].min():.1f} to {merged['soc_g_kg'].max():.1f} g/kg")

    return merged, covar_map


def extract_mir(merged, verbose=True):
    mir_cols = sorted(
        [c for c in merged.columns if c.startswith("scan_mir.")],
        key=lambda c: float(c.replace("scan_mir.", "").replace("_abs", ""))
    )
    wavenumbers = np.array([float(c.replace("scan_mir.", "").replace("_abs", "")) for c in mir_cols])
    X = merged[mir_cols].values.astype(float)
    if verbose:
        print(f"[mir] {len(mir_cols)} MIR bands, {wavenumbers.min():.0f}-{wavenumbers.max():.0f} cm-1")
    return X, wavenumbers


def run_mir_transferability(lu_splits, verbose=True):
    lu_codes = sorted(lu_splits.keys())
    rpd_matrix = pd.DataFrame(index=lu_codes, columns=lu_codes, dtype=float)

    splits = {}
    for lu in lu_codes:
        X = lu_splits[lu]["X"]
        y = lu_splits[lu]["y"]
        idx = np.arange(len(y))
        np.random.seed(42)
        np.random.shuffle(idx)
        n_test = max(int(len(idx) * 0.2), 3)
        te, tr = idx[:n_test], idx[n_test:]
        splits[lu] = {
            "X_train": preprocess_spectra(X[tr]), "y_train": y[tr],
            "X_test":  preprocess_spectra(X[te]), "y_test":  y[te],
        }

    models = {}
    for src in lu_codes:
        Xtr, ytr = splits[src]["X_train"], splits[src]["y_train"]
        nc = select_n_components(Xtr, ytr, max_components=min(15, len(ytr) - 1), verbose=False)
        m = NIPALS_PLSR(n_components=nc)
        m.fit(Xtr, ytr)
        models[src] = m
        if verbose:
            print(f"[mir] Trained {src} (n_train={len(ytr)}, ncomp={nc})")

    for src in lu_codes:
        for tgt in lu_codes:
            Xte, yte = splits[tgt]["X_test"], splits[tgt]["y_test"]
            rpd = compute_rpd(yte, models[src].predict(Xte))
            rpd_matrix.loc[src, tgt] = rpd
            if verbose:
                flag = " [IN-DOMAIN]" if src == tgt else ""
                print(f"  {src}->{tgt}{flag}: RPD={rpd:.2f}")
    return rpd_matrix


def fig_mir_vs_vnir(mir_matrix, vnir_csv, out_path):
    try:
        vnir = pd.read_csv(vnir_csv, index_col=0)
    except FileNotFoundError:
        print(f"[fig] VNIR matrix not found, skipping.")
        return
    common = [lu for lu in sorted(mir_matrix.index) if lu in vnir.index]
    if len(common) < 2:
        print("[fig] Not enough common classes.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    vmin, vmax = 0.5, 3.5
    for ax, mat, title in [
        (axes[0], vnir.loc[common, common], "VNIR PLSR (LUCAS 2015)"),
        (axes[1], mir_matrix.loc[common, common], "MIR PLSR (OSSL LUCAS)"),
    ]:
        vals = mat.values.astype(float)
        im = ax.imshow(vals, vmin=vmin, vmax=vmax, cmap=RPD_CMAP, aspect="auto")
        n = len(common)
        for i in range(n):
            for j in range(n):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                b = (v - vmin) / (vmax - vmin)
                tc = "white" if b < 0.32 or b > 0.76 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color=tc, fontweight="bold" if i == j else "normal")
        for k in range(n):
            ax.add_patch(plt.Rectangle((k-0.5, k-0.5), 1, 1, fill=False,
                                        edgecolor="black", linewidth=2.2))
        labels = [f"{lu}\n({LU_NAMES.get(lu,lu)})" for lu in common]
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Target (test)", fontsize=9)
        ax.set_ylabel("Source (training)", fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        for yval, col, ls in [(1.0, "#C0392B", "-"), (1.4, "black", "--"), (2.0, "black", "--")]:
            cb.ax.axhline(y=(yval-vmin)/(vmax-vmin), color=col, linewidth=1.2, linestyle=ls)
    legend = [Patch(facecolor="#C0392B", label="Poor (<1.4)"),
              Patch(facecolor="#E67E22", label="Moderate (1.4-2.0)"),
              Patch(facecolor="#0d7a7a", label="Good (>=2.0)")]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8.5,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Figure 14: VNIR vs. MIR Transferability Matrix\n"
        "Does the cropland (B) transfer-failure pattern persist with higher-fidelity MIR?",
        fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min-class-size", type=int, default=20)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    print("=" * 60)
    print("SOLUM: OSSL MIR Transferability (v2, self-contained)")
    print("=" * 60)

    print("\n[Step 1] Loading OSSL data...")
    dfs = load_ossl(verbose=args.verbose)

    print("\n[Step 2] Merging (OSSL-native, no LUCAS 2015 join)...")
    merged, covar_map = merge_ossl(dfs, verbose=args.verbose)
    merged.to_csv(os.path.join(OUT, "ossl_lucas_mir_merged.csv"), index=False)

    print("\n[Step 3] Extracting MIR spectra...")
    X_mir, wavenumbers = extract_mir(merged, verbose=args.verbose)

    # Filter to valid samples
    retain = {"B", "C", "D", "E", "F"}
    mask = (
        merged["lu_class"].isin(retain)
        & merged["soc_g_kg"].notna()
        & ~np.isnan(X_mir).all(axis=1)
    )
    merged_f = merged[mask].reset_index(drop=True)
    X_f = X_mir[mask.values]
    if args.verbose:
        print(f"\n[Step 3] {len(merged_f)} valid samples")
        print(merged_f["lu_class"].value_counts().to_string())

    lu_splits = {}
    for lu in sorted(retain):
        m = merged_f["lu_class"] == lu
        n = int(m.sum())
        if n < args.min_class_size:
            print(f"[Step 3] Dropping {lu}: only {n} samples (min={args.min_class_size})")
            continue
        lu_splits[lu] = {"X": X_f[m.values],
                         "y": merged_f.loc[m, "soc_g_kg"].values.astype(float)}
        print(f"  {lu} ({LU_NAMES.get(lu,lu)}): n={n}, SOC mean={lu_splits[lu]['y'].mean():.1f} g/kg")

    if len(lu_splits) < 2:
        print("\nNot enough classes for a transferability matrix.")
        print("This likely means the MIR subset (from 2009-2012 archive) has")
        print("limited land use coverage. The merged CSV is saved for inspection.")
        return

    print(f"\n[Step 4] Running MIR transferability ({len(lu_splits)} classes)...")
    mir_matrix = run_mir_transferability(lu_splits, verbose=args.verbose)
    mir_matrix.to_csv(os.path.join(OUT, "transferability_matrix_mir_plsr_rpd.csv"))
    print("\n[Step 4] MIR PLSR Transferability Matrix:")
    print(mir_matrix.to_string(float_format="{:.2f}".format))

    print("\n[Step 5] MIR vs VNIR comparison figure...")
    fig_mir_vs_vnir(mir_matrix,
                    os.path.join(OUT, "transferability_matrix_plsr_rpd.csv"),
                    os.path.join(FIGS, "fig14_mir_vs_vnir_comparison.png"))

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()