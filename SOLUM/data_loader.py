"""
data_loader.py
==============
Loads and merges the LUCAS 2015 Topsoil dataset for SOLUM.

LUCAS 2015 comes as two separate downloads that must be merged:
  1. Main soil chemistry CSV (LUCAS_Topsoil_2015_20200323.csv)
     Contains: POINT_ID, OC, LC, pH, N, etc.
  2. Per-country spectra CSVs in a folder (e.g. data/spectra/)
     Files named spectra_AT.csv, spectra_BE.csv, etc.
     Each row = one sample; columns = POINT_ID + wavelength absorbance values.

This module loads both, concatenates the per-country spectra into one
dataframe, merges on POINT_ID, then filters and splits by land use class.

LUCAS 2015 land use codes (first character of LC column):
  A  - Cropland (arable)
  B  - Cropland (permanent)
  C  - Woodland / forest  ← NOTE: in 2015, C codes include woodland (C2x, C3x)
  D  - Shrubland
  E  - Grassland
  F  - Bare land / sparse vegetation
  G  - Water / wetland (excluded — too few samples)
  H  - Artificial land (excluded)

The major class groupings used by SOLUM are derived from the first character
of the LC code, with C further split into C1x (grassland-type) and C2x/C3x
(woodland-type) based on LC0_Desc. See LUCAS_CLASS_MAP below.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# ─────────────────────────── configuration ───────────────────────────────────

# Column name for soil organic carbon (g/kg) in the chemistry CSV
SOC_COL = "OC"

# Column name for land use/cover code in the chemistry CSV
LC_COL = "LC1"

# Column name for point ID — used to merge chemistry and spectra
POINT_ID_COL = "POINT_ID"

# Minimum samples per land use class to include in the matrix
MIN_CLASS_SIZE = 200

# LUCAS 2015 maps LC first-character to a simplified class label.
# C1x = grassland-type; C2x/C3x = woodland/forest-type.
# We use LC0_Desc to distinguish these where available.
LUCAS_CLASS_MAP = {
    "A": "Cropland (Arable)",
    "B": "Cropland (Permanent)",
    "C": "Woodland",       # C2x, C3x codes
    "E": "Grassland",      # C1x codes get remapped to E in preprocessing
    "D": "Shrubland",
    "F": "Bare Land",
}

LU_NAMES = LUCAS_CLASS_MAP  # alias used by other modules


# ─────────────────────────── spectra loading ─────────────────────────────────


def load_spectra_folder(spectra_dir: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load and concatenate all per-country spectra CSV files from a folder.

    Each file (e.g. spectra_AT.csv) has:
      - First column: sample ID (various names: POINT_ID, point_id, ID, etc.)
      - Remaining columns: absorbance values at wavelengths (numeric column names
        like 400.0, 400.5, 401.0, ... up to 2499.5, or integers 400, 401, ...)

    Parameters
    ----------
    spectra_dir : str
        Path to folder containing spectra_XX.csv files.
    verbose : bool

    Returns
    -------
    spectra_df : pd.DataFrame
        Combined spectra with POINT_ID as index and wavelength columns.
    """
    pattern = os.path.join(spectra_dir, "spectra_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        # Also try without the spectra_ prefix
        pattern2 = os.path.join(spectra_dir, "*.csv")
        files = sorted(glob.glob(pattern2))

    if not files:
        raise FileNotFoundError(
            f"No spectra CSV files found in: {spectra_dir}\n"
            "Expected files named spectra_AT.csv, spectra_BE.csv, etc."
        )

    if verbose:
        print(f"[data_loader] Found {len(files)} spectra files in {spectra_dir}")

    dfs = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"[data_loader] Warning: could not read {fpath}: {e}")

    if not dfs:
        raise RuntimeError("No spectra files could be loaded.")

    spectra_all = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"[data_loader] Combined spectra shape: {spectra_all.shape}")

    # ── Identify the ID column ──────────────────────────────────────────────
    # Different files may name it differently
    id_candidates = ["POINT_ID", "point_id", "Point_ID", "ID", "id",
                     "PointID", "POINTID", "sample_id"]
    id_col_found = None
    for cand in id_candidates:
        if cand in spectra_all.columns:
            id_col_found = cand
            break

    if id_col_found is None:
        # First column is usually the ID
        id_col_found = spectra_all.columns[0]
        if verbose:
            print(f"[data_loader] Using first column as ID: '{id_col_found}'")

    # Rename to POINT_ID for consistency
    spectra_all = spectra_all.rename(columns={id_col_found: POINT_ID_COL})

    # ── Identify spectral columns ───────────────────────────────────────────
    spectral_cols = []
    for col in spectra_all.columns:
        if col == POINT_ID_COL:
            continue
        try:
            float(col)
            spectral_cols.append(col)
        except (ValueError, TypeError):
            pass

    if len(spectral_cols) < 100:
        raise ValueError(
            f"Only {len(spectral_cols)} spectral columns detected. "
            "Expected ~4,200. Check that the spectra CSV files are correct."
        )

    if verbose:
        print(f"[data_loader] Detected {len(spectral_cols)} spectral bands")
        print(f"[data_loader] Wavelength range: {spectral_cols[0]} – {spectral_cols[-1]}")

    # Keep only ID and spectral columns
    spectra_all = spectra_all[[POINT_ID_COL] + spectral_cols].copy()

    # Handle duplicate POINT_IDs (two scan replicates per sample in LUCAS 2015)
    # Average the two replicates
    n_before = len(spectra_all)
    spectra_all = spectra_all.groupby(POINT_ID_COL, as_index=False)[spectral_cols].mean()
    n_after = len(spectra_all)
    if verbose and n_before != n_after:
        print(f"[data_loader] Averaged {n_before - n_after} replicate scans → {n_after} unique samples")

    return spectra_all, spectral_cols


# ─────────────────────────── main loader ─────────────────────────────────────


def load_lucas(
    chemistry_csv: str,
    spectra_dir: str,
    min_class_size: int = MIN_CLASS_SIZE,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], np.ndarray, Dict[str, pd.DataFrame]]:
    """
    Load LUCAS 2015 topsoil data by merging chemistry CSV with spectra CSVs.

    Parameters
    ----------
    chemistry_csv : str
        Path to LUCAS_Topsoil_2015_20200323.csv
    spectra_dir : str
        Path to folder containing spectra_AT.csv, spectra_BE.csv, etc.
    min_class_size : int
        Minimum samples per land use class.
    verbose : bool

    Returns
    -------
    df_clean : pd.DataFrame
    spectral_cols : list of str
    wavelengths : np.ndarray
    lu_splits : dict {class_code: DataFrame}
    """
    # ── Load chemistry ───────────────────────────────────────────────────────
    if not os.path.exists(chemistry_csv):
        raise FileNotFoundError(f"Chemistry CSV not found: {chemistry_csv}")

    if verbose:
        print(f"[data_loader] Loading chemistry: {chemistry_csv}")

    chem = pd.read_csv(chemistry_csv, low_memory=False)

    if verbose:
        print(f"[data_loader] Chemistry shape: {chem.shape}")
        print(f"[data_loader] Chemistry columns: {list(chem.columns[:10])}...")

    # Normalise the point ID column name in chemistry
    for cand in ["POINT_ID", "point_id", "Point_ID", "POINTID", "PointID"]:
        if cand in chem.columns:
            chem = chem.rename(columns={cand: POINT_ID_COL})
            break

    # ── Load spectra ─────────────────────────────────────────────────────────
    spectra_df, spectral_cols = load_spectra_folder(spectra_dir, verbose=verbose)

    # ── Merge on POINT_ID ────────────────────────────────────────────────────
    if verbose:
        print(f"[data_loader] Merging chemistry ({len(chem)} rows) with spectra ({len(spectra_df)} rows)...")

    # Ensure POINT_ID types match for merge
    chem[POINT_ID_COL] = chem[POINT_ID_COL].astype(str).str.strip()
    spectra_df[POINT_ID_COL] = spectra_df[POINT_ID_COL].astype(str).str.strip()

    df = pd.merge(chem, spectra_df, on=POINT_ID_COL, how="inner")

    if verbose:
        print(f"[data_loader] After merge: {len(df)} samples with both chemistry and spectra")

    if len(df) == 0:
        raise RuntimeError(
            "Merge produced zero rows. The POINT_ID values in the chemistry "
            "and spectra files do not match. Check both files."
        )

    # ── Validate SOC ─────────────────────────────────────────────────────────
    if SOC_COL not in df.columns:
        soc_candidates = [c for c in df.columns if "OC" in c.upper() or "carbon" in c.lower()]
        raise KeyError(
            f"SOC column '{SOC_COL}' not found after merge. "
            f"Candidates: {soc_candidates}"
        )

    df[SOC_COL] = pd.to_numeric(df[SOC_COL], errors="coerce")

    # ── Validate LC column ───────────────────────────────────────────────────
    if LC_COL not in df.columns:
        lc_candidates = [c for c in df.columns if "LC" in c.upper() or "land" in c.lower()]
        raise KeyError(
            f"Land use column '{LC_COL}' not found. Candidates: {lc_candidates}"
        )

    # ── Assign simplified land use class ─────────────────────────────────────
    # LUCAS 2015: C1x = grassland-type, C2x/C3x = woodland/forest-type
    # We check LC0_Desc if available to split C codes correctly
    def assign_lu_class(row):
        lc = str(row[LC_COL]).strip()
        if not lc or lc == "nan":
            return None
        first = lc[0].upper()
        second = lc[1] if len(lc) > 1 else ""

        if first == "C":
            # C1x = grassland (map to E); C2x/C3x = woodland (keep as C)
            if second == "1":
                return "E"   # grassland
            elif second in ("2", "3"):
                return "C"   # woodland/forest
            else:
                return "C"   # default C to woodland
        elif first in ("A", "B", "D", "E", "F"):
            return first
        else:
            return None  # G (water), H (artificial) — excluded

    df["_lu_major"] = df.apply(assign_lu_class, axis=1)

    # ── Filter to valid samples ───────────────────────────────────────────────
    df[spectral_cols] = df[spectral_cols].apply(pd.to_numeric, errors="coerce")

    mask_soc = df[SOC_COL].notna() & (df[SOC_COL] >= 0)
    mask_spectra = df[spectral_cols].notna().all(axis=1) & (df[spectral_cols] >= 0).all(axis=1)
    mask_lu = df["_lu_major"].notna()

    df_clean = df[mask_soc & mask_spectra & mask_lu].copy()

    if verbose:
        n_dropped = len(df) - len(df_clean)
        print(f"[data_loader] Retained {len(df_clean)} / {len(df)} samples "
              f"({n_dropped} dropped for missing SOC, spectra, or LU code)")

    # ── Enforce minimum class size ────────────────────────────────────────────
    class_counts = df_clean["_lu_major"].value_counts()
    if verbose:
        print("[data_loader] Class counts before filtering:")
        for lu, count in class_counts.items():
            name = LU_NAMES.get(lu, lu)
            flag = "" if count >= min_class_size else f"  ← DROPPED (< {min_class_size})"
            print(f"  {lu} ({name}): {count}{flag}")

    valid_classes = class_counts[class_counts >= min_class_size].index.tolist()
    df_clean = df_clean[df_clean["_lu_major"].isin(valid_classes)].copy()

    if verbose:
        print(f"[data_loader] Classes retained: {sorted(valid_classes)}")
        print(f"[data_loader] Final dataset: {len(df_clean)} samples")

    # ── Wavelengths as numeric array ──────────────────────────────────────────
    wavelengths = np.array([float(c) for c in spectral_cols])

    # ── Per-class splits ──────────────────────────────────────────────────────
    lu_splits = {
        lu: df_clean[df_clean["_lu_major"] == lu].copy()
        for lu in sorted(valid_classes)
    }

    return df_clean, spectral_cols, wavelengths, lu_splits


def get_X_y(
    df: pd.DataFrame,
    spectral_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract spectral matrix X and SOC target y."""
    X = df[spectral_cols].values.astype(np.float64)
    y = df[SOC_COL].values.astype(np.float64)
    return X, y