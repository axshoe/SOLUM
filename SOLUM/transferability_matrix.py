"""
transferability_matrix.py
==========================
Cross-land-use evaluation loop producing the SOLUM Transferability Matrix.

For each ordered pair (source, target) of land use classes:
  - The source-trained model (both PLSR and RF) is evaluated on the target
    test set
  - R², RMSE, and RPD are computed

The matrix is N x N where N = number of retained land use classes.
Diagonal entries = in-domain performance (source == target).
Off-diagonal entries = transfer performance (source != target).

RPD (Ratio of Performance to Deviation) is the standard metric in soil
spectroscopy for benchmarking predictive performance:
  RPD = SD(y_true) / RMSE
  RPD > 2.0 : good predictive performance
  RPD 1.4-2.0 : moderate (approximate quantitative predictions)
  RPD < 1.4 : poor (not useful for quantitative prediction)

Reference:
  Chang, C.-W., Laird, D.A., Mausbach, M.J., & Hurburgh, C.R. (2001).
  Near-infrared reflectance spectroscopy–principal components regression
  analyses of soil properties. Soil Science Society of America Journal,
  65(2), 480–490.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split

from data_loader import LU_NAMES, get_X_y
from spectral_preprocessing import preprocess_spectra
from plsr import NIPALS_PLSR, select_n_components
from rf_model import RFModel


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²). Can be negative for poor transfers."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Ratio of Performance to Deviation.
    RPD = SD(y_true) / RMSE(y_true, y_pred)
    """
    rmse = compute_rmse(y_true, y_pred)
    sd = float(np.std(y_true, ddof=1))
    if rmse == 0:
        return float("inf")
    if sd == 0:
        return float("nan")
    return sd / rmse


def classify_rpd(rpd: float) -> str:
    """Return performance tier based on RPD value."""
    if np.isnan(rpd) or np.isinf(rpd):
        return "N/A"
    if rpd >= 2.0:
        return "Good"
    elif rpd >= 1.4:
        return "Moderate"
    else:
        return "Poor"


# ─────────────────────────────────────────────────────────────────────────────
# Train-test splitting per class
# ─────────────────────────────────────────────────────────────────────────────


def stratified_split(
    df_class: pd.DataFrame,
    spectral_cols: List[str],
    test_size: float = 0.2,
    n_quartiles: int = 4,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    80/20 train/test split stratified by SOC quartile.

    Stratification by SOC quartile ensures that the test set has a similar
    SOC distribution to the training set, which is important for RPD
    calculations to be representative.

    Parameters
    ----------
    df_class : pd.DataFrame
        DataFrame for one land use class.
    spectral_cols : list of str
    test_size : float
    n_quartiles : int
        Number of quartile bins for stratification. Default: 4.
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    from data_loader import SOC_COL

    X, y = get_X_y(df_class, spectral_cols)

    # Create SOC quartile labels for stratification
    quartile_labels = pd.qcut(y, q=n_quartiles, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=quartile_labels,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Main transferability matrix builder
# ─────────────────────────────────────────────────────────────────────────────


def build_transferability_matrix(
    lu_splits: Dict[str, pd.DataFrame],
    spectral_cols: List[str],
    sg_window: int = 11,
    sg_poly: int = 2,
    plsr_max_components: int = 20,
    rf_grid_search: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Build the full N x N Transferability Matrix.

    For each source class S:
      1. Preprocess X_train_S with SG + SNV
      2. Select PLSR components via LOO-CV
      3. Fit PLSR and RF on X_train_S, y_train_S

    For each (S, T) pair:
      1. Preprocess X_test_T with the same pipeline
      2. Evaluate S-trained PLSR and RF on X_test_T
      3. Record R², RMSE, RPD

    Parameters
    ----------
    lu_splits : dict
        {lu_code: DataFrame} from data_loader.load_lucas
    spectral_cols : list of str
    sg_window : int
    sg_poly : int
    plsr_max_components : int
    rf_grid_search : bool
    random_state : int
    verbose : bool

    Returns
    -------
    plsr_matrix : pd.DataFrame
        RPD matrix for PLSR (rows=source, cols=target)
    rf_matrix : pd.DataFrame
        RPD matrix for RF
    results_dict : dict
        Full results including R², RMSE, RPD for both models and all pairs
    """
    lu_codes = sorted(lu_splits.keys())
    n_classes = len(lu_codes)

    if verbose:
        print(f"[matrix] Building {n_classes}x{n_classes} transferability matrix")
        print(f"[matrix] Classes: {lu_codes}")

    # ── Step 1: split, preprocess, and store test sets ───────────────────────
    splits = {}
    for lu in lu_codes:
        df_lu = lu_splits[lu]
        X_train, X_test, y_train, y_test = stratified_split(
            df_lu, spectral_cols, random_state=random_state
        )

        # Preprocess: SG + SNV
        X_train_proc = preprocess_spectra(X_train, sg_window=sg_window, sg_poly=sg_poly)
        X_test_proc = preprocess_spectra(X_test, sg_window=sg_window, sg_poly=sg_poly)

        splits[lu] = {
            "X_train": X_train_proc,
            "X_test": X_test_proc,
            "y_train": y_train,
            "y_test": y_test,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

        if verbose:
            print(
                f"[matrix] {lu} ({LU_NAMES.get(lu, lu)}): "
                f"n_train={len(y_train)}, n_test={len(y_test)}, "
                f"SOC mean={y_train.mean():.1f} g/kg"
            )

    # ── Step 2: train source models ───────────────────────────────────────────
    plsr_models = {}
    rf_models = {}

    for src in lu_codes:
        X_tr = splits[src]["X_train"]
        y_tr = splits[src]["y_train"]

        if verbose:
            print(f"\n[matrix] Training models on source: {src} ({LU_NAMES.get(src, src)})")

        # PLSR: select n_components via CV
        n_comp = select_n_components(
            X_tr, y_tr,
            max_components=plsr_max_components,
            verbose=verbose,
        )
        plsr = NIPALS_PLSR(n_components=n_comp)
        plsr.fit(X_tr, y_tr)
        plsr_models[src] = plsr

        if verbose:
            print(f"[matrix]   PLSR fitted with n_components={n_comp}")

        # RF: fit with optional grid search
        rf = RFModel(random_state=random_state)
        rf.fit(X_tr, y_tr, grid_search=rf_grid_search, verbose=verbose)
        rf_models[src] = rf

        if verbose:
            print(f"[matrix]   RF fitted with params={rf.best_params_}")

    # ── Step 3: evaluate all (source, target) pairs ──────────────────────────
    results = {}
    plsr_rpd_matrix = pd.DataFrame(index=lu_codes, columns=lu_codes, dtype=float)
    rf_rpd_matrix = pd.DataFrame(index=lu_codes, columns=lu_codes, dtype=float)

    for src in lu_codes:
        for tgt in lu_codes:
            X_test = splits[tgt]["X_test"]
            y_test = splits[tgt]["y_test"]

            # PLSR evaluation
            y_pred_plsr = plsr_models[src].predict(X_test)
            plsr_r2 = compute_r2(y_test, y_pred_plsr)
            plsr_rmse = compute_rmse(y_test, y_pred_plsr)
            plsr_rpd = compute_rpd(y_test, y_pred_plsr)

            # RF evaluation
            y_pred_rf = rf_models[src].predict(X_test)
            rf_r2 = compute_r2(y_test, y_pred_rf)
            rf_rmse = compute_rmse(y_test, y_pred_rf)
            rf_rpd = compute_rpd(y_test, y_pred_rf)

            results[(src, tgt)] = {
                "plsr_r2": plsr_r2,
                "plsr_rmse": plsr_rmse,
                "plsr_rpd": plsr_rpd,
                "plsr_tier": classify_rpd(plsr_rpd),
                "rf_r2": rf_r2,
                "rf_rmse": rf_rmse,
                "rf_rpd": rf_rpd,
                "rf_tier": classify_rpd(rf_rpd),
                "y_pred_plsr": y_pred_plsr,
                "y_pred_rf": y_pred_rf,
                "y_test": y_test,
            }

            plsr_rpd_matrix.loc[src, tgt] = plsr_rpd
            rf_rpd_matrix.loc[src, tgt] = rf_rpd

            if verbose:
                diagonal = " [IN-DOMAIN]" if src == tgt else ""
                print(
                    f"  {src}→{tgt}{diagonal}: "
                    f"PLSR RPD={plsr_rpd:.2f} ({classify_rpd(plsr_rpd)}), "
                    f"RF RPD={rf_rpd:.2f} ({classify_rpd(rf_rpd)})"
                )

    return plsr_rpd_matrix, rf_rpd_matrix, results, plsr_models, rf_models, splits


def results_to_dataframe(results: Dict, lu_codes: List[str]) -> pd.DataFrame:
    """
    Flatten results dict into a long-format DataFrame for export.
    """
    rows = []
    for (src, tgt), metrics in results.items():
        rows.append({
            "source_class": src,
            "source_name": LU_NAMES.get(src, src),
            "target_class": tgt,
            "target_name": LU_NAMES.get(tgt, tgt),
            "is_in_domain": src == tgt,
            "plsr_r2": metrics["plsr_r2"],
            "plsr_rmse": metrics["plsr_rmse"],
            "plsr_rpd": metrics["plsr_rpd"],
            "plsr_tier": metrics["plsr_tier"],
            "rf_r2": metrics["rf_r2"],
            "rf_rmse": metrics["rf_rmse"],
            "rf_rpd": metrics["rf_rpd"],
            "rf_tier": metrics["rf_tier"],
        })
    return pd.DataFrame(rows).sort_values(["source_class", "target_class"])
