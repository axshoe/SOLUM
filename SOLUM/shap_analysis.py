"""
shap_analysis.py
================
SHAP-based wavelength attribution for transfer failure analysis.

For each RF model trained on source class S, computes SHAP values on the
target class T test set. SHAP (SHapley Additive exPlanations) assigns each
spectral band a contribution value toward each prediction.

Transfer failure attribution:
  We define "SHAP discrepancy" between source and target as:
    delta_SHAP(wavelength) = |mean_SHAP_target - mean_SHAP_source|
  High discrepancy in a wavelength region indicates that band drives
  predictions differently when the model is applied to a different ecosystem,
  which is the spectral signature of domain shift.

For failed transfers (RF RPD < 1.4), the wavelength bands with highest
SHAP discrepancy are flagged as the likely spectral sources of the failure.

Reference:
  Lundberg, S.M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. Advances in Neural Information Processing Systems, 30.
  (NeurIPS 2017)

  Shapley, L.S. (1953). A value for n-person games. In H.W. Kuhn &
  A.W. Tucker (Eds.), Contributions to the Theory of Games (Vol. 2,
  pp. 307–317). Princeton University Press.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import shap

from data_loader import LU_NAMES
from rf_model import RFModel


# ─────────────────────────────────────────────────────────────────────────────
# SHAP computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_shap_values(
    rf_model: RFModel,
    X: np.ndarray,
    n_background: int = 100,
    check_additivity: bool = False,
) -> np.ndarray:
    """
    Compute SHAP values for a set of samples using TreeExplainer.

    Parameters
    ----------
    rf_model : RFModel
        Fitted RFModel instance.
    X : np.ndarray, shape (n_samples, n_bands)
        Samples to explain.
    n_background : int
        Number of background samples for the TreeExplainer. Smaller values
        are faster; 100 is usually sufficient for spectral data.
    check_additivity : bool
        SHAP internal consistency check. Set False for speed.

    Returns
    -------
    shap_values : np.ndarray, shape (n_samples, n_bands)
        SHAP values for each sample and each spectral band.
    """
    explainer = shap.TreeExplainer(
        rf_model.sklearn_model,
        feature_perturbation="tree_path_dependent",
    )
    # For large X, subsample for efficiency
    if X.shape[0] > 500:
        idx = np.random.choice(X.shape[0], 500, replace=False)
        X_explain = X[idx]
    else:
        X_explain = X

    sv = explainer.shap_values(X_explain, check_additivity=check_additivity)
    return sv  # shape (n_samples, n_bands)


def compute_mean_abs_shap(shap_values: np.ndarray) -> np.ndarray:
    """
    Mean absolute SHAP value per band (global feature importance).

    Parameters
    ----------
    shap_values : np.ndarray, shape (n_samples, n_bands)

    Returns
    -------
    mean_abs_shap : np.ndarray, shape (n_bands,)
    """
    return np.mean(np.abs(shap_values), axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP discrepancy between source and target
# ─────────────────────────────────────────────────────────────────────────────


def compute_shap_discrepancy(
    src_shap: np.ndarray,
    tgt_shap: np.ndarray,
) -> np.ndarray:
    """
    Per-band SHAP discrepancy between source in-domain SHAP and
    cross-domain (target) SHAP for the same model.

    delta_SHAP[band] = |mean_abs_SHAP(target) - mean_abs_SHAP(source)|

    A large delta indicates that band is used differently (or more
    erratically) when the model crosses the domain boundary.

    Parameters
    ----------
    src_shap : np.ndarray, shape (n_src, n_bands)
        SHAP values from source model evaluated on SOURCE test set.
    tgt_shap : np.ndarray, shape (n_tgt, n_bands)
        SHAP values from source model evaluated on TARGET test set.

    Returns
    -------
    discrepancy : np.ndarray, shape (n_bands,)
    """
    src_mean = compute_mean_abs_shap(src_shap)
    tgt_mean = compute_mean_abs_shap(tgt_shap)
    return np.abs(tgt_mean - src_mean)


def top_discrepant_bands(
    discrepancy: np.ndarray,
    wavelengths: np.ndarray,
    n_top: int = 20,
) -> pd.DataFrame:
    """
    Return the wavelength bands with highest SHAP discrepancy.

    Parameters
    ----------
    discrepancy : np.ndarray, shape (n_bands,)
    wavelengths : np.ndarray, shape (n_bands,)
    n_top : int

    Returns
    -------
    df : pd.DataFrame with columns [wavelength_nm, discrepancy, rank]
    """
    top_idx = np.argsort(discrepancy)[::-1][:n_top]
    return pd.DataFrame({
        "wavelength_nm": wavelengths[top_idx],
        "shap_discrepancy": discrepancy[top_idx],
        "rank": np.arange(1, n_top + 1),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Full SHAP analysis pipeline for all transfer pairs
# ─────────────────────────────────────────────────────────────────────────────


def run_shap_analysis(
    rf_models: Dict[str, RFModel],
    splits: Dict[str, Dict],
    results: Dict,
    wavelengths: np.ndarray,
    lu_codes: List[str],
    rpd_failure_threshold: float = 1.4,
    verbose: bool = True,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run SHAP analysis for all transfer pairs, with discrepancy computation
    for failed transfers (RF RPD < threshold).

    Parameters
    ----------
    rf_models : dict {lu_code: RFModel}
    splits : dict {lu_code: {X_train, X_test, y_train, y_test}}
    results : dict {(src, tgt): metrics}
    wavelengths : np.ndarray
    lu_codes : list of str
    rpd_failure_threshold : float
    verbose : bool

    Returns
    -------
    shap_data : dict
        Keyed by (src, tgt) -> {'src_shap', 'tgt_shap', 'discrepancy',
                                 'top_bands', 'failed_transfer'}
    discrepancy_matrix : pd.DataFrame
        Mean SHAP discrepancy per band summed across all failed pairs,
        indexed by wavelength band.
    """
    shap_data = {}
    # Matrix: rows = (src, tgt) pair label, cols = wavelength bands
    # We only populate entries where RPD < threshold
    discrepancy_rows = []

    for src in lu_codes:
        # Compute in-domain (source test set) SHAP as baseline
        X_src_test = splits[src]["X_test"]
        if verbose:
            print(f"[SHAP] Computing source SHAP: {src}")
        src_shap = compute_shap_values(rf_models[src], X_src_test)

        for tgt in lu_codes:
            if src == tgt:
                # Diagonal: in-domain, still compute for reference
                shap_data[(src, tgt)] = {
                    "src_shap": src_shap,
                    "tgt_shap": src_shap,
                    "discrepancy": np.zeros(len(wavelengths)),
                    "top_bands": None,
                    "failed_transfer": False,
                }
                continue

            rf_rpd = results[(src, tgt)]["rf_rpd"]
            is_failure = rf_rpd < rpd_failure_threshold

            X_tgt_test = splits[tgt]["X_test"]
            if verbose:
                fail_str = " [FAILURE]" if is_failure else ""
                print(f"[SHAP] {src}→{tgt} (RPD={rf_rpd:.2f}){fail_str}")

            tgt_shap = compute_shap_values(rf_models[src], X_tgt_test)
            discrepancy = compute_shap_discrepancy(src_shap, tgt_shap)
            top_bands = top_discrepant_bands(discrepancy, wavelengths)

            shap_data[(src, tgt)] = {
                "src_shap": src_shap,
                "tgt_shap": tgt_shap,
                "discrepancy": discrepancy,
                "top_bands": top_bands,
                "failed_transfer": is_failure,
            }

            if is_failure:
                row = {
                    "pair": f"{src}→{tgt}",
                    "source": src,
                    "target": tgt,
                    "rf_rpd": rf_rpd,
                }
                for k, wl in enumerate(wavelengths):
                    row[f"wl_{int(wl)}"] = discrepancy[k]
                discrepancy_rows.append(row)

    if discrepancy_rows:
        discrepancy_df = pd.DataFrame(discrepancy_rows).set_index("pair")
    else:
        discrepancy_df = pd.DataFrame()

    return shap_data, discrepancy_df
