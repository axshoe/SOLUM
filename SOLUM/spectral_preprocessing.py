"""
spectral_preprocessing.py
==========================
Spectral preprocessing routines for Vis-NIR reflectance data, implemented
from scratch to demonstrate understanding of the underlying mathematics.

Implements:
  1. Savitzky-Golay smoothing — noise removal via local polynomial fitting
  2. Standard Normal Variate (SNV) transformation — scatter correction

Both are standard in the soil spectroscopy literature and are applied
sequentially (SG first, then SNV) before model training.

References:
  Savitzky, A., & Golay, M.J.E. (1964). Smoothing and differentiation of data
    by simplified least squares procedures. Analytical Chemistry, 36(8), 1627–1639.
  Barnes, R.J., Dhanoa, M.S., & Lister, S.J. (1989). Standard normal variate
    transformation and de-trending of near-infrared diffuse reflectance spectra.
    Applied Spectroscopy, 43(5), 772–777.
"""

import numpy as np
import math
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Savitzky-Golay smoothing
# ─────────────────────────────────────────────────────────────────────────────


def _sg_coefficients(window_size: int, poly_order: int, deriv: int = 0) -> np.ndarray:
    """
    Compute Savitzky-Golay filter coefficients via least-squares.

    The SG filter fits a polynomial of degree `poly_order` to a moving window
    of `window_size` points and evaluates the derivative of that polynomial at
    the center point. For deriv=0, this gives the smoothed value.

    Parameters
    ----------
    window_size : int
        Number of points in the sliding window. Must be odd and > poly_order.
    poly_order : int
        Degree of the fitting polynomial.
    deriv : int
        Derivative order (0 = smoothed values, 1 = first derivative, etc.)

    Returns
    -------
    coeffs : np.ndarray, shape (window_size,)
        Convolution coefficients to apply at each window position.
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")
    if window_size <= poly_order:
        raise ValueError(f"window_size ({window_size}) must be > poly_order ({poly_order})")

    half_win = window_size // 2
    # x positions relative to center: [-half_win, ..., 0, ..., half_win]
    pos = np.arange(-half_win, half_win + 1, dtype=float)

    # Vandermonde matrix: each row is [1, x, x^2, ..., x^poly_order]
    A = np.vander(pos, N=poly_order + 1, increasing=True)  # shape (window_size, poly_order+1)

    # Least-squares solution: B = (A^T A)^{-1} A^T
    # The coefficients for deriv-th derivative at center (pos=0) are:
    #   c = B[deriv, :] * deriv!
    # For deriv=0, c = B[0, :] which is the first row of the pseudoinverse.
    B = np.linalg.pinv(A)  # shape (poly_order+1, window_size)

    # Scale for derivative: deriv! factor (for deriv=0, this is 1)
    factorial = float(math.factorial(deriv))
    coeffs = B[deriv, :] * factorial
    return coeffs


def savitzky_golay(
    X: np.ndarray,
    window_size: int = 11,
    poly_order: int = 2,
    deriv: int = 0,
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to a matrix of spectra.

    Each row of X is treated as an independent spectrum. The filter is applied
    along the spectral (wavelength) axis. Edges are handled by padding with
    the nearest valid value (edge padding).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_bands)
        Matrix of raw reflectance spectra. Each row is one sample.
    window_size : int
        Sliding window width. Must be odd, >= 3. Default: 11.
    poly_order : int
        Polynomial degree for local fitting. Default: 2.
    deriv : int
        Derivative order. 0 = smoothed values, 1 = first derivative.

    Returns
    -------
    X_smooth : np.ndarray, shape (n_samples, n_bands)
        Smoothed spectra. Shape identical to input.

    Notes
    -----
    Using scipy.signal.savgol_filter would be equivalent but we implement
    from scratch to demonstrate the convolution approach explicitly.
    """
    coeffs = _sg_coefficients(window_size, poly_order, deriv)
    half_win = window_size // 2

    n_samples, n_bands = X.shape
    X_smooth = np.empty_like(X, dtype=np.float64)

    # Pad each spectrum at both edges by repeating the edge value
    # (mode='edge' equivalent)
    X_pad = np.pad(X, ((0, 0), (half_win, half_win)), mode="edge")

    for i in range(n_samples):
        spectrum = X_pad[i]
        for j in range(n_bands):
            window = spectrum[j: j + window_size]
            X_smooth[i, j] = np.dot(coeffs, window)

    return X_smooth


def savitzky_golay_fast(
    X: np.ndarray,
    window_size: int = 11,
    poly_order: int = 2,
    deriv: int = 0,
) -> np.ndarray:
    """
    Vectorized Savitzky-Golay smoothing using np.convolve for each sample.
    Equivalent to savitzky_golay() but faster for large datasets.

    The convolution is performed in 'same' mode, then edge corrections are
    applied using values computed from full-window positions only.
    """
    coeffs = _sg_coefficients(window_size, poly_order, deriv)
    half_win = window_size // 2

    n_samples, n_bands = X.shape
    X_smooth = np.empty_like(X, dtype=np.float64)

    # Reverse coefficients for convolution (np.convolve flips the kernel)
    coeffs_rev = coeffs[::-1]

    for i in range(n_samples):
        spectrum_pad = np.pad(X[i], (half_win, half_win), mode="edge")
        smoothed = np.convolve(spectrum_pad, coeffs_rev, mode="valid")
        X_smooth[i] = smoothed

    return X_smooth


# ─────────────────────────────────────────────────────────────────────────────
# Standard Normal Variate (SNV) transformation
# ─────────────────────────────────────────────────────────────────────────────


def snv(X: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate (SNV) transformation.

    Corrects for multiplicative scatter effects caused by differences in
    particle size, path length, and surface heterogeneity between samples.
    Each spectrum is independently mean-centered and divided by its standard
    deviation, producing a spectrum with mean=0 and std=1.

    Mathematically, for sample i:
        X_snv[i, :] = (X[i, :] - mean(X[i, :])) / std(X[i, :])

    This is applied per sample (row-wise), not across samples.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_bands)
        Spectra to transform. Can be raw or SG-smoothed.

    Returns
    -------
    X_snv : np.ndarray, shape (n_samples, n_bands)
        SNV-transformed spectra.

    Raises
    ------
    ValueError
        If any sample has zero standard deviation (constant spectrum).

    References
    ----------
    Barnes, R.J., Dhanoa, M.S., & Lister, S.J. (1989). Standard normal variate
    transformation and de-trending of near-infrared diffuse reflectance spectra.
    Applied Spectroscopy, 43(5), 772–777.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)      # (n_samples, 1)
    sigma = X.std(axis=1, ddof=1, keepdims=True)  # (n_samples, 1), unbiased std

    zero_std = (sigma == 0).flatten()
    if np.any(zero_std):
        raise ValueError(
            f"{zero_std.sum()} sample(s) have zero spectral variance and "
            "cannot be SNV-normalized. Remove or inspect these samples."
        )

    X_snv = (X - mu) / sigma
    return X_snv


# ─────────────────────────────────────────────────────────────────────────────
# Combined preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_spectra(
    X: np.ndarray,
    sg_window: int = 11,
    sg_poly: int = 2,
    apply_snv: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a spectral matrix.

    Step 1: Savitzky-Golay smoothing (removes sensor noise while
            preserving spectral features).
    Step 2: SNV transformation (removes multiplicative scatter effects).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_bands)
    sg_window : int
        SG window width (odd integer). Default: 11.
    sg_poly : int
        SG polynomial order. Default: 2.
    apply_snv : bool
        Whether to apply SNV after SG smoothing. Default: True.
    verbose : bool

    Returns
    -------
    X_proc : np.ndarray, shape (n_samples, n_bands)
    """
    if verbose:
        print(f"[preprocessing] Input shape: {X.shape}")

    X_sg = savitzky_golay_fast(X, window_size=sg_window, poly_order=sg_poly)

    if verbose:
        print(f"[preprocessing] After SG smoothing (window={sg_window}, poly={sg_poly}): {X_sg.shape}")

    if apply_snv:
        X_proc = snv(X_sg)
        if verbose:
            print(f"[preprocessing] After SNV: {X_proc.shape}")
    else:
        X_proc = X_sg

    return X_proc
