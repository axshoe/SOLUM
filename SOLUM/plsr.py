"""
plsr.py
=======
Partial Least Squares Regression (PLSR) implemented from scratch using the
NIPALS (Nonlinear Iterative Partial Least Squares) algorithm.

PLSR is the standard workhorse of chemometrics and soil spectroscopy. It finds
latent variables (components) that simultaneously maximize variance in X and
covariance with y. For high-dimensional spectral data (thousands of bands,
hundreds of samples), PLSR is far more numerically stable than OLS regression
and handles collinearity gracefully.

Why from scratch: using scikit-learn's PLSRegression would be simpler, but
implementing NIPALS explicitly forces a confrontation with what the algorithm
actually does — how it extracts latent structure, why deflation works, and why
the number of components is a meaningful hyperparameter rather than a tuning
knob to optimize blindly.

Reference:
  Wold, H. (1966). Estimation of principal components and related models by
  iterative least squares. In P.R. Krishnaiah (Ed.), Multivariate Analysis
  (pp. 391–420). Academic Press.

  Geladi, P., & Kowalski, B.R. (1986). Partial least-squares regression: a
  tutorial. Analytica Chimica Acta, 185, 1–17.
"""

import numpy as np
from typing import Optional, Tuple


class NIPALS_PLSR:
    """
    Partial Least Squares Regression via NIPALS.

    Fits a PLS model with `n_components` latent variables by alternately
    regressing X and y scores until convergence, then deflating both matrices.

    Parameters
    ----------
    n_components : int
        Number of PLS components (latent variables). Selected via LOO-CV
        on the training set; see `select_n_components`.
    max_iter : int
        Maximum NIPALS iterations per component. Default: 500.
    tol : float
        Convergence tolerance on the change in x-scores. Default: 1e-6.
    scale : bool
        If True, scale X columns to unit variance (in addition to centering).
        SNV-preprocessed spectra are already mean-centered per sample, but
        column-scaling further stabilizes numerical behavior. Default: True.

    Attributes (after fitting)
    --------------------------
    x_weights_ : np.ndarray, shape (n_bands, n_components)
        X loading weights (W matrix).
    x_loadings_ : np.ndarray, shape (n_bands, n_components)
        X loadings (P matrix).
    y_loadings_ : np.ndarray, shape (1, n_components)
        Y loadings (Q matrix, scalar per component for univariate y).
    x_scores_ : np.ndarray, shape (n_train, n_components)
        Training X scores (T matrix).
    coef_ : np.ndarray, shape (n_bands,)
        Regression coefficients in the original X space.
    x_mean_ : np.ndarray, shape (n_bands,)
    x_std_ : np.ndarray, shape (n_bands,)
    y_mean_ : float
    """

    def __init__(
        self,
        n_components: int = 10,
        max_iter: int = 500,
        tol: float = 1e-6,
        scale: bool = True,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NIPALS_PLSR":
        """
        Fit the PLSR model to training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_bands)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_bands = X.shape

        # ── centering and optional scaling ───────────────────────────────────
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0, ddof=1)
        self.x_std_[self.x_std_ == 0] = 1.0  # avoid division by zero

        self.y_mean_ = y.mean()

        X_c = X - self.x_mean_
        if self.scale:
            X_c = X_c / self.x_std_

        y_c = y - self.y_mean_

        # Storage
        W = np.zeros((n_bands, self.n_components))   # X weights
        P = np.zeros((n_bands, self.n_components))   # X loadings
        Q = np.zeros((1, self.n_components))          # Y loadings (scalar y)
        T = np.zeros((n_samples, self.n_components)) # X scores

        E = X_c.copy()  # residual X block
        f = y_c.copy()  # residual y block

        for k in range(self.n_components):
            # NIPALS inner loop for component k
            # Initialize x-weights as the cross-product of residuals
            w = E.T @ f
            w_norm = np.linalg.norm(w)
            if w_norm < 1e-12:
                # Residual exhausted; stop early
                self.n_components = k
                break
            w = w / w_norm

            # Compute x-scores
            t = E @ w  # (n_samples,)

            # Compute y-loadings (scalar for univariate y)
            q = (f @ t) / (t @ t)

            # Compute x-loadings
            p = (E.T @ t) / (t @ t)

            # Deflate X and y residuals
            E = E - np.outer(t, p)
            f = f - q * t

            # Store
            W[:, k] = w
            P[:, k] = p
            Q[0, k] = q
            T[:, k] = t

        self.x_weights_ = W[:, : self.n_components]
        self.x_loadings_ = P[:, : self.n_components]
        self.y_loadings_ = Q[:, : self.n_components]
        self.x_scores_ = T[:, : self.n_components]

        # Compute regression coefficients in original space:
        # B = W (P^T W)^{-1} Q^T
        # This allows direct prediction without re-running the score loop.
        PtW = self.x_loadings_.T @ self.x_weights_       # (n_comp, n_comp)
        B_latent = np.linalg.solve(PtW, self.y_loadings_.T)  # (n_comp, 1)
        self.coef_ = (self.x_weights_ @ B_latent).ravel()     # (n_bands,)

        # Scale coefficients back to original X space if scaling was applied
        if self.scale:
            self.coef_ = self.coef_ / self.x_std_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict SOC for new samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_bands)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        X_c = X - self.x_mean_
        return X_c @ self.coef_ + self.y_mean_

    def get_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Project X into the latent score space.

        Returns
        -------
        T : np.ndarray, shape (n_samples, n_components)
        """
        X = np.asarray(X, dtype=np.float64)
        X_c = X - self.x_mean_
        if self.scale:
            X_c = X_c / self.x_std_
        # Project via x_weights (not x_loadings)
        return X_c @ self.x_weights_


# ─────────────────────────────────────────────────────────────────────────────
# Component selection via leave-one-out cross-validation
# ─────────────────────────────────────────────────────────────────────────────


def select_n_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_components: int = 20,
    scale: bool = True,
    verbose: bool = False,
) -> int:
    """
    Select the optimal number of PLSR components via leave-one-out CV.

    For each candidate n_comp in [1, max_components], fits PLSR to
    (n_train - 1) samples and evaluates on the held-out sample.
    Repeats for all samples and computes RMSECV (root mean squared error
    of cross-validation). Returns the n_comp minimizing RMSECV.

    LOO-CV is expensive for large datasets. For n_train > 300, a block
    k-fold (k=5) is used instead to keep runtime tractable.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, n_bands)
    y_train : np.ndarray, shape (n_samples,)
    max_components : int
    scale : bool
    verbose : bool

    Returns
    -------
    n_components : int
        Optimal number of components.
    """
    n_samples = X_train.shape[0]
    rmsecv = np.full(max_components + 1, np.inf)

    if n_samples > 300:
        # 5-fold CV for efficiency
        k_folds = 5
        fold_size = n_samples // k_folds
        fold_indices = [
            np.arange(i * fold_size, (i + 1) * fold_size if i < k_folds - 1 else n_samples)
            for i in range(k_folds)
        ]
    else:
        # LOO-CV
        k_folds = n_samples
        fold_indices = [np.array([i]) for i in range(n_samples)]

    for n_comp in range(1, max_components + 1):
        sq_errors = []
        for fold_idx in fold_indices:
            train_idx = np.setdiff1d(np.arange(n_samples), fold_idx)
            X_tr = X_train[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_train[fold_idx]
            y_val = y_train[fold_idx]

            model = NIPALS_PLSR(n_components=n_comp, scale=scale)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            sq_errors.extend((y_val - y_pred) ** 2)

        rmsecv[n_comp] = np.sqrt(np.mean(sq_errors))

    best_n = int(np.argmin(rmsecv[1:]) + 1)

    if verbose:
        print(f"[PLSR] RMSECV by n_components:")
        for k in range(1, min(max_components + 1, 11)):
            marker = " <-- selected" if k == best_n else ""
            print(f"  n={k:2d}: RMSECV={rmsecv[k]:.4f}{marker}")

    return best_n
