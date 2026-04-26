"""
rf_model.py
===========
Random Forest regression wrapper using scikit-learn.

Unlike PLSR, Random Forest is not implemented from scratch here. The decision
is deliberate: ensemble methods involve bootstrapping, tree construction, and
aggregation steps that are individually well-understood but collectively
represent thousands of lines of optimized code. Implementing them from scratch
would not add clarity and would introduce numerical differences relative to the
established benchmark. scikit-learn's RandomForestRegressor is the de-facto
standard and is used across the spectroscopy literature.

What this module provides is a consistent interface to match the NIPALS_PLSR
class, plus grid search for hyperparameter tuning (n_estimators, max_features).

Key hyperparameters:
  n_estimators : number of trees. More is better up to a point; runtime scales
    linearly. We use 200 as a default with grid search over [100, 200, 500].
  max_features : fraction or number of features considered at each split.
    Spectral data is highly collinear, so "sqrt" or "log2" prevents individual
    trees from exploiting correlated bands in identical ways.

Reference:
  Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold


# ─────────────────────────────────────────────────────────────────────────────
# Default hyperparameter grid
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_features": ["sqrt", "log2", 0.1, 0.3],
    "min_samples_leaf": [1, 2, 5],
}

# Lighter grid for faster runs (e.g., during debugging)
LIGHT_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_features": ["sqrt", 0.1],
}


# ─────────────────────────────────────────────────────────────────────────────
# RF model class
# ─────────────────────────────────────────────────────────────────────────────


class RFModel:
    """
    Random Forest regressor for SOC prediction from Vis-NIR spectra.

    Wraps scikit-learn's RandomForestRegressor with:
    - Grid search hyperparameter tuning
    - Consistent predict() interface matching NIPALS_PLSR
    - SHAP value support via the underlying sklearn model

    Parameters
    ----------
    n_estimators : int
        Default number of trees (used when grid_search=False). Default: 200.
    max_features : str or float
        Default max_features (used when grid_search=False). Default: 'sqrt'.
    random_state : int
        Seed for reproducibility. Default: 42.
    n_jobs : int
        Parallel jobs (-1 = all cores). Default: -1.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: Optional[RandomForestRegressor] = None
        self.best_params_: Optional[Dict] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        grid_search: bool = True,
        param_grid: Optional[Dict] = None,
        cv_folds: int = 5,
        verbose: bool = False,
    ) -> "RFModel":
        """
        Fit Random Forest to training data, optionally with grid search CV.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_bands)
        y_train : np.ndarray, shape (n_samples,)
        grid_search : bool
            If True, tune hyperparameters via 5-fold CV grid search.
        param_grid : dict or None
            Parameter grid for grid search. Defaults to LIGHT_PARAM_GRID.
        cv_folds : int
            Number of CV folds for grid search. Default: 5.
        verbose : bool

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()

        if grid_search:
            if param_grid is None:
                param_grid = LIGHT_PARAM_GRID

            base_rf = RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

            gs = GridSearchCV(
                base_rf,
                param_grid,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=1 if verbose else 0,
                refit=True,
            )
            gs.fit(X_train, y_train)
            self.model_ = gs.best_estimator_
            self.best_params_ = gs.best_params_

            if verbose:
                print(f"[RF] Best params: {self.best_params_}")
                print(f"[RF] Best CV RMSE: {-gs.best_score_:.4f}")
        else:
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            self.model_.fit(X_train, y_train)
            self.best_params_ = {
                "n_estimators": self.n_estimators,
                "max_features": self.max_features,
            }

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict SOC values.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_bands)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model_.predict(np.asarray(X, dtype=np.float64))

    def feature_importances(self) -> np.ndarray:
        """
        Return RF feature importances (mean decrease in impurity).

        Returns
        -------
        importances : np.ndarray, shape (n_bands,)
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        return self.model_.feature_importances_

    @property
    def sklearn_model(self) -> RandomForestRegressor:
        """Access underlying sklearn model (e.g., for SHAP)."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        return self.model_
