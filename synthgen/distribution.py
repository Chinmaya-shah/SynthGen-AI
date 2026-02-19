"""
Distribution Learning Module for SynthGen AI.

Learns multivariate Gaussian distributions for numerical features
and probability distributions for categorical features.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DistributionLearner:
    """Learns statistical distributions from structured tabular data.

    For numerical features, fits a multivariate Gaussian by computing
    the mean vector (μ) and covariance matrix (Σ), validating that Σ
    is positive semi-definite.

    For categorical features, computes normalized probability
    distributions per column.

    Attributes:
        mean_vector: Mean vector for numerical columns.
        covariance_matrix: Covariance matrix for numerical columns.
        categorical_distributions: Per-column category probability maps.
    """

    def __init__(self) -> None:
        self.mean_vector: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None
        self.numerical_columns: List[str] = []
        self.categorical_distributions: Dict[str, Dict[str, Any]] = {}

    def fit_numerical_distribution(
        self, df: pd.DataFrame, numerical_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a multivariate Gaussian to the numerical columns.

        Computes the mean vector μ and covariance matrix Σ,
        then validates that Σ is positive semi-definite.

        Args:
            df: Input DataFrame.
            numerical_columns: List of numerical column names.

        Returns:
            Tuple of (mean_vector, covariance_matrix).

        Raises:
            ValueError: If covariance matrix computation fails.
        """
        self.numerical_columns = numerical_columns

        if not numerical_columns:
            logger.warning("No numerical columns provided for distribution fitting")
            self.mean_vector = np.array([])
            self.covariance_matrix = np.array([[]])
            return self.mean_vector, self.covariance_matrix

        logger.info(
            "Fitting multivariate Gaussian on %d numerical columns",
            len(numerical_columns),
        )

        data = df[numerical_columns].values.astype(np.float64)

        # Compute mean vector μ
        self.mean_vector = np.mean(data, axis=0)
        logger.info("Mean vector computed: shape=%s", self.mean_vector.shape)

        # Compute covariance matrix Σ
        self.covariance_matrix = np.cov(data, rowvar=False)

        # Handle single-column edge case
        if self.covariance_matrix.ndim == 0:
            self.covariance_matrix = np.array([[float(self.covariance_matrix)]])

        logger.info(
            "Covariance matrix computed: shape=%s", self.covariance_matrix.shape
        )

        # Validate positive semi-definiteness
        self.covariance_matrix = self._ensure_positive_semi_definite(
            self.covariance_matrix
        )

        return self.mean_vector, self.covariance_matrix

    def _ensure_positive_semi_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Validate and correct the covariance matrix to be PSD.

        Uses eigenvalue decomposition. If any eigenvalue is negative,
        clamps it to zero and reconstructs the matrix.

        Args:
            matrix: Square covariance matrix.

        Returns:
            Corrected positive semi-definite matrix.

        Raises:
            ValueError: If the matrix cannot be corrected.
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"Eigenvalue decomposition failed on covariance matrix: {exc}"
            ) from exc

        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue < 0:
            logger.warning(
                "Covariance matrix is not PSD (min eigenvalue=%.6e). "
                "Applying nearest PSD correction.",
                min_eigenvalue,
            )
            # Clamp negative eigenvalues to a small positive epsilon
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Ensure symmetry
            matrix = (matrix + matrix.T) / 2.0
            logger.info("Covariance matrix corrected to nearest PSD matrix")
        else:
            logger.info(
                "Covariance matrix is positive semi-definite "
                "(min eigenvalue=%.6e)",
                min_eigenvalue,
            )

        return matrix

    def fit_categorical_distribution(
        self, df: pd.DataFrame, categorical_columns: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Learn probability distributions for categorical columns.

        For each categorical column, computes normalized probabilities
        from observed frequencies.

        Args:
            df: Input DataFrame.
            categorical_columns: List of categorical column names.

        Returns:
            Dictionary mapping column names to their distribution info
            (categories and probabilities).
        """
        if not categorical_columns:
            logger.warning("No categorical columns provided")
            return {}

        logger.info(
            "Learning categorical distributions for %d columns",
            len(categorical_columns),
        )

        for col in categorical_columns:
            value_counts = df[col].value_counts(normalize=True)
            categories = value_counts.index.tolist()
            probabilities = value_counts.values.astype(np.float64)

            # Normalize to ensure probabilities sum to exactly 1.0
            probabilities = probabilities / probabilities.sum()

            self.categorical_distributions[col] = {
                "categories": [str(c) for c in categories],
                "probabilities": probabilities.tolist(),
            }

            logger.info(
                "Column '%s': %d unique categories", col, len(categories)
            )

        return self.categorical_distributions

    def get_distributions(self) -> Dict[str, Any]:
        """Return all learned distributions.

        Returns:
            Dictionary with numerical and categorical distributions.
        """
        return {
            "numerical": {
                "mean_vector": (
                    self.mean_vector.tolist()
                    if self.mean_vector is not None
                    else []
                ),
                "covariance_matrix": (
                    self.covariance_matrix.tolist()
                    if self.covariance_matrix is not None
                    else []
                ),
                "columns": self.numerical_columns,
            },
            "categorical": self.categorical_distributions,
        }
