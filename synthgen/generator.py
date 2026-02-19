"""
Synthetic Data Generation Module for SynthGen AI.

Generates synthetic numerical features via multivariate normal sampling
and categorical features via learned probability distributions.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticGenerator:
    """Generates synthetic tabular data from learned distributions.

    Uses multivariate Gaussian sampling for numerical columns and
    weighted random choice for categorical columns, then combines
    both into a unified synthetic dataset.

    Attributes:
        random_seed: Seed for reproducibility.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def generate_numerical_data(
        self,
        mean_vector: np.ndarray,
        covariance_matrix: np.ndarray,
        size: int,
        column_names: List[str],
        column_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic numerical features via multivariate normal sampling.

        Uses ``numpy.random.multivariate_normal`` as specified in the PRD.
        Values are clipped to observed [min, max] ranges when statistics
        are provided.

        Args:
            mean_vector: Mean vector μ for the multivariate Gaussian.
            covariance_matrix: Covariance matrix Σ (must be PSD).
            size: Number of synthetic rows to generate.
            column_names: Names for the generated numerical columns.
            column_stats: Optional dict of per-column stats with 'min'/'max'
                keys for value clipping.

        Returns:
            DataFrame with synthetic numerical data.
        """
        if len(mean_vector) == 0:
            logger.warning("Empty mean vector; skipping numerical generation")
            return pd.DataFrame()

        logger.info("Generating %d synthetic numerical rows", size)

        # Use numpy.random.multivariate_normal as specified by PRD
        synthetic_data = np.random.multivariate_normal(
            mean=mean_vector,
            cov=covariance_matrix,
            size=size,
        )

        df_synthetic = pd.DataFrame(synthetic_data, columns=column_names)

        # Clip values to observed min/max ranges
        if column_stats:
            for col in column_names:
                if col in column_stats:
                    col_min = column_stats[col].get("min")
                    col_max = column_stats[col].get("max")
                    if col_min is not None and col_max is not None:
                        df_synthetic[col] = df_synthetic[col].clip(
                            lower=col_min, upper=col_max
                        )

        # Prevent unrealistic negative values for columns that should
        # be non-negative (heuristic: if original min >= 0, clip to 0)
        if column_stats:
            for col in column_names:
                if col in column_stats:
                    if column_stats[col].get("min", -1) >= 0:
                        df_synthetic[col] = df_synthetic[col].clip(lower=0)

        logger.info(
            "Numerical data generated: shape=%s", df_synthetic.shape
        )
        return df_synthetic

    def generate_categorical_data(
        self,
        categorical_distributions: Dict[str, Dict[str, Any]],
        size: int,
    ) -> pd.DataFrame:
        """Generate synthetic categorical features via weighted random choice.

        Uses ``numpy.random.choice`` with learned probabilities as
        specified in the PRD.

        Args:
            categorical_distributions: Per-column distribution info with
                'categories' and 'probabilities' keys.
            size: Number of synthetic rows to generate.

        Returns:
            DataFrame with synthetic categorical data.
        """
        if not categorical_distributions:
            logger.warning("No categorical distributions; skipping categorical generation")
            return pd.DataFrame()

        logger.info("Generating %d synthetic categorical rows", size)

        cat_data: Dict[str, np.ndarray] = {}

        for col, dist_info in categorical_distributions.items():
            categories = dist_info["categories"]
            probabilities = np.array(dist_info["probabilities"], dtype=np.float64)

            # Ensure probabilities sum to exactly 1.0
            probabilities = probabilities / probabilities.sum()

            # Use numpy.random.choice as specified by PRD
            cat_data[col] = np.random.choice(
                categories, size=size, p=probabilities
            )

        df_categorical = pd.DataFrame(cat_data)
        logger.info(
            "Categorical data generated: shape=%s", df_categorical.shape
        )
        return df_categorical

    def combine_data(
        self,
        numerical_df: pd.DataFrame,
        categorical_df: pd.DataFrame,
        original_column_order: List[str],
    ) -> pd.DataFrame:
        """Combine numerical and categorical DataFrames into a unified dataset.

        Preserves the original column ordering from the source data.

        Args:
            numerical_df: DataFrame with synthetic numerical columns.
            categorical_df: DataFrame with synthetic categorical columns.
            original_column_order: Column order from the original dataset.

        Returns:
            Combined DataFrame with columns in original order.
        """
        logger.info("Combining numerical and categorical synthetic data")

        if numerical_df.empty and categorical_df.empty:
            logger.warning("Both numerical and categorical DataFrames are empty")
            return pd.DataFrame()

        if numerical_df.empty:
            combined = categorical_df.copy()
        elif categorical_df.empty:
            combined = numerical_df.copy()
        else:
            combined = pd.concat(
                [numerical_df, categorical_df], axis=1
            )

        # Reorder columns to match original dataset
        available_cols = [
            c for c in original_column_order if c in combined.columns
        ]
        combined = combined[available_cols]

        logger.info("Combined synthetic dataset: shape=%s", combined.shape)
        return combined
