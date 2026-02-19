"""
Data Profiling Module for SynthGen AI.

Automatically detects column types, computes summary statistics,
and builds correlation matrices for downstream distribution learning.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProfiler:
    """Profiles structured tabular data to extract statistical metadata.

    Attributes:
        df: The DataFrame to profile.
        numerical_columns: List of detected numerical column names.
        categorical_columns: List of detected categorical column names.
        summary_stats: Dictionary of computed summary statistics.
        correlation_matrix: Correlation matrix for numerical columns.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.summary_stats: Dict[str, Any] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None

    def detect_column_types(self) -> Tuple[List[str], List[str]]:
        """Detect and classify columns as numerical or categorical.

        Returns:
            Tuple of (numerical_columns, categorical_columns).
        """
        self.numerical_columns = (
            self.df.select_dtypes(include=[np.number]).columns.tolist()
        )
        self.categorical_columns = (
            self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        )

        logger.info(
            "Detected %d numerical columns: %s",
            len(self.numerical_columns),
            self.numerical_columns,
        )
        logger.info(
            "Detected %d categorical columns: %s",
            len(self.categorical_columns),
            self.categorical_columns,
        )

        return self.numerical_columns, self.categorical_columns

    def compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for all columns.

        For numerical columns: mean, variance, std, min, max.
        For categorical columns: frequency distributions.

        Returns:
            Dictionary containing per-column statistics.
        """
        logger.info("Computing summary statistics")

        stats: Dict[str, Any] = {"numerical": {}, "categorical": {}}

        # Numerical statistics — vectorized computation
        if self.numerical_columns:
            num_df = self.df[self.numerical_columns]
            means = num_df.mean()
            variances = num_df.var()
            stds = num_df.std()
            mins = num_df.min()
            maxs = num_df.max()

            for col in self.numerical_columns:
                stats["numerical"][col] = {
                    "mean": float(means[col]),
                    "variance": float(variances[col]),
                    "std": float(stds[col]),
                    "min": float(mins[col]),
                    "max": float(maxs[col]),
                }

        # Categorical frequency distributions
        for col in self.categorical_columns:
            freq = self.df[col].value_counts(normalize=False).to_dict()
            freq_pct = self.df[col].value_counts(normalize=True).to_dict()
            stats["categorical"][col] = {
                "frequencies": {str(k): int(v) for k, v in freq.items()},
                "proportions": {str(k): float(v) for k, v in freq_pct.items()},
                "unique_count": int(self.df[col].nunique()),
            }

        self.summary_stats = stats
        logger.info("Summary statistics computed for all columns")
        return stats

    def compute_correlation_matrix(self) -> pd.DataFrame:
        """Compute Pearson correlation matrix for numerical columns.

        Returns:
            DataFrame containing the correlation matrix.
        """
        if not self.numerical_columns:
            logger.warning("No numerical columns found for correlation computation")
            self.correlation_matrix = pd.DataFrame()
            return self.correlation_matrix

        logger.info("Computing correlation matrix for %d numerical columns",
                     len(self.numerical_columns))

        self.correlation_matrix = self.df[self.numerical_columns].corr()
        return self.correlation_matrix

    def get_metadata(self) -> Dict[str, Any]:
        """Return a consolidated metadata dictionary.

        Returns:
            Dictionary containing column types, summary stats,
            and correlation matrix.
        """
        return {
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "summary_stats": self.summary_stats,
            "correlation_matrix": (
                self.correlation_matrix.to_dict()
                if self.correlation_matrix is not None
                else {}
            ),
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
        }

    def run(self) -> Dict[str, Any]:
        """Execute the full profiling pipeline.

        Returns:
            Consolidated metadata dictionary.
        """
        self.detect_column_types()
        self.compute_summary_stats()
        self.compute_correlation_matrix()
        return self.get_metadata()
