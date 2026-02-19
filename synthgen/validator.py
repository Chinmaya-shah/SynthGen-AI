"""
Statistical Validation Module for SynthGen AI.

Computes comprehensive comparison metrics between original and
synthetic datasets to validate statistical fidelity.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class Validator:
    """Validates statistical similarity between original and synthetic data.

    Computes mean difference percentages, variance ratios, correlation
    matrix Frobenius norm difference, Kolmogorov-Smirnov tests, and
    categorical frequency comparisons.

    Attributes:
        report: Structured validation report dictionary.
    """

    def __init__(self) -> None:
        self.report: Dict[str, Any] = {}

    def compare_means(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        numerical_columns: List[str],
    ) -> Dict[str, float]:
        """Compute mean percentage difference per numerical column.

        Args:
            original: Original dataset.
            synthetic: Synthetic dataset.
            numerical_columns: Numerical column names.

        Returns:
            Dict mapping column name to mean percentage difference.
        """
        logger.info("Comparing means between original and synthetic data")
        results: Dict[str, float] = {}

        orig_means = original[numerical_columns].mean()
        synth_means = synthetic[numerical_columns].mean()

        for col in numerical_columns:
            orig_m = orig_means[col]
            synth_m = synth_means[col]
            if abs(orig_m) > 1e-10:
                pct_diff = abs(synth_m - orig_m) / abs(orig_m) * 100
            else:
                pct_diff = abs(synth_m - orig_m) * 100
            results[col] = round(float(pct_diff), 4)

        self.report["mean_difference_pct"] = results
        logger.info("Mean comparison complete")
        return results

    def compare_variances(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        numerical_columns: List[str],
    ) -> Dict[str, float]:
        """Compute variance ratio per numerical column.

        Ratio = synthetic_variance / original_variance.
        A ratio of 1.0 indicates perfect match.

        Args:
            original: Original dataset.
            synthetic: Synthetic dataset.
            numerical_columns: Numerical column names.

        Returns:
            Dict mapping column name to variance ratio.
        """
        logger.info("Comparing variances between original and synthetic data")
        results: Dict[str, float] = {}

        orig_vars = original[numerical_columns].var()
        synth_vars = synthetic[numerical_columns].var()

        for col in numerical_columns:
            orig_v = orig_vars[col]
            if abs(orig_v) > 1e-10:
                ratio = float(synth_vars[col]) / float(orig_v)
            else:
                ratio = float("inf") if synth_vars[col] > 0 else 1.0
            results[col] = round(ratio, 4)

        self.report["variance_ratio"] = results
        logger.info("Variance comparison complete")
        return results

    def correlation_difference(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        numerical_columns: List[str],
    ) -> float:
        """Compute Frobenius norm difference between correlation matrices.

        Args:
            original: Original dataset.
            synthetic: Synthetic dataset.
            numerical_columns: Numerical column names.

        Returns:
            Frobenius norm of the difference between correlation matrices.
        """
        logger.info("Computing correlation matrix Frobenius norm difference")

        if not numerical_columns:
            logger.warning("No numerical columns for correlation comparison")
            self.report["correlation_frobenius_norm"] = 0.0
            return 0.0

        orig_corr = original[numerical_columns].corr().values
        synth_corr = synthetic[numerical_columns].corr().values

        # Replace any NaN with 0 for comparison
        orig_corr = np.nan_to_num(orig_corr)
        synth_corr = np.nan_to_num(synth_corr)

        frob_norm = float(np.linalg.norm(orig_corr - synth_corr, "fro"))
        self.report["correlation_frobenius_norm"] = round(frob_norm, 6)

        logger.info("Frobenius norm difference: %.6f", frob_norm)
        return frob_norm

    def ks_test(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        numerical_columns: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Perform Kolmogorov-Smirnov test per numerical column.

        Uses ``scipy.stats.ks_2samp`` to compare distributions.

        Args:
            original: Original dataset.
            synthetic: Synthetic dataset.
            numerical_columns: Numerical column names.

        Returns:
            Dict mapping column name to KS statistic and p-value.
        """
        logger.info("Running Kolmogorov-Smirnov tests")
        results: Dict[str, Dict[str, float]] = {}

        for col in numerical_columns:
            ks_stat, p_value = stats.ks_2samp(
                original[col].dropna().values,
                synthetic[col].dropna().values,
            )
            results[col] = {
                "ks_statistic": round(float(ks_stat), 6),
                "p_value": round(float(p_value), 6),
            }
            logger.info(
                "KS test '%s': statistic=%.6f, p_value=%.6f",
                col,
                ks_stat,
                p_value,
            )

        self.report["ks_tests"] = results
        return results

    def compare_categories(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        categorical_columns: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Compare categorical frequency distributions.

        Args:
            original: Original dataset.
            synthetic: Synthetic dataset.
            categorical_columns: Categorical column names.

        Returns:
            Dict mapping column name to frequency comparison data.
        """
        logger.info("Comparing categorical frequency distributions")
        results: Dict[str, Dict[str, Any]] = {}

        for col in categorical_columns:
            orig_freq = original[col].value_counts(normalize=True)
            synth_freq = synthetic[col].value_counts(normalize=True)

            # Align categories
            all_categories = sorted(
                set(orig_freq.index.tolist() + synth_freq.index.tolist())
            )

            comparison: Dict[str, Dict[str, float]] = {}
            for cat in all_categories:
                orig_p = float(orig_freq.get(cat, 0))
                synth_p = float(synth_freq.get(cat, 0))
                comparison[str(cat)] = {
                    "original_proportion": round(orig_p, 6),
                    "synthetic_proportion": round(synth_p, 6),
                    "absolute_difference": round(abs(orig_p - synth_p), 6),
                }

            results[col] = comparison

        self.report["categorical_comparison"] = results
        logger.info("Categorical comparison complete")
        return results

    def generate_report(
        self,
        output_path: str,
        privacy_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate and save a structured validation report as JSON.

        Args:
            output_path: File path for the JSON report.
            privacy_metrics: Optional privacy enforcement metrics.

        Returns:
            The complete validation report dictionary.
        """
        if privacy_metrics:
            self.report["privacy_metrics"] = privacy_metrics

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(self.report, fh, indent=2, default=str)

        logger.info("Validation report saved to %s", output_path)
        return self.report

    def run(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        numerical_columns: List[str],
        categorical_columns: List[str],
        output_path: str,
        privacy_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the full validation pipeline.

        Args:
            original: Original dataset.
            synthetic: Synthetic dataset.
            numerical_columns: Numerical column names.
            categorical_columns: Categorical column names.
            output_path: Path for the JSON report.
            privacy_metrics: Optional privacy metrics.

        Returns:
            Structured validation report.
        """
        self.compare_means(original, synthetic, numerical_columns)
        self.compare_variances(original, synthetic, numerical_columns)
        self.correlation_difference(original, synthetic, numerical_columns)
        self.ks_test(original, synthetic, numerical_columns)
        self.compare_categories(original, synthetic, categorical_columns)
        return self.generate_report(output_path, privacy_metrics)
