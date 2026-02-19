"""
Privacy Protection Layer for SynthGen AI.

Ensures synthetic data does not replicate original records through
exact row duplication detection and normalized distance-based
similarity checks.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PrivacyGuard:
    """Enforces privacy constraints between original and synthetic datasets.

    Detects exact row duplicates and rows with high similarity using
    standardized Euclidean distance on numerical features, then
    triggers regeneration for any violating rows.

    Attributes:
        similarity_threshold: Minimum normalized Euclidean distance
            threshold. Rows with min-distance below this are flagged.
        max_attempts: Maximum regeneration attempts.
        regeneration_count: Running count of regenerated rows.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.05,
        max_attempts: int = 5,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_attempts = max_attempts
        self.regeneration_count: int = 0

    def check_exact_duplicates(
        self,
        original_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
    ) -> Tuple[bool, List[int]]:
        """Detect exact row duplicates between original and synthetic data.

        Uses string-based row hashing for fast set-membership checks.

        Args:
            original_df: Original dataset.
            synthetic_df: Synthetic dataset.

        Returns:
            Tuple of (has_duplicates, list of duplicate synthetic row indices).
        """
        logger.info("Checking for exact row duplicates")

        # Convert to rounded string representations for exact matching
        # Round numerical columns to avoid floating-point precision issues
        orig_rounded = original_df.copy()
        synth_rounded = synthetic_df.copy()

        for col in orig_rounded.select_dtypes(include=[np.number]).columns:
            orig_rounded[col] = orig_rounded[col].round(6)
        for col in synth_rounded.select_dtypes(include=[np.number]).columns:
            synth_rounded[col] = synth_rounded[col].round(6)

        original_set = set(
            orig_rounded.astype(str).apply(lambda row: "|".join(row), axis=1)
        )
        synthetic_rows = synth_rounded.astype(str).apply(
            lambda row: "|".join(row), axis=1
        )

        duplicate_indices = [
            idx for idx, row in synthetic_rows.items() if row in original_set
        ]

        has_dupes = len(duplicate_indices) > 0
        if has_dupes:
            logger.warning(
                "Found %d exact duplicate rows in synthetic data",
                len(duplicate_indices),
            )
        else:
            logger.info("No exact duplicate rows detected")

        return has_dupes, duplicate_indices

    def check_similarity(
        self,
        original_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        numerical_columns: List[str],
        sample_size: int = 5000,
    ) -> Tuple[List[int], List[float]]:
        """Check similarity using standardized Euclidean distance.

        Standardizes features (Z-score) then computes minimum Euclidean
        distance from each synthetic row to sampled original rows.
        Rows closer than the threshold are flagged.

        Args:
            original_df: Original dataset.
            synthetic_df: Synthetic dataset.
            numerical_columns: Numerical column names.
            sample_size: Number of original rows to sample for comparison
                (for memory efficiency with large datasets).

        Returns:
            Tuple of (flagged_indices, min_distance_scores).
        """
        if not numerical_columns:
            logger.info("No numerical columns for similarity check; skipping")
            return [], []

        logger.info(
            "Running distance-based similarity check (threshold=%.4f)",
            self.similarity_threshold,
        )

        orig_num = original_df[numerical_columns].values.astype(np.float64)
        synth_num = synthetic_df[numerical_columns].values.astype(np.float64)

        # Standardize features using original data statistics
        means = np.mean(orig_num, axis=0)
        stds = np.std(orig_num, axis=0)
        stds = np.where(stds == 0, 1.0, stds)  # avoid division by zero

        orig_standardized = (orig_num - means) / stds
        synth_standardized = (synth_num - means) / stds

        # Sample original rows if dataset is large
        if len(orig_standardized) > sample_size:
            sample_indices = np.random.choice(
                len(orig_standardized), size=sample_size, replace=False
            )
            orig_sample = orig_standardized[sample_indices]
        else:
            orig_sample = orig_standardized

        flagged_indices: List[int] = []
        min_distances: List[float] = []

        # Process synthetic rows in batches for memory efficiency
        batch_size = 1000
        n_synth = len(synth_standardized)

        for start_idx in range(0, n_synth, batch_size):
            end_idx = min(start_idx + batch_size, n_synth)
            batch = synth_standardized[start_idx:end_idx]

            # Compute pairwise squared distances: ||a - b||^2
            # = ||a||^2 + ||b||^2 - 2 * a.b
            batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)
            orig_sq = np.sum(orig_sample ** 2, axis=1, keepdims=True).T
            dist_sq = batch_sq + orig_sq - 2 * (batch @ orig_sample.T)
            dist_sq = np.maximum(dist_sq, 0)  # numerical stability

            # Normalize distances by number of features
            min_dist = np.sqrt(np.min(dist_sq, axis=1)) / np.sqrt(len(numerical_columns))

            for i, d in enumerate(min_dist):
                global_idx = start_idx + i
                min_distances.append(float(d))
                if d < self.similarity_threshold:
                    flagged_indices.append(global_idx)

        logger.info(
            "Similarity check complete: %d rows flagged out of %d "
            "(avg min-distance: %.4f)",
            len(flagged_indices),
            n_synth,
            np.mean(min_distances) if min_distances else 0.0,
        )

        return flagged_indices, min_distances

    def enforce_privacy(
        self,
        original_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        numerical_columns: List[str],
        mean_vector: np.ndarray,
        covariance_matrix: np.ndarray,
        column_stats: Optional[Dict[str, Dict[str, float]]] = None,
        categorical_distributions: Optional[Dict[str, Dict[str, Any]]] = None,
        categorical_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enforce privacy by regenerating violating rows.

        Iteratively checks for exact duplicates and high-similarity rows,
        re-generating them until they pass or max attempts are reached.

        Args:
            original_df: Original dataset.
            synthetic_df: Synthetic dataset (modified in place).
            numerical_columns: Numerical column names.
            mean_vector: Mean vector for numerical regeneration.
            covariance_matrix: Covariance matrix for numerical regeneration.
            column_stats: Per-column min/max stats for clipping.
            categorical_distributions: Category probabilities for regeneration.
            categorical_columns: Categorical column names.

        Returns:
            Tuple of (privacy-safe synthetic DataFrame, metrics dict).
        """
        logger.info("Enforcing privacy constraints")
        self.regeneration_count = 0
        total_flagged = 0
        all_flagged: List[int] = []

        for attempt in range(self.max_attempts):
            # Check exact duplicates
            has_dupes, dupe_indices = self.check_exact_duplicates(
                original_df, synthetic_df
            )

            # Check similarity
            flagged_indices, _ = self.check_similarity(
                original_df, synthetic_df, numerical_columns
            )

            # Merge all violating indices
            all_flagged = list(set(dupe_indices + flagged_indices))

            if not all_flagged:
                logger.info(
                    "Privacy enforcement passed on attempt %d", attempt + 1
                )
                break

            total_flagged += len(all_flagged)
            logger.info(
                "Attempt %d: Regenerating %d flagged rows",
                attempt + 1,
                len(all_flagged),
            )

            # Regenerate flagged rows
            n_regen = len(all_flagged)
            self.regeneration_count += n_regen

            if len(numerical_columns) > 0 and len(mean_vector) > 0:
                new_numerical = np.random.multivariate_normal(
                    mean=mean_vector, cov=covariance_matrix, size=n_regen
                )
                new_num_df = pd.DataFrame(
                    new_numerical, columns=numerical_columns
                )

                # Clip values
                if column_stats:
                    for col in numerical_columns:
                        if col in column_stats:
                            stats = column_stats[col]
                            new_num_df[col] = new_num_df[col].clip(
                                lower=stats.get("min", None),
                                upper=stats.get("max", None),
                            )
                            if stats.get("min", -1) >= 0:
                                new_num_df[col] = new_num_df[col].clip(lower=0)

                for i, idx in enumerate(all_flagged):
                    for col in numerical_columns:
                        synthetic_df.at[idx, col] = new_num_df.iloc[i][col]

            if categorical_distributions and categorical_columns:
                for col in categorical_columns:
                    if col in categorical_distributions:
                        dist = categorical_distributions[col]
                        probs = np.array(dist["probabilities"])
                        probs = probs / probs.sum()
                        new_cats = np.random.choice(
                            dist["categories"], size=n_regen, p=probs
                        )
                        for i, idx in enumerate(all_flagged):
                            synthetic_df.at[idx, col] = new_cats[i]
        else:
            logger.warning(
                "Max regeneration attempts (%d) reached. "
                "%d rows may still have high similarity.",
                self.max_attempts,
                len(all_flagged),
            )

        metrics = {
            "total_regenerated_rows": self.regeneration_count,
            "total_flagged_rows": total_flagged,
            "similarity_threshold": self.similarity_threshold,
            "max_attempts": self.max_attempts,
            "privacy_check_passed": self.regeneration_count == 0
            or len(all_flagged) == 0,
        }

        logger.info(
            "Privacy enforcement complete. Total regenerated: %d",
            self.regeneration_count,
        )
        return synthetic_df, metrics
