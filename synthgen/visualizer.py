"""
Visualization Module for SynthGen AI.

Generates comparative plots between original and synthetic datasets
including histograms, correlation heatmaps, and categorical bar charts.
"""

import logging
import os
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI usage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


class Visualizer:
    """Generates comparison visualizations between original and synthetic data.

    All plots are saved as PNG files to the configured output directory.

    Attributes:
        output_dir: Directory where plot files are saved.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_histograms(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        numerical_columns: List[str],
    ) -> List[str]:
        """Generate overlaid histogram comparisons for numerical columns.

        Each numerical column gets its own subplot comparing the
        original vs synthetic distribution.

        Args:
            original: Original dataset.
            synthetic: Synthetic dataset.
            numerical_columns: List of numerical column names.

        Returns:
            List of saved file paths.
        """
        if not numerical_columns:
            logger.warning("No numerical columns for histogram plotting")
            return []

        logger.info("Generating histogram comparisons for %d columns",
                     len(numerical_columns))

        n_cols = min(3, len(numerical_columns))
        n_rows = (len(numerical_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            squeeze=False,
        )

        for idx, col in enumerate(numerical_columns):
            row_idx = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row_idx][col_idx]

            ax.hist(
                original[col].dropna(), bins=50, alpha=0.5,
                label="Original", color="#2196F3", density=True,
            )
            ax.hist(
                synthetic[col].dropna(), bins=50, alpha=0.5,
                label="Synthetic", color="#FF5722", density=True,
            )
            ax.set_title(f"{col}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(len(numerical_columns), n_rows * n_cols):
            row_idx = idx // n_cols
            col_idx = idx % n_cols
            axes[row_idx][col_idx].set_visible(False)

        fig.suptitle("Histogram Comparison: Original vs Synthetic",
                      fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, "histogram_comparison.png")
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

        logger.info("Histogram comparison saved to %s", filepath)
        return [filepath]

    def plot_heatmaps(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        numerical_columns: List[str],
    ) -> List[str]:
        """Generate side-by-side correlation heatmap comparison.

        Args:
            original: Original dataset.
            synthetic: Synthetic dataset.
            numerical_columns: List of numerical column names.

        Returns:
            List of saved file paths.
        """
        if not numerical_columns or len(numerical_columns) < 2:
            logger.warning("Insufficient numerical columns for heatmap")
            return []

        logger.info("Generating correlation heatmap comparison")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        orig_corr = original[numerical_columns].corr()
        synth_corr = synthetic[numerical_columns].corr()

        # Determine shared color scale
        vmin = min(orig_corr.min().min(), synth_corr.min().min())
        vmax = max(orig_corr.max().max(), synth_corr.max().max())

        sns.heatmap(
            orig_corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax1, vmin=vmin, vmax=vmax, square=True,
            annot_kws={"size": 7},
        )
        ax1.set_title("Original Data Correlation", fontweight="bold")

        sns.heatmap(
            synth_corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax2, vmin=vmin, vmax=vmax, square=True,
            annot_kws={"size": 7},
        )
        ax2.set_title("Synthetic Data Correlation", fontweight="bold")

        fig.suptitle("Correlation Heatmap Comparison",
                      fontsize=14, fontweight="bold")
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, "correlation_heatmap.png")
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

        logger.info("Correlation heatmap saved to %s", filepath)
        return [filepath]

    def plot_category_distributions(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        categorical_columns: List[str],
    ) -> List[str]:
        """Generate grouped bar charts comparing categorical distributions.

        Args:
            original: Original dataset.
            synthetic: Synthetic dataset.
            categorical_columns: List of categorical column names.

        Returns:
            List of saved file paths.
        """
        if not categorical_columns:
            logger.warning("No categorical columns for distribution plotting")
            return []

        logger.info("Generating categorical distribution comparisons for %d columns",
                     len(categorical_columns))

        saved_paths: List[str] = []

        for col in categorical_columns:
            orig_counts = original[col].value_counts(normalize=True).sort_index()
            synth_counts = synthetic[col].value_counts(normalize=True).sort_index()

            # Align categories
            all_cats = sorted(
                set(orig_counts.index.tolist() + synth_counts.index.tolist())
            )
            orig_vals = [float(orig_counts.get(c, 0)) for c in all_cats]
            synth_vals = [float(synth_counts.get(c, 0)) for c in all_cats]

            # Truncate labels if too many categories
            display_cats = all_cats
            if len(all_cats) > 20:
                # Show top 20 by original frequency
                top_cats = original[col].value_counts().head(20).index.tolist()
                display_cats = sorted(top_cats)
                orig_vals = [float(orig_counts.get(c, 0)) for c in display_cats]
                synth_vals = [float(synth_counts.get(c, 0)) for c in display_cats]

            x = np.arange(len(display_cats))
            width = 0.35

            fig, ax = plt.subplots(figsize=(max(8, len(display_cats) * 0.6), 5))
            ax.bar(x - width / 2, orig_vals, width, label="Original",
                   color="#2196F3", alpha=0.8)
            ax.bar(x + width / 2, synth_vals, width, label="Synthetic",
                   color="#FF5722", alpha=0.8)

            ax.set_xlabel("Category")
            ax.set_ylabel("Proportion")
            ax.set_title(f"Category Distribution: {col}", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(display_cats, rotation=45, ha="right", fontsize=8)
            ax.legend()

            plt.tight_layout()

            safe_col = col.replace("/", "_").replace("\\", "_").replace(" ", "_")
            filepath = os.path.join(
                self.output_dir, f"category_dist_{safe_col}.png"
            )
            fig.savefig(filepath, bbox_inches="tight")
            plt.close(fig)

            saved_paths.append(filepath)
            logger.info("Category distribution plot saved: %s", filepath)

        return saved_paths

    def run(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        numerical_columns: List[str],
        categorical_columns: List[str],
    ) -> List[str]:
        """Execute the full visualization pipeline.

        Args:
            original: Original dataset.
            synthetic: Synthetic dataset.
            numerical_columns: Numerical column names.
            categorical_columns: Categorical column names.

        Returns:
            List of all saved file paths.
        """
        all_paths: List[str] = []
        all_paths.extend(
            self.plot_histograms(original, synthetic, numerical_columns)
        )
        all_paths.extend(
            self.plot_heatmaps(original, synthetic, numerical_columns)
        )
        all_paths.extend(
            self.plot_category_distributions(
                original, synthetic, categorical_columns
            )
        )
        logger.info("All visualizations generated: %d plots", len(all_paths))
        return all_paths
