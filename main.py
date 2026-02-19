"""
SynthGen AI — Privacy-Preserving Synthetic Data Generator

Main pipeline orchestrator. Executes the full end-to-end pipeline:
  1. Load configuration
  2. Download and preprocess dataset
  3. Profile data
  4. Learn distributions
  5. Generate synthetic data
  6. Enforce privacy constraints
  7. Validate statistical similarity
  8. Generate visualizations
  9. Export results

Usage:
    python main.py
    python main.py --config config.yaml
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from synthgen.data_loader import DataLoader
from synthgen.profiler import DataProfiler
from synthgen.distribution import DistributionLearner
from synthgen.generator import SyntheticGenerator
from synthgen.privacy import PrivacyGuard
from synthgen.validator import Validator
from synthgen.visualizer import Visualizer


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the pipeline.

    Args:
        level: Logging level string (e.g., 'INFO', 'DEBUG').
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    return config


def run_pipeline(config: Dict[str, Any]) -> None:
    """Execute the full SynthGen AI pipeline.

    Args:
        config: Loaded configuration dictionary.
    """
    logger = logging.getLogger("synthgen.pipeline")
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("  SynthGen AI — Privacy-Preserving Synthetic Data Generator")
    logger.info("=" * 70)

    # Set random seed for reproducibility
    random_seed = config.get("random_seed", 42)
    np.random.seed(random_seed)
    logger.info("Random seed set to %d", random_seed)

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Download & Preprocess Dataset
    # ─────────────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 1: Dataset Download & Preprocessing")
    logger.info("─" * 50)

    dataset_cfg = config["dataset"]
    missing_cfg = config.get("missing_values", {})

    data_loader = DataLoader(
        raw_path=dataset_cfg["raw_path"],
        processed_path=dataset_cfg["processed_path"],
        missing_strategy=missing_cfg.get("strategy", "impute"),
        column_names=dataset_cfg.get("column_names"),
    )

    df_original, processed_path = data_loader.run(
        url=dataset_cfg["download_url"]
    )

    logger.info(
        "Dataset ready: %d rows × %d columns",
        len(df_original),
        len(df_original.columns),
    )

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Data Profiling
    # ─────────────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 2: Data Profiling")
    logger.info("─" * 50)

    profiler = DataProfiler(df_original)
    metadata = profiler.run()

    numerical_cols = metadata["numerical_columns"]
    categorical_cols = metadata["categorical_columns"]

    logger.info(
        "Profile complete: %d numerical, %d categorical columns",
        len(numerical_cols),
        len(categorical_cols),
    )

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Distribution Learning
    # ─────────────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 3: Distribution Learning")
    logger.info("─" * 50)

    dist_learner = DistributionLearner()

    mean_vector, covariance_matrix = dist_learner.fit_numerical_distribution(
        df_original, numerical_cols
    )

    cat_distributions = dist_learner.fit_categorical_distribution(
        df_original, categorical_cols
    )

    logger.info("Distributions learned successfully")

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Synthetic Data Generation
    # ─────────────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 4: Synthetic Data Generation")
    logger.info("─" * 50)

    synth_cfg = config.get("synthetic", {})
    num_synthetic_rows = synth_cfg.get("num_rows", 0)
    if num_synthetic_rows <= 0:
        num_synthetic_rows = len(df_original)

    logger.info("Generating %d synthetic rows", num_synthetic_rows)

    generator = SyntheticGenerator(random_seed=random_seed)

    # Get column statistics for clipping
    column_stats = metadata["summary_stats"].get("numerical", {})

    numerical_df = generator.generate_numerical_data(
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        size=num_synthetic_rows,
        column_names=numerical_cols,
        column_stats=column_stats,
    )

    categorical_df = generator.generate_categorical_data(
        categorical_distributions=cat_distributions,
        size=num_synthetic_rows,
    )

    original_column_order = df_original.columns.tolist()
    df_synthetic = generator.combine_data(
        numerical_df, categorical_df, original_column_order
    )

    logger.info("Synthetic dataset shape: %s", df_synthetic.shape)

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: Privacy Protection
    # ─────────────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 5: Privacy Protection")
    logger.info("─" * 50)

    privacy_cfg = config.get("privacy", {})
    privacy_guard = PrivacyGuard(
        similarity_threshold=privacy_cfg.get("similarity_threshold", 0.95),
        max_attempts=privacy_cfg.get("max_regeneration_attempts", 10),
    )

    df_synthetic, privacy_metrics = privacy_guard.enforce_privacy(
        original_df=df_original,
        synthetic_df=df_synthetic,
        numerical_columns=numerical_cols,
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        column_stats=column_stats,
        categorical_distributions=cat_distributions,
        categorical_columns=categorical_cols,
    )

    logger.info(
        "Privacy check complete. Rows regenerated: %d",
        privacy_metrics["total_regenerated_rows"],
    )

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: Save Synthetic Dataset
    # ─────────────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 6: Export Synthetic Dataset")
    logger.info("─" * 50)

    synth_output_path = synth_cfg.get(
        "output_path", "data/synthetic/synthetic_data.csv"
    )
    os.makedirs(os.path.dirname(synth_output_path), exist_ok=True)
    df_synthetic.to_csv(synth_output_path, index=False)
    logger.info("Synthetic dataset saved to %s", synth_output_path)

    # ─────────────────────────────────────────────────────────────────
    # STEP 7: Statistical Validation
    # ─────────────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 7: Statistical Validation")
    logger.info("─" * 50)

    reports_cfg = config.get("reports", {})
    validation_path = reports_cfg.get(
        "validation_report", "reports/validation_report.json"
    )

    validator = Validator()
    report = validator.run(
        original=df_original,
        synthetic=df_synthetic,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        output_path=validation_path,
        privacy_metrics=privacy_metrics,
    )

    logger.info("Validation report saved to %s", validation_path)

    # Print summary metrics
    logger.info("─" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("─" * 50)

    if "mean_difference_pct" in report:
        avg_mean_diff = np.mean(
            list(report["mean_difference_pct"].values())
        )
        logger.info("  Average mean difference: %.4f%%", avg_mean_diff)

    if "variance_ratio" in report:
        avg_var_ratio = np.mean(
            list(report["variance_ratio"].values())
        )
        logger.info("  Average variance ratio: %.4f", avg_var_ratio)

    if "correlation_frobenius_norm" in report:
        logger.info(
            "  Correlation Frobenius norm diff: %.6f",
            report["correlation_frobenius_norm"],
        )

    if "ks_tests" in report:
        avg_ks = np.mean(
            [v["ks_statistic"] for v in report["ks_tests"].values()]
        )
        logger.info("  Average KS statistic: %.6f", avg_ks)

    # ─────────────────────────────────────────────────────────────────
    # STEP 8: Visualization
    # ─────────────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 8: Visualization")
    logger.info("─" * 50)

    output_dir = reports_cfg.get("output_dir", "reports")
    visualizer = Visualizer(output_dir=output_dir)
    plot_files = visualizer.run(
        original=df_original,
        synthetic=df_synthetic,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
    )

    logger.info("Generated %d visualization plots", len(plot_files))

    # ─────────────────────────────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE  (%.2f seconds)", elapsed)
    logger.info("=" * 70)

    # Print synthetic data preview
    logger.info("\nSynthetic Data Preview (first 5 rows):")
    print(df_synthetic.head().to_string(index=False))

    logger.info("\nOutput Files:")
    logger.info("  ├── %s", processed_path)
    logger.info("  ├── %s", synth_output_path)
    logger.info("  ├── %s", validation_path)
    for pf in plot_files:
        logger.info("  ├── %s", pf)
    logger.info("  └── Done!")


def main() -> None:
    """CLI entry point for SynthGen AI."""
    parser = argparse.ArgumentParser(
        description="SynthGen AI — Privacy-Preserving Synthetic Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    log_cfg = config.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"))

    # Run pipeline
    try:
        run_pipeline(config)
    except Exception as exc:
        logging.getLogger("synthgen.pipeline").error(
            "Pipeline failed: %s", exc, exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
