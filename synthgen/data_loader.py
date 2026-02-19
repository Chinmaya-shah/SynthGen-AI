"""
Data Loader Module for SynthGen AI.

Handles dataset downloading, CSV loading, and preprocessing
(missing value imputation, format validation).
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class DataLoader:
    """Downloads, loads, and preprocesses structured CSV datasets.

    Attributes:
        raw_path: Destination path for the raw downloaded dataset.
        processed_path: Destination path for the cleaned dataset.
        missing_strategy: Strategy for handling missing values ('impute' or 'drop').
        column_names: Optional list of column names for headerless datasets.
    """

    def __init__(
        self,
        raw_path: str,
        processed_path: str,
        missing_strategy: str = "impute",
        column_names: Optional[List[str]] = None,
    ) -> None:
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.missing_strategy = missing_strategy
        self.column_names = column_names

    def download_dataset(self, url: str) -> str:
        """Download a CSV dataset from a public URL.

        Args:
            url: Public URL pointing to the dataset.

        Returns:
            Path to the saved raw CSV file.

        Raises:
            ConnectionError: If the download fails.
        """
        logger.info("Downloading dataset from %s", url)
        os.makedirs(os.path.dirname(self.raw_path), exist_ok=True)

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectionError(f"Failed to download dataset: {exc}") from exc

        with open(self.raw_path, "wb") as fh:
            fh.write(response.content)

        logger.info("Raw dataset saved to %s", self.raw_path)
        return self.raw_path

    def load_csv(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load a CSV file into a DataFrame.

        Args:
            path: Path to the CSV file. Defaults to ``self.raw_path``.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid CSV.
        """
        path = path or self.raw_path

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        if not path.lower().endswith(".csv"):
            raise ValueError(f"Unsupported file format. Expected CSV: {path}")

        logger.info("Loading CSV from %s", path)

        try:
            df = pd.read_csv(
                path,
                names=self.column_names if self.column_names else None,
                header=0 if not self.column_names else None,
                na_values=["?", " ?", "? ", " ? ", "NA", "N/A", "", " "],
                skipinitialspace=True,
            )
        except Exception as exc:
            raise ValueError(f"Failed to parse CSV file: {exc}") from exc

        logger.info(
            "Loaded dataset with %d rows and %d columns", len(df), len(df.columns)
        )

        if len(df) < 20:
            logger.warning(
                "Dataset has fewer than 20 rows (%d). Results may be unreliable.",
                len(df),
            )

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the DataFrame.

        Handles missing values according to the configured strategy,
        strips whitespace from string columns, and validates the data.

        Args:
            df: Raw DataFrame to preprocess.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Preprocessing dataset (strategy=%s)", self.missing_strategy)

        # Strip whitespace from object columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
            # Restore NaN for values that were originally missing
            df[col] = df[col].replace({"nan": np.nan, "": np.nan})

        initial_missing = df.isnull().sum().sum()
        logger.info("Total missing values before handling: %d", initial_missing)

        if self.missing_strategy == "drop":
            df = df.dropna().reset_index(drop=True)
            logger.info("Dropped rows with missing values. Remaining: %d rows", len(df))
        elif self.missing_strategy == "impute":
            # Impute numerical columns with mean
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    logger.info("Imputed column '%s' with mean=%.4f", col, mean_val)

            # Impute categorical columns with mode
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                    logger.info("Imputed column '%s' with mode='%s'", col, mode_val)
        else:
            raise ValueError(
                f"Unknown missing value strategy: '{self.missing_strategy}'. "
                "Use 'impute' or 'drop'."
            )

        final_missing = df.isnull().sum().sum()
        logger.info("Missing values after handling: %d", final_missing)

        return df

    def save_processed(self, df: pd.DataFrame) -> str:
        """Save the processed DataFrame to CSV.

        Args:
            df: Processed DataFrame.

        Returns:
            Path to the saved processed CSV.
        """
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        logger.info("Processed dataset saved to %s", self.processed_path)
        return self.processed_path

    def run(self, url: str) -> Tuple[pd.DataFrame, str]:
        """Execute the full data loading pipeline.

        Downloads, loads, preprocesses, and saves the dataset.

        Args:
            url: Public URL to download the dataset from.

        Returns:
            Tuple of (processed DataFrame, path to processed CSV).
        """
        self.download_dataset(url)
        df = self.load_csv()
        df = self.preprocess(df)
        processed_path = self.save_processed(df)
        return df, processed_path
