# SynthGen AI -- Privacy-Preserving Synthetic Data Generator

## Product Requirements Document (PRD)

**Version:** 1.1\
**Language:** Python\
**Project Type:** Pure AI/ML Engineering System

------------------------------------------------------------------------

# 1. Project Vision

SynthGen AI is a modular machine learning system designed to generate
statistically accurate synthetic tabular datasets from structured CSV
input data.

The system must:

-   Learn statistical distributions from real structured datasets
-   Preserve correlations between numerical variables
-   Preserve probability distributions of categorical variables
-   Generate synthetic rows without replicating original records
-   Provide statistical similarity validation metrics
-   Maintain privacy-safe generation guarantees

This document defines the full technical implementation requirements for
the AI coding agent.

------------------------------------------------------------------------

# 2. Primary Objective

Build a complete Python-based AI system that:

1.  Accepts structured CSV datasets
2.  Automatically profiles column types
3.  Learns statistical properties
4.  Generates synthetic tabular datasets
5.  Validates statistical similarity
6.  Ensures no row duplication
7.  Exports synthetic dataset
8.  Generates statistical visualization reports

No frontend or backend is required in this phase.

------------------------------------------------------------------------

# 3. Functional Requirements

## 3.1 Data Input Module

-   Accept input CSV file path
-   Handle datasets containing:
    -   Numerical columns (int, float)
    -   Categorical columns (object, category)
-   Handle missing values (drop or impute mean/mode)
-   Reject unsupported formats

------------------------------------------------------------------------

## 3.2 Data Profiling Module

Must:

-   Detect column types automatically
-   Compute numerical statistics:
    -   Mean
    -   Variance
    -   Standard deviation
    -   Min / Max
-   Compute categorical frequency distributions
-   Compute correlation matrix for numerical columns
-   Store metadata for downstream modules

------------------------------------------------------------------------

## 3.3 Distribution Learning Module

### Numerical Features

-   Compute mean vector (μ)
-   Compute covariance matrix (Σ)
-   Validate covariance matrix is positive semi-definite
-   Support multivariate Gaussian modeling

### Categorical Features

-   Learn category probability distribution
-   Store normalized probabilities

------------------------------------------------------------------------

## 3.4 Synthetic Data Generation Module

### Numerical Generation

Use:

``` python
numpy.random.multivariate_normal(mean_vector, covariance_matrix, size=N)
```

-   Clip values within observed min/max if needed
-   Prevent unrealistic negative values where domain-specific

### Categorical Generation

Use:

``` python
numpy.random.choice(categories, p=probabilities)
```

Combine numerical and categorical outputs into final dataset.

------------------------------------------------------------------------

## 3.5 Privacy Protection Layer

-   Ensure no exact row match between original and synthetic data
-   Implement row similarity detection
-   If similarity exceeds threshold, regenerate row
-   Log regeneration count

------------------------------------------------------------------------

## 3.6 Statistical Validation Module

Must compute:

-   Mean difference percentage
-   Variance ratio
-   Frobenius norm difference between correlation matrices
-   Kolmogorov-Smirnov test per numerical column
-   Category frequency comparison
-   Structured validation report output

------------------------------------------------------------------------

## 3.7 Visualization Module

Generate:

-   Histogram comparison (original vs synthetic)
-   Correlation heatmap comparison
-   Category distribution bar charts
-   Save plots as PNG

------------------------------------------------------------------------

## 3.8 Export Module

-   Save synthetic dataset as CSV
-   Save validation report as JSON
-   Save visualization outputs

------------------------------------------------------------------------

# 4. Non-Functional Requirements

-   Follow PEP8 standards
-   Modular architecture
-   Clear docstrings
-   Deterministic random seed support
-   Clean project structure
-   Separation of concerns

------------------------------------------------------------------------

# 5. Project Architecture

    synthgen_ai/
    │
    ├── data/
    │   └── sample_input.csv
    │
    ├── synthgen/
    │   ├── profiler.py
    │   ├── distribution.py
    │   ├── generator.py
    │   ├── validator.py
    │   ├── visualizer.py
    │
    ├── main.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# 6. Module Definitions

## profiler.py

Class: DataProfiler\
Methods: - detect_column_types() - compute_summary_stats() -
compute_correlation_matrix()

## distribution.py

Class: DistributionLearner\
Methods: - fit_numerical_distribution() - fit_categorical_distribution()

## generator.py

Class: SyntheticGenerator\
Methods: - generate_numerical_data() - generate_categorical_data() -
combine_data()

## validator.py

Class: Validator\
Methods: - compare_means() - compare_variances() -
correlation_difference() - ks_test()

## visualizer.py

Class: Visualizer\
Methods: - plot_histograms() - plot_heatmaps() -
plot_category_distributions()

## main.py

-   CLI entry point
-   Accept dataset path
-   Accept synthetic sample size
-   Execute full pipeline

------------------------------------------------------------------------

# 7. Machine Learning Concepts Used

-   Multivariate Gaussian Distribution
-   Covariance Matrix
-   Correlation Preservation
-   Distribution Similarity Testing
-   Kolmogorov-Smirnov Statistical Test
-   Privacy via Non-replication constraint

------------------------------------------------------------------------

# 8. Performance Considerations

-   Support datasets up to 100k rows
-   Use vectorized NumPy operations
-   Avoid excessive memory duplication

------------------------------------------------------------------------

# 9. Error Handling

-   Raise errors for covariance failures
-   Warn if dataset \< 20 rows
-   Validate input format
-   Log validation metrics clearly

------------------------------------------------------------------------

# 10. Future Extensions

-   Replace Gaussian model with GAN-based generator
-   Add Differential Privacy noise
-   Build REST API wrapper
-   Add Web dashboard frontend
-   Add Docker containerization

------------------------------------------------------------------------

# END OF PRD
