# SynthGen AI — Technical Documentation (Part 1)
# Project Overview & System Architecture

---

## 1. Project Overview

### What the Project Does

SynthGen AI is a **privacy-preserving synthetic data generation system** built in Python. It takes a real-world structured CSV dataset (the UCI Adult Census Income dataset — 32,561 rows × 15 columns), learns the underlying statistical distributions of both numerical and categorical features, generates a completely new synthetic dataset of equal size, enforces privacy constraints to ensure no synthetic row is a copy or near-duplicate of any original record, then validates the statistical fidelity of the output through rigorous quantitative metrics and visual comparisons.

The entire pipeline executes with a single command: `python main.py`.

### Real-World Problem It Solves

Organizations in healthcare, finance, government, and tech routinely need realistic datasets for:

- **Development & testing** — QA teams need production-like data without accessing real customer records
- **Machine learning model training** — Data scientists need large labeled datasets, but real data may be restricted by GDPR, HIPAA, or CCPA
- **Cross-team sharing** — Analytics teams cannot share PII-laden datasets across departments or with external vendors
- **Data augmentation** — Imbalanced or small datasets can be supplemented with statistically faithful synthetic records

Using production data directly creates legal liability, compliance violations, and reputational risk. SynthGen AI eliminates this by generating data that *looks* real but *is not* real.

### Business Relevance

- **Compliance**: Enables data usage without violating GDPR Article 89, HIPAA Safe Harbor, or CCPA Right to Deletion
- **Cost reduction**: Eliminates the need for expensive data anonymization pipelines
- **Speed**: Generates 32,561 synthetic rows in ~40 seconds on commodity hardware
- **Auditability**: Produces a structured JSON validation report proving statistical fidelity

### Why Synthetic Data Is Important

Synthetic data is projected to be used in 60% of AI development by 2025 (Gartner). It solves the fundamental tension between data utility and data privacy. Unlike anonymization (which can be reversed) or aggregation (which loses granularity), synthetic data preserves the *statistical properties* of the original dataset while containing zero actual individual records.

---

## 2. System Architecture

### Folder Structure

```
SynthAI/
├── config.yaml                    # Centralized YAML configuration
├── main.py                        # CLI entry point & 8-step pipeline orchestrator
├── requirements.txt               # 7 Python dependencies
├── README.md                      # Project documentation
├── SynthGen_AI_PRD.md             # Product Requirements Document
│
├── synthgen/                      # Core ML package (7 modules)
│   ├── __init__.py                # Package init, version="1.1.0"
│   ├── data_loader.py             # Download + CSV loading + preprocessing
│   ├── profiler.py                # Column type detection + statistics + correlation
│   ├── distribution.py            # Multivariate Gaussian fitting + categorical probs
│   ├── generator.py               # MVN sampling + categorical choice + combining
│   ├── privacy.py                 # Exact dedup + standardized Euclidean distance
│   ├── validator.py               # KS test + mean diff + variance ratio + Frobenius
│   └── visualizer.py              # Histograms + heatmaps + bar charts
│
├── data/
│   ├── raw/                       # Downloaded adult_census.csv (3.97 MB)
│   ├── processed/                 # Cleaned adult_census_processed.csv (3.59 MB)
│   └── synthetic/                 # Generated synthetic_data.csv (6.10 MB)
│
└── reports/                       # All output artifacts
    ├── validation_report.json     # Quantitative metrics (17 KB)
    ├── histogram_comparison.png   # Overlaid numerical distributions
    ├── correlation_heatmap.png    # Side-by-side correlation matrices
    └── category_dist_*.png        # 9 categorical comparison bar charts
```

### Role of Each Module

| Module | Class | Responsibility | Key Methods |
|--------|-------|----------------|-------------|
| `data_loader.py` | `DataLoader` | Downloads dataset via HTTP, loads CSV with custom NA values, imputes missing values (mean for numerical, mode for categorical) or drops rows, strips whitespace, saves processed CSV | `download_dataset()`, `load_csv()`, `preprocess()`, `run()` |
| `profiler.py` | `DataProfiler` | Auto-detects numerical vs categorical columns via pandas dtypes, computes per-column summary stats (mean, variance, std, min, max for numerical; frequency distributions for categorical), builds Pearson correlation matrix | `detect_column_types()`, `compute_summary_stats()`, `compute_correlation_matrix()`, `run()` |
| `distribution.py` | `DistributionLearner` | Fits multivariate Gaussian by computing mean vector μ and covariance matrix Σ using `np.cov()`, validates PSD via eigenvalue decomposition with clamping, learns categorical probability distributions from `value_counts(normalize=True)` | `fit_numerical_distribution()`, `_ensure_positive_semi_definite()`, `fit_categorical_distribution()` |
| `generator.py` | `SyntheticGenerator` | Generates numerical features via `np.random.multivariate_normal()`, categorical features via `np.random.choice()` with learned probabilities, clips to observed [min, max] ranges, combines into unified DataFrame preserving original column order | `generate_numerical_data()`, `generate_categorical_data()`, `combine_data()` |
| `privacy.py` | `PrivacyGuard` | Detects exact duplicates via string hashing with set membership, computes standardized Euclidean distance on Z-score normalized numerical features with batched processing and sampling (5000 original rows), iteratively regenerates flagged rows up to max_attempts | `check_exact_duplicates()`, `check_similarity()`, `enforce_privacy()` |
| `validator.py` | `Validator` | Computes 5 metric categories: mean % difference, variance ratio, correlation Frobenius norm, per-column KS tests via `scipy.stats.ks_2samp()`, categorical frequency absolute differences. Outputs structured JSON | `compare_means()`, `compare_variances()`, `correlation_difference()`, `ks_test()`, `compare_categories()` |
| `visualizer.py` | `Visualizer` | Generates overlaid density histograms (blue=original, orange=synthetic), side-by-side correlation heatmaps with shared color scale using seaborn, grouped bar charts per categorical column. Uses `Agg` backend for headless operation | `plot_histograms()`, `plot_heatmaps()`, `plot_category_distributions()` |

### Data Flow

```
[UCI URL] ──HTTP GET──▶ [raw/adult_census.csv]
                              │
                     DataLoader.preprocess()
                     (impute NaN, strip whitespace)
                              │
                              ▼
                   [processed/adult_census_processed.csv]
                              │
                     DataProfiler.run()
                     (detect types, compute stats, correlation)
                              │
                              ▼
                   {metadata: numerical_cols, categorical_cols,
                    summary_stats, correlation_matrix}
                              │
                     DistributionLearner
                     ├── fit_numerical: μ ∈ ℝ⁶, Σ ∈ ℝ⁶ˣ⁶
                     └── fit_categorical: P(cat) per column
                              │
                     SyntheticGenerator
                     ├── MVN sampling → numerical DataFrame
                     ├── Weighted choice → categorical DataFrame
                     └── combine_data() → unified synthetic DataFrame
                              │
                     PrivacyGuard.enforce_privacy()
                     ├── String hash dedup check
                     ├── Z-score normalized Euclidean distance
                     └── Iterative regeneration (up to 5 attempts)
                              │
                              ▼
                   [synthetic/synthetic_data.csv] (32,561 rows)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     Validator.run()                 Visualizer.run()
     ├── Mean % diff                ├── histogram_comparison.png
     ├── Variance ratio             ├── correlation_heatmap.png
     ├── Frobenius norm             └── 9 × category_dist_*.png
     ├── KS tests
     └── Category freq
              │
              ▼
   [validation_report.json]
```

### Design Decisions and Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Multivariate Gaussian** over per-column univariate fits | Preserves inter-column correlations (e.g., age↔hours_per_week). A per-column approach would destroy all pairwise relationships | Cannot model non-linear dependencies, multi-modal distributions, or zero-inflated features (capital_gain/capital_loss) |
| **Independent categorical sampling** | Simple, fast, and preserves marginal distributions exactly | Does not capture conditional dependencies (e.g., occupation given education level) |
| **Standardized Euclidean distance** over cosine similarity for privacy | Cosine similarity was dominated by large-magnitude features (fnlwgt ~180K), flagging all 32,561 rows. Z-score normalization equalizes feature scales | Requires computing mean/std of original data; sampling 5000 rows adds approximation error |
| **Eigenvalue clamping for PSD** over Cholesky-based near-PSD | Simple to implement, mathematically rigorous (nearest PSD matrix), and handles any numerical instability in covariance computation | Slightly modifies the learned covariance matrix; not the theoretically optimal nearest PSD in Frobenius norm |
| **YAML configuration** over CLI arguments | All 15+ parameters in one file; easy to version-control and reproduce experiments | Requires YAML dependency; slightly more complex than flat CLI flags for simple usage |
| **Batch processing** in privacy checks (batch_size=1000) | Prevents memory explosion when computing pairwise distances for 32K×5K matrices | Adds code complexity; batch boundaries could theoretically miss edge cases (they don't in this implementation) |
| **Value clipping** to observed [min, max] | Prevents unrealistic values (e.g., negative ages, hours_per_week > 99) | Slightly truncates the tails of the Gaussian, reducing variance |
