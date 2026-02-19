# SynthGen AI — Privacy-Preserving Synthetic Data Generator

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modular machine learning system that generates statistically accurate synthetic tabular datasets from structured CSV data while preserving privacy through non-replication constraints.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [ML Methodology](#ml-methodology)
- [Mathematical Foundation](#mathematical-foundation)
- [Validation Methodology](#validation-methodology)
- [Privacy Safeguards](#privacy-safeguards)
- [Execution Instructions](#execution-instructions)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Dataset Information](#dataset-information)
- [Limitations & Future Improvements](#limitations--future-improvements)

---

## Problem Statement

Organizations frequently need realistic datasets for testing, development, analytics, and machine learning, yet using production data directly poses significant privacy and compliance risks. SynthGen AI addresses this by learning the statistical properties of real structured datasets and generating synthetic equivalents that:

- **Preserve statistical fidelity** — distributions, means, variances, and inter-column correlations closely match the original data
- **Guarantee privacy** — no synthetic row is an exact copy or near-duplicate of any original record
- **Scale efficiently** — handles datasets up to 100,000 rows using vectorized NumPy operations

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     main.py (Pipeline Orchestrator)       │
│                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │ DataLoader   │───▶│ DataProfiler│───▶│ Distribution │  │
│  │ (download,   │    │ (types,     │    │ Learner      │  │
│  │  preprocess) │    │  stats,     │    │ (μ, Σ,       │  │
│  └─────────────┘    │  corr)      │    │  P(cat))     │  │
│                      └─────────────┘    └──────┬───────┘  │
│                                                 │          │
│  ┌─────────────┐    ┌─────────────┐    ┌───────▼───────┐ │
│  │ Visualizer   │◀──│ Validator   │◀───│ Synthetic     │ │
│  │ (histograms, │    │ (KS test,  │    │ Generator     │ │
│  │  heatmaps,   │    │  means,    │    │ (MVN sample,  │ │
│  │  bar charts) │    │  Frobenius)│    │  cat choice)  │ │
│  └─────────────┘    └─────────────┘    └───────┬───────┘ │
│                                                 │          │
│                                        ┌───────▼───────┐  │
│                                        │ PrivacyGuard  │  │
│                                        │ (dedup,       │  │
│                                        │  similarity)  │  │
│                                        └───────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## ML Methodology

### Multivariate Gaussian Modeling (Numerical Features)

For numerical columns, the system fits a **multivariate Gaussian (normal) distribution** to jointly model the data, preserving inter-column correlations. Synthetic samples are drawn using:

```python
numpy.random.multivariate_normal(mean_vector, covariance_matrix, size=N)
```

This approach captures the full joint distribution, not just marginal statistics per column, ensuring that relationships (e.g., age ↔ hours_per_week) are preserved in synthetic output.

### Categorical Probability Modeling

For categorical columns, the system learns the **empirical probability distribution** from observed frequencies and generates synthetic values using weighted random sampling:

```python
numpy.random.choice(categories, p=probabilities)
```

---

## Mathematical Foundation

### Mean Vector (μ)

For *d* numerical features measured across *n* observations:

$$\mu = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i \in \mathbb{R}^d$$

### Covariance Matrix (Σ)

The covariance matrix captures pairwise linear relationships:

$$\Sigma_{jk} = \frac{1}{n-1} \sum_{i=1}^{n} (x_{ij} - \mu_j)(x_{ik} - \mu_k)$$

Where Σ ∈ ℝ^(d×d) is symmetric and must be **positive semi-definite** (PSD). The system validates PSD through eigenvalue decomposition:

1. Compute eigenvalues λ₁, ..., λ_d via `numpy.linalg.eigh`
2. If any λᵢ < 0, clamp to ε = 10⁻¹⁰
3. Reconstruct: Σ' = QΛ'Qᵀ and symmetrize

### Multivariate Normal Sampling

Synthetic samples **x** ~ N(μ, Σ) are generated via Cholesky decomposition internally by NumPy:

$$\mathbf{x} = \mu + L\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$$

where Σ = LLᵀ (Cholesky factorization).

### Correlation Preservation

Because we model the full covariance matrix (not just per-column variances), the correlation structure:

$$\rho_{jk} = \frac{\Sigma_{jk}}{\sqrt{\Sigma_{jj} \cdot \Sigma_{kk}}}$$

is explicitly preserved in the synthetic data.

---

## Validation Methodology

The system computes five categories of metrics to validate fidelity:

| Metric | Formula | Goal |
|--------|---------|------|
| **Mean % Difference** | \|μ_synth - μ_orig\| / \|μ_orig\| × 100 | < 5% |
| **Variance Ratio** | σ²_synth / σ²_orig | ≈ 1.0 |
| **Frobenius Norm** | ‖R_orig - R_synth‖_F | Low value |
| **KS Test** | sup\|F_orig(x) - F_synth(x)\| | Low statistic, high p-value |
| **Category Frequency** | \|P_orig(c) - P_synth(c)\| per category | Small differences |

All results are saved as a structured JSON report in `reports/validation_report.json`.

---

## Privacy Safeguards

SynthGen AI implements a multi-layer privacy protection system:

1. **Exact Duplication Detection** — Converts each row to a string representation and checks for exact matches between original and synthetic data using set membership.

2. **Similarity Detection** — Computes cosine similarity on numerical features between each synthetic row and all original rows. Rows exceeding the configurable threshold (default: 0.95) are flagged.

3. **Iterative Regeneration** — Flagged rows are regenerated using the same distribution parameters and re-checked, up to a configurable maximum number of attempts.

4. **Metrics Logging** — Total flagged rows, regenerated rows, and final privacy status are logged and included in the validation report.

---

## Execution Instructions

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Internet connection (for dataset download)

### Setup & Run

```bash
# 1. Navigate to project directory
cd SynthAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete pipeline
python main.py
```

The pipeline will automatically:
1. Download the Adult Census Income dataset (~32,561 rows)
2. Preprocess and clean the data
3. Profile column types and compute statistics
4. Learn multivariate Gaussian and categorical distributions
5. Generate synthetic data matching original size
6. Enforce privacy constraints
7. Compute validation metrics
8. Generate visualization plots
9. Export all results

### Custom Configuration

```bash
python main.py --config config.yaml
```

---

## Project Structure

```
SynthAI/
│
├── config.yaml                    # Pipeline configuration
├── main.py                        # CLI entry point & pipeline orchestrator
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── SynthGen_AI_PRD.md             # Product Requirements Document
│
├── synthgen/                      # Core ML modules
│   ├── __init__.py
│   ├── data_loader.py             # Dataset download & preprocessing
│   ├── profiler.py                # Data profiling & statistics
│   ├── distribution.py            # Distribution learning (MVN + categorical)
│   ├── generator.py               # Synthetic data generation
│   ├── privacy.py                 # Privacy protection layer
│   ├── validator.py               # Statistical validation
│   └── visualizer.py              # Visualization generation
│
├── data/
│   ├── raw/                       # Downloaded raw datasets
│   ├── processed/                 # Cleaned & preprocessed data
│   └── synthetic/                 # Generated synthetic datasets
│
└── reports/                       # Validation reports & visualizations
    ├── validation_report.json
    ├── histogram_comparison.png
    ├── correlation_heatmap.png
    └── category_dist_*.png
```

---

## Configuration

All pipeline parameters are configurable via `config.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `random_seed` | Seed for reproducibility | 42 |
| `dataset.download_url` | Public dataset URL | UCI Adult Census |
| `synthetic.num_rows` | Synthetic rows (0 = match original) | 0 |
| `privacy.similarity_threshold` | Cosine similarity flag threshold | 0.95 |
| `privacy.max_regeneration_attempts` | Max regen iterations | 10 |
| `missing_values.strategy` | How to handle NaN: `impute` or `drop` | impute |

---

## Dataset Information

**Adult Census Income Dataset** (UCI Machine Learning Repository)

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Records**: 32,561 rows
- **Features**: 15 columns (6 numerical, 9 categorical)
- **Domain**: US Census demographic and income data
- **Numerical**: age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week
- **Categorical**: workclass, education, marital_status, occupation, relationship, race, sex, native_country, income

This dataset was selected for its realistic mix of numerical and categorical features, sufficient volume (>10,000 rows), public availability without authentication, and relevance to business/financial domains.

---

## Limitations & Future Improvements

### Current Limitations

- **Gaussian assumption** — Numerical columns are modeled as multivariate Gaussian, which may not capture non-linear relationships, multi-modal distributions, or skewed features accurately
- **Independent categorical generation** — Categorical columns are sampled independently, without modeling inter-categorical or numerical-categorical dependencies
- **Scalability** — Cosine similarity privacy checks have O(n²) complexity, which becomes expensive for very large datasets
- **No time-series support** — The system is designed for i.i.d. tabular data only

### Future Extensions

- Replace Gaussian model with **GAN-based generator** (CTGAN, TableGAN) for non-linear distributions
- Add **Differential Privacy** noise (ε-DP) for formal privacy guarantees
- Build **REST API wrapper** for service-oriented deployment
- Add **web dashboard frontend** for interactive exploration
- Add **Docker containerization** for portable deployment
- Implement **conditional generation** to capture categorical-numerical dependencies
- Add **automated hyperparameter tuning** for distribution fitting

---

## License

This project is provided for educational and portfolio demonstration purposes.
