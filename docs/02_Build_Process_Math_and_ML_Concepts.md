# SynthGen AI — Technical Documentation (Part 2)
# Build Process, Mathematical Foundations & Core ML Concepts

---

## 3. Step-by-Step Build Process

### 3.1 Dataset Selection

**Chosen**: UCI Adult Census Income Dataset (32,561 rows × 15 columns)

**Why this dataset:**
- **Mixed data types**: 6 numerical + 9 categorical columns — exercises both distribution learners
- **Real-world scale**: 32K rows is large enough to learn meaningful distributions but small enough for fast iteration
- **Missing values present**: `?` markers in workclass, occupation, native_country — exercises the preprocessing pipeline
- **No authentication required**: Direct HTTP download from UCI ML Repository
- **Domain relevance**: Financial/demographic data is a common synthetic data use case

### 3.2 Preprocessing Decisions

In `DataLoader.preprocess()` (data_loader.py:114-167):

1. **Whitespace stripping**: The UCI dataset uses ` Private` instead of `Private`. All object columns are stripped via `str.strip()`
2. **NA markers**: `?`, ` ?`, `NA`, `N/A`, empty strings are all recognized as NaN via `pd.read_csv(na_values=[...])`
3. **Imputation strategy** (configurable):
   - **Numerical**: Mean imputation — preserves the column mean (used for capital_gain, capital_loss)
   - **Categorical**: Mode imputation — fills with most frequent category (e.g., `Private` for workclass)
   - Alternative: `drop` strategy removes rows with any NaN
4. **Column names injected**: The UCI dataset has no header row, so 15 column names are provided via config.yaml

### 3.3 Statistical Modeling Decisions

**Numerical modeling** (distribution.py:39-92):
- Fit a single multivariate Gaussian: μ ∈ ℝ⁶, Σ ∈ ℝ⁶ˣ⁶
- `np.cov(data, rowvar=False)` computes the sample covariance matrix with Bessel's correction (N-1 denominator)
- PSD validation via `np.linalg.eigh()` — clamps any negative eigenvalue to 1e-10

**Categorical modeling** (distribution.py:139-181):
- `value_counts(normalize=True)` extracts empirical probability distributions
- Probabilities are force-normalized to sum to exactly 1.0 (avoiding floating-point drift)

### 3.4 Privacy Mechanism Implementation

**Evolution**: The initial implementation used cosine similarity (threshold=0.95) but it flagged ALL 32,561 rows because large-magnitude features like `fnlwgt` (~180,000) dominated the cosine computation, making all rows appear similar.

**Final implementation** (privacy.py:91-178):
1. Extract numerical features as float64 arrays
2. **Z-score standardize** using original data's mean and std: `(x - μ) / σ`
3. Sample 5,000 original rows (if > 5,000) for comparison efficiency
4. Compute pairwise squared Euclidean distances using the expansion: `||a-b||² = ||a||² + ||b||² - 2a·b`
5. Normalize by `√(num_features)` to make threshold interpretable across different dimensionalities
6. Flag rows with min-distance < 0.05
7. Regenerate flagged rows with fresh MVN samples + categorical choices
8. Repeat up to 5 iterations

### 3.5 Validation Strategy

Five complementary metrics (validator.py:34-224):
1. **Mean % difference**: How much the column mean shifted (target: < 5%)
2. **Variance ratio**: σ²_synth / σ²_orig (target: ≈ 1.0)
3. **Frobenius norm**: ||R_orig - R_synth||_F measures correlation matrix drift (target: low)
4. **KS test**: Per-column distributional comparison via `scipy.stats.ks_2samp`
5. **Categorical frequency**: Absolute difference in proportions per category

### 3.6 Visualization Generation

Three chart types (visualizer.py):
- **Overlaid histograms**: 50-bin density histograms, blue(original) + orange(synthetic), one subplot per numerical column
- **Side-by-side heatmaps**: Annotated correlation matrices with shared `coolwarm` colorscale
- **Grouped bar charts**: One PNG per categorical column, sorted categories, capped at 20 categories for readability

### 3.7 Final Integration

`main.py` (369 lines) orchestrates 8 steps sequentially:
1. Load YAML config → 2. Download & preprocess → 3. Profile → 4. Learn distributions → 5. Generate synthetic → 6. Enforce privacy → 7. Validate → 8. Visualize → Export

Uses `argparse` for CLI, `logging` for structured output, `time.time()` for runtime measurement. Top-level try/except with `sys.exit(1)` on failure.

---

## 4. Mathematical Foundations

### 4.1 Multivariate Gaussian Distribution

A multivariate Gaussian (normal) distribution describes the joint probability of *d* continuous random variables. For a random vector **x** ∈ ℝᵈ:

```
p(x) = (2π)^(-d/2) |Σ|^(-1/2) exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))
```

- **μ** (mean vector): The center of the distribution in d-dimensional space
- **Σ** (covariance matrix): Encodes both the spread (variance) of each feature and the linear relationships between features

**In this project**: d=6 (age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week). The system computes μ ∈ ℝ⁶ and Σ ∈ ℝ⁶ˣ⁶ from the original data, then draws synthetic samples from this learned distribution.

### 4.2 Covariance Matrix

The covariance matrix Σ is computed as:

```
Σ_jk = (1/(n-1)) Σᵢ (xᵢⱼ - μⱼ)(xᵢₖ - μₖ)
```

- Diagonal entries Σ_jj = variance of feature j
- Off-diagonal entries Σ_jk = covariance between features j and k
- The matrix is symmetric: Σ_jk = Σ_kj
- Size: d × d (here, 6 × 6 = 36 entries, but only 21 unique due to symmetry)

**Implementation**: `np.cov(data, rowvar=False)` computes this with Bessel's correction (n-1 denominator for unbiased estimation).

### 4.3 Why Covariance Must Be Positive Semi-Definite (PSD)

A matrix Σ is PSD if and only if all eigenvalues λᵢ ≥ 0. This is required because:

1. **Physical meaning**: Variance cannot be negative. For any direction vector **v**, the variance along that direction is **v**ᵀΣ**v** ≥ 0
2. **Sampling requirement**: `numpy.random.multivariate_normal` uses Cholesky decomposition (Σ = LLᵀ), which requires PSD
3. **Numerical issues**: Floating-point arithmetic can introduce tiny negative eigenvalues (~10⁻¹⁵)

**Implementation** (distribution.py:94-137): Uses eigenvalue decomposition via `np.linalg.eigh()`, clamps negative eigenvalues to ε=10⁻¹⁰, reconstructs Σ' = QΛ'Qᵀ, then symmetrizes as (Σ' + Σ'ᵀ)/2.

### 4.4 How Correlation Preservation Works

Because the system models the **full** covariance matrix (not just per-column variances), the Pearson correlation:

```
ρ_jk = Σ_jk / √(Σ_jj · Σ_kk)
```

is implicitly preserved. When we sample from N(μ, Σ), the generated data will exhibit the same pairwise correlations as the original.

**Measured result**: Frobenius norm of correlation difference = **0.0396** (near-zero), confirming excellent correlation preservation across all 6 numerical columns.

### 4.5 Kolmogorov-Smirnov (KS) Test

The KS test compares two empirical cumulative distribution functions (ECDFs):

```
KS statistic = sup_x |F_orig(x) - F_synth(x)|
```

- Measures the maximum vertical distance between the two ECDFs
- Range: [0, 1] — lower is better
- **p-value**: Probability that two datasets with identical underlying distributions would produce a KS statistic this large
- Non-parametric: makes no assumption about the shape of the distribution

**In this project**: KS test is applied per numerical column. Results show age KS=0.062 (good), capital_gain KS=0.483 (poor — expected due to zero-inflation).

### 4.6 Frobenius Norm

The Frobenius norm measures the element-wise magnitude of a matrix:

```
||A||_F = √(Σᵢ Σⱼ |aᵢⱼ|²)
```

Applied to the **difference** between original and synthetic correlation matrices, it gives a single scalar measuring how much the correlation structure has drifted.

**Result**: ||R_orig - R_synth||_F = **0.0396** — this is extremely low for a 6×6 matrix, indicating near-perfect correlation preservation.

### 4.7 Statistical Assumptions & Limitations

1. **Gaussian assumption**: Numerical features are assumed to follow a multivariate Gaussian. This works well for age, fnlwgt, hours_per_week (roughly bell-shaped) but fails for capital_gain and capital_loss which are **zero-inflated** (>90% of values are 0, with a long right tail)
2. **Linear correlations only**: Covariance captures only linear relationships. Non-linear dependencies (e.g., U-shaped relationship between age and income) are lost
3. **Independent categoricals**: Each categorical column is sampled independently. Conditional dependencies (e.g., P(occupation | education)) are not modeled
4. **Stationarity**: The model assumes the data is i.i.d. (independent and identically distributed). Time-series or sequential data cannot be modeled

---

## 5. Core ML Concepts Used

### 5.1 Generative Modeling

SynthGen AI is a **generative model** — it learns to produce new data points from the same distribution as the training data. Unlike discriminative models (which learn P(y|x)), generative models learn P(x) or the joint distribution P(x,y).

The specific approach used is **parametric density estimation**: fitting a known parametric family (multivariate Gaussian) to the data by estimating its sufficient statistics (μ, Σ).

### 5.2 Distribution Learning

Distribution learning is the process of estimating the parameters of a probability distribution from observed data:
- **Numerical**: Maximum Likelihood Estimation (MLE) of μ and Σ for a multivariate Gaussian. The sample mean and sample covariance are the MLE estimators
- **Categorical**: Empirical frequency estimation (also MLE for the categorical/multinomial distribution)

### 5.3 Feature Correlation

Correlation measures the strength and direction of linear relationships between features. The covariance matrix captures all pairwise correlations simultaneously. By using `np.cov()` on the full feature matrix, the system preserves the correlation structure in synthetic output.

### 5.4 Sampling Methods

- **Numerical**: `np.random.multivariate_normal(μ, Σ, size=N)` — internally uses Cholesky decomposition to transform independent standard normal samples into correlated samples
- **Categorical**: `np.random.choice(categories, p=probabilities)` — inverse transform sampling from learned categorical distributions

### 5.5 Bias and Variance in Synthetic Data

- **Bias**: The sample mean is an unbiased estimator of the population mean. However, clipping to [min, max] introduces a small bias by truncating the Gaussian tails
- **Variance**: The sample covariance with Bessel's correction (n-1) is unbiased. The variance ratio metric validates this: hours_per_week ratio = 1.013 (nearly perfect), capital_gain ratio = 0.401 (poor — Gaussian underfits the zero-inflated distribution)

### 5.6 Overfitting Risk in Synthetic Modeling

If the learned distribution too closely fits the training data, the synthetic data may inadvertently memorize individual records. The PrivacyGuard module mitigates this by:
- Checking for exact duplicate rows (string hash comparison)
- Computing minimum standardized Euclidean distance to detect near-copies
- Regenerating any synthetic row that falls within 0.05 normalized distance units of any original row
