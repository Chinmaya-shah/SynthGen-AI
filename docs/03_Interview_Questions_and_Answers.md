# SynthGen AI — Technical Documentation (Part 3)
# Deep Technical Interview Questions & Model Answers

---

## 6. Interview Questions (35 Questions with Answers)

### Category A: Basic AI/ML Concepts (5 Questions)

**Q1. What is generative modeling and how does it differ from discriminative modeling?**

**A**: Generative modeling learns the joint probability distribution P(X) of the data so it can generate new data points. Discriminative models learn P(Y|X) — the boundary between classes. In SynthGen AI, I implemented a generative model: I learned μ and Σ of a multivariate Gaussian from census data, then sampled new data points from that distribution. A discriminative model like logistic regression would classify income (≤50K vs >50K) but could not generate new census records.

**Q2. What is Maximum Likelihood Estimation (MLE) and where does your project use it?**

**A**: MLE finds parameter values that maximize the probability of observing the data. The sample mean and sample covariance matrix are the MLE estimators for a multivariate Gaussian. In `distribution.py`, `np.mean(data, axis=0)` computes the MLE for μ, and `np.cov(data, rowvar=False)` computes the MLE for Σ (with Bessel's correction for unbiasedness). For categorical columns, `value_counts(normalize=True)` computes the MLE for the multinomial distribution parameters.

**Q3. What is the difference between overfitting and underfitting in the context of synthetic data?**

**A**: Overfitting in synthetic data generation means the model memorizes individual records — the synthetic data becomes too similar to the original, violating privacy. Underfitting means the model fails to capture the data's statistics — synthetic data has poor fidelity. My PrivacyGuard module detects overfitting (near-duplicate rows), while my Validator module detects underfitting (large KS statistics, poor variance ratios). The capital_gain column underfits (KS=0.483) because the Gaussian model can't represent its zero-inflated distribution.

**Q4. What are parametric vs non-parametric models?**

**A**: Parametric models assume a fixed functional form (e.g., Gaussian) and learn a fixed number of parameters. Non-parametric models make fewer assumptions and their complexity grows with data size (e.g., KDE, decision trees). SynthGen AI uses a parametric approach: 6 parameters for μ plus 21 unique covariance entries (6×6 symmetric matrix). A non-parametric alternative like Kernel Density Estimation would work better for skewed features but would be harder to sample from in high dimensions.

**Q5. What is the curse of dimensionality and how does it affect this project?**

**A**: As dimensions increase, data becomes sparse and distances lose meaning. With 6 numerical features, we're in a low-dimensional regime where multivariate Gaussian works well. If we had 100+ features, the covariance matrix (100×100 = 10,000 parameters) would be poorly estimated from 32K samples, and the Gaussian assumption would break down. The privacy check would also suffer because Euclidean distances become less meaningful in high dimensions.

---

### Category B: NumPy and Pandas Questions (5 Questions)

**Q6. Why does your code use `np.cov(data, rowvar=False)` instead of `np.cov(data)`?**

**A**: By default, `np.cov()` treats each row as a variable. Setting `rowvar=False` tells it each column is a variable and each row is an observation — which matches our data layout where rows are census records and columns are features. Without this flag, we'd get a 32561×32561 matrix instead of the desired 6×6 matrix.

**Q7. How does your privacy check compute pairwise distances efficiently?**

**A**: Instead of nested loops (O(n²) with high constant), I use the algebraic expansion: `||a-b||² = ||a||² + ||b||² - 2a·b`. In privacy.py lines 156-158, `batch @ orig_sample.T` is a single matrix multiplication that computes all pairwise dot products simultaneously. This leverages BLAS-optimized linear algebra, making it ~100x faster than Python loops. I also batch the synthetic rows (1000 at a time) to control memory usage.

**Q8. Why do you use `pd.read_csv(na_values=["?", " ?"])` with a custom NA list?**

**A**: The UCI Adult dataset uses `?` and ` ?` (with leading space) as missing value indicators — not the standard NaN/NA. Without specifying these custom NA values, pandas would treat `?` as a valid category string. We also include `skipinitialspace=True` to handle whitespace-padded values.

**Q9. Explain the `value_counts(normalize=True)` pattern used for categorical distributions.**

**A**: `value_counts(normalize=True)` returns relative frequencies (proportions summing to 1.0) instead of absolute counts. This directly gives us the empirical probability distribution for each category. For example, for the `sex` column: Male=0.669, Female=0.331. These probabilities are then passed to `np.random.choice(p=probabilities)` for synthesis.

**Q10. How does your clipping logic work and why is it important?**

**A**: In generator.py lines 73-89, after generating numerical values from the Gaussian, I clip each column to its observed [min, max] range using `df[col].clip(lower=min, upper=max)`. Without this, the Gaussian's infinite tails could produce impossible values like age=-5 or hours_per_week=200. There's also a heuristic: if the original min ≥ 0, clip to 0 (prevents negative capital_gain). This is a post-hoc correction that introduces slight bias but ensures domain validity.

---

### Category C: Statistical Questions (5 Questions)

**Q11. Why must the covariance matrix be positive semi-definite? What happens if it isn't?**

**A**: PSD means **v**ᵀΣ**v** ≥ 0 for all vectors **v** — variance in any direction is non-negative. `np.random.multivariate_normal` uses Cholesky decomposition (Σ = LLᵀ) internally, which requires PSD; it throws `LinAlgError` otherwise. In distribution.py:94-137, I validate PSD via eigenvalue decomposition. If any eigenvalue is negative (from floating-point error), I clamp it to 1e-10 and reconstruct the matrix. In practice, with 32K rows and 6 features, the matrix was naturally PSD, but the safeguard is essential for numerical robustness.

**Q12. Interpret the KS test results from your project. Why is capital_gain KS=0.483?**

**A**: The KS statistic measures the maximum gap between the original and synthetic CDFs. For `age` (KS=0.062), the distributions nearly overlap. For `capital_gain` (KS=0.483), nearly half the CDF is misaligned. This happens because capital_gain is 91.7% zeros with occasional large values (up to $99,999). A Gaussian centered at the mean (~$1,078) generates a bell curve — it cannot produce the spike at zero or the sparse outliers. This is a documented limitation requiring a more sophisticated model (e.g., zero-inflated mixture model or GAN).

**Q13. What does a variance ratio of 0.40 for capital_gain tell you?**

**A**: The synthetic variance is only 40% of the original variance, meaning the synthetic data's spread is significantly narrower. This happens because the Gaussian underestimates the extreme values: the original has values from 0 to 99,999 with 91.7% at zero, creating huge variance. The clipped Gaussian produces values closer to the mean, reducing variance. Ideal ratio is 1.0; age (0.896) and education_num (0.98) are much closer to ideal.

**Q14. Why did you choose Frobenius norm instead of other matrix norms for correlation comparison?**

**A**: The Frobenius norm treats all off-diagonal correlation differences equally, giving a single scalar that's easy to interpret and compare. Alternatives: spectral norm (maximum singular value) would only capture the worst-case direction, L1 norm would weight all entries equally without penalizing large individual errors. Our result of 0.0396 for a 6×6 matrix (42 entries) means the average per-entry difference is ~0.006 — excellent correlation preservation.

**Q15. Explain the difference between sample covariance (N-1 denominator) and population covariance (N denominator).**

**A**: `np.cov()` uses N-1 (Bessel's correction) because we're estimating the population covariance from a sample. Dividing by N would systematically underestimate the true variance (biased estimator). With N=32,561, the difference is negligible (~0.003%), but using N-1 is mathematically correct for inference. If we were computing statistics of the entire population (not a sample), N would be appropriate.

---

### Category D: Project-Specific Deep Questions (8 Questions)

**Q16. Walk through exactly what happens when you call `enforce_privacy()`. What is the iteration logic?**

**A**: In privacy.py:180-301: (1) Set regeneration_count=0. (2) Loop up to max_attempts=5 times. (3) Each iteration: first call `check_exact_duplicates()` — converts all rows to pipe-delimited strings with rounded numerics, builds a set of original row strings, checks each synthetic row for membership. Then call `check_similarity()` — Z-score standardize numerical features using original mean/std, sample 5000 original rows, compute pairwise Euclidean distances in batches of 1000, flag rows with min-distance < 0.05. (4) Merge flagged indices from both checks. (5) If no flags, break (privacy passed). (6) Otherwise, regenerate flagged rows: draw new MVN samples for numerical columns, clip to [min, max], draw new categorical values via weighted choice. (7) Increment counter. (8) After loop, return metrics dict.

**Q17. Why did cosine similarity fail and what specifically did you replace it with?**

**A**: Cosine similarity measures angle between vectors: cos(a,b) = (a·b)/(||a||·||b||). For census data, the `fnlwgt` column (~180,000) dominates the dot product, making almost all pairs appear similar (cos > 0.95) regardless of actual record similarity. I replaced it with **standardized Euclidean distance**: first Z-score normalize (subtract mean, divide by std) so each feature has equal weight, then compute Euclidean distance and normalize by √(num_features). The threshold changed from 0.95 (cosine, higher = more similar) to 0.05 (distance, lower = more similar).

**Q18. How does your system handle a column that has zero variance (all identical values)?**

**A**: In privacy.py line 129: `stds = np.where(stds == 0, 1.0, stds)` — if a column's standard deviation is 0, we replace it with 1.0 before Z-score normalization. This prevents division by zero and effectively ignores that feature in similarity computation (all rows have the same standardized value of 0 for that feature). In `_ensure_positive_semi_definite`, a zero-variance column creates a zero eigenvalue, which remains at 0 (it's ≥ 0, so no clamping needed).

**Q19. Explain the end-to-end data flow for a single categorical column (e.g., `occupation`).**

**A**: (1) `DataLoader.preprocess()`: strip whitespace, impute missing `occupation` values with the mode (e.g., "Prof-specialty"). (2) `DataProfiler.detect_column_types()`: classified as categorical because its dtype is `object`. (3) `DataProfiler.compute_summary_stats()`: compute `value_counts()` — 14 categories with their counts and proportions. (4) `DistributionLearner.fit_categorical_distribution()`: extract proportions, normalize to sum=1.0. Store: {"categories": ["Adm-clerical", ...], "probabilities": [0.1158, ...]}. (5) `SyntheticGenerator.generate_categorical_data()`: `np.random.choice(14_categories, size=32561, p=probabilities)`. (6) `PrivacyGuard`: categorical columns aren't directly used in distance computation, but flagged rows get new categorical values. (7) `Validator.compare_categories()`: compare original vs synthetic proportions — e.g., "Adm-clerical": 0.1158 vs 0.1157 (diff=0.0001). (8) `Visualizer.plot_category_distributions()`: grouped bar chart saved as `category_dist_occupation.png`.

**Q20. Why does your pipeline generate 32,561 rows and not some other number?**

**A**: In config.yaml, `synthetic.num_rows: 0`. In main.py line 170-171: `if num_synthetic_rows <= 0: num_synthetic_rows = len(df_original)`. Setting 0 means "match original dataset size." This is a deliberate design choice: same-size synthetic data allows direct statistical comparison. Users can set any positive integer to generate different sizes.

**Q21. What is Bessel's correction and where does it appear in your code?**

**A**: Bessel's correction (n-1 denominator instead of n) provides an unbiased estimate of population variance from a sample. It appears in `np.cov(data, rowvar=False)` (distribution.py:77) which defaults to `bias=False` (uses n-1). It also appears in `pd.DataFrame.var()` (validator.py:91) which defaults to `ddof=1` (uses n-1). Without it, variance estimates would be systematically too small, leading to synthetic data with insufficient spread.

**Q22. How does the `combine_data()` method ensure columns appear in the correct order?**

**A**: In generator.py:140-180, numerical and categorical DataFrames are concatenated with `pd.concat([numerical_df, categorical_df], axis=1)`. Then columns are reordered: `combined = combined[available_cols]` where `available_cols` is filtered from `original_column_order`. This ensures the synthetic CSV has columns in the exact same order as the original (age, workclass, fnlwgt, ..., income) regardless of how they were generated internally.

**Q23. What happens if the dataset download fails?**

**A**: In data_loader.py:56-60, `requests.get(url, timeout=120)` raises `requests.RequestException` on network failure, which is caught and re-raised as `ConnectionError`. This propagates up to main.py:358-364 where the top-level try/except logs the error with full traceback (`exc_info=True`) and exits with `sys.exit(1)`. The 120-second timeout prevents infinite hangs on unresponsive servers.

---

### Category E: Edge-Case Scenario Questions (6 Questions)

**Q24. What happens if you run the pipeline on a dataset with only categorical columns?**

**A**: The system handles this gracefully. In distribution.py:59-63, if `numerical_columns` is empty, `fit_numerical_distribution()` returns empty arrays. In generator.py:57-59, `generate_numerical_data()` returns an empty DataFrame. The `combine_data()` method detects `numerical_df.empty` and returns only the categorical DataFrame. Privacy's `check_similarity()` returns empty lists when no numerical columns exist (privacy.py:114-116). Validation's `correlation_difference()` returns 0.0 (validator.py:124-127). The system would produce valid synthetic data from categorical distributions alone.

**Q25. What if two columns have identical values (perfect correlation)?**

**A**: Perfect correlation means one column is a linear function of another (e.g., column B = 2×column A). The covariance matrix would have a determinant of 0 (singular), and one eigenvalue would be exactly 0. The PSD check in `_ensure_positive_semi_definite()` would not clamp it (0 ≥ 0). However, `multivariate_normal` might struggle because the distribution degenerates into a lower-dimensional subspace. In practice, the function adds a tiny nugget to the diagonal internally.

**Q26. What if the dataset has only 10 rows?**

**A**: In data_loader.py:106-110, a warning is logged: "Dataset has fewer than 20 rows. Results may be unreliable." The covariance matrix from 10 rows with 6 features would be poorly estimated (only 10 data points for 21 covariance parameters). The system would still run, but synthetic data quality would be low. The privacy check would also be more likely to flag rows because with few original rows, random Gaussian samples may accidentally fall close to them.

**Q27. What if all missing values are in a single column?**

**A**: The imputation strategy handles each column independently. If `capital_gain` has 1000 NaN values and all other columns have none, only `capital_gain` gets imputed with its column mean. This preserves the other columns exactly while filling the gaps. If using `drop` strategy instead, any row with a NaN in that column would be removed — potentially reducing the dataset significantly.

**Q28. What if a categorical column has 1000 unique categories?**

**A**: The distribution learner would compute probabilities for all 1000 categories. `np.random.choice` handles any number of categories. The visualizer caps at 20 categories per chart (visualizer.py:209-214): it selects the top 20 by frequency and shows only those in the bar chart. The validation report would still compare all 1000 categories. Memory usage would remain manageable since categorical probabilities are stored as simple lists.

**Q29. What if the user provides a malformed CSV with inconsistent column counts?**

**A**: `pd.read_csv()` would raise a `ParserError` (e.g., "Expected 15 fields, saw 17"). This is caught by the generic except block in data_loader.py:99-100 and re-raised as `ValueError("Failed to parse CSV file: ...")`. The main.py error handler logs the error and exits with code 1.

---

### Category F: System Design Questions (6 Questions)

**Q30. Why did you choose a modular architecture with separate classes instead of a single script?**

**A**: Separation of concerns: each module has a single responsibility (SRP). DataLoader doesn't know about distributions; Generator doesn't know about validation. This enables: (1) independent testing — each class can be unit-tested in isolation; (2) reusability — the Validator can compare any two DataFrames; (3) extensibility — replacing the Gaussian generator with a GAN requires changing only generator.py; (4) maintainability — bugs are localized to specific modules.

**Q31. How would you scale this system to handle 10 million rows?**

**A**: (1) DataLoader: use `pd.read_csv(chunksize=100000)` for streaming. (2) Profiler: online mean/variance computation (Welford's algorithm). (3) Distribution: covariance matrix is O(n×d²) — still feasible for d=6 but would need tiling for d>100. (4) Generator: generate in chunks, not all at once. (5) Privacy: the O(n×m) distance computation (n=10M synth × m=5K sample) would take ~50 billion operations per batch. Solution: use approximate nearest neighbor (ANN) libraries like FAISS for O(n log n) nearest neighbor queries. (6) Validation: KS test handles any n natively. Memory: ~10M × 15 columns × 8 bytes ≈ 1.2 GB raw — fits in memory.

**Q32. Why YAML for configuration instead of JSON, TOML, or environment variables?**

**A**: YAML supports comments (JSON doesn't), is human-readable (unlike env vars for nested structures), and has excellent Python support via PyYAML. TOML would also work but is less widely used for data science configurations. The hierarchical structure (dataset → download_url, raw_path; privacy → threshold, attempts) maps naturally to YAML's indentation-based nesting.

**Q33. How would you add logging to file instead of just stdout?**

**A**: In main.py:40-54, add a `FileHandler`: `handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("pipeline.log")]`. The existing `format` and `datefmt` would apply to both. For production, add log rotation: `from logging.handlers import RotatingFileHandler` with `maxBytes` and `backupCount`.

**Q34. Why use Matplotlib's Agg backend?**

**A**: In visualizer.py:13, `matplotlib.use("Agg")` selects the non-interactive Anti-Grain Geometry backend, which renders to image files without requiring a display server (X11, Wayland). This is essential for: headless servers, CI/CD pipelines, Docker containers, and SSH sessions. Without it, `plt.show()` would raise `TclError: no display name`.

**Q35. How would you implement this as a REST API?**

**A**: (1) Wrap `run_pipeline()` in a Flask/FastAPI endpoint: `POST /generate` accepting config JSON and returning a job ID. (2) Use Celery + Redis for async job processing (pipeline takes ~40s). (3) `GET /status/{job_id}` returns progress. (4) `GET /download/{job_id}` returns the synthetic CSV. (5) Add authentication via JWT. (6) Serve validation report and plots via static file endpoints. The modular architecture makes this straightforward — each class is already stateless and composable.
