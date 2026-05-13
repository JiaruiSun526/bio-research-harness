# Data Quality Control and Auditing

## Description

Systematic assessment of dataset quality before analysis. Covers missing
values, distributions, outliers, data types, sample-level QC, and consistency
checks. Applies to tabular data (clinical, expression, experimental).

## When to Use

- As the first step of any analysis project, before any statistical or
  computational methods are applied.
- When receiving a new dataset or combining multiple data sources.
- After data transformations to verify integrity is maintained.

## Method

Data QC is not a single algorithm but a structured checklist applied to the
dataset. The output is a QC report documenting findings and flagging issues
that require attention before downstream analysis.

### Step 1: Schema Inspection

```python
import pandas as pd

df = pd.read_csv("data.csv")  # or .tsv, .parquet, etc.

# Basic shape and types
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# First few rows
df.head()
```

### Step 2: Missing Value Assessment

```python
# Per-column missing counts and percentages
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct,
}).query("missing_count > 0").sort_values("missing_pct", ascending=False)
```

### Step 3: Distribution Analysis

```python
# Numeric columns: summary statistics
df.describe()

# Categorical columns: value counts
for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head(10))
```

### Step 4: Outlier Detection

```python
import numpy as np

# IQR-based outlier detection for numeric columns
for col in df.select_dtypes(include=np.number).columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    if outliers > 0:
        print(f"{col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")
```

### Step 5: Sample-Level QC (for expression/omics data)

```python
# Total counts per sample (for RNA-seq count matrices)
sample_totals = df.sum(axis=0)  # or axis=1, depending on orientation
print(f"Library size range: {sample_totals.min():.0f} - {sample_totals.max():.0f}")
print(f"Samples below 1M reads: {(sample_totals < 1e6).sum()}")

# PCA to detect batch effects or sample swaps
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(df.T)  # samples as rows
```

## Critical Checks

1. **Data types**: Numeric columns stored as strings (e.g., "1,234" with
   comma separators, or "NA" as string). Detect with `df.dtypes` and
   sample value inspection.
2. **Duplicate rows/IDs**: Check for duplicate sample IDs or row identifiers.
   `df.duplicated().sum()` and `df[id_col].duplicated().sum()`.
3. **Constant columns**: Columns with zero variance provide no information.
   `(df.nunique() == 1).sum()`.
4. **ID column alignment**: When merging clinical and expression data, verify
   sample IDs match exactly (case, whitespace, prefix differences).
5. **Value ranges**: Check that values are biologically plausible (e.g., ages
   0-120, gene expression counts non-negative, percentages 0-100).

## Known Pitfalls

- **Silent type coercion**: pandas reads mixed-type columns as object dtype.
  A column with 99% integers and one "NA" string becomes all strings.
- **Index vs column**: Gene/sample identifiers may be in the index or a column
  depending on how the file was saved. Always check `df.index` explicitly.
- **Compressed file encoding**: `.gz` files may have different line endings or
  encoding than expected. Always specify encoding when reading.
- **Missing value representations**: Different sources use different missing
  markers: NA, N/A, NaN, "", ".", -999, 0. Identify the convention used.
- **Log-transformed vs raw**: Expression values that are all positive floats
  in the range 0-20 are likely log-transformed. Integer counts in hundreds
  to millions are likely raw. This distinction is critical for tool choice
  (DESeq2 requires raw counts).

## Expected Outputs

- **QC report** (Markdown or text): Summary of dataset properties, missing
  values, distributions, outliers, and flagged issues.
- **Missing value heatmap** (PNG): Visual pattern of missingness across
  samples and features.
- **Distribution plots** (PNG): Histograms or boxplots for key numeric columns.
- **Sample correlation/PCA plot** (PNG): For omics data, to detect outlier
  samples or batch effects.
- **Actionable recommendations**: Specific steps to address identified issues
  before downstream analysis.

## Validation

- QC report that flags zero issues on a real-world dataset is suspicious —
  real data almost always has quirks.
- Missing value percentages should be consistent with the data source's
  documentation or known collection limitations.
- Outlier counts exceeding 10% of samples suggest the threshold may be
  inappropriate for the data distribution, not that 10% of data is bad.
