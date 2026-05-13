# Stage 1: T Cell Differential Expression

## Objective
Identify differentially expressed genes between treatment and control in T cell samples using DESeq2.

## Data
- Input: data/tcell_counts.csv (10 genes x 6 samples)
- Metadata: data/sample_metadata.csv

## Method
- DESeq2 with FDR < 0.05 correction
- Volcano plot of results
