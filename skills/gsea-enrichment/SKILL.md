# GSEA and Pathway Enrichment Analysis

## Description

Gene Set Enrichment Analysis (GSEA) and Over-Representation Analysis (ORA)
using gseapy. Determines whether predefined gene sets show statistically
significant enrichment in ranked gene lists or differentially expressed genes.

## When to Use

- After differential expression analysis to interpret biological meaning of
  DEG lists or ranked gene lists.
- When you have a ranked gene list (by fold change, p-value, or other metric)
  and want to test pathway/gene set enrichment.
- ORA: when you have a discrete gene list (e.g., significant DEGs).
- GSEA (preranked): when you have a continuous ranking metric for all genes.

## Method

### GSEA (Preranked)

Uses the full ranked gene list. Does not require an arbitrary significance
cutoff. Preferred when a continuous ranking metric is available.

```python
import gseapy as gp

# gene_ranking: pandas Series or DataFrame with gene names as index,
# ranking metric (e.g., -log10(pvalue) * sign(log2FC)) as values.
result = gp.prerank(
    rnk=gene_ranking,
    gene_sets="MSigDB_Hallmark_2020",  # or path to .gmt file
    outdir="gsea_output",
    seed=42,
    permutation_num=1000,
)
```

### ORA (Enrichr-style)

Tests a discrete gene list against background using Fisher's exact test.

```python
result = gp.enrich(
    gene_list=deg_list,       # list of gene symbols
    gene_sets="GO_Biological_Process_2023",
    background=background_genes,  # all tested genes (not just DEGs)
    outdir="ora_output",
)
```

## Gene Set Sources

- **MSigDB collections**: Hallmark, C2 (curated), C5 (GO), C7 (immunologic).
  Use via gseapy's built-in names or download .gmt files from MSigDB.
- **Enrichr libraries**: GO_Biological_Process, KEGG, Reactome, WikiPathways.
  gseapy supports Enrichr library names directly.
- **Custom .gmt files**: For organism-specific or project-specific gene sets.

## Critical Checks

1. **Gene ID format**: Gene set databases use gene symbols (BRCA1, TP53).
   If your DEG list uses Ensembl IDs, convert first (see gget skill).
2. **Background gene list for ORA**: Must be all genes that were tested in the
   differential expression analysis, not all genes in the genome. Using the
   wrong background inflates significance.
3. **Ranking metric for preranked GSEA**: Use a signed metric that captures
   both direction and significance. Common choice: `-log10(pvalue) * sign(log2FC)`.
   Never use unsigned p-values alone (loses directionality).
4. **Multiple testing**: gseapy applies FDR correction. Use `fdr < 0.25` for
   GSEA (standard threshold) and `adjusted_p_value < 0.05` for ORA.

## Known Pitfalls

- **Redundant gene sets**: GO terms are hierarchical; enrichment results often
  contain many overlapping terms. Consider filtering by term size or using
  semantic similarity to reduce redundancy.
- **Small gene sets**: Gene sets with fewer than 15 genes have low statistical
  power. Gene sets with more than 500 genes are too broad to be informative.
  Filter by size: `min_size=15, max_size=500`.
- **Leading edge genes**: The leading edge subset (genes driving enrichment)
  is often more informative than the enrichment score itself. Always report it.
- **Organism mismatch**: Ensure gene set database matches your organism.
  Human gene sets applied to mouse data will miss genes.
- **gseapy version differences**: API has changed across versions. Check
  `gp.__version__` and consult current documentation.

## Expected Outputs

- **Enrichment table** (CSV): term, es/nes, pvalue, fdr, matched_genes,
  gene_set_size, leading_edge_genes.
- **Enrichment dot plot** (PNG): Top enriched terms with NES/fold enrichment
  on x-axis, term name on y-axis, dot size = gene count, color = FDR.
- **GSEA enrichment plots** (PNG): Running enrichment score plots for top terms.
- **Summary statistics**: Total terms tested, significant terms (FDR < threshold),
  top 5 terms with NES and FDR.

## Validation

- Zero significant pathways with a large DEG list (>500 genes) suggests a
  problem: wrong gene ID format, missing background, or wrong gene set database.
- Enrichment of very generic terms only (e.g., "metabolic process" with 2000
  genes) without specific pathways suggests the analysis lacks resolution —
  try more specific gene set collections.
- NES values close to zero for all terms suggests the ranking metric has
  insufficient signal.
