---
name: database-api-reference
description: "Quick-reference curl/Python templates for 77+ public life science databases. Use when querying genomics, proteomics, pathway, drug, clinical, or literature databases via REST API. Covers NCBI E-utilities, Ensembl, UniProt, GTEx, ChEMBL, PubChem, OpenTargets, ClinicalTrials.gov, STRING, Enrichr, Reactome, and more."
license: MIT
metadata:
    skill-author: BioCompute (consolidated from ScienceClaw SCIENCE.md templates)
---

# Database API Quick Reference

## Overview

This skill provides ready-to-use curl and Python templates for querying 77+ public life science databases via their REST APIs. Use this as a quick-lookup reference when you need to programmatically access biological data. For detailed database-specific guidance (query syntax, pagination, advanced filters), refer to the individual database skills (e.g., `pubmed-database`, `ensembl-database`).

## When to Use This Skill

- You need a quick curl template to query a specific database
- You want to combine data from multiple databases in a single script
- You need to check which API endpoint to use for a given data type
- The individual database skill is not available or you need a quick reminder

## Genomics & Transcriptomics

### NCBI E-utilities (Gene, PubMed, GEO, ClinVar, etc.)

Base URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`

```bash
# Search for a gene
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmode=json&term=BRCA1+AND+human[orgn]"

# Search PubMed
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=20&sort=relevance&term=QUERY"

# Fetch PubMed abstracts by PMID
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id=PMID1,PMID2,PMID3"

# Search GEO datasets
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&retmode=json&term=QUERY"

# Fetch GEO dataset metadata
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&retmode=json&id=GDS_ID"

# Search ClinVar
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=clinvar&retmode=json&term=GENE"

# Fetch gene summary
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&retmode=json&id=GENE_ID"

# Link PubMed articles to gene
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gene&db=pubmed&retmode=json&id=GENE_ID"
```

**Tip**: Add `&api_key=YOUR_KEY` for higher rate limits (10 req/s vs 3 req/s without key). Register at https://www.ncbi.nlm.nih.gov/account/

### Ensembl REST API

Base URL: `https://rest.ensembl.org/`

```bash
# Gene lookup by symbol
curl -s "https://rest.ensembl.org/lookup/symbol/homo_sapiens/BRCA1?content-type=application/json;expand=1"

# Gene lookup by Ensembl ID
curl -s "https://rest.ensembl.org/lookup/id/ENSG00000012048?content-type=application/json"

# Get gene sequence
curl -s "https://rest.ensembl.org/sequence/id/ENSG00000012048?content-type=application/json;type=genomic"

# Cross-references (IDs mapping)
curl -s "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/BRCA1?content-type=application/json"

# Variant consequences (VEP)
curl -s "https://rest.ensembl.org/vep/human/hgvs/BRAF:p.Val600Glu?content-type=application/json"

# Regulatory features in a region
curl -s "https://rest.ensembl.org/overlap/region/human/7:140424943-140624564?feature=regulatory;content-type=application/json"
```

### GTEx (Gene Expression)

```bash
# Median gene expression across tissues
curl -s "https://gtexportal.org/api/v2/expression/medianGeneExpression?gencodeId=ENSG00000012048.23&datasetId=gtex_v8"

# Single-tissue eQTLs
curl -s "https://gtexportal.org/api/v2/association/singleTissueEqtl?gencodeId=ENSG00000012048.23&tissueSiteDetailId=Liver&datasetId=gtex_v8"

# Top expressed genes in a tissue
curl -s "https://gtexportal.org/api/v2/expression/topExpressedGene?tissueSiteDetailId=Brain_Cortex&datasetId=gtex_v8&sortBy=median&sortDirection=desc&page=0&pageSize=50"
```

### UCSC Genome Browser

```bash
# Get sequence for a region
curl -s "https://api.genome.ucsc.edu/getData/sequence?genome=hg38&chrom=chr7&start=140424943&end=140624564"

# List available tracks
curl -s "https://api.genome.ucsc.edu/list/tracks?genome=hg38"
```

## Proteomics & Structure

### UniProt

```bash
# Search by gene name
curl -s "https://rest.uniprot.org/uniprotkb/search?query=gene_exact:BRCA1+AND+organism_id:9606&format=json&size=5"

# Fetch by accession
curl -s "https://rest.uniprot.org/uniprotkb/P38398?format=json"

# ID mapping (e.g., gene name to UniProt)
curl -s "https://rest.uniprot.org/idmapping/run" -d "from=Gene_Name&to=UniProtKB&ids=BRCA1&taxId=9606"
```

### PDB (Protein Data Bank)

```bash
# Search structures
curl -s "https://search.rcsb.org/rcsbsearch/v2/query" -H "Content-Type: application/json" -d '{
  "query": {"type": "terminal", "service": "text", "parameters": {"value": "BRCA1"}},
  "return_type": "entry",
  "request_options": {"results_content_type": ["experimental"], "return_all_hits": false, "pager": {"start": 0, "rows": 10}}
}'

# Fetch structure summary
curl -s "https://data.rcsb.org/rest/v1/core/entry/1JM7"

# Fetch polymer entity (chain info)
curl -s "https://data.rcsb.org/rest/v1/core/polymer_entity/1JM7/1"
```

### AlphaFold Database

```bash
# Get predicted structure by UniProt ID
curl -s "https://alphafold.ebi.ac.uk/api/prediction/P38398"

# Download PDB file
curl -sO "https://alphafold.ebi.ac.uk/files/AF-P38398-F1-model_v4.pdb"

# Download PAE (Predicted Aligned Error) JSON
curl -sO "https://alphafold.ebi.ac.uk/files/AF-P38398-F1-predicted_aligned_error_v4.json"
```

### STRING (Protein-Protein Interactions)

```bash
# Get interaction network
curl -s "https://string-db.org/api/json/network?identifiers=BRCA1&species=9606"

# Get interaction partners (sorted by score)
curl -s "https://string-db.org/api/json/interaction_partners?identifiers=BRCA1&species=9606&limit=20"

# Functional enrichment
curl -s "https://string-db.org/api/json/enrichment?identifiers=BRCA1%0dTP53%0dATM&species=9606"

# Resolve protein name to STRING ID
curl -s "https://string-db.org/api/json/get_string_ids?identifiers=BRCA1&species=9606"
```

### InterPro (Protein Domains)

```bash
# Search by UniProt accession
curl -s "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/P38398?page_size=200"
```

## Chemistry & Drugs

### ChEMBL

```bash
# Search molecule by name
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=aspirin&limit=5"

# Get molecule by ChEMBL ID
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json"

# Get bioactivities for a target
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL1824&limit=20"

# Search target by gene name
curl -s "https://www.ebi.ac.uk/chembl/api/data/target/search.json?q=EGFR&limit=5"
```

### PubChem

```bash
# Search compound by name
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/JSON"

# Get compound properties
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount/JSON"

# Search by SMILES substructure
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/substructure/smiles/c1ccccc1/JSON"

# Get bioassay results
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/assaysummary/JSON"
```

### DrugBank (limited free access)

```bash
# DrugBank public API is limited. Use the open data download:
# https://go.drugbank.com/releases/latest

# Alternative: OpenFDA for drug labels
curl -s "https://api.fda.gov/drug/label.json?search=openfda.brand_name:aspirin&limit=3"
```

### OpenTargets

```bash
# GraphQL: Gene-disease associations
curl -s -X POST "https://api.platform.opentargets.org/api/v4/graphql" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ target(ensemblId: \"ENSG00000012048\") { id approvedSymbol associatedDiseases(page: {index: 0, size: 10}) { rows { disease { id name } score } } } }"}'

# GraphQL: Drug mechanisms for a target
curl -s -X POST "https://api.platform.opentargets.org/api/v4/graphql" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ target(ensemblId: \"ENSG00000146648\") { id approvedSymbol knownDrugs(size: 10) { rows { drug { id name } phase mechanismOfAction } } } }"}'
```

## Pathway & Enrichment

### KEGG

```bash
# Find pathway by keyword
curl -s "https://rest.kegg.jp/find/pathway/apoptosis"

# Get pathway entry
curl -s "https://rest.kegg.jp/get/hsa04210"

# List genes in a pathway
curl -s "https://rest.kegg.jp/link/hsa/hsa04210"

# Get pathway image
curl -sO "https://rest.kegg.jp/get/hsa04210/image"

# Convert gene IDs
curl -s "https://rest.kegg.jp/conv/ncbi-geneid/hsa:672"
```

### Reactome

```bash
# Search pathways
curl -s "https://reactome.org/ContentService/search/query?query=apoptosis&types=Pathway&species=Homo+sapiens&cluster=true"

# Get pathway details
curl -s "https://reactome.org/ContentService/data/query/R-HSA-109581"

# Pathway analysis (gene list enrichment) — POST
curl -s -X POST "https://reactome.org/AnalysisService/identifiers/?pageSize=20&page=1" \
  -H "Content-Type: text/plain" \
  -d "BRCA1
TP53
ATM
CHEK2"
```

### Enrichr

```bash
# Step 1: Submit gene list
curl -s -X POST "https://maayanlab.cloud/Enrichr/addList" \
  -F "list=BRCA1\nTP53\nATM\nCHEK2\nRAD51" \
  -F "description=my gene list"
# Returns: {"shortId": "abc123", "userListId": 12345}

# Step 2: Get enrichment results
curl -s "https://maayanlab.cloud/Enrichr/enrich?userListId=12345&backgroundType=KEGG_2021_Human"
```

## Clinical & Disease

### ClinicalTrials.gov (API v2)

```bash
# Search trials
curl -s "https://clinicaltrials.gov/api/v2/studies?query.term=BRCA1+breast+cancer&pageSize=10"

# Get specific trial
curl -s "https://clinicaltrials.gov/api/v2/studies/NCT04072718"

# Filter by status and phase
curl -s "https://clinicaltrials.gov/api/v2/studies?query.term=pembrolizumab&filter.overallStatus=RECRUITING&filter.phase=PHASE3&pageSize=10"
```

### ClinVar (via E-utilities)

```bash
# Search variants for a gene
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=clinvar&retmode=json&term=BRCA1[gene]+AND+pathogenic[clinsig]&retmax=20"

# Fetch variant details
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=clinvar&retmode=json&id=CLINVAR_ID"
```

### OMIM (registration required)

```bash
# Search OMIM
curl -s "https://api.omim.org/api/entry/search?search=BRCA1&format=json&apiKey=YOUR_KEY"
```

### DisGeNET (registration required)

```bash
# Gene-disease associations
curl -s -H "Authorization: Bearer YOUR_TOKEN" \
  "https://www.disgenet.org/api/gda/gene/BRCA1?source=ALL&format=json"
```

### GWAS Catalog

```bash
# Search associations by gene
curl -s "https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms/search/findByGene?geneName=BRCA1"

# Search by trait
curl -s "https://www.ebi.ac.uk/gwas/rest/api/efoTraits/search/findBySearchQuery?query=breast+cancer"
```

### OpenFDA (Drug Labels, Adverse Events)

```bash
# Drug labels
curl -s "https://api.fda.gov/drug/label.json?search=openfda.generic_name:metformin&limit=3"

# Adverse events
curl -s "https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:metformin&count=patient.reaction.reactionmeddrapt.exact&limit=10"
```

## Literature Search APIs

### PubMed (see NCBI E-utilities above)

### OpenAlex

```bash
# Search works
curl -s "https://api.openalex.org/works?search=BRCA1+breast+cancer&per_page=10&sort=relevance_score:desc&select=id,title,authorships,publication_year,cited_by_count,doi,primary_location"

# Search by author
curl -s "https://api.openalex.org/authors?search=John+Smith&per_page=5"

# Get author profile with works
curl -s "https://api.openalex.org/authors/A5023888391?select=id,display_name,works_count,cited_by_count,summary_stats"
```

### Semantic Scholar

```bash
# Search papers
curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=BRCA1+breast+cancer&limit=10&fields=title,authors,year,abstract,citationCount,externalIds,url"

# Get paper details by PMID
curl -s "https://api.semanticscholar.org/graph/v1/paper/PMID:12345678?fields=title,authors,year,abstract,citationCount,references,citations"

# Forward citations
curl -s "https://api.semanticscholar.org/graph/v1/paper/PMID:12345678/citations?fields=title,authors,year,citationCount&limit=10"

# References (backward citations)
curl -s "https://api.semanticscholar.org/graph/v1/paper/PMID:12345678/references?fields=title,authors,year,citationCount&limit=10"

# Author search
curl -s "https://api.semanticscholar.org/graph/v1/author/search?query=John+Smith&fields=name,hIndex,paperCount,citationCount"
```

### Europe PMC

```bash
# Search articles
curl -s "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=BRCA1&format=json&pageSize=10&sort=CITED+desc"

# Get full text XML (open access)
curl -s "https://www.ebi.ac.uk/europepmc/webservices/rest/PMID/fullTextXML"

# Get citations
curl -s "https://www.ebi.ac.uk/europepmc/webservices/rest/MED/PMID/citations?format=json&page=1&pageSize=25"
```

### Full Text via Jina Reader

```bash
# Read full text of any URL (paper, documentation, etc.)
curl -s "https://r.jina.ai/https://doi.org/10.1234/example.doi"
```

## Metabolomics & Small Molecules

### HMDB (Human Metabolome Database)

```bash
# Search metabolite
curl -s "https://hmdb.ca/metabolites.xml?utf8=✓&query=glucose&search_type=metabolites&button="

# Get metabolite by HMDB ID (XML)
curl -s "https://hmdb.ca/metabolites/HMDB0000122.xml"
```

### ZINC (Purchasable Compounds)

```bash
# Search by SMILES
curl -s "https://zinc15.docking.org/substances/search/?q=c1ccccc1&output_format=json"

# Get substance info
curl -s "https://zinc15.docking.org/substances/ZINC000000000001.json"
```

## Multi-Source Query Patterns

### Parallel search (recommended for literature review)

```bash
echo "=== PubMed ===" && \
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=20&sort=relevance&term=QUERY" && \
echo -e "\n=== OpenAlex ===" && \
curl -s "https://api.openalex.org/works?search=QUERY&per_page=10&sort=relevance_score:desc&select=id,title,authorships,publication_year,cited_by_count,doi,primary_location" && \
echo -e "\n=== Semantic Scholar ===" && \
curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=QUERY&limit=10&fields=title,authors,year,abstract,citationCount,externalIds,url"
```

### Gene annotation pipeline

```bash
GENE="BRCA1"
echo "=== Ensembl ===" && \
curl -s "https://rest.ensembl.org/lookup/symbol/homo_sapiens/${GENE}?content-type=application/json" && \
echo -e "\n=== UniProt ===" && \
curl -s "https://rest.uniprot.org/uniprotkb/search?query=gene_exact:${GENE}+AND+organism_id:9606&format=json&size=1" && \
echo -e "\n=== STRING interactions ===" && \
curl -s "https://string-db.org/api/json/interaction_partners?identifiers=${GENE}&species=9606&limit=10" && \
echo -e "\n=== OpenTargets diseases ===" && \
curl -s -X POST "https://api.platform.opentargets.org/api/v4/graphql" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"{ search(queryString: \\\"${GENE}\\\", entityNames: [\\\"target\\\"]) { hits { id ... on Target { approvedSymbol associatedDiseases { count } } } } }\"}"
```

## Tips

- **Rate limits**: Most APIs have rate limits. NCBI allows 3 req/s without API key, 10 req/s with key. Semantic Scholar allows ~100 req/5min.
- **Pagination**: Most APIs support pagination. Check `nextCursorMark`, `offset`, `page` parameters.
- **JSON parsing**: Use `python3 -c "import sys,json; ..."` or `jq` to parse responses in bash.
- **For detailed usage**: Refer to individual database skills (e.g., `pubmed-database`, `ensembl-database`) for advanced query syntax, field tags, and pagination patterns.
