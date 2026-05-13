---
name: web-browse
description: Web search (Google/Google Scholar) and page browsing via web_skills_base_tools. Use when a task needs to fetch content from web pages, search for gene/pathway information online, or access databases like GeneCards, MSigDB, NCBI, Ensembl that lack a programmatic API for the needed data.
---

# Web Browse & Search

Python toolkit for web information gathering: **search** (Google / Google Scholar)
and **browse** web pages/PDFs. All functions are designed to be called inside
`execute_code` blocks.

Package: `web_skills_base_tools` (pre-installed).
Required env vars (pre-configured): `JINA_API_KEY`, `SERP_API_URL`,
`SERP_DEV_KEY`, `SPIDER_API_URL`.

## Quick Start

```python
from web_skills_base_tools import (
    search_serp, search_serp_scholar,   # search
    browse_spider, browse_jina,          # browse
)
```

## 1. Search

### search_serp — Google Search

```python
result = search_serp(query, timeout=20, num=10, start=1)
# Returns: {"status": int, "output": [{"title", "link", "snippet", "date"}, ...]}
```

### search_serp_scholar — Google Scholar

```python
result = search_serp_scholar(query, timeout=20, num=10, start=1)
# Returns: {"status": int, "output": [{"title", "link", "authors", "snippet", "cited_num", "date", "pdf_link"}, ...]}
```

- `num`: max results per call (1–10).
- `start`: 1-based offset for pagination.

## 2. Browse

Two browse backends with the same interface. Fallback priority:

**`browse_spider` → `browse_jina`**

- `browse_spider` — Server-side crawl. Tries multiple download strategies in
  parallel and returns the longest result. Good general success rate.
- `browse_jina` — Jina Reader API. Needs longer timeouts (60–100s) for heavy
  pages but handles Cloudflare-protected sites (e.g., GeneCards) well.

### Interface

```python
result = browse_spider(url, view="text", timeout=30)
result = browse_jina(url, view="text", timeout=100)
# Returns: {"status": int, "page": str}
```

### View Modes

| view | output | use case |
|------|--------|----------|
| `"raw"` | Original HTML | Parse specific elements (tables, JSON-LD) |
| `"text"` | Full page text, links as `text [url]` | General reading, following links |
| `"main"` | Main content only, markdown | Clean article body |

### Fallback Pattern

```python
from web_skills_base_tools import browse_spider, browse_jina

def browse(url, view="text"):
    """Spider → Jina fallback with escalating timeouts."""
    chain = [
        (browse_spider, 30),
        (browse_spider, 60),
        (browse_jina,   60),
        (browse_jina,  100),
    ]
    for func, timeout in chain:
        r = func(url, view=view, timeout=timeout)
        if r["status"] == 200 and len(r.get("page", "")) > 200:
            return r
    return r  # return last attempt even if failed
```

Adapt the chain to the task. For Cloudflare-heavy sites (GeneCards), start
with Jina directly.

## 3. Bioinformatics-Specific Guidance

### GeneCards

GeneCards pages are behind Cloudflare. `browse_jina` with timeout ≥ 100s
works reliably. `browse_spider` sometimes succeeds, sometimes gets blocked.

```python
# Fetch a specific gene page
r = browse_jina(
    "https://www.genecards.org/cgi-bin/carddisp.pl?gene=TP53",
    view="text", timeout=100,
)
```

GeneCards pages are very large (200–500 KB text). Save to file and parse
programmatically rather than trying to read the full output:

```python
page = r["page"]
with open("genecards_tp53.txt", "w") as f:
    f.write(page)

# Extract specific sections by searching for section headers
import re
# Find the "Function" section
match = re.search(r'Function\s*\n(.*?)(?=\n[A-Z][a-z]+ \n|\Z)', page, re.DOTALL)
```

### MSigDB / GSEA

MSigDB pages work well with `browse_spider`:

```python
r = browse_spider(
    "https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_INFLAMMATORY_RESPONSE.html",
    view="text", timeout=60,
)
```

### General Bioinformatics Databases

Most public databases (NCBI, Ensembl, UniProt, STRING) work with
`browse_spider` at default timeouts. Use `search_serp` to find the right
URL first if needed.

## 4. Working with Large Results

Browse results can be very long. Save to file and explore programmatically:

```python
r = browse_spider(url, view="text")
with open("page.txt", "w") as f:
    f.write(r["page"])

# Preview structure
print(r["page"][:2000])

# Search for specific content
import re
for m in re.finditer(r'inflammatory|inflammation', r["page"], re.IGNORECASE):
    start = max(0, m.start() - 100)
    end = min(len(r["page"]), m.end() + 100)
    print(f"...{r['page'][start:end]}...")
    print("---")
```

## 5. Search → Browse → Extract Pattern

```python
from web_skills_base_tools import search_serp, browse_jina

# Step 1: Find relevant pages
results = search_serp("site:genecards.org inflammation liver cancer", num=10)
urls = [item["link"] for item in results["output"]]

# Step 2: Browse each page
for url in urls[:3]:
    r = browse_jina(url, view="text", timeout=100)
    if r["status"] == 200:
        # Step 3: Extract what you need
        with open(f"data/{url.split('gene=')[-1]}.txt", "w") as f:
            f.write(r["page"])
```
