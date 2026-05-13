"""Generate a scenario TOML from a paper and data directory.

Usage:
    python scripts/prepare_scenario.py \
        --name phagecounting \
        --paper references/user_paper/paper_text.txt \
        --data-dir references/user_paper/data \
        [--model openrouter/openai/gpt-5.4] \
        [--max-turns 50]

Pipeline:
    paper text/PDF + data directory
      → LLM call 1: generate SimulatedUser research_brief
      → LLM call 2: generate data catalog entries
      → assemble scenarios/<name>.toml
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "llm.toml"


def _load_openrouter_config() -> dict[str, str]:
    """Load [openrouter] section from config/llm.toml.

    Returns dict with keys: api_key, proxy, default_model.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    if not CONFIG_PATH.exists():
        log.error("Config file not found: %s (copy from %s.example)", CONFIG_PATH, CONFIG_PATH)
        sys.exit(1)

    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    section = config.get("openrouter")
    if not section:
        log.error("[openrouter] section missing from %s", CONFIG_PATH)
        sys.exit(1)

    return {
        "api_key": section["api_key"],
        "proxy": section.get("proxy", ""),
        "default_model": section.get("default_model", "openrouter/openai/gpt-5.4"),
    }


def _build_client(api_key: str, proxy: str) -> OpenAI:
    """Build OpenAI client for OpenRouter with optional proxy."""
    client_kwargs: dict = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": api_key,
    }
    if proxy:
        import httpx

        client_kwargs["http_client"] = httpx.Client(proxy=proxy)

    return OpenAI(**client_kwargs)


# ---------------------------------------------------------------------------
# File scanning (adapted from scripts_for_assets/generate_data_catalog.py)
# ---------------------------------------------------------------------------

COLLAPSE_THRESHOLD = 10

BINARY_EXTENSIONS = {
    ".bam", ".bai", ".cram", ".crai", ".bcf", ".sra",
    ".pkl", ".pickle", ".npy", ".npz", ".pt", ".pth", ".onnx",
    ".h5", ".hdf5", ".hdf", ".zarr", ".db", ".sqlite", ".sqlite3", ".lmdb",
    ".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".bmp", ".svg",
    ".tar", ".zip", ".bz2", ".xz", ".7z", ".rar",
    ".part", ".partial",
    ".pdf", ".docx", ".xlsx", ".pptx", ".whl", ".so", ".dylib", ".exe",
    ".mat",
}


@dataclass
class FileEntry:
    """A single file or collapsed directory of same-extension files."""

    rel_path: str
    file_count: int
    total_bytes: int
    ext: str
    content_sample: str
    tabular_meta: str | None
    sample_filenames: list[str] = field(default_factory=list)


def _is_text_file(path: Path) -> bool:
    name = path.name.lower()
    if any(name.endswith(ext) for ext in BINARY_EXTENSIONS):
        return False
    if name.endswith(".gz") and "." in name[:-3]:
        inner = name[:-3]
        if any(inner.endswith(ext) for ext in BINARY_EXTENSIONS):
            return False
    return True


def _read_sample(path: Path, max_bytes: int) -> str:
    name = path.name.lower()
    try:
        if name.endswith(".gz"):
            with gzip.open(path, "rt", errors="replace") as fh:
                return fh.read(max_bytes)
        else:
            with open(path, "r", errors="replace") as fh:
                return fh.read(max_bytes)
    except Exception as exc:
        return f"[could not read: {exc}]"


def _get_rich_ext(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gz") and "." in name[:-3]:
        return "." + name.split(".", 1)[1]
    return path.suffix.lower()


def _size_label(nbytes: int) -> str:
    if nbytes >= 1024 ** 3:
        return f"{nbytes / 1024**3:.1f} GB"
    if nbytes >= 1024 ** 2:
        return f"{nbytes / 1024**2:.1f} MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes} bytes"


def _compute_tabular_metadata(path: Path) -> str | None:
    """Compute column names, dtypes, row count for CSV/TSV/Excel files."""
    name_lower = path.name.lower()

    # Route to Excel handler for .xlsx/.xls files
    if name_lower.endswith((".xlsx", ".xls")):
        return _compute_xlsx_metadata(path)

    if ".tsv" in name_lower:
        sep = "\t"
    elif ".csv" in name_lower:
        sep = ","
    else:
        return None

    try:
        import pandas as pd
    except ImportError:
        return None

    try:
        df_head = pd.read_csv(path, sep=sep, nrows=20)
    except Exception:
        return None

    parts: list[str] = []
    parts.append(f"Columns ({len(df_head.columns)}): {list(df_head.columns)}")
    parts.append(f"Dtypes: {dict(df_head.dtypes.astype(str))}")

    numeric_cols = df_head.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        for col in numeric_cols[:5]:
            vals = df_head[col].dropna()
            if len(vals) == 0:
                continue
            int_frac = (vals == vals.astype(int)).mean()
            parts.append(
                f"  Column '{col}': integer_fraction={int_frac:.2f}, "
                f"min={vals.min()}, max={vals.max()}, "
                f"sample_values={vals.head(3).tolist()}"
            )

    file_size = path.stat().st_size
    if not name_lower.endswith(".gz") and file_size < 50 * 1024 * 1024:
        try:
            with open(path) as fh:
                row_count = sum(1 for _ in fh) - 1
            parts.append(f"Row count: {row_count}")
        except Exception:
            pass

    return "\n".join(parts)


def _compute_xlsx_metadata(path: Path) -> str | None:
    """Compute sheet names, column names, dtypes, row count for Excel files."""
    try:
        import pandas as pd
    except ImportError:
        return None

    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return None

    parts: list[str] = []
    parts.append(f"Sheets: {xl.sheet_names}")

    for sheet_name in xl.sheet_names[:3]:
        try:
            df = xl.parse(sheet_name)
        except Exception:
            parts.append(f"Sheet '{sheet_name}': [could not parse]")
            continue

        parts.append(
            f"Sheet '{sheet_name}': {len(df)} rows × {len(df.columns)} columns"
        )
        parts.append(f"  Columns: {list(df.columns)}")
        parts.append(f"  Dtypes: {dict(df.dtypes.astype(str))}")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            for col in numeric_cols[:3]:
                vals = df[col].dropna()
                if len(vals) == 0:
                    continue
                parts.append(
                    f"  Column '{col}': min={vals.min()}, max={vals.max()}, "
                    f"sample_values={vals.head(3).tolist()}"
                )

        # Show a few sample values from non-numeric columns for context
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        for col in obj_cols[:3]:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 20:
                parts.append(f"  Column '{col}': unique_values={list(unique_vals)}")
            else:
                parts.append(
                    f"  Column '{col}': {len(unique_vals)} unique values, "
                    f"sample={list(unique_vals[:5])}"
                )

    return "\n".join(parts)


def scan_data_dir(data_dir: Path, max_sample_bytes: int = 2048) -> list[FileEntry]:
    """Walk data_dir and return FileEntry list with collapse for large dirs."""
    entries: list[FileEntry] = []
    _scan_recursive(data_dir, data_dir, max_sample_bytes, entries)
    return entries


def _scan_recursive(
    dirpath: Path, data_dir: Path, max_sample_bytes: int, out: list[FileEntry],
) -> None:
    direct_files = sorted(f for f in dirpath.iterdir() if f.is_file())
    subdirs = sorted(d for d in dirpath.iterdir() if d.is_dir())

    for subdir in subdirs:
        _scan_recursive(subdir, data_dir, max_sample_bytes, out)

    if not direct_files:
        return

    by_ext: dict[str, list[Path]] = {}
    for f in direct_files:
        ext = _get_rich_ext(f)
        by_ext.setdefault(ext, []).append(f)

    for ext, files in sorted(by_ext.items()):
        if len(files) > COLLAPSE_THRESHOLD:
            out.append(_make_collapsed_entry(files, ext, dirpath, data_dir, max_sample_bytes))
        else:
            for f in files:
                out.append(_make_file_entry(f, data_dir, max_sample_bytes))


def _make_file_entry(path: Path, data_dir: Path, max_sample_bytes: int) -> FileEntry:
    rel = str(path.relative_to(data_dir))
    ext = _get_rich_ext(path)
    size = path.stat().st_size

    if _is_text_file(path):
        sample = _read_sample(path, max_sample_bytes)
        tabular = _compute_tabular_metadata(path)
    elif ext in (".xlsx", ".xls"):
        # Excel files are binary but we can extract structured metadata
        sample = "[Excel file — see tabular metadata below]"
        tabular = _compute_tabular_metadata(path)
    else:
        sample = "[binary file — content not shown]"
        tabular = None

    return FileEntry(
        rel_path=rel, file_count=1, total_bytes=size, ext=ext,
        content_sample=sample[:max_sample_bytes], tabular_meta=tabular,
        sample_filenames=[path.name],
    )


def _make_collapsed_entry(
    files: list[Path], ext: str, dirpath: Path, data_dir: Path, max_sample_bytes: int,
) -> FileEntry:
    rel = str(dirpath.relative_to(data_dir))
    total_size = sum(f.stat().st_size for f in files)
    sample_names = [f.name for f in files[:3]]

    first = files[0]
    if _is_text_file(first):
        sample = _read_sample(first, max_sample_bytes)
        tabular = _compute_tabular_metadata(first)
    else:
        sample = "[binary file — content not shown]"
        tabular = None

    return FileEntry(
        rel_path=f"{rel}/*{ext}", file_count=len(files), total_bytes=total_size,
        ext=ext, content_sample=sample[:max_sample_bytes], tabular_meta=tabular,
        sample_filenames=sample_names,
    )


def _format_file_entry(entry: FileEntry) -> str:
    lines: list[str] = []
    lines.append(f"=== {entry.rel_path} ===")
    lines.append(f"Files: {entry.file_count} | Size: {_size_label(entry.total_bytes)} | Ext: {entry.ext}")

    if entry.file_count > 1:
        lines.append(f"Sample filenames: {', '.join(entry.sample_filenames)}")

    if entry.tabular_meta:
        lines.append(f"Tabular metadata:\n{entry.tabular_meta}")
        budget = 512
    else:
        budget = 1024

    sample = entry.content_sample[:budget]
    if len(entry.content_sample) > budget:
        sample += "\n[... truncated]"
    lines.append(f"Content sample:\n{sample}")
    return "\n".join(lines)


def _build_file_inventory(entries: list[FileEntry]) -> str:
    return "\n\n".join(_format_file_entry(e) for e in entries)


def _collect_source_paths(data_dir: Path) -> list[str]:
    """Collect all file paths under data_dir, relative to PROJECT_ROOT.

    These become [data].sources in the scenario TOML — files copied into
    the workspace at startup. Paths must be relative to the project root
    because run_validation.py resolves them via PROJECT_ROOT / source.
    """
    data_dir = data_dir.resolve()
    paths: list[str] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            try:
                rel = str(path.relative_to(PROJECT_ROOT))
            except ValueError:
                rel = str(path)
            paths.append(rel)
    return paths


# ---------------------------------------------------------------------------
# PDF / text handling
# ---------------------------------------------------------------------------

def _encode_pdf_base64(pdf_path: Path) -> str:
    return base64.standard_b64encode(pdf_path.read_bytes()).decode("ascii")


def _build_paper_content(paper_path: Path) -> list[dict]:
    """Build user message content for the paper — PDF multimodal or plain text."""
    if paper_path.suffix.lower() == ".pdf":
        pdf_b64 = _encode_pdf_base64(paper_path)
        return [
            {
                "type": "file",
                "file": {
                    "filename": paper_path.name,
                    "file_data": f"data:application/pdf;base64,{pdf_b64}",
                },
            },
        ]
    else:
        text = paper_path.read_text()
        return [{"type": "text", "text": f"Paper text:\n\n{text}"}]


# ---------------------------------------------------------------------------
# Notebook ground truth extraction
# ---------------------------------------------------------------------------


def _extract_notebook_code(notebooks_dir: Path) -> str:
    """Concatenate code cells from all .ipynb files, stripped of outputs.

    Returns a single string with notebook filenames as section headers,
    suitable for LLM consumption. Skips notebooks that fail to parse.
    """
    sections: list[str] = []
    for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
        try:
            nb = json.loads(nb_path.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("Skipping unparseable notebook: %s", nb_path.name)
            continue
        code_cells = [c for c in nb.get("cells", []) if c["cell_type"] == "code"]
        if not code_cells:
            continue
        code_text = "\n\n".join("".join(c["source"]) for c in code_cells)
        sections.append(f"=== {nb_path.name} ===\n{code_text}")
    return "\n\n".join(sections)


LAYOUT_PROMPT = """\
You are extracting experimental design information from analysis notebooks. \
These notebooks were written by the paper's authors and contain hardcoded \
mappings between samples (e.g. microplate wells, patient IDs, experimental \
groups) and experimental conditions that are NOT available in the raw data \
files.

Your task: find every sample-to-condition mapping in the code (dict literals, \
list assignments, index selections) and extract it as structured JSON. These \
mappings are the researcher's "lab notebook" — prior knowledge about which \
sample received which treatment.

For each distinct experiment or plate you find, output:
{{
  "experiment": "<descriptive name>",
  "notebook": "<source notebook filename>",
  "data_file": "<data filename loaded by this notebook, if identifiable>",
  "figure": "<paper figure, if identifiable from filename or comments>",
  "conditions": {{
    "<condition label with units>": ["<sample1>", "<sample2>", ...],
    ...
  }},
  "control_samples": ["<sample>", ...],
  "notes": "<relevant context: temperature, medium, strain, cohort, etc.>"
}}

Rules:
- Extract ONLY mappings explicitly written in the code as dict/list \
  literals or direct assignments. Do NOT infer mappings from analysis logic.
- Condition labels must include physical meaning and units where available \
  (e.g. "1e10 PFU/mL" not just "1e10", "1h post-infection" not just "1h").
- If the same mapping appears in multiple notebooks, include it once and \
  list all source notebooks in the "notebook" field (comma-separated).
- If a notebook loads a specific data file (via read_csv, read_excel, \
  loadmat, etc.), record that filename in "data_file".
- Use standard sample identifiers as they appear in the code (e.g. A1-F8 \
  for well plates).

Output ONLY the JSON array. No preamble, no code fences.
"""

LAYOUT_PROSE_PROMPT = """\
Convert the following JSON plate/sample layout data into a concise \
Experimental Design section for a research brief. Write in first person \
as a researcher describing their experimental setup to a collaborator. \
Be specific about sample identifiers, conditions, and controls.

Format: 1-3 sentences per experiment. Include data file names, condition \
ranges, and control identifiers. Do NOT include analysis methods.

Output ONLY the markdown text starting with the first experiment description. \
No section header, no preamble.
"""


def extract_plate_layouts(
    notebooks_dir: Path,
    client: OpenAI,
    model: str,
) -> list[dict]:
    """Extract sample-to-condition mappings from notebook code."""
    notebook_code = _extract_notebook_code(notebooks_dir)
    if not notebook_code:
        log.info("No notebook code found in %s", notebooks_dir)
        return []

    log.info(
        "Extracting plate layouts from notebooks (%d chars of code)...",
        len(notebook_code),
    )
    create_kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": LAYOUT_PROMPT},
            {"role": "user", "content": notebook_code},
        ],
        "max_tokens": 16384,
        "temperature": 0.1,
    }
    if "openai/" in model or "gpt" in model.lower():
        create_kwargs["extra_body"] = {"reasoning_effort": "high"}

    response = client.chat.completions.create(**create_kwargs)
    raw = response.choices[0].message.content or ""
    if not raw:
        log.warning("Empty layout response, retrying without reasoning_effort")
        create_kwargs.pop("extra_body", None)
        response = client.chat.completions.create(**create_kwargs)
        raw = response.choices[0].message.content or ""

    log.info("Layout response received (%d chars)", len(raw))

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        layouts = json.loads(raw)
    except json.JSONDecodeError:
        # Truncated JSON from reasoning models — try to salvage by closing
        # the array at the last complete object
        log.warning("JSON parse failed, attempting truncation recovery")
        last_brace = raw.rfind("}")
        if last_brace > 0:
            salvaged = raw[:last_brace + 1] + "]"
            layouts = json.loads(salvaged)
        else:
            log.error("Cannot recover layout JSON")
            return []
    log.info("Extracted %d plate/sample layouts", len(layouts))
    return layouts


def format_layouts_for_brief(
    layouts: list[dict],
    client: OpenAI,
    model: str,
) -> str:
    """Convert structured layouts JSON into natural prose for the brief."""
    if not layouts:
        return ""

    log.info("Converting %d layouts to prose for brief...", len(layouts))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LAYOUT_PROSE_PROMPT},
            {"role": "user", "content": json.dumps(layouts, indent=2)},
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""
    if not raw:
        # Fallback: build a minimal summary ourselves
        parts = []
        for layout in layouts:
            exp = layout.get("experiment", "unknown")
            df = layout.get("data_file", "")
            n_cond = len(layout.get("conditions", {}))
            parts.append(f"{exp}: {n_cond} conditions" + (f" ({df})" if df else ""))
        raw = "\n".join(f"- {p}" for p in parts)

    log.info("Layout prose received (%d chars)", len(raw))
    return raw.strip()


# ---------------------------------------------------------------------------
# LLM call 1: Research brief generation
# ---------------------------------------------------------------------------

BRIEF_PROMPT = """\
You are a scientific paper analyst. Your task is to read a research paper and \
produce a SimulatedUser research brief that will guide an AI research agent \
through reproducing the paper's computational analyses.

IMPORTANT FRAMING — design decisions vs experimental results:
The brief must read like a RESEARCH DESIGN document, not a paper summary. \
Imagine a PI who has designed the study, chosen the methods, and secured \
the data — but has NOT yet run any analysis. Include method choices and \
parameters (design decisions) but do NOT include specific numerical results.

{data_inventory}

CRITICAL — language and package fidelity:
The agent's code execution sandbox supports BOTH Python and R. The agent \
will probe available packages at runtime and adapt accordingly. Your job \
is to write the brief FAITHFULLY to the paper's original methods — use \
the same language (Python or R) and packages the paper uses.

Rules:
1. If the paper uses R, write the brief with R packages and idioms. \
If the paper uses Python, write it with Python packages. If mixed, \
reflect that.
2. Reference the EXACT packages and functions the paper uses (e.g. \
"survival::survfit" not "fit a survival model", "TumorImmuneModels::m1" \
not "run the ODE model"). The agent needs precise references.
3. Do NOT substitute packages preemptively. The agent will handle \
unavailable packages at runtime based on what is actually installed. \
Your brief should describe the INTENDED method, not a fallback.
4. For proprietary or highly specialized tools that are clearly \
unavailable in a standard Python/R environment (commercial software, \
hardware-specific tools), note them and suggest the closest open-source \
equivalent.

Produce the brief with EXACTLY these sections:

## Background
2-3 sentences of domain context. Describe the dataset, its source, and the \
overall research question. Write as if addressing a computational biologist \
who will carry out the analysis.

## Research Goals
Numbered list of stages from the paper's analysis pipeline. Each stage is \
an independent work unit. Frame as actions to perform, not results to obtain. \
Do NOT include specific result numbers or gene names from model outputs.

## Methodological Preferences
Bullet list of specific methods, thresholds, tools, and parameters from the \
paper. Include: statistical methods and their configurations, package \
preferences, cross-validation strategy, evaluation metrics, plot types. \
Be concrete — "use survival::survfit for Kaplan-Meier" or \
"use scipy.optimize.curve_fit with Monod equation", not "fit a model".

## What to Approve
Bullet list of conditions under which the SimulatedUser should approve \
an agent's plan. Include: correct method choices, proper data handling, \
reasonable parameter ranges.

## What to Push Back On
Bullet list of conditions that should trigger pushback. Include: skipping \
validation steps, using wrong methods, missing key analyses from the paper, \
reporting insufficient metrics.

## Domain Interests
Bullet list of domain-specific quality signals. What would make the results \
more convincing? What known patterns should the agent look for?

## Implementation Specifications
For each research goal that involves a non-trivial method, provide the \
EXACT technical specification — equations, parameters, thresholds — that \
the SimulatedUser will give to the agent when approving the plan. These \
specs prevent the agent from guessing wrong and wasting turns.

Format as a numbered list matching the Research Goals numbering. Skip \
goals that are straightforward. For each non-trivial goal, include ALL \
of these that apply:
- **Equations**: Complete with ALL variable names and definitions
- **Parameters**: Name, meaning, bounds, units, fitting vs fixed
- **Thresholds**: Exact values with units
- **Implementation constraints**: Compartment numbers, optimizer, CV folds

CRITICAL — be maximally specific. The level of detail should be enough \
for a programmer to implement the method without referring back to the \
paper. Example of the REQUIRED level of detail:

> ### Goal 6: Fit infection ODE model
> **Equations**:
> - dN/dt = -e·(U + ΣIᵢ)·v·N/(N+K)
> - dU/dt = U·v·N/(N+K) - r·U·P
> - dI₁/dt = r·U·P - (M/τ)·I₁
> - dIᵢ/dt = (M/τ)·(Iᵢ₋₁ - Iᵢ) for i=2..M
> - dP/dt = B·(M/τ)·I_M - r·(U+ΣIᵢ)·P
> **Parameters**: r (adsorption rate, ~10⁻⁹ mL/min), B (burst size, \
> ~100-200), τ (latent period, ~40-60 min), M=5 (Erlang stages, fixed)
> **Optimizer**: simulated annealing (scipy.optimize.dual_annealing) for \
> best fit, then 200-member perturbation ensemble for uncertainty
> **Fitting**: Joint across 6 MOI conditions, minimize normalized MSE
> **Variants**: null (constant r,B,τ), r-model (r=max(0,rk·φ+r0)), \
> B-model (B=max(0,Bk·φ+B0)), τ-model (τ=max(20,τk·φ+τ0))

Write specifications using the SAME packages and functions as the paper. \
The agent will adapt at runtime if a package is unavailable.

## Communication Style
Always include these exact bullets:
- You are the researcher who designed these experiments. When approving \
a plan, include ALL implementation-critical specifications the agent \
needs: exact equations, parameter constraints, sample assignments, \
threshold values. Front-load the specs — do not wait to see if the \
agent gets it right.
- After each stage, proactively state the next research direction with \
specific methods and parameters.
- Give specific, actionable feedback when reviewing plans.
- Approve explicitly with "approved" or "go ahead" when satisfied.
- Keep responses to 3-5 sentences. Specs can be longer when needed.
- Ask for clarification on vague method descriptions.
- NEVER reference "the paper", "the brief", or "the protocol". Speak \
from your own research experience as if you designed everything yourself.
- NEVER attempt to use tools, read files, or call functions. You can \
only communicate through text responses to the agent.
- When ALL research goals from this brief have been addressed (even if \
some results are imperfect), include the marker [FINISH] at the END of \
your response. This signals the system to end the run. Do NOT include \
[FINISH] if there are remaining goals that haven't been attempted.

Output ONLY the Markdown content starting with '## Background'. \
No preamble, no code fences around the entire output.
"""


def generate_research_brief(
    paper_path: Path,
    data_inventory: str,
    client: OpenAI,
    model: str,
) -> str:
    """LLM call 1: generate SimulatedUser research_brief from the paper."""
    inventory_block = (
        f"\nThe following data files are available locally:\n"
        f"```\n{data_inventory}\n```\n"
        f"Reference these in the brief where applicable."
        if data_inventory
        else ""
    )

    system_prompt = BRIEF_PROMPT.format(data_inventory=inventory_block)
    paper_content = _build_paper_content(paper_path)
    paper_content.append({
        "type": "text",
        "text": (
            "Read the paper above and generate a SimulatedUser research brief "
            "following the schema in the system prompt."
        ),
    })

    log.info("LLM call 1: generating research brief (model=%s, paper=%s)...", model, paper_path.name)
    create_kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": paper_content},
        ],
        "max_tokens": 32768,
        "temperature": 0.2,
    }
    if "openai/" in model or "gpt" in model.lower():
        create_kwargs["extra_body"] = {"reasoning_effort": "high"}

    response = client.chat.completions.create(**create_kwargs)
    msg = response.choices[0].message
    content = msg.content or ""
    # Some reasoning models may return empty content — log usage for debugging
    if not content:
        log.warning(
            "LLM returned empty content. usage=%s, finish_reason=%s",
            response.usage, response.choices[0].finish_reason,
        )

    # Detect truncation: the brief must end with Communication Style section
    if content and "## Communication Style" not in content:
        log.warning(
            "Research brief appears TRUNCATED (%d chars, missing Communication Style section). "
            "The model may have exhausted max_tokens on reasoning. "
            "Consider increasing max_tokens or reducing reasoning_effort.",
            len(content),
        )

    log.info("Research brief received (%d chars)", len(content))
    return content.strip()


# ---------------------------------------------------------------------------
# LLM call 2: Data catalog entries
# ---------------------------------------------------------------------------

CATALOG_PROMPT = """\
You are writing data catalog entries for an AI agent's scenario configuration. \
The agent uses this catalog to find the right file for a task without guessing.

Below is a file inventory with sampled content and metadata. For each \
semantically distinct data file or group, produce a JSON object with:
- "name": short descriptive title (e.g. "OD time series measurements")
- "path": relative path as it will appear in the workspace data/ directory \
  (e.g. "data/TECAN_230114.csv"). MUST be a single string, not a list.
- "description": 2-4 sentences describing content, structure, and how to use it

Grouping rules:
- Files with the same schema CAN share one entry, but only if they serve the \
  same analytical purpose. Same schema alone is not enough — scientific data \
  files often share a format but represent different experiments, conditions, \
  or time points. When grouping, list every filename in the description and \
  note each file's distinguishing context (date, condition, identifier in \
  the filename). The agent must know WHICH file to use for WHICH analysis.
- Files with different schemas get separate entries.
- Include ALL files — do not silently skip any.

Structure inference rules:
- Infer what rows and columns represent from the CONTENT SAMPLE values, not \
  from column names alone. Column names like "1", "2", "3" or "A1", "B2" \
  are ambiguous — they could be samples, wells, features, patients, or \
  time points depending on the domain. Look at the actual cell values and \
  row labels in the sample to determine the axis semantics, and state them \
  explicitly (e.g. "rows = observations, columns = features").
- Describe only what the evidence supports. If the content sample does not \
  contain enough information to determine the meaning of an axis, say so \
  rather than guessing. "Column semantics unclear from sample — inspect \
  file header" is more useful than a wrong guess.
- For binary or unreadable files, describe ONLY what is evident from the \
  filename and file size. Do NOT speculate about contents. Instead, state \
  how to inspect the file (e.g. "load with scipy.io.loadmat() to list \
  variables" or "use pd.ExcelFile() to list sheet names").

Output a JSON array of objects. Example:
[
  {{"name": "OD time series", "path": "data/TECAN_230114.csv", "description": "TECAN plate reader exports. Other files in this group: data/TECAN_230119.csv, data/TECAN_230212.csv. ..."}},
  {{"name": "Calibration data", "path": "data/calibration.xlsx", "description": "..."}}
]

Output ONLY the JSON array. No preamble, no code fences.
"""


def generate_catalog_entries(
    file_inventory: str,
    client: OpenAI,
    model: str,
) -> list[dict[str, str]]:
    """LLM call 2: generate data catalog entries from file inventory."""
    log.info("LLM call 2: generating data catalog entries (model=%s)...", model)
    create_kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": CATALOG_PROMPT},
            {"role": "user", "content": file_inventory},
        ],
        "max_tokens": 16384,
        "temperature": 0.1,
    }
    if "openai/" in model or "gpt" in model.lower():
        create_kwargs["extra_body"] = {"reasoning_effort": "high"}

    response = client.chat.completions.create(**create_kwargs)
    msg = response.choices[0].message
    raw = msg.content or ""
    # Some reasoning models (o-series, gpt-5.x) may return output in a
    # reasoning field or as empty content. Fall back to checking reasoning.
    if not raw and hasattr(msg, "reasoning_content") and msg.reasoning_content:
        log.warning("content empty, extracting from reasoning_content")
        raw = msg.reasoning_content
    if not raw and hasattr(msg, "reasoning") and msg.reasoning:
        log.warning("content empty, extracting from reasoning field")
        raw = msg.reasoning
    if not raw:
        log.warning(
            "LLM returned empty content. finish_reason=%s, usage=%s",
            response.choices[0].finish_reason, response.usage,
        )
        # Retry once without reasoning_effort
        log.info("Retrying catalog generation without reasoning_effort...")
        create_kwargs.pop("extra_body", None)
        response = client.chat.completions.create(**create_kwargs)
        raw = response.choices[0].message.content or ""
    log.info("Catalog response received (%d chars)", len(raw))

    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    entries = json.loads(raw)
    # Post-process: ensure each path is a single string, not a list
    for entry in entries:
        path_val = entry.get("path", "")
        if isinstance(path_val, list):
            entry["path"] = path_val[0]
            rest = ", ".join(path_val[1:])
            if rest:
                entry["description"] = entry.get("description", "") + f" Other files in this group: {rest}."
            log.info("Flattened list path to single: %s", entry["path"])
    log.info("Parsed %d catalog entries", len(entries))
    return entries


# ---------------------------------------------------------------------------
# LLM call 3: Researcher-perspective initial message
# ---------------------------------------------------------------------------

INITIAL_MSG_PROMPT = """\
You are writing the opening message from a researcher to their AI research \
assistant. The researcher has a clear vision and specific goals — they are \
NOT asking the assistant to figure out what to do.

Based on the research brief below, write a 2-4 sentence opening message that:
1. States what the researcher wants to accomplish (their research goal)
2. Gives a concrete first step direction (what to start with)
3. Mentions specific methods or approaches they want to use

Do NOT mention "data catalog", "please plan", or ask the assistant to \
figure out what datasets exist. The researcher knows their data and their \
methods — they are giving direction, not asking questions.

Write ONLY the message text, no quotes or markdown formatting.
"""


def generate_initial_message(
    research_brief: str,
    client: OpenAI,
    model: str,
) -> str:
    """LLM call 3: generate a researcher-perspective opening message."""
    log.info("LLM call 3: generating initial message (model=%s)...", model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INITIAL_MSG_PROMPT},
            {"role": "user", "content": research_brief},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    raw = response.choices[0].message.content or ""
    if not raw:
        # Fallback for reasoning models
        log.warning("Empty initial message, retrying without reasoning_effort")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INITIAL_MSG_PROMPT},
                {"role": "user", "content": research_brief},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        raw = response.choices[0].message.content or ""
    log.info("Initial message received (%d chars)", len(raw))
    return raw.strip()


# ---------------------------------------------------------------------------
# TOML assembly
# ---------------------------------------------------------------------------

def _escape_toml_literal(s: str) -> str:
    """Escape a string for use in a TOML literal triple-quoted value ('''...''').

    Literal strings don't process backslash escapes, which is critical since
    LLM output often contains LaTeX (e.g. \\(x\\)). The only forbidden
    sequence inside '''...''' is three consecutive single quotes.
    """
    return s.replace("'''", "'''\"'''\"'''")  # break the triple-quote sequence


def assemble_toml(
    name: str,
    sources: list[str],
    catalog_entries: list[dict[str, str]],
    research_brief: str,
    max_turns: int,
    experimental_design: str = "",
    plate_layouts_json: str = "",
) -> str:
    """Assemble the final scenario TOML content."""
    lines: list[str] = []

    # Header
    lines.append(f'name = "{name}"')
    lines.append("")

    # Data sources
    lines.append("# -- Data sources: copied into workspace/data/ at startup --")
    lines.append("")
    lines.append("[data]")
    lines.append("sources = [")
    for src in sources:
        lines.append(f'    "{src}",')
    lines.append("]")
    lines.append("")

    # Data catalog
    lines.append("# -- Data catalog: describes datasets for the main agent --")
    lines.append("")
    for entry in catalog_entries:
        lines.append("[[data.catalog]]")
        lines.append(f'name = "{entry["name"]}"')
        lines.append(f'path = "{entry["path"]}"')
        desc = _escape_toml_literal(entry["description"].strip())
        lines.append(f"description = '''{desc}'''")
        lines.append("")

    # Plate layouts JSON (appended to [data] section above)
    if plate_layouts_json:
        lines.append("# -- Plate/sample layouts extracted from notebooks --")
        lines.append(f"plate_layouts_json = '''{_escape_toml_literal(plate_layouts_json)}'''")
        lines.append("")

    # SimulatedUser
    lines.append("# -- SimulatedUser configuration --")
    lines.append("")
    lines.append("[simulated_user]")
    # Inject experimental design into brief if available
    full_brief = research_brief
    if experimental_design:
        # Insert after ## Background section
        bg_end = full_brief.find("\n## Research Goals")
        if bg_end != -1:
            full_brief = (
                full_brief[:bg_end]
                + "\n\n## Experimental Design\n"
                + experimental_design
                + full_brief[bg_end:]
            )
        else:
            full_brief += "\n\n## Experimental Design\n" + experimental_design
    brief = _escape_toml_literal(full_brief)
    lines.append("research_brief = '''")
    lines.append(f"# Research Brief: {name}")
    lines.append("")
    lines.append(brief)
    lines.append("'''")
    lines.append("")

    # Run configuration
    lines.append("# -- Run configuration --")
    lines.append("")
    lines.append("[run]")
    lines.append(f"max_turns = {max_turns}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a scenario TOML from a paper and data directory",
    )
    parser.add_argument(
        "--name", required=True,
        help="Scenario name (used for output filename and TOML name field)",
    )
    parser.add_argument(
        "--paper", type=Path, required=True,
        help="Path to the paper (.pdf or .txt)",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model (default: from config/llm.toml [openrouter].default_model)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=50,
        help="Max conversation turns (default: 50)",
    )
    parser.add_argument(
        "--notebooks-dir", type=Path, default=None,
        help="Path to notebooks directory (optional — extracts experimental design mappings)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: scenarios/<name>.toml)",
    )
    parser.add_argument(
        "--generate-checklist", action="store_true",
        help="Also generate evaluation_checklist.json alongside the scenario TOML",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.paper.exists():
        log.error("Paper not found: %s", args.paper)
        sys.exit(1)
    if not args.data_dir.exists():
        log.error("Data directory not found: %s", args.data_dir)
        sys.exit(1)

    # Load config and build client
    config = _load_openrouter_config()
    model = args.model or config["default_model"]
    # Strip "openrouter/" prefix — litellm convention not used by OpenRouter API
    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]
    client = _build_client(config["api_key"], config["proxy"])

    # Scan data directory
    log.info("Scanning data directory: %s", args.data_dir)
    entries = scan_data_dir(args.data_dir)
    log.info("Found %d file entries", len(entries))
    file_inventory = _build_file_inventory(entries)
    log.info("File inventory: %d chars", len(file_inventory))

    # Collect source paths for TOML
    sources = _collect_source_paths(args.data_dir)
    log.info("Data sources: %d files", len(sources))

    # LLM call 1: research brief
    research_brief = generate_research_brief(args.paper, file_inventory, client, model)

    # LLM call 2: catalog entries
    catalog_entries = generate_catalog_entries(file_inventory, client, model)

    # Optional: extract experimental design from notebooks
    experimental_design = ""
    plate_layouts_json = ""
    if args.notebooks_dir and args.notebooks_dir.is_dir():
        layouts = extract_plate_layouts(args.notebooks_dir, client, model)
        if layouts:
            experimental_design = format_layouts_for_brief(layouts, client, model)
            plate_layouts_json = json.dumps(layouts, indent=2, ensure_ascii=False)
    elif args.notebooks_dir:
        log.warning("Notebooks directory not found: %s", args.notebooks_dir)

    # Assemble TOML
    toml_content = assemble_toml(
        name=args.name,
        sources=sources,
        catalog_entries=catalog_entries,
        research_brief=research_brief,
        max_turns=args.max_turns,
        experimental_design=experimental_design,
        plate_layouts_json=plate_layouts_json,
    )

    # Write output
    output_path = args.output or (PROJECT_ROOT / "scenarios" / f"{args.name}.toml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(toml_content)
    log.info("Scenario written to: %s", output_path)
    print(
        f"Generated {output_path} ({len(toml_content)} chars, "
        f"{len(catalog_entries)} catalog entries)",
        file=sys.stderr,
    )

    # Optional: generate evaluation checklist
    if args.generate_checklist:
        from generate_checklist import generate_checklist_from_paper

        checklist_path = output_path.with_name(f"{args.name}_checklist.json")
        checklist_json = generate_checklist_from_paper(
            paper_path=args.paper,
            data_dir=args.data_dir,
            client=client,
            model=model,
        )
        checklist_path.write_text(checklist_json, encoding="utf-8")
        log.info("Checklist written to: %s", checklist_path)
        print(f"Generated {checklist_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
