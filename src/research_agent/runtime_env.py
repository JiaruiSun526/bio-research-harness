"""Runtime environment probing for sub-agent code execution.

This module defines explicit contracts for the Python and R runtimes
used by sub agents. The rest of the system consumes validated
RuntimeProbeResult / RProbeResult instead of guessing which interpreters
or packages are available.

Python is required; R is optional. When Rscript is not found,
probe_r_environment() raises RuntimeError and the harness sets
r_probe_result = None — sub agents simply cannot use language="r".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_PYTHON_CANDIDATES: tuple[str, ...] = (
    "/usr/bin/python3",
    "/usr/bin/python",
    sys.executable,
)
"""Ordered Python interpreter candidates for runtime probing."""

DEFAULT_REQUIRED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
)
"""Packages that must be present for the fixed runtime strategy."""

DEFAULT_OPTIONAL_PACKAGES: tuple[str, ...] = (
    "seaborn",
    "statsmodels",
    "sklearn",
    "gseapy",
    "pydeseq2",
    "adjustText",
    "xgboost",
    "shap",
    "lightgbm",
    "emcee",
    "lmfit",
    "optuna",
    "plotly",
    "umap",
    "alphastats",
    "torch",
    "networkx",
    "sympy",
    "natsort",
    "joblib",
    "xlsxwriter",
    "corner",
    "ddeint",
    "bct",
    "reservoirpy",
    "inmoose",
    "matplotlib_venn",
    "pimmslearn",
    "tellurium",
    "rrplugins",
    "sobol",
    "sobol_seq",
    "halo",
)
"""Packages that may be used when the runtime probe confirms availability."""


class RuntimeProbeResult(BaseModel):
    """Validated runtime metadata for sub-agent Python code execution."""

    python_path: str
    python_version: str
    available_packages: dict[str, str] = Field(default_factory=dict)
    missing_packages: list[str] = Field(default_factory=list)


# ── R runtime constants ──

DEFAULT_RSCRIPT_CANDIDATES: tuple[str, ...] = (
    "/usr/bin/Rscript",
    "/usr/local/bin/Rscript",
)
"""Ordered Rscript interpreter candidates for runtime probing."""

DEFAULT_R_OPTIONAL_PACKAGES: tuple[str, ...] = (
    # tidyverse core
    "ggplot2", "dplyr", "tidyr", "readr", "stringr", "purrr", "tibble",
    # visualization
    "pheatmap", "EnhancedVolcano", "ComplexHeatmap", "ggrepel", "gridExtra",
    # Bioconductor — differential expression
    "DESeq2", "edgeR", "limma",
    # Bioconductor — pathway / enrichment
    "clusterProfiler", "org.Hs.eg.db", "DOSE", "enrichplot", "fgsea",
    # single-cell
    "Seurat", "SingleCellExperiment", "scran",
    # survival analysis
    "survival", "survminer",
    # general statistics
    "lme4", "car", "broom", "caret",
    # Bioconductor infrastructure
    "BiocGenerics", "SummarizedExperiment", "GenomicRanges", "Biobase",
)
"""R packages probed when an Rscript interpreter is found.

All are optional — R support works with a bare R installation.
The probe reports which are actually installed so the sub-agent
prompt and library preamble reflect the real environment.
"""


class RProbeResult(BaseModel):
    """Validated runtime metadata for sub-agent R code execution."""

    rscript_path: str
    r_version: str
    available_packages: dict[str, str] = Field(default_factory=dict)
    missing_packages: list[str] = Field(default_factory=list)


def probe_environment(
    python_candidates: list[str] | tuple[str, ...] | None = None,
    required_packages: list[str] | tuple[str, ...] | None = None,
    optional_packages: list[str] | tuple[str, ...] | None = None,
) -> RuntimeProbeResult:
    """Find a Python interpreter with all required research packages.

    The probe runs a small Python script inside each interpreter
    candidate and validates imports there, rather than inspecting the
    current process. This avoids false positives from the caller's
    environment.

    Args:
        python_candidates: Candidate interpreter paths to try in order.
        required_packages: Packages that must be importable.
        optional_packages: Packages that are useful when available.

    Returns:
        A RuntimeProbeResult for the first interpreter that satisfies all
        required packages.

    Raises:
        RuntimeError: If no interpreter candidate has every required
            package available.
    """

    candidate_list = _deduplicate_items(
        python_candidates or list(DEFAULT_PYTHON_CANDIDATES)
    )
    required_list = _deduplicate_items(
        required_packages or list(DEFAULT_REQUIRED_PACKAGES)
    )
    optional_list = _deduplicate_items(
        optional_packages or list(DEFAULT_OPTIONAL_PACKAGES)
    )
    probe_script = _build_probe_script(required_list, optional_list)
    probe_env = build_isolated_python_env()
    failure_reasons: list[str] = []

    for candidate in candidate_list:
        candidate_path = Path(candidate)
        if not candidate or not candidate_path.is_file():
            failure_reasons.append(f"{candidate}: interpreter not found")
            continue

        try:
            result = subprocess.run(
                [candidate, "-I", "-c", probe_script],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=probe_env,
            )
        except OSError as error:
            failure_reasons.append(f"{candidate}: failed to execute ({error})")
            continue
        except subprocess.TimeoutExpired:
            failure_reasons.append(f"{candidate}: probe timed out")
            continue

        if result.returncode != 0:
            stderr = result.stderr.strip() or "no stderr"
            failure_reasons.append(
                f"{candidate}: probe exited with code {result.returncode} ({stderr})"
            )
            continue

        try:
            payload = _parse_probe_payload(candidate, result.stdout)
        except RuntimeError as error:
            failure_reasons.append(str(error))
            continue
        probe_result = RuntimeProbeResult(
            python_path=candidate,
            python_version=payload["python_version"],
            available_packages=payload["available_packages"],
            missing_packages=payload["missing_packages"],
        )
        missing_required = [
            package
            for package in required_list
            if package not in probe_result.available_packages
        ]
        if not missing_required:
            return probe_result

        failure_reasons.append(
            f"{candidate}: missing required packages {', '.join(missing_required)}"
        )

    reason_text = "; ".join(failure_reasons) if failure_reasons else "no candidates tried"
    raise RuntimeError(
        "No Python interpreter satisfied the required runtime packages. "
        f"Required: {', '.join(required_list)}. Details: {reason_text}"
    )


def probe_r_environment(
    rscript_candidates: list[str] | tuple[str, ...] | None = None,
    optional_packages: list[str] | tuple[str, ...] | None = None,
) -> RProbeResult:
    """Find an Rscript interpreter and probe installed R packages.

    Unlike the Python probe, R has no "required" packages — base R
    (stats, utils, methods) is always present. All packages in the
    optional list are probed and their availability reported.

    Args:
        rscript_candidates: Candidate Rscript paths to try in order.
        optional_packages: R packages to check for availability.

    Returns:
        An RProbeResult for the first working Rscript interpreter.

    Raises:
        RuntimeError: If no Rscript candidate is found or executes
            successfully.
    """
    candidate_list = _deduplicate_items(
        rscript_candidates or list(DEFAULT_RSCRIPT_CANDIDATES)
    )
    optional_list = _deduplicate_items(
        optional_packages or list(DEFAULT_R_OPTIONAL_PACKAGES)
    )
    probe_script = _build_r_probe_script(optional_list)
    probe_env = build_isolated_r_env()
    failure_reasons: list[str] = []

    for candidate in candidate_list:
        candidate_path = Path(candidate)
        if not candidate or not candidate_path.is_file():
            failure_reasons.append(f"{candidate}: Rscript not found")
            continue

        try:
            result = subprocess.run(
                [candidate, "--vanilla", "-e", probe_script],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env=probe_env,
            )
        except OSError as error:
            failure_reasons.append(f"{candidate}: failed to execute ({error})")
            continue
        except subprocess.TimeoutExpired:
            failure_reasons.append(f"{candidate}: probe timed out")
            continue

        if result.returncode != 0:
            stderr = result.stderr.strip() or "no stderr"
            failure_reasons.append(
                f"{candidate}: probe exited with code {result.returncode} ({stderr})"
            )
            continue

        try:
            payload = _parse_probe_payload(candidate, result.stdout)
        except RuntimeError as error:
            failure_reasons.append(str(error))
            continue

        return RProbeResult(
            rscript_path=candidate,
            r_version=str(payload.get("r_version", "unknown")),
            available_packages={
                str(k): str(v) for k, v in (payload.get("available_packages") or {}).items()
            },
            missing_packages=[str(p) for p in (payload.get("missing_packages") or [])],
        )

    reason_text = "; ".join(failure_reasons) if failure_reasons else "no candidates tried"
    raise RuntimeError(
        f"No Rscript interpreter found. Details: {reason_text}"
    )


def _build_r_probe_script(optional_packages: list[str]) -> str:
    """Build an R script that probes installed packages and emits JSON to stdout.

    The script avoids depending on any R package (not even jsonlite) by
    constructing JSON manually with cat() and paste(). This ensures the
    probe works on a bare R installation.
    """
    # Encode package names as an R character vector literal
    r_packages = ", ".join(f'"{pkg}"' for pkg in optional_packages)
    return f"""\
pkgs <- c({r_packages})
installed <- rownames(installed.packages())
avail_keys <- c()
avail_vals <- c()
missing <- c()
for (pkg in pkgs) {{
  if (pkg %in% installed) {{
    ver <- tryCatch(as.character(packageVersion(pkg)), error = function(e) "unknown")
    avail_keys <- c(avail_keys, pkg)
    avail_vals <- c(avail_vals, ver)
  }} else {{
    missing <- c(missing, pkg)
  }}
}}
# Build JSON manually (no jsonlite dependency)
avail_entries <- paste0('"', avail_keys, '":"', avail_vals, '"')
avail_json <- paste0("{{", paste(avail_entries, collapse = ","), "}}")
if (length(avail_keys) == 0) avail_json <- "{{}}"
missing_entries <- paste0('"', missing, '"')
missing_json <- paste0("[", paste(missing_entries, collapse = ","), "]")
if (length(missing) == 0) missing_json <- "[]"
r_ver <- paste0(R.version$major, ".", R.version$minor)
cat(paste0(
  '{{"r_version":"', r_ver,
  '","available_packages":', avail_json,
  ',"missing_packages":', missing_json, "}}"
))
"""


def build_isolated_r_env() -> dict[str, str]:
    """Build a subprocess environment for Rscript without parent overrides."""

    env = os.environ.copy()
    # Remove R_HOME if set by a parent environment to avoid conflicts.
    # Keep R_LIBS_USER — Bioconductor and user-installed packages live there.
    env.pop("R_HOME", None)
    return env


def _build_probe_script(required_packages: list[str], optional_packages: list[str]) -> str:
    """Build the isolated probe script executed inside each candidate interpreter."""

    all_packages = required_packages + [
        package for package in optional_packages if package not in required_packages
    ]
    encoded_packages = json.dumps(all_packages)
    return f"""\
import importlib
import importlib.metadata
import json
import sys

packages = {encoded_packages}
available_packages = {{}}
missing_packages = []

for package_name in packages:
    try:
        module = importlib.import_module(package_name)
    except Exception:
        missing_packages.append(package_name)
        continue

    try:
        version = importlib.metadata.version(package_name)
    except Exception:
        version = getattr(module, "__version__", "unknown")

    available_packages[package_name] = str(version)

print(
    json.dumps(
        {{
            "python_version": sys.version.split()[0],
            "available_packages": available_packages,
            "missing_packages": missing_packages,
        }}
    )
)
"""


def build_isolated_python_env() -> dict[str, str]:
    """Build a subprocess environment without parent virtualenv overrides."""

    env = os.environ.copy()
    for variable in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        env.pop(variable, None)
    return env


def _deduplicate_items(items: list[str] | tuple[str, ...]) -> list[str]:
    """Return items in order with duplicates removed."""

    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def _parse_probe_payload(candidate: str, stdout: str) -> dict[str, object]:
    """Parse and validate the JSON payload emitted by the probe script."""

    payload_text = stdout.strip()
    if not payload_text:
        raise RuntimeError(f"{candidate}: probe returned empty stdout")

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{candidate}: probe returned invalid JSON: {payload_text}"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(f"{candidate}: probe payload is not a JSON object")

    return payload
