"""Shared pytest fixtures for runtime probing and test harness setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.runtime_env import (
    RProbeResult,
    RuntimeProbeResult,
    probe_environment,
    probe_r_environment,
)


def build_stub_probe_result(
    *,
    python_path: str = "/usr/bin/python3",
    python_version: str = "3.11.0",
    available_packages: dict[str, str] | None = None,
    missing_packages: list[str] | None = None,
) -> RuntimeProbeResult:
    """Build a deterministic RuntimeProbeResult for tests that avoid real probing."""

    return RuntimeProbeResult(
        python_path=python_path,
        python_version=python_version,
        available_packages=available_packages
        or {
            "numpy": "1.0.0",
            "pandas": "1.0.0",
            "scipy": "1.0.0",
            "matplotlib": "1.0.0",
        },
        missing_packages=missing_packages
        or [
            "seaborn",
            "statsmodels",
            "sklearn",
            "gseapy",
            "pydeseq2",
            "adjustText",
        ],
    )


@pytest.fixture
def stub_probe_result() -> RuntimeProbeResult:
    """Provide a deterministic probe result for unit tests."""

    return build_stub_probe_result()


@pytest.fixture(scope="session")
def real_probe_result() -> RuntimeProbeResult:
    """Probe the real runtime once for tests that execute scientific Python code."""

    return probe_environment()


# ── R runtime fixtures ──


def build_stub_r_probe_result(
    *,
    rscript_path: str = "/usr/bin/Rscript",
    r_version: str = "4.3.1",
    available_packages: dict[str, str] | None = None,
    missing_packages: list[str] | None = None,
) -> RProbeResult:
    """Build a deterministic RProbeResult for tests that avoid real R probing."""

    return RProbeResult(
        rscript_path=rscript_path,
        r_version=r_version,
        available_packages=available_packages
        if available_packages is not None
        else {
            "ggplot2": "3.4.0",
            "dplyr": "1.1.0",
            "tidyr": "1.3.0",
            "DESeq2": "1.40.0",
        },
        missing_packages=missing_packages
        if missing_packages is not None
        else [
            "Seurat",
            "clusterProfiler",
            "edgeR",
        ],
    )


@pytest.fixture
def stub_r_probe_result() -> RProbeResult:
    """Provide a deterministic R probe result for unit tests."""

    return build_stub_r_probe_result()


@pytest.fixture(scope="session")
def real_r_probe_result() -> RProbeResult | None:
    """Probe the real R runtime once; None if Rscript not found."""

    try:
        return probe_r_environment()
    except RuntimeError:
        return None
