"""Tests for runtime environment probing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.runtime_env import (
    RProbeResult,
    RuntimeProbeResult,
    probe_environment,
    probe_r_environment,
)


def test_probe_environment_returns_first_candidate_with_required_packages(
    tmp_path: Path,
) -> None:
    """probe_environment returns the first interpreter that satisfies required packages."""

    first_candidate = tmp_path / "python_a"
    second_candidate = tmp_path / "python_b"
    first_candidate.write_text("", encoding="utf-8")
    second_candidate.write_text("", encoding="utf-8")

    run_results = [
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "python_version": "3.11.1",
                    "available_packages": {
                        "numpy": "1.26.0",
                        "pandas": "2.0.0",
                    },
                    "missing_packages": ["scipy", "matplotlib"],
                }
            ),
            stderr="",
        ),
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "python_version": "3.11.7",
                    "available_packages": {
                        "numpy": "1.26.0",
                        "pandas": "2.0.0",
                        "scipy": "1.11.0",
                        "matplotlib": "3.8.0",
                    },
                    "missing_packages": ["seaborn"],
                }
            ),
            stderr="",
        ),
    ]

    with patch("research_agent.runtime_env.subprocess.run", side_effect=run_results):
        result = probe_environment(
            python_candidates=[str(first_candidate), str(second_candidate)],
            required_packages=["numpy", "pandas", "scipy", "matplotlib"],
            optional_packages=["seaborn"],
        )

    assert result == RuntimeProbeResult(
        python_path=str(second_candidate),
        python_version="3.11.7",
        available_packages={
            "numpy": "1.26.0",
            "pandas": "2.0.0",
            "scipy": "1.11.0",
            "matplotlib": "3.8.0",
        },
        missing_packages=["seaborn"],
    )


def test_probe_environment_raises_when_no_candidate_meets_requirements(
    tmp_path: Path,
) -> None:
    """probe_environment raises RuntimeError when all candidates miss required packages."""

    candidate = tmp_path / "python_missing"
    candidate.write_text("", encoding="utf-8")

    run_result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(
            {
                "python_version": "3.11.7",
                "available_packages": {"numpy": "1.26.0"},
                "missing_packages": ["pandas", "scipy", "matplotlib"],
            }
        ),
        stderr="",
    )

    with patch("research_agent.runtime_env.subprocess.run", return_value=run_result):
        try:
            probe_environment(
                python_candidates=[str(candidate)],
                required_packages=["numpy", "pandas"],
                optional_packages=[],
            )
        except RuntimeError as error:
            message = str(error)
        else:
            raise AssertionError("Expected RuntimeError")

    assert "No Python interpreter satisfied the required runtime packages" in message
    assert "missing required packages pandas" in message


# ── R probe tests ──


def test_probe_r_environment_parses_json_from_rscript(tmp_path: Path) -> None:
    """probe_r_environment parses the JSON output from the R probe script."""

    candidate = tmp_path / "Rscript"
    candidate.write_text("", encoding="utf-8")

    run_result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(
            {
                "r_version": "4.3.1",
                "available_packages": {"ggplot2": "3.4.0", "dplyr": "1.1.0"},
                "missing_packages": ["DESeq2", "Seurat"],
            }
        ),
        stderr="",
    )

    with patch("research_agent.runtime_env.subprocess.run", return_value=run_result):
        result = probe_r_environment(
            rscript_candidates=[str(candidate)],
            optional_packages=["ggplot2", "dplyr", "DESeq2", "Seurat"],
        )

    assert result == RProbeResult(
        rscript_path=str(candidate),
        r_version="4.3.1",
        available_packages={"ggplot2": "3.4.0", "dplyr": "1.1.0"},
        missing_packages=["DESeq2", "Seurat"],
    )


def test_probe_r_environment_raises_when_no_rscript_found() -> None:
    """probe_r_environment raises RuntimeError when no Rscript candidate exists."""

    try:
        probe_r_environment(
            rscript_candidates=["/nonexistent/Rscript"],
            optional_packages=["ggplot2"],
        )
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("Expected RuntimeError")

    assert "No Rscript interpreter found" in message


def test_probe_r_environment_skips_failed_candidate_and_uses_next(
    tmp_path: Path,
) -> None:
    """probe_r_environment skips candidates that fail and tries the next one."""

    bad_candidate = tmp_path / "Rscript_bad"
    good_candidate = tmp_path / "Rscript_good"
    bad_candidate.write_text("", encoding="utf-8")
    good_candidate.write_text("", encoding="utf-8")

    run_results = [
        subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error loading"
        ),
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "r_version": "4.2.0",
                    "available_packages": {"survival": "3.5.0"},
                    "missing_packages": [],
                }
            ),
            stderr="",
        ),
    ]

    with patch("research_agent.runtime_env.subprocess.run", side_effect=run_results):
        result = probe_r_environment(
            rscript_candidates=[str(bad_candidate), str(good_candidate)],
            optional_packages=["survival"],
        )

    assert result.rscript_path == str(good_candidate)
    assert result.r_version == "4.2.0"
