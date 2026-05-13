"""Tool definitions for main agent and sub agent.

Main agent tools (project management):
    read_file, save_plan, approve_plan, dispatch_subagent,
    save_conclusion, escalate_to_user

Sub agent tools (execution):
    read_file, write_file, run_code

Tools capture their dependencies (workspace, user_agent, dispatch_fn)
via closures in the factory functions. This avoids circular imports
between tools.py and loop.py — the dispatch_fn is injected by harness.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .runtime_env import (
    RProbeResult,
    RuntimeProbeResult,
    build_isolated_python_env,
    build_isolated_r_env,
)
from .schemas import ReviewContext, SubAgentCompletion, SubAgentRole, TaskResult, TaskSpec
from .tool_registry import ToolError, ToolRegistry
from .workspace import WorkspaceError, resolve_path_within_root

if TYPE_CHECKING:
    from .user_agent.base import UserAgent
    from .workspace import Workspace


def create_main_agent_tools(
    workspace: Workspace,
    user_agent: UserAgent,
    dispatch_fn: Callable[[TaskSpec], TaskResult],
) -> ToolRegistry:
    """Create the tool set for the main agent.

    Each tool is a closure over workspace / user_agent / dispatch_fn.
    The LLM only sees tool name, description, and parameter schema.

    Tools and their engineering constraints (per 03/08 docs):

    | Tool              | Precondition enforced by harness                    |
    |-------------------|-----------------------------------------------------|
    | read_file         | Path must be within workspace                       |
    | save_plan         | Cannot overwrite an approved plan                   |
    | approve_plan      | Plan must exist, be 'drafting', and have a user review round |
    | dispatch_subagent | Current stage's plan must be 'approved'             |
    | save_conclusion   | Stage must be in 'executing' status                 |
    | escalate_to_user  | Marks drafting plans as user-reviewed before approval |

    Args:
        workspace: Workspace manager instance.
        user_agent: UserAgent (real or simulated) for escalation routing.
        dispatch_fn: Callable that launches a sub agent loop — provided
            by ResearchHarness to break the circular dependency.

    Returns:
        ToolRegistry populated with main agent tools.
    """
    registry = ToolRegistry()

    def _run_workspace_operation(operation: Callable[[], str]) -> str:
        """Execute a workspace-backed tool operation with consistent error mapping."""

        try:
            return operation()
        except WorkspaceError as error:
            raise ToolError(str(error)) from error

    @registry.register(
        name="read_file",
        description=(
            "Read a file from the workspace by relative path. "
            "Small files are returned in full. "
            "Large data files (CSV/TSV/Excel >50KB) return a structured preview "
            "(shape, columns, dtypes, sample rows) — use run_code with pandas "
            "to analyze the full data."
        ),
        parameters={
            "path": {"type": "string", "description": "Workspace-relative file path"},
        },
    )
    def read_file(path: str) -> str:
        """Read file contents from the workspace."""

        return _run_workspace_operation(lambda: workspace.read_file(path))

    @registry.register(
        name="save_plan",
        description="Save a stage plan draft in Markdown.",
        parameters={
            "stage_id": {"type": "string", "description": "Stage identifier"},
            "content": {"type": "string", "description": "Plan content in Markdown"},
        },
    )
    def save_plan(stage_id: str, content: str) -> str:
        """Write a plan draft for a stage."""

        def operation() -> str:
            workspace.write_plan(stage_id, content)
            return f"Plan saved to plans/{stage_id}_plan.md. Status: drafting."

        return _run_workspace_operation(operation)

    @registry.register(
        name="approve_plan",
        description="Approve a drafted plan so the stage can execute.",
        parameters={
            "stage_id": {"type": "string", "description": "Stage identifier"},
        },
    )
    def approve_plan(stage_id: str) -> str:
        """Approve a stage plan."""

        def operation() -> str:
            workspace.approve_plan(stage_id)
            return f"Plan approved for {stage_id}. Status: approved. Ready for execution."

        return _run_workspace_operation(operation)

    @registry.register(
        name="dispatch_subagent",
        description="Dispatch a sub agent task for the current stage.",
        parameters={
            "task_id": {"type": "string", "description": "Sub agent task identifier"},
            "stage_id": {"type": "string", "description": "Stage identifier"},
            "task_description": {
                "type": "string",
                "description": "Natural-language task instructions for the sub agent",
            },
            "role": {
                "type": "string",
                "description": "Sub agent role: general, data_analyst, or visualization",
                "enum": ["general", "data_analyst", "visualization"],
            },
            "max_turns": {
                "type": "integer",
                "description": "Maximum tool/LLM turns for the sub agent",
            },
        },
    )
    def dispatch_subagent(
        task_id: str,
        stage_id: str,
        task_description: str,
        role: SubAgentRole = "general",
        max_turns: int = 30,
    ) -> str:
        """Build a TaskSpec, dispatch it, and format the structured result."""

        def operation() -> str:
            task_spec = TaskSpec(
                task_id=task_id,
                stage_id=stage_id,
                task_description=task_description,
                role=role,
                max_turns=max_turns,
            )
            task_result = dispatch_fn(task_spec)
            return (
                f"Task {task_id} {task_result.status}.\n"
                f"Summary: {task_result.summary}\n"
                f"Artifacts: {task_result.artifact_paths}\n"
                f"Blockers: {task_result.blockers}\n"
                f"Suggestions: {task_result.suggestions}"
            )

        return _run_workspace_operation(operation)

    @registry.register(
        name="save_conclusion",
        description="Save the conclusion for an executing stage.",
        parameters={
            "stage_id": {"type": "string", "description": "Stage identifier"},
            "content": {"type": "string", "description": "Conclusion content in Markdown"},
        },
    )
    def save_conclusion(stage_id: str, content: str) -> str:
        """Write a stage conclusion."""

        def operation() -> str:
            workspace.write_conclusion(stage_id, content)
            return (
                f"Conclusion saved for {stage_id}. "
                f"Stage is now complete. Present your findings to the user via escalate_to_user."
            )

        return _run_workspace_operation(operation)

    @registry.register(
        name="escalate_to_user",
        description="Ask the user for review, feedback, or a decision.",
        parameters={
            "summary": {"type": "string", "description": "Summary for the user"},
            "stage_id": {"type": "string", "description": "Related stage identifier"},
            "artifact_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Workspace-relative artifact paths for user review",
            },
        },
    )
    def escalate_to_user(
        summary: str,
        stage_id: str | None = None,
        artifact_paths: list[str] | None = None,
    ) -> str:
        """Route a summary and optional review context to the user agent."""

        if stage_id is not None:
            _run_workspace_operation(lambda: workspace.mark_plan_reviewed(stage_id))

        valid_paths: list[str] = []
        warnings: list[str] = []
        for path in artifact_paths or []:
            try:
                absolute_path = resolve_path_within_root(workspace.root, path)
            except WorkspaceError:
                warnings.append(f"Warning: artifact path escapes workspace root: {path}")
                continue

            if absolute_path.is_file():
                valid_paths.append(path)
            else:
                warnings.append(f"Warning: artifact not found: {path}")

        context = ReviewContext(
            stage_id=stage_id,
            artifact_paths=valid_paths,
        )
        user_response = user_agent.respond(summary, context)
        if warnings:
            user_response = user_response + "\n\n" + "\n".join(warnings)

        # Detect user finish signal: SimulatedUser includes [FINISH] when
        # all research goals have been addressed.
        if "[FINISH]" in user_response:
            return f"[FINISH_SIGNAL] {user_response}"

        # Nudge: if the stage has completed tasks but no conclusion,
        # remind the agent to save_conclusion before moving on.
        if stage_id is not None:
            state = workspace.get_state()
            stage = state.stages.get(stage_id)
            if (
                stage is not None
                and stage.task_ids
                and not stage.has_conclusion
            ):
                user_response += (
                    "\n\n[Note: This stage has completed tasks but no "
                    "conclusion yet. Call save_conclusion for this stage "
                    "before starting the next one.]"
                )

        return user_response

    @registry.register(
        name="finish_run",
        description="Signal the end of the current research workflow.",
        parameters={
            "final_summary": {
                "type": "string",
                "description": "Summary of the completed research workflow",
            },
        },
        terminal=True,
    )
    def finish_run(final_summary: str) -> str:
        """Terminal tool for the main agent loop."""

        return final_summary

    return registry


def create_sub_agent_tools(
    workspace: Workspace,
    stage_id: str,
    probe_result: RuntimeProbeResult,
    r_probe_result: RProbeResult | None = None,
) -> ToolRegistry:
    """Create the tool set for a sub agent.

    Sub agents only get execution tools — no project management tools.
    All write operations are scoped to stages/{stage_id}/outputs/.

    Tools:
    - read_file: Read any file in the workspace (same as main agent).
    - write_file: Write a file to the current stage's outputs directory.
    - run_code: Execute Python or R code in a subprocess whose cwd is the
      stage outputs directory. Files written by to_csv()/savefig()/ggsave()
      land directly in the workspace. Standard imports/library() calls are
      auto-prepended.
    - finish_task: Submit the structured task result and terminate the subagent loop.

    Args:
        workspace: Workspace manager instance.
        stage_id: Current stage — write_file and run_code are scoped to
            this stage's outputs directory.
        probe_result: Validated Python runtime description for subprocess
            execution and package-aware import preamble generation.
        r_probe_result: Optional validated R runtime description. When None,
            language="r" is structurally rejected by run_code.

    Returns:
        ToolRegistry populated with sub agent tools.
    """
    registry = ToolRegistry()

    # Ensure outputs directory exists for run_code cwd
    outputs_dir = workspace.root / "stages" / stage_id / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    def _run_workspace_operation(operation: Callable[[], str]) -> str:
        """Execute a workspace-backed tool operation with consistent error mapping."""

        try:
            return operation()
        except WorkspaceError as error:
            raise ToolError(str(error)) from error

    @registry.register(
        name="read_file",
        description=(
            "Read a file from the workspace by relative path. "
            "Small files are returned in full. "
            "Large data files (CSV/TSV/Excel >50KB) return a structured preview "
            "(shape, columns, dtypes, sample rows) — use run_code with pandas "
            "to analyze the full data."
        ),
        parameters={
            "path": {"type": "string", "description": "Workspace-relative file path"},
        },
    )
    def read_file(path: str) -> str:
        """Read file contents from the workspace."""

        return _run_workspace_operation(lambda: workspace.read_file(path))

    @registry.register(
        name="write_file",
        description="Write an artifact file into the current stage outputs directory.",
        parameters={
            "filename": {"type": "string", "description": "Artifact filename"},
            "content": {"type": "string", "description": "Artifact file content"},
        },
    )
    def write_file(filename: str, content: str) -> str:
        """Write a text artifact for the active stage."""

        def operation() -> str:
            relative_path = workspace.write_artifact(stage_id, filename, content)
            return f"File written: {relative_path}"

        return _run_workspace_operation(operation)

    # Build run_code description and language enum dynamically
    _r_available = r_probe_result is not None
    if _r_available:
        _run_code_description = (
            "Execute Python or R code and return stdout/stderr. "
            "Set language='python' or language='r'. "
            "A dynamic preamble is added for packages confirmed in the Runtime "
            "Environment section. "
            "The result shows the working directory and any new files created."
        )
        _language_enum = ["python", "r"]
    else:
        _run_code_description = (
            "Execute Python code and return stdout/stderr using the probed runtime. "
            "Only Python is supported. "
            "A dynamic import preamble is added for packages confirmed in the runtime "
            "environment section. "
            "The result shows the working directory and any new files created."
        )
        _language_enum = ["python"]

    @registry.register(
        name="run_code",
        description=_run_code_description,
        parameters={
            "code": {"type": "string", "description": "Source code to execute"},
            "language": {
                "type": "string",
                "description": f"Programming language: {' or '.join(_language_enum)}",
                "enum": _language_enum,
            },
        },
    )
    def run_code(code: str, language: str = "python") -> str:
        """Execute Python or R code in an isolated subprocess.

        - cwd is set to the stage outputs directory so file writes land in the workspace.
        - Standard imports (Python) or library() calls (R) are auto-prepended for
          packages confirmed by the probe.
        - After execution, any new files in the outputs directory are reported.
        """
        normalized_language = language.strip().lower() if language else "python"
        if normalized_language == "r":
            return _run_r_code(code, r_probe_result, outputs_dir, workspace)
        if normalized_language == "python":
            return _run_python_code(code, probe_result, outputs_dir, workspace)
        supported = "python or r" if _r_available else "python"
        raise ToolError(
            f"Unsupported language '{language}'. Use {supported}. "
            "Check the Runtime Environment section for available packages."
        )

    @registry.register(
        name="finish_task",
        description=(
            "Submit the final structured task result and terminate the subagent workflow. "
            "Use this instead of a plain-text completion."
        ),
        parameters={
            "summary": {
                "type": "string",
                "description": "Concise summary of what was completed and the key findings.",
            },
            "blockers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Remaining blockers that prevented full completion.",
            },
            "suggestions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Suggested follow-up actions for the main agent.",
            },
            "artifact_descriptions": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": (
                    "Map of filename to one-line description for each output file produced. "
                    "Example: {\"metrics.csv\": \"Model performance metrics for 3 classifiers\", "
                    "\"roc_curves.png\": \"ROC curves comparing models on test set\"}"
                ),
            },
        },
        terminal=True,
    )
    def finish_task(
        summary: str,
        blockers: list[str] | None = None,
        suggestions: list[str] | None = None,
        artifact_descriptions: dict[str, str] | None = None,
    ) -> str:
        """Return the structured completion payload for harness-side parsing."""

        completion = SubAgentCompletion(
            summary=summary,
            blockers=blockers or [],
            suggestions=suggestions or [],
            artifact_descriptions=artifact_descriptions or {},
        )
        return json.dumps(completion.model_dump())

    return registry


def _run_python_code(
    code: str,
    probe_result: RuntimeProbeResult,
    outputs_dir: Path,
    workspace: Workspace,
) -> str:
    """Execute Python code in an isolated subprocess."""

    files_before = _list_files(outputs_dir)
    full_code = _build_import_preamble(probe_result) + code
    try:
        result = subprocess.run(
            [probe_result.python_path, "-I", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=1200,
            check=False,
            cwd=str(outputs_dir),
            env=build_isolated_python_env(),
        )
    except subprocess.TimeoutExpired:
        return "Code execution timed out after 600 seconds."

    files_after = _list_files(outputs_dir)
    new_files = sorted(files_after - files_before)

    cwd_relative = str(outputs_dir.relative_to(workspace.root))
    output = (
        f"[cwd: {cwd_relative}/]\n"
        f"[python: {probe_result.python_path} ({probe_result.python_version})]\n"
        f"Exit code: {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    if "ModuleNotFoundError" in result.stderr:
        output += (
            "\n--- runtime packages ---\n"
            f"Available: {_format_available_packages(probe_result.available_packages)}\n"
            f"Missing: {_format_missing_packages(probe_result.missing_packages)}"
        )
    if new_files:
        listing = "\n".join(
            f"  {f} ({os.path.getsize(outputs_dir / f)} bytes)" for f in new_files
        )
        output += f"\n--- new files in outputs/ ---\n{listing}"

    return output


def _run_r_code(
    code: str,
    r_probe_result: RProbeResult | None,
    outputs_dir: Path,
    workspace: Workspace,
) -> str:
    """Execute R code in an isolated Rscript subprocess.

    Writes code to a temporary .R file (in system temp dir, not outputs),
    runs it with cwd=outputs_dir, then cleans up. This avoids shell
    quoting issues with Rscript -e for complex multi-line code.
    """
    if r_probe_result is None:
        raise ToolError(
            "R is not available in this environment. "
            "Use language='python' and rewrite your code in Python."
        )

    files_before = _list_files(outputs_dir)
    full_code = _build_r_library_preamble(r_probe_result) + code

    # Write to temp file (system temp dir — avoids polluting outputs/)
    script_fd, script_path = tempfile.mkstemp(suffix=".R")
    try:
        with os.fdopen(script_fd, "w", encoding="utf-8") as script_file:
            script_file.write(full_code)

        try:
            result = subprocess.run(
                [r_probe_result.rscript_path, "--vanilla", script_path],
                capture_output=True,
                text=True,
                timeout=1200,
                check=False,
                cwd=str(outputs_dir),
                env=build_isolated_r_env(),
            )
        except subprocess.TimeoutExpired:
            return "R code execution timed out after 600 seconds."
    finally:
        # Always clean up temp file
        try:
            os.unlink(script_path)
        except OSError:
            pass

    files_after = _list_files(outputs_dir)
    new_files = sorted(files_after - files_before)

    cwd_relative = str(outputs_dir.relative_to(workspace.root))
    output = (
        f"[cwd: {cwd_relative}/]\n"
        f"[R: {r_probe_result.rscript_path} ({r_probe_result.r_version})]\n"
        f"Exit code: {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    if "there is no package called" in result.stderr:
        output += (
            "\n--- R packages ---\n"
            f"Available: {_format_available_packages(r_probe_result.available_packages)}\n"
            f"Missing: {_format_missing_packages(r_probe_result.missing_packages)}"
        )
    if new_files:
        listing = "\n".join(
            f"  {f} ({os.path.getsize(outputs_dir / f)} bytes)" for f in new_files
        )
        output += f"\n--- new files in outputs/ ---\n{listing}"

    return output


def _list_files(directory: Path) -> set[str]:
    """List files in a directory tree as directory-relative paths."""

    if not directory.is_dir():
        return set()
    return {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }


def _build_import_preamble(probe_result: RuntimeProbeResult) -> str:
    """Build a Python preamble that only imports probe-confirmed packages."""

    lines = [
        "import json",
        "import math",
        "import os",
        "import sys",
    ]
    available = probe_result.available_packages

    if "numpy" in available:
        lines.append("import numpy as np")
    if "pandas" in available:
        lines.append("import pandas as pd")
    if "scipy" in available:
        lines.append("import scipy")
        lines.append("from scipy import stats")
    if "matplotlib" in available:
        lines.append("import matplotlib")
        lines.append("matplotlib.use('Agg')")
        lines.append("import matplotlib.pyplot as plt")
    if "seaborn" in available:
        lines.append("import seaborn as sns")
    if "statsmodels" in available:
        lines.append("import statsmodels.api as sm")
    if "sklearn" in available:
        lines.append("import sklearn")
    if "gseapy" in available:
        lines.append("import gseapy as gp")
    if "pydeseq2" in available:
        lines.append("import pydeseq2")
    if "adjustText" in available:
        lines.append("from adjustText import adjust_text")

    return "\n".join(lines) + "\n\n"


def _build_r_library_preamble(r_probe_result: RProbeResult) -> str:
    """Build an R preamble that loads probe-confirmed packages.

    Uses suppressPackageStartupMessages() to keep stderr clean from
    R's verbose package loading messages.
    """
    available = r_probe_result.available_packages
    if not available:
        return "options(warn = 1)\n\n"

    library_calls = "\n".join(
        f"  library({package})" for package in sorted(available)
    )
    return (
        "suppressPackageStartupMessages({\n"
        f"{library_calls}\n"
        "})\n"
        "options(warn = 1)\n\n"
    )


def _format_available_packages(available_packages: dict[str, str]) -> str:
    """Render available packages as a stable, human-readable list."""

    if not available_packages:
        return "(none)"
    return ", ".join(
        f"{package}=={version}"
        for package, version in sorted(available_packages.items())
    )


def _format_missing_packages(missing_packages: list[str]) -> str:
    """Render missing packages as a stable, human-readable list."""

    if not missing_packages:
        return "(none)"
    return ", ".join(sorted(missing_packages))
