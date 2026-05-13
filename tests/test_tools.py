"""Tests for tool factory functions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import shutil

from conftest import build_stub_probe_result, build_stub_r_probe_result

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.runtime_env import RProbeResult, RuntimeProbeResult
from research_agent.schemas import ReviewContext, TaskResult, TaskSpec
from research_agent.tools import (
    _build_r_library_preamble,
    create_main_agent_tools,
    create_sub_agent_tools,
)
from research_agent.workspace import Workspace
from research_agent.workspace import WorkspaceError


def _make_task_result() -> TaskResult:
    """Build a representative TaskResult for dispatch tool tests."""

    return TaskResult(
        task_id="task_001",
        status="success",
        summary="Collected the key findings.",
        artifact_paths=["stages/stage_01/outputs/findings.md"],
        blockers=["Need final user confirmation."],
        suggestions=["Review the artifact before approval."],
    )


def test_create_main_agent_tools_returns_registry_with_seven_tools() -> None:
    """Main agent tool factory registers the expected seven tools."""

    registry = create_main_agent_tools(
        workspace=MagicMock(),
        user_agent=MagicMock(),
        dispatch_fn=MagicMock(),
    )

    definitions = registry.get_definitions()

    assert len(definitions) == 7
    assert [definition["function"]["name"] for definition in definitions] == [
        "read_file",
        "save_plan",
        "approve_plan",
        "dispatch_subagent",
        "save_conclusion",
        "escalate_to_user",
        "finish_run",
    ]


def test_read_file_tool_calls_workspace_read_file() -> None:
    """read_file delegates to Workspace.read_file and returns file contents."""

    workspace = MagicMock()
    workspace.read_file.return_value = "alpha\nbeta"
    registry = create_main_agent_tools(workspace, MagicMock(), MagicMock())

    result, is_error = registry.execute("read_file", {"path": "plans/stage_01_plan.md"})

    assert (result, is_error) == ("alpha\nbeta", False)
    workspace.read_file.assert_called_once_with("plans/stage_01_plan.md")


def test_save_plan_tool_calls_workspace_write_plan() -> None:
    """save_plan persists a draft plan and returns the status message."""

    workspace = MagicMock()
    registry = create_main_agent_tools(workspace, MagicMock(), MagicMock())

    result, is_error = registry.execute(
        "save_plan",
        {"stage_id": "stage_01", "content": "# Draft Plan"},
    )

    assert (result, is_error) == ("Plan saved to plans/stage_01_plan.md. Status: drafting.", False)
    workspace.write_plan.assert_called_once_with("stage_01", "# Draft Plan")


def test_approve_plan_tool_calls_workspace_approve_plan() -> None:
    """approve_plan delegates to Workspace.approve_plan."""

    workspace = MagicMock()
    registry = create_main_agent_tools(workspace, MagicMock(), MagicMock())

    result, is_error = registry.execute("approve_plan", {"stage_id": "stage_01"})

    assert (
        result,
        is_error,
    ) == ("Plan approved for stage_01. Status: approved. Ready for execution.", False)
    workspace.approve_plan.assert_called_once_with("stage_01")


def test_dispatch_subagent_tool_builds_task_spec_and_calls_dispatch() -> None:
    """dispatch_subagent passes structured arguments into a TaskSpec."""

    dispatch_fn = MagicMock(return_value=_make_task_result())
    registry = create_main_agent_tools(MagicMock(), MagicMock(), dispatch_fn)

    result, is_error = registry.execute(
        "dispatch_subagent",
        {
            "task_id": "task_001",
            "stage_id": "stage_01",
            "task_description": "Analyze the dataset for anomalies.",
            "max_turns": 12,
        },
    )

    assert is_error is False
    task_spec = dispatch_fn.call_args.args[0]
    assert isinstance(task_spec, TaskSpec)
    assert task_spec.task_id == "task_001"
    assert task_spec.stage_id == "stage_01"
    assert task_spec.task_description == "Analyze the dataset for anomalies."
    assert task_spec.role == "general"
    assert task_spec.max_turns == 12
    assert result == (
        "Task task_001 success.\n"
        "Summary: Collected the key findings.\n"
        "Artifacts: ['stages/stage_01/outputs/findings.md']\n"
        "Blockers: ['Need final user confirmation.']\n"
        "Suggestions: ['Review the artifact before approval.']"
    )


def test_dispatch_subagent_passes_role_to_task_spec() -> None:
    """dispatch_subagent forwards the role parameter into the TaskSpec."""

    dispatch_fn = MagicMock(return_value=_make_task_result())
    registry = create_main_agent_tools(MagicMock(), MagicMock(), dispatch_fn)

    result, is_error = registry.execute(
        "dispatch_subagent",
        {
            "task_id": "task_003",
            "stage_id": "stage_01",
            "task_description": "Create a volcano plot.",
            "role": "visualization",
        },
    )

    assert is_error is False
    task_spec = dispatch_fn.call_args.args[0]
    assert task_spec.role == "visualization"
    assert "Task task_003 success." in result


def test_dispatch_subagent_rejects_invalid_role() -> None:
    """dispatch_subagent should reject unsupported sub-agent roles."""

    dispatch_fn = MagicMock(return_value=_make_task_result())
    registry = create_main_agent_tools(MagicMock(), MagicMock(), dispatch_fn)

    result, is_error = registry.execute(
        "dispatch_subagent",
        {
            "task_id": "task_004",
            "stage_id": "stage_01",
            "task_description": "Do the work.",
            "role": "bioinformatics",
        },
    )

    assert is_error is True
    assert "role" in result
    dispatch_fn.assert_not_called()


def test_dispatch_subagent_rejects_invalid_max_turns_type() -> None:
    """dispatch_subagent surfaces invalid protocol arguments as a tool error."""

    dispatch_fn = MagicMock(return_value=_make_task_result())
    registry = create_main_agent_tools(MagicMock(), MagicMock(), dispatch_fn)

    result, is_error = registry.execute(
        "dispatch_subagent",
        {
            "task_id": "task_002",
            "stage_id": "stage_02",
            "task_description": "Summarize the literature review.",
            "max_turns": "not-an-int",
        },
    )

    assert is_error is True
    assert "max_turns" in result
    dispatch_fn.assert_not_called()


def test_save_conclusion_tool_calls_workspace_write_conclusion() -> None:
    """save_conclusion delegates to Workspace.write_conclusion."""

    workspace = MagicMock()
    registry = create_main_agent_tools(workspace, MagicMock(), MagicMock())

    result, is_error = registry.execute(
        "save_conclusion",
        {"stage_id": "stage_01", "content": "## Final conclusion"},
    )

    assert (
        result,
        is_error,
    ) == (
        "Conclusion saved for stage_01. "
        "Stage is now complete. Present your findings to the user via escalate_to_user.",
        False,
    )
    workspace.write_conclusion.assert_called_once_with("stage_01", "## Final conclusion")


def test_escalate_to_user_routes_to_user_agent_with_review_context(tmp_path: Path) -> None:
    """escalate_to_user filters missing artifacts and forwards ReviewContext."""

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    workspace.write_plan("stage_03", "# Plan")
    artifact_path = workspace.root / "stages" / "stage_03" / "report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("# Report", encoding="utf-8")
    user_agent = MagicMock()
    user_agent.respond.return_value = "Please revise section 2."
    registry = create_main_agent_tools(workspace, user_agent, MagicMock())

    result, is_error = registry.execute(
        "escalate_to_user",
        {
            "summary": "Need approval for the current draft.",
            "stage_id": "stage_03",
            "artifact_paths": ["stages/stage_03/report.md", "stages/stage_03/missing.csv"],
        },
    )

    assert is_error is False
    assert "Please revise section 2." in result
    assert "Warning: artifact not found: stages/stage_03/missing.csv" in result
    message, context = user_agent.respond.call_args.args
    assert message == "Need approval for the current draft."
    assert isinstance(context, ReviewContext)
    assert context.stage_id == "stage_03"
    assert context.artifact_paths == ["stages/stage_03/report.md"]
    assert workspace.get_state().stages["stage_03"].plan_reviewed is True


def test_approve_plan_requires_review_round_via_escalation_tool(tmp_path: Path) -> None:
    """Main-agent tools should structurally require a user-review round before approval."""

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    workspace.write_plan("stage_01", "# Plan")
    user_agent = MagicMock()
    user_agent.respond.return_value = "Looks good."
    registry = create_main_agent_tools(workspace, user_agent, MagicMock())

    result, is_error = registry.execute("approve_plan", {"stage_id": "stage_01"})

    assert is_error is True
    assert "reviewed with the user" in result

    review_result, review_error = registry.execute(
        "escalate_to_user",
        {"summary": "Please review.", "stage_id": "stage_01"},
    )
    assert review_error is False
    assert review_result == "Looks good."

    approve_result, approve_error = registry.execute("approve_plan", {"stage_id": "stage_01"})
    assert approve_error is False
    assert "Ready for execution" in approve_result


def test_escalate_to_user_rejects_artifact_paths_outside_workspace(tmp_path: Path) -> None:
    """escalate_to_user should not pass escaped paths into review context."""

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    user_agent = MagicMock()
    user_agent.respond.return_value = "Reviewed."
    registry = create_main_agent_tools(workspace, user_agent, MagicMock())

    result, is_error = registry.execute(
        "escalate_to_user",
        {
            "summary": "Please review.",
            "artifact_paths": ["../outside.txt"],
        },
    )

    assert is_error is False
    assert "escapes workspace root" in result
    _, context = user_agent.respond.call_args.args
    assert isinstance(context, ReviewContext)
    assert context.artifact_paths == []


def test_finish_run_returns_final_summary() -> None:
    """finish_run should terminate through a structured tool call payload."""

    registry = create_main_agent_tools(MagicMock(), MagicMock(), MagicMock())

    result, is_error = registry.execute(
        "finish_run",
        {"final_summary": "Research workflow finished."},
    )

    assert (result, is_error) == ("Research workflow finished.", False)


def test_workspace_error_is_converted_to_tool_error() -> None:
    """WorkspaceError is mapped into a tool error result by the registry."""

    workspace = MagicMock()
    workspace.read_file.side_effect = WorkspaceError("File not found: missing.txt")
    registry = create_main_agent_tools(workspace, MagicMock(), MagicMock())

    result, is_error = registry.execute("read_file", {"path": "missing.txt"})

    assert (result, is_error) == ("File not found: missing.txt", True)


def test_create_sub_agent_tools_returns_registry_with_three_tools(
    stub_probe_result: RuntimeProbeResult,
) -> None:
    """Sub agent tool factory registers the expected four tools."""

    registry = create_sub_agent_tools(MagicMock(), "stage_01", stub_probe_result)

    definitions = registry.get_definitions()

    assert len(definitions) == 4
    assert [definition["function"]["name"] for definition in definitions] == [
        "read_file",
        "write_file",
        "run_code",
        "finish_task",
    ]


def test_write_file_tool_calls_workspace_write_artifact(
    stub_probe_result: RuntimeProbeResult,
) -> None:
    """write_file writes into the current stage outputs directory."""

    workspace = MagicMock()
    workspace.write_artifact.return_value = "stages/stage_04/outputs/report.md"
    registry = create_sub_agent_tools(workspace, "stage_04", stub_probe_result)

    result, is_error = registry.execute(
        "write_file",
        {"filename": "report.md", "content": "# Report"},
    )

    assert (result, is_error) == ("File written: stages/stage_04/outputs/report.md", False)
    workspace.write_artifact.assert_called_once_with("stage_04", "report.md", "# Report")


def test_run_code_tool_executes_python_code_and_returns_output(
    tmp_path: Path,
    real_probe_result: RuntimeProbeResult,
) -> None:
    """run_code executes in stage outputs with probe-confirmed imports and file detection."""

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    registry = create_sub_agent_tools(workspace, "stage_01", real_probe_result)
    outputs_dir = workspace.root / "stages" / "stage_01" / "outputs"

    result, is_error = registry.execute(
        "run_code",
        {
            "code": (
                'print(os.getcwd())\n'
                'print(int(pd.Series([1, 2, 3]).sum()))\n'
                'pd.DataFrame({"value": [1, 2]}).to_csv("summary.csv", index=False)'
            ),
            "language": "python",
        },
    )

    assert is_error is False
    assert "Exit code: 0" in result
    assert "--- stdout ---" in result
    assert f"[python: {real_probe_result.python_path} ({real_probe_result.python_version})]" in result
    assert str(outputs_dir) in result
    assert "\n6\n" in result
    assert "--- stderr ---" in result
    assert "--- new files in outputs/ ---" in result
    assert "summary.csv" in result


def test_run_code_reports_available_and_missing_packages_on_import_error(
    tmp_path: Path,
) -> None:
    """run_code appends probe details when a module import fails at runtime."""

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    probe_result = build_stub_probe_result(
        available_packages={"numpy": "1.0.0"},
        missing_packages=["pandas", "scipy"],
    )
    registry = create_sub_agent_tools(workspace, "stage_01", probe_result)

    result, is_error = registry.execute(
        "run_code",
        {
            "code": "import totally_missing_package",
            "language": "python",
        },
    )

    assert is_error is False
    assert "ModuleNotFoundError" in result
    assert "--- runtime packages ---" in result
    assert "Available: numpy==1.0.0" in result
    assert "Missing: pandas, scipy" in result


def test_finish_task_returns_structured_completion_payload(
    stub_probe_result: RuntimeProbeResult,
) -> None:
    """finish_task should return a JSON payload the harness can parse deterministically."""

    registry = create_sub_agent_tools(MagicMock(), "stage_01", stub_probe_result)

    result, is_error = registry.execute(
        "finish_task",
        {
            "summary": "Analysis complete.",
            "blockers": ["Need user sign-off."],
            "suggestions": ["Review the generated CSV."],
        },
    )

    assert is_error is False
    assert json.loads(result) == {
        "summary": "Analysis complete.",
        "blockers": ["Need user sign-off."],
        "suggestions": ["Review the generated CSV."],
        "artifact_descriptions": {},
    }


# ── R language support tests ──


def test_run_code_rejects_r_when_r_unavailable(
    stub_probe_result: RuntimeProbeResult,
) -> None:
    """run_code with language='r' returns a tool error when R is not available."""

    workspace = MagicMock()
    workspace.root = Path("/fake/workspace")
    registry = create_sub_agent_tools(
        workspace, "stage_01", stub_probe_result, r_probe_result=None,
    )

    result, is_error = registry.execute("run_code", {"code": 'cat("hello")', "language": "r"})

    assert is_error is True
    assert "R is not available" in result


def test_run_code_tool_description_includes_r_when_available(
    stub_probe_result: RuntimeProbeResult,
    stub_r_probe_result: RProbeResult,
) -> None:
    """Tool description and language enum include R when r_probe_result is provided."""

    registry = create_sub_agent_tools(
        MagicMock(), "stage_01", stub_probe_result, stub_r_probe_result,
    )
    definitions = registry.get_definitions()
    run_code_def = next(d for d in definitions if d["function"]["name"] == "run_code")
    params = run_code_def["function"]["parameters"]["properties"]

    assert "r" in params["language"]["enum"]
    assert "R" in run_code_def["function"]["description"]


def test_run_code_tool_description_excludes_r_when_unavailable(
    stub_probe_result: RuntimeProbeResult,
) -> None:
    """Tool description and language enum exclude R when r_probe_result is None."""

    registry = create_sub_agent_tools(
        MagicMock(), "stage_01", stub_probe_result, r_probe_result=None,
    )
    definitions = registry.get_definitions()
    run_code_def = next(d for d in definitions if d["function"]["name"] == "run_code")
    params = run_code_def["function"]["parameters"]["properties"]

    assert params["language"]["enum"] == ["python"]
    assert "Only Python" in run_code_def["function"]["description"]


def test_run_code_rejects_unsupported_language(
    stub_probe_result: RuntimeProbeResult,
) -> None:
    """run_code rejects languages other than python/r."""

    workspace = MagicMock()
    workspace.root = Path("/fake/workspace")
    registry = create_sub_agent_tools(
        workspace, "stage_01", stub_probe_result, r_probe_result=None,
    )

    result, is_error = registry.execute("run_code", {"code": "code", "language": "julia"})

    assert is_error is True
    assert "Unsupported language" in result


def test_build_r_library_preamble_with_packages() -> None:
    """R preamble loads confirmed packages in suppressPackageStartupMessages."""

    r_probe = build_stub_r_probe_result(
        available_packages={"ggplot2": "3.4.0", "dplyr": "1.1.0"},
        missing_packages=[],
    )
    preamble = _build_r_library_preamble(r_probe)

    assert "suppressPackageStartupMessages" in preamble
    assert "library(dplyr)" in preamble
    assert "library(ggplot2)" in preamble
    assert "options(warn = 1)" in preamble


def test_build_r_library_preamble_empty_packages() -> None:
    """R preamble with no available packages just sets options."""

    r_probe = build_stub_r_probe_result(
        available_packages={}, missing_packages=["ggplot2"],
    )
    preamble = _build_r_library_preamble(r_probe)

    assert "suppressPackageStartupMessages" not in preamble
    assert "options(warn = 1)" in preamble


import pytest


@pytest.mark.skipif(not shutil.which("Rscript"), reason="Rscript not available")
def test_run_code_r_executes_and_returns_output(
    tmp_path: Path,
    real_probe_result: RuntimeProbeResult,
    real_r_probe_result: RProbeResult | None,
) -> None:
    """run_code(language='r') executes R code and captures stdout."""

    assert real_r_probe_result is not None, "Rscript found but probe failed"
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    registry = create_sub_agent_tools(
        workspace, "stage_01", real_probe_result, real_r_probe_result,
    )

    result, is_error = registry.execute(
        "run_code",
        {"code": 'cat("hello from R\\n")', "language": "r"},
    )

    assert is_error is False
    assert "Exit code: 0" in result
    assert "hello from R" in result
    assert f"[R: {real_r_probe_result.rscript_path}" in result


@pytest.mark.skipif(not shutil.which("Rscript"), reason="Rscript not available")
def test_run_code_r_writes_files_to_outputs_dir(
    tmp_path: Path,
    real_probe_result: RuntimeProbeResult,
    real_r_probe_result: RProbeResult | None,
) -> None:
    """R code writing files in cwd should be detected as new outputs."""

    assert real_r_probe_result is not None
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    registry = create_sub_agent_tools(
        workspace, "stage_01", real_probe_result, real_r_probe_result,
    )

    result, is_error = registry.execute(
        "run_code",
        {
            "code": 'write.csv(data.frame(x=1:3, y=4:6), "test_output.csv", row.names=FALSE)',
            "language": "r",
        },
    )

    assert is_error is False
    assert "test_output.csv" in result
    assert "new files in outputs" in result
