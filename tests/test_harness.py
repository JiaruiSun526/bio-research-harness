"""Tests for the research harness composition root."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import build_stub_probe_result

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.harness import ResearchHarness
from research_agent.runtime_env import RuntimeProbeResult
from research_agent.schemas import AgentLoopResult, TaskResult, TaskSpec
from research_agent.workspace import Workspace, WorkspaceError


class DummyUserAgent:
    """Minimal user agent stub used by harness tests."""

    def respond(self, message: str, context: object | None = None) -> str:
        """Return a fixed response for escalation calls."""

        del message, context
        return "Acknowledged."


def make_workspace(tmp_path: Path) -> Workspace:
    """Create and initialize a real workspace for harness tests."""

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    return workspace


def make_harness(
    tmp_path: Path,
    *,
    llm_client: MagicMock | None = None,
    max_turns: int = 9,
    probe_result: RuntimeProbeResult | None = None,
) -> tuple[ResearchHarness, Workspace]:
    """Construct a harness bound to a real temporary workspace."""

    workspace = make_workspace(tmp_path)
    harness = ResearchHarness(
        model="test-model",
        workspace=workspace,
        user_agent=DummyUserAgent(),
        llm_client=llm_client,
        max_turns=max_turns,
        probe_result=probe_result or build_stub_probe_result(),
    )
    return harness, workspace


def prepare_approved_stage(workspace: Workspace, stage_id: str) -> None:
    """Create a stage with an approved plan so subagent dispatch is allowed."""

    workspace.write_plan(stage_id, "# Approved Plan")
    workspace.mark_plan_reviewed(stage_id)
    workspace.approve_plan(stage_id)


def test_run_returns_run_result(tmp_path: Path) -> None:
    """run returns a RunResult populated from AgentLoopResult plus workspace metrics."""

    harness, workspace = make_harness(tmp_path)
    workspace.write_plan("stage_01", "# Plan")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")
    workspace.write_conclusion("stage_01", "# Conclusion")

    with patch("research_agent.harness.run_agent_loop") as mock_run_agent_loop:
        mock_run_agent_loop.return_value = AgentLoopResult(
            stop_reason="completed",
            final_response="Finished.",
            tool_call_count=3,
            turn_count=2,
            prompt_tokens=120,
            completion_tokens=40,
        )

        result = harness.run()

    assert result.stop_reason == "completed"
    assert result.stages_completed == 1
    assert result.tool_call_count == 3
    assert result.turn_count == 2
    assert result.escalation_count == 0
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 40


def test_run_builds_correct_initial_messages(tmp_path: Path) -> None:
    """run generates the initial user message from the SimulatedUser, not from the caller."""

    harness, workspace = make_harness(tmp_path)
    workspace.write_plan("stage_01", "# Plan")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")

    captured_messages: list[dict[str, object]] = []

    def fake_run_agent_loop(**kwargs: object) -> AgentLoopResult:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        captured_messages.extend(messages)
        return AgentLoopResult(stop_reason="completed")

    with patch("research_agent.harness.run_agent_loop", side_effect=fake_run_agent_loop):
        harness.run()

    assert captured_messages[0]["role"] == "system"
    system_message = captured_messages[0]["content"]
    assert isinstance(system_message, str)
    assert "## Current Project State" in system_message
    assert "Current stage: stage_01" in system_message
    assert "stage_01 | plan_status=approved | plan_reviewed=True | stage_status=executing" in system_message
    # Initial user message comes from DummyUserAgent.respond()
    assert captured_messages[1] == {
        "role": "user",
        "content": "Acknowledged.",
    }


def test_dispatch_subagent_registers_task_and_returns_result(tmp_path: Path) -> None:
    """_dispatch_subagent registers the task, extracts artifacts, and persists the TaskResult."""

    harness, workspace = make_harness(tmp_path)
    prepare_approved_stage(workspace, "stage_01")

    task_spec = TaskSpec(
        task_id="task_001",
        stage_id="stage_01",
        task_description="Write the findings artifact.",
        max_turns=5,
    )

    def fake_run_agent_loop(**kwargs: object) -> AgentLoopResult:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        assert kwargs["trap_errors"] is True
        assert kwargs["require_terminal_tool"] is True
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"filename":"findings.md","content":"# Findings"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "File written: stages/stage_01/outputs/findings.md",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "finish_task",
                                "arguments": (
                                    '{"summary":"Task finished successfully.",'
                                    '"blockers":[],"suggestions":[]}'
                                ),
                            },
                        }
                    ],
                },
            ]
        )
        return AgentLoopResult(
            stop_reason="completed",
            final_response='{"summary": "Task finished successfully.", "blockers": [], "suggestions": []}',
            tool_call_count=2,
            turn_count=2,
        )

    with patch("research_agent.harness.run_agent_loop", side_effect=fake_run_agent_loop):
        result = harness._dispatch_subagent(task_spec)

    assert result == TaskResult(
        task_id="task_001",
        status="success",
        summary="Task finished successfully.",
        artifact_paths=["stages/stage_01/outputs/findings.md"],
        blockers=[],
        suggestions=[],
        error=None,
    )

    state = workspace.get_state()
    assert state.stages["stage_01"].task_ids == ["task_001"]

    result_path = workspace.root / "stages" / "stage_01" / "tasks" / "task_001_result.json"
    persisted_result = TaskResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    assert persisted_result == result


def test_dispatch_subagent_raises_on_invalid_completion_payload(tmp_path: Path) -> None:
    """Completed subagent runs must end with a valid structured completion payload."""

    harness, workspace = make_harness(tmp_path)
    prepare_approved_stage(workspace, "stage_01")
    task_spec = TaskSpec(
        task_id="task_bad_completion",
        stage_id="stage_01",
        task_description="Return a malformed terminal payload.",
    )

    with patch("research_agent.harness.run_agent_loop") as mock_run_agent_loop:
        mock_run_agent_loop.return_value = AgentLoopResult(
            stop_reason="completed",
            final_response="not-json",
        )

        with pytest.raises(ValueError, match="invalid completion payload"):
            harness._dispatch_subagent(task_spec)


def test_dispatch_subagent_maps_stop_reasons(tmp_path: Path) -> None:
    """_dispatch_subagent maps loop stop reasons into success, failure, and partial statuses."""

    stop_reason_to_status = {
        "completed": "success",
        "error": "failure",
        "max_turns_reached": "partial",
    }

    for index, (stop_reason, expected_status) in enumerate(stop_reason_to_status.items(), start=1):
        harness, workspace = make_harness(tmp_path / f"case_{index}")
        stage_id = f"stage_{index:02d}"
        prepare_approved_stage(workspace, stage_id)
        task_spec = TaskSpec(
            task_id=f"task_{index:03d}",
            stage_id=stage_id,
            task_description="Execute the requested research task.",
            max_turns=4,
        )

        with patch("research_agent.harness.run_agent_loop") as mock_run_agent_loop:
            mock_run_agent_loop.return_value = AgentLoopResult(
                stop_reason=stop_reason,
                final_response=(
                    '{"summary": "Completed task.", "blockers": [], "suggestions": []}'
                    if stop_reason == "completed"
                    else None
                ),
                error_message="RuntimeError: subagent failed" if stop_reason == "error" else None,
            )

            result = harness._dispatch_subagent(task_spec)

        assert result.status == expected_status
        expected_summary = "Completed task." if stop_reason == "completed" else "No response from sub agent."
        assert result.summary == expected_summary
        if stop_reason == "error":
            assert result.error == "RuntimeError: subagent failed"
        else:
            assert result.error is None


def test_dispatch_subagent_uses_role_based_prompt(tmp_path: Path) -> None:
    """_dispatch_subagent selects the system prompt based on task_spec.role."""

    from research_agent.prompts import SUB_AGENT_PROMPTS

    probe_result = build_stub_probe_result()
    harness, workspace = make_harness(tmp_path, probe_result=probe_result)
    prepare_approved_stage(workspace, "stage_01")

    task_spec = TaskSpec(
        task_id="task_viz",
        stage_id="stage_01",
        task_description="Create a volcano plot.",
        role="visualization",
    )

    captured_messages: list[dict[str, object]] = []

    def fake_run_agent_loop(**kwargs: object) -> AgentLoopResult:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        captured_messages.extend(messages)
        return AgentLoopResult(
            stop_reason="completed",
            final_response='{"summary": "Done.", "blockers": [], "suggestions": []}',
        )

    with patch("research_agent.harness.run_agent_loop", side_effect=fake_run_agent_loop):
        harness._dispatch_subagent(task_spec)

    system_msg = captured_messages[0]
    assert system_msg["role"] == "system"
    system_content = str(system_msg["content"])
    assert system_content.startswith(SUB_AGENT_PROMPTS["visualization"])
    assert "## Runtime Environment" in system_content
    assert probe_result.python_path in system_content
    assert "numpy==1.0.0" in system_content


def test_build_workspace_manifest_injects_data_outputs_and_task_summaries(
    tmp_path: Path,
) -> None:
    """_dispatch_subagent injects a workspace manifest with current stage context."""

    harness, workspace = make_harness(tmp_path)
    prepare_approved_stage(workspace, "stage_01")

    data_path = workspace.root / "data" / "raw.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text("gene,count\nA,1\n", encoding="utf-8")

    prior_output_path = workspace.root / "stages" / "stage_01" / "outputs" / "prior.txt"
    prior_output_path.parent.mkdir(parents=True, exist_ok=True)
    prior_output_path.write_text("existing output", encoding="utf-8")

    prior_result = TaskResult(
        task_id="task_prev",
        status="success",
        summary="Prepared baseline metrics.",
    )
    workspace.register_task("stage_01", "task_prev")
    workspace.write_task_result("stage_01", "task_prev", prior_result)

    captured_messages: list[dict[str, object]] = []
    task_spec = TaskSpec(
        task_id="task_manifest",
        stage_id="stage_01",
        task_description="Inspect the available workspace files.",
    )

    def fake_run_agent_loop(**kwargs: object) -> AgentLoopResult:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        captured_messages.extend(messages)
        return AgentLoopResult(
            stop_reason="completed",
            final_response='{"summary": "Done.", "blockers": [], "suggestions": []}',
        )

    with patch("research_agent.harness.run_agent_loop", side_effect=fake_run_agent_loop):
        harness._dispatch_subagent(task_spec)

    task_message = captured_messages[1]
    assert task_message["role"] == "user"
    assert "## Workspace Files" in str(task_message["content"])
    assert "data/raw.csv" in str(task_message["content"])
    assert "stages/stage_01/outputs/prior.txt" in str(task_message["content"])
    assert "task_prev [success]: Prepared baseline metrics." in str(task_message["content"])


def test_dispatch_subagent_system_prompt_overrides_role(tmp_path: Path) -> None:
    """TaskSpec.system_prompt takes priority over role-based template."""

    probe_result = build_stub_probe_result()
    harness, workspace = make_harness(tmp_path, probe_result=probe_result)
    prepare_approved_stage(workspace, "stage_01")

    custom_prompt = "You are a specialized bioinformatics agent."
    task_spec = TaskSpec(
        task_id="task_bio",
        stage_id="stage_01",
        task_description="Run DESeq2 analysis.",
        role="data_analyst",
        system_prompt=custom_prompt,
    )

    captured_messages: list[dict[str, object]] = []

    def fake_run_agent_loop(**kwargs: object) -> AgentLoopResult:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        captured_messages.extend(messages)
        return AgentLoopResult(
            stop_reason="completed",
            final_response='{"summary": "Done.", "blockers": [], "suggestions": []}',
        )

    with patch("research_agent.harness.run_agent_loop", side_effect=fake_run_agent_loop):
        harness._dispatch_subagent(task_spec)

    system_content = str(captured_messages[0]["content"])
    assert system_content.startswith(custom_prompt)
    assert "## Runtime Environment" in system_content
    assert probe_result.python_version in system_content


def test_dispatch_subagent_raises_on_invalid_prior_task_result(tmp_path: Path) -> None:
    """Corrupted persisted task results should fail loudly during manifest construction."""

    harness, workspace = make_harness(tmp_path)
    prepare_approved_stage(workspace, "stage_01")
    tasks_dir = workspace.root / "stages" / "stage_01" / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / "task_bad_result.json").write_text("{not-json}", encoding="utf-8")

    task_spec = TaskSpec(
        task_id="task_001",
        stage_id="stage_01",
        task_description="Inspect the current workspace.",
    )

    with pytest.raises(WorkspaceError, match="Invalid task result file"):
        harness._dispatch_subagent(task_spec)


def test_dispatch_subagent_collects_nested_run_code_artifacts(tmp_path: Path) -> None:
    """Artifacts created in nested output directories should be surfaced to the main agent."""

    harness, workspace = make_harness(tmp_path)
    prepare_approved_stage(workspace, "stage_01")
    task_spec = TaskSpec(
        task_id="task_nested",
        stage_id="stage_01",
        task_description="Create a nested artifact.",
        max_turns=5,
    )

    def fake_run_agent_loop(**kwargs: object) -> AgentLoopResult:
        outputs_dir = workspace.root / "stages" / "stage_01" / "outputs"
        nested_dir = outputs_dir / "plots"
        nested_dir.mkdir(parents=True, exist_ok=True)
        (nested_dir / "summary.txt").write_text("nested artifact", encoding="utf-8")
        return AgentLoopResult(
            stop_reason="completed",
            final_response=(
                '{"summary": "Nested artifact created.", '
                '"blockers": [], "suggestions": []}'
            ),
        )

    with patch("research_agent.harness.run_agent_loop", side_effect=fake_run_agent_loop):
        result = harness._dispatch_subagent(task_spec)

    assert result.artifact_paths == ["stages/stage_01/outputs/plots/summary.txt"]


def test_harness_probes_runtime_when_probe_result_not_provided(tmp_path: Path) -> None:
    """ResearchHarness auto-probes the runtime environment when probe_result is omitted."""

    workspace = make_workspace(tmp_path)
    probed_result = build_stub_probe_result(python_version="3.11.9")

    with patch("research_agent.harness.probe_environment", return_value=probed_result):
        harness = ResearchHarness(
            model="test-model",
            workspace=workspace,
            user_agent=DummyUserAgent(),
            llm_client=MagicMock(),
            max_turns=3,
        )

    assert harness.probe_result == probed_result


def test_escalation_count(tmp_path: Path) -> None:
    """run counts escalate_to_user tool calls from the main agent conversation."""

    harness, _ = make_harness(tmp_path)

    def fake_run_agent_loop(**kwargs: object) -> AgentLoopResult:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "Need user input.",
                    "tool_calls": [
                        {
                            "id": "esc_1",
                            "type": "function",
                            "function": {
                                "name": "escalate_to_user",
                                "arguments": (
                                    '{"summary":"Review plan","stage_id":"stage_01",'
                                    '"artifact_paths":""}'
                                ),
                            },
                        },
                        {
                            "id": "esc_2",
                            "type": "function",
                            "function": {
                                "name": "escalate_to_user",
                                "arguments": (
                                    '{"summary":"Confirm artifact","stage_id":"stage_01",'
                                    '"artifact_paths":"a.md"}'
                                ),
                            },
                        },
                    ],
                }
            ]
        )
        return AgentLoopResult(stop_reason="completed")

    with patch("research_agent.harness.run_agent_loop", side_effect=fake_run_agent_loop):
        result = harness.run("Request the missing approvals.")

    assert result.escalation_count == 2
